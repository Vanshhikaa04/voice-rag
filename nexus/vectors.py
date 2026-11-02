"""Qdrant access. Hybrid search lives here, and so does the permission filter: no caller can skip it."""
import threading
import uuid
import zlib
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client import models as m

from voicerag.text import tokens

from . import llm
from .config import settings
from .security import Principal

_write_lock = threading.Lock()
_DENSE, _SPARSE = "dense", "sparse"


@dataclass
class Hit:
    id: str
    document_id: str
    title: str
    url: str
    text: str
    score: float            # RRF rank score
    cosine: float           # true similarity to the query, used by the relevance gate
    coverage: float         # share of the query's terms that this chunk contains


@lru_cache
def client() -> QdrantClient:
    s = settings()
    if s.qdrant_url:
        return QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key or None)
    return QdrantClient(path=str(s.data_dir / "qdrant"))


def _chunks_name() -> str:
    e = llm.embedder()
    return f"chunks_{e.name}_{e.dim}"


def _memory_name() -> str:
    e = llm.embedder()
    return f"memories_{e.name}_{e.dim}"


@lru_cache
def _ensure() -> bool:
    c, dim = client(), llm.embedder().dim
    for name in (_chunks_name(), _memory_name()):
        if not c.collection_exists(name):
            c.create_collection(name, vectors_config={_DENSE: m.VectorParams(size=dim, distance=m.Distance.COSINE)},
                                sparse_vectors_config={_SPARSE: m.SparseVectorParams(modifier=m.Modifier.IDF)})
            if settings().qdrant_url:             # payload indexes are a no-op in embedded mode
                for field in ("workspace_id", "owner_id", "document_id", "visibility", "user_id"):
                    c.create_payload_index(name, field, m.PayloadSchemaType.KEYWORD)
    return True


def sparse(text: str) -> m.SparseVector:
    """Hashed term frequencies; Qdrant applies IDF at query time, which together gives BM25-style scoring."""
    counts = Counter(zlib.crc32(t.encode()) & 0x7fffffff for t in tokens(text))
    return m.SparseVector(indices=list(counts), values=[float(1 + (v - 1) * 0.3) for v in counts.values()])


def _point_id(key: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def _access(p: Principal) -> m.Filter:
    """Workspace-wide content plus this user's private content. Nothing else is ever searched."""
    return m.Filter(
        must=[m.FieldCondition(key="workspace_id", match=m.MatchValue(value=p.workspace_id))],
        should=[m.FieldCondition(key="visibility", match=m.MatchValue(value="workspace")),
                m.FieldCondition(key="owner_id", match=m.MatchValue(value=p.id))],
    )


def replace_document(doc, chunks: list[str]) -> None:
    """Atomic from the reader's view of one document: old chunks go, then new ones arrive."""
    _ensure()
    vecs = llm.embedder().encode([f"{doc.title}. {c}" for c in chunks], doc.workspace_id) if chunks else []
    points = [m.PointStruct(
        id=_point_id(f"{doc.id}:{i}"),
        vector={_DENSE: vecs[i].tolist(), _SPARSE: sparse(f"{doc.title}. {c}")},
        payload={"workspace_id": doc.workspace_id, "owner_id": doc.owner_id, "visibility": doc.visibility,
                 "document_id": doc.id, "title": doc.title, "url": doc.url, "text": c, "n": i},
    ) for i, c in enumerate(chunks)]
    with _write_lock:
        delete_document(doc.id)
        for i in range(0, len(points), 128):
            client().upsert(_chunks_name(), points[i:i + 128], wait=True)


def delete_document(document_id: str) -> None:
    _ensure()
    client().delete(_chunks_name(), m.FilterSelector(filter=m.Filter(must=[
        m.FieldCondition(key="document_id", match=m.MatchValue(value=document_id))])), wait=True)


def set_visibility(document_id: str, visibility: str) -> None:
    _ensure()
    client().set_payload(_chunks_name(), {"visibility": visibility}, points=m.Filter(must=[
        m.FieldCondition(key="document_id", match=m.MatchValue(value=document_id))]), wait=True)


def search(p: Principal, query: str, limit: int) -> list[Hit]:
    _ensure()
    dense = llm.embed_query(query, p.workspace_id).tolist()
    prefetch = [m.Prefetch(query=dense, using=_DENSE, limit=limit, filter=_access(p))]
    sp = sparse(query)
    if sp.indices:
        prefetch.append(m.Prefetch(query=sp, using=_SPARSE, limit=limit, filter=_access(p)))
    res = client().query_points(_chunks_name(), prefetch=prefetch, query=m.FusionQuery(fusion=m.Fusion.RRF),
                                limit=limit, with_payload=True, with_vectors=[_DENSE]).points
    terms = set(tokens(query))
    hits = []
    for pt in res:
        pl, v = pt.payload, pt.vector[_DENSE]
        cosine = sum(a * b for a, b in zip(dense, v)) / max(_norm(dense) * _norm(v), 1e-9)
        have = set(tokens(f"{pl['title']} {pl['text']}"))
        hits.append(Hit(str(pt.id), pl["document_id"], pl["title"], pl.get("url", ""), pl["text"], pt.score, cosine,
                        len(terms & have) / len(terms) if terms else 0.0))
    return hits


def _norm(v) -> float:
    return sum(x * x for x in v) ** 0.5


def upsert_memory(memory) -> None:
    _ensure()
    vec = llm.embedder().encode([memory.content], memory.workspace_id)[0].tolist()
    with _write_lock:
        client().upsert(_memory_name(), [m.PointStruct(
            id=_point_id(memory.id), vector={_DENSE: vec, _SPARSE: sparse(memory.content)},
            payload={"workspace_id": memory.workspace_id, "user_id": memory.user_id, "content": memory.content},
        )], wait=True)


def delete_memory(memory_id: str) -> None:
    _ensure()
    with _write_lock:
        client().delete(_memory_name(), m.PointIdsList(points=[_point_id(memory_id)]), wait=True)


def search_memories(p: Principal, query: str, limit: int = 4, floor: float = 0.3) -> list[str]:
    """Memories are private to one user, regardless of role."""
    _ensure()
    res = client().query_points(_memory_name(), query=llm.embed_query(query, p.workspace_id).tolist(), using=_DENSE, limit=limit,
                                query_filter=m.Filter(must=[
                                    m.FieldCondition(key="workspace_id", match=m.MatchValue(value=p.workspace_id)),
                                    m.FieldCondition(key="user_id", match=m.MatchValue(value=p.id))]),
                                with_payload=True).points
    return [pt.payload["content"] for pt in res if pt.score >= floor]
