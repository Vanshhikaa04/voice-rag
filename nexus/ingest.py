"""Index one piece of content. Unchanged content is skipped by hash, so re-syncing costs nothing."""
import hashlib
import logging
from dataclasses import dataclass

from sqlalchemy.orm import Session

from voicerag.chunking import STRATEGIES

from . import graph, vectors
from .db import Document, now, uid

log = logging.getLogger("nexus.ingest")
CHUNK_SIZE, CHUNK_OVERLAP = 900, 120


@dataclass
class Item:
    source_key: str
    title: str
    text: str
    url: str = ""
    kind: str = "file"
    mime: str = ""


def worth_reading(doc: Document) -> bool:
    """Whether a document merits model calls for the graph and signals. Source files and one-line
    issue comments would cost one call each and fill the graph with noise such as usernames."""
    if doc.kind == "code":
        return False
    return doc.kind != "issue" or len(doc.text) >= 300


def chunk(text: str) -> list[str]:
    return [c for c in STRATEGIES["recursive"](text, CHUNK_SIZE, CHUNK_OVERLAP) if c.strip()]


def upsert(db: Session, workspace_id: str, owner_id: str, item: Item, *, visibility: str = "workspace",
           connector_id: str | None = None, enrich: bool = True) -> tuple[Document, str]:
    """Returns (document, "created" | "updated" | "unchanged")."""
    digest = hashlib.sha256(item.text.encode("utf-8", "replace")).hexdigest()
    key = item.source_key
    if len(key) > 512:                           # long URLs: stay unique within the column limit
        key = key[:440] + hashlib.sha256(key.encode()).hexdigest()
    doc = db.query(Document).filter_by(workspace_id=workspace_id, source_key=key).one_or_none()
    if doc and doc.content_hash == digest:
        return doc, "unchanged"
    state = "updated" if doc else "created"
    is_new = not doc
    if is_new:
        doc = Document(id=uid(), workspace_id=workspace_id, owner_id=owner_id, source_key=key, connector_id=connector_id)
    doc.title, doc.url, doc.kind, doc.mime = item.title[:500], item.url, item.kind, item.mime[:120]
    doc.visibility, doc.content_hash, doc.text, doc.updated_at = visibility, digest, item.text, now()
    pieces = chunk(item.text)
    doc.chunk_count = len(pieces)
    try:
        vectors.replace_document(doc, pieces)    # no SQL transaction is open here: embedding is slow and meters usage
    except Exception:
        db.rollback()
        raise
    if is_new:
        db.add(doc)
    db.commit()
    if enrich and worth_reading(doc):
        try:
            graph.extract(db, doc)
        except Exception as e:                   # the graph is an enrichment; the document stays searchable
            log.warning("graph extraction skipped for %s: %s", doc.id, e)
    return doc, state


def remove(db: Session, doc: Document) -> None:
    vectors.delete_document(doc.id)
    db.delete(doc)
    db.commit()
