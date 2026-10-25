"""Hybrid vector index: dense cosine similarity + BM25, fused, with per-hit confidence signals."""
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .chunking import Chunk, chunk_document
from .config import Config
from .embeddings import HashingEmbedder
from .text import tokens

K1, B = 1.5, 0.75


@dataclass
class Hit:
    chunk: Chunk
    dense: float        # raw cosine similarity (used for the confidence guardrail)
    bm25: float
    score: float        # fused ranking score in [0, 1]
    coverage: float     # idf-weighted share of the question's terms present in this chunk


class Index:
    def __init__(self, cfg: Config, embedder=None):
        self.cfg = cfg
        self.embedder = embedder or HashingEmbedder(cfg.dim)
        self.chunks: list[Chunk] = []

    # ---- build -------------------------------------------------------------------------------
    @classmethod
    def from_dir(cls, cfg: Config, embedder=None, strategy: str | None = None) -> "Index":
        idx = cls(cfg, embedder)
        chunks: list[Chunk] = []
        for path in sorted(Path(cfg.docs_dir).glob("*.md")):
            chunks += chunk_document(
                path.stem, path.read_text(encoding="utf-8"),
                strategy or cfg.chunk_strategy, cfg.chunk_size, cfg.chunk_overlap,
            )
        idx.build(chunks)
        return idx

    def build(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        texts = [c.embed_text for c in chunks]
        self.embedder.fit(texts)
        self.matrix = self.embedder.encode(texts)                   # (N, dim), unit rows

        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.term_sets: list[set[str]] = []
        lengths, df = [], Counter()
        for i, t in enumerate(texts):
            toks = tokens(t)
            lengths.append(len(toks))
            self.term_sets.append(set(toks))
            for term, f in Counter(toks).items():
                self.postings[term].append((i, f))
                df[term] += 1
        n = len(chunks)
        self.lengths = np.array(lengths, dtype=np.float32)
        self.avg_len = float(self.lengths.mean()) if n else 1.0
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        self.max_idf = math.log(1 + n)                               # weight for terms never seen

    def weight(self, term: str) -> float:
        return self.idf.get(term, self.max_idf)

    # ---- query -------------------------------------------------------------------------------
    def search(self, query: str, k: int | None = None) -> list[Hit]:
        k = min(k or self.cfg.top_k, len(self.chunks))
        q_terms = set(tokens(query))
        dense = np.maximum(self.matrix @ self.embedder.encode([query])[0], 0.0)

        bm = np.zeros(len(self.chunks), dtype=np.float32)
        for term in q_terms:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, f in self.postings[term]:
                bm[i] += idf * f * (K1 + 1) / (f + K1 * (1 - B + B * self.lengths[i] / self.avg_len))

        d_norm = dense / dense.max() if dense.max() > 0 else dense
        b_norm = bm / bm.max() if bm.max() > 0 else bm
        fused = self.cfg.alpha * d_norm + (1 - self.cfg.alpha) * b_norm

        top = np.argpartition(-fused, k - 1)[:k]
        top = top[np.argsort(-fused[top])]
        total_w = sum(self.weight(t) for t in q_terms) or 1.0
        hits = []
        for i in top:
            covered = sum(self.weight(t) for t in q_terms & self.term_sets[i])
            hits.append(Hit(self.chunks[i], float(dense[i]), float(bm[i]), float(fused[i]), covered / total_w))
        return hits
