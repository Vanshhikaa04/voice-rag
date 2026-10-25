"""Embedders. The default is a hashed TF-IDF embedder: no model download, ~microseconds per query.

Swap in SentenceTransformerEmbedder for true semantic vectors; the rest of the pipeline only needs
`fit(texts)` and `encode(texts) -> L2-normalised float32 matrix`.
"""
import zlib
from typing import Sequence

import numpy as np

from .text import tokens


class HashingEmbedder:
    """Unigram + bigram feature hashing with signed buckets and learned IDF weights."""

    def __init__(self, dim: int = 2048):
        self.dim = dim
        self.idf = np.ones(dim, dtype=np.float32)

    def _buckets(self, text: str) -> list[tuple[int, int]]:
        toks = tokens(text)
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        out = []
        for f in feats:
            h = zlib.crc32(f.encode())
            out.append((h % self.dim, 1 if (h >> 31) & 1 else -1))
        return out

    def fit(self, texts: Sequence[str]) -> "HashingEmbedder":
        df = np.zeros(self.dim, dtype=np.float32)
        for t in texts:
            for idx in {i for i, _ in self._buckets(t)}:
                df[idx] += 1
        self.idf = (np.log((len(texts) + 1) / (df + 1)) + 1).astype(np.float32)
        return self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        m = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, t in enumerate(texts):
            for idx, sign in self._buckets(t):
                m[row, idx] += sign
        m = np.sign(m) * np.log1p(np.abs(m)) * self.idf      # sublinear tf * idf
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        return m / np.maximum(norms, 1e-9)


class SentenceTransformerEmbedder:
    """Optional: pip install sentence-transformers. Same interface, real semantic vectors."""

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model)

    def fit(self, texts):
        return self

    def encode(self, texts):
        return self.model.encode(list(texts), normalize_embeddings=True).astype(np.float32)
