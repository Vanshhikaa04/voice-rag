"""Minimal fake embedder for testing eval/target.py's interface contract --
deterministic hash-based vectors, no real model, no config.py needed."""
import hashlib

import numpy as np

DIM = 64


def _vec(text: str) -> np.ndarray:
    h = hashlib.sha256(text.encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "big"))
    v = rng.standard_normal(DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def embed_one(text: str) -> np.ndarray:
    return _vec(text)


def embed(texts: list[str]) -> np.ndarray:
    return np.vstack([_vec(t) for t in texts]) if texts else np.zeros((0, DIM), dtype=np.float32)


def get_model():
    return "fake-model"
