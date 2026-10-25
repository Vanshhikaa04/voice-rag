"""embed / embed_one / get_model for the eval suite.

Default: this project's own hashed TF-IDF embedder (no download, English tokens only).
Set VOICERAG_EVAL_EMBEDDER=e5 for multilingual-e5-small, which also covers the Indic passages.
"""
import os

import numpy as np

from voicerag.config import Config
from voicerag.embeddings import HashingEmbedder, SentenceTransformerEmbedder

_E5 = os.environ.get("VOICERAG_EVAL_EMBEDDER", "hash").lower() == "e5"
_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformerEmbedder("intfloat/multilingual-e5-small") if _E5 else HashingEmbedder(Config().dim)
    return _model


def embed(texts: list[str]) -> np.ndarray:
    return get_model().encode([("passage: " + t) if _E5 else t for t in texts])


def embed_one(text: str) -> np.ndarray:
    return get_model().encode([("query: " + text) if _E5 else text])[0]
