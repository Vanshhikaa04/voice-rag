"""Four chunking strategies, from naive baseline to structure/meaning aware.

fixed      - raw character windows (the "naive split" baseline; cuts mid-word)
sentence   - whole sentences packed up to a size budget, with sentence-level overlap
recursive  - split on the biggest natural boundary first (heading > paragraph > line > sentence > word)
semantic   - sentence packing that also breaks where adjacent sentences stop being similar
"""
import re
from dataclasses import dataclass

import numpy as np

from .embeddings import HashingEmbedder
from .text import split_sentences


@dataclass(frozen=True)
class Chunk:
    id: str
    doc: str
    title: str
    text: str
    strategy: str

    @property
    def embed_text(self) -> str:
        # contextual chunk: prefix the document title so "reset" inside the Hub doc means *Hub* reset
        return f"{self.title}. {self.text}"


def fixed(text: str, size: int, overlap: int) -> list[str]:
    text = " ".join(text.split())
    out, i = [], 0
    while i < len(text):
        j = min(i + size, len(text))
        out.append(text[i:j].strip())
        if j == len(text):
            break
        i = j - overlap
    return out


def by_sentence(text: str, size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    cur: list[str] = []
    n = 0
    for s in split_sentences(text):
        if cur and n + len(s) + 1 > size:
            chunks.append(" ".join(cur))
            carry, c = [], 0
            for t in reversed(cur):                 # re-use trailing sentences as overlap
                if c + len(t) > overlap:
                    break
                carry.insert(0, t)
                c += len(t) + 1
            cur, n = carry, c
        cur.append(s)
        n += len(s) + 1
    if cur:
        chunks.append(" ".join(cur))
    return chunks


_LEVELS = [r"\n(?=#{1,6} )", r"\n\s*\n", r"\n", r"(?<=[.!?])\s+", r"\s+"]


def _pieces(text: str, size: int, level: int = 0) -> list[str]:
    if len(text) <= size:
        return [text]
    if level >= len(_LEVELS):
        return [text[i : i + size] for i in range(0, len(text), size)]
    parts = [p for p in re.split(_LEVELS[level], text) if p.strip()]
    if len(parts) == 1:
        return _pieces(text, size, level + 1)
    out: list[str] = []
    for p in parts:
        out.extend(_pieces(p, size, level + 1) if len(p) > size else [p])
    return out


def recursive(text: str, size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    cur = ""
    for p in _pieces(text.strip(), size):
        p = p.strip()
        if cur and len(cur) + len(p) + 1 > size:
            chunks.append(cur)
            tail = cur[-overlap:] if overlap else ""
            tail = tail[tail.find(" ") + 1 :] if " " in tail else ""     # snap to a word boundary
            cur = (tail + "\n" + p).strip()      # newline keeps headings at the start of their line
        else:
            cur = (cur + "\n" + p).strip() if cur else p
    if cur:
        chunks.append(cur)
    return chunks


def semantic(text: str, size: int, overlap: int, percentile: float = 30) -> list[str]:
    sents = split_sentences(text)
    if len(sents) < 3:
        return sents
    emb = HashingEmbedder(512).fit(sents)
    v = emb.encode(sents)
    sims = (v[:-1] * v[1:]).sum(axis=1)             # sims[i] = similarity(sent i, sent i+1)
    cutoff = np.percentile(sims, percentile)
    chunks: list[str] = []
    cur: list[str] = []
    n = 0
    for i, s in enumerate(sents):
        topic_shift = i > 0 and sims[i - 1] <= cutoff and n >= size * 0.4
        if cur and (n + len(s) + 1 > size or topic_shift):
            chunks.append(" ".join(cur))
            cur, n = [], 0
        cur.append(s)
        n += len(s) + 1
    if cur:
        chunks.append(" ".join(cur))
    return chunks


STRATEGIES = {"fixed": fixed, "sentence": by_sentence, "recursive": recursive, "semantic": semantic}


def chunk_document(doc: str, text: str, strategy: str, size: int, overlap: int) -> list[Chunk]:
    first = next((l for l in text.splitlines() if l.strip()), doc)
    title = first.lstrip("# ").strip()
    pieces = STRATEGIES[strategy](text, size, overlap)
    return [Chunk(f"{doc}#{i}", doc, title, p, strategy) for i, p in enumerate(pieces) if p.strip()]
