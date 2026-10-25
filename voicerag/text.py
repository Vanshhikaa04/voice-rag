"""Tiny, dependency-free text utilities shared by every stage."""
import re

STOPWORDS = frozenset(
    "a an the of to in on at for and or but is are was were be been being do does did can could should "
    "would will may might i you he she it we they my your our their this that these those with from by as "
    "if then than so not no what which who whom when where why how about into over under up down out off "
    "again just also there here have has had get got me us them its please tell say said "
    # question scaffolding: says how the user asks, not what they ask about
    "long many much often need want like know work works without after before".split()
)

_WORD = re.compile(r"[a-z0-9]+")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_MD_PREFIX = re.compile(r"^\s*(#{1,6}\s+|[-*]\s+|\d+\.\s+)")


def stem(word: str) -> str:
    """Crude suffix stripper: good enough to match reset/resetting, device/devices."""
    if word.endswith("ss"):
        return word
    for suffix in ("ing", "ed", "es", "s", "ly"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            if len(word) > 3 and word[-1] == word[-2] and word[-1] not in "ls":
                word = word[:-1]            # resett -> reset
            break
    if len(word) > 3 and word.endswith("e"):
        word = word[:-1]                    # device -> devic (matches devices)
    return word


def tokens(text: str) -> list[str]:
    """Lower-cased, stop-word-free, stemmed content terms."""
    return [stem(w) for w in _WORD.findall(text.lower()) if w not in STOPWORDS]


def split_sentences(text: str) -> list[str]:
    """Sentence splitter that also treats markdown lines/bullets as their own units."""
    out: list[str] = []
    for line in text.splitlines():
        line = _MD_PREFIX.sub("", line).strip()
        if line:
            out.extend(s.strip() for s in _SENT_SPLIT.split(line) if s.strip())
    return out
