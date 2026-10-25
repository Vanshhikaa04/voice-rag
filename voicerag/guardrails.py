"""Guardrails: decide when NOT to answer. Three checkpoints around the pipeline.

1. input      - empty/unclear speech, low STT confidence, prompt-injection attempts, absurd length
2. retrieval  - is the evidence strong enough? (cosine + idf-weighted term coverage)
3. output     - is every answer sentence actually supported by the retrieved context?
Plus: retrieved chunks that themselves contain instructions are dropped (indirect injection).
"""
import re
from dataclasses import dataclass
from enum import Enum

from .config import Config
from .text import split_sentences, tokens


class Status(str, Enum):
    ANSWERED = "answered"
    NO_QUERY = "refused_no_query"              # silence / garbled / too short
    UNSAFE = "refused_unsafe"                  # prompt injection etc.
    LOW_CONFIDENCE = "refused_low_confidence"  # question not covered by the knowledge base
    UNGROUNDED = "refused_ungrounded"          # generator said something the context doesn't support
    ERROR = "error"


@dataclass
class Verdict:
    ok: bool
    status: Status | None = None
    reason: str = ""


_INJECTION = re.compile(
    r"ignore (all |any |the )?(previous|prior|above|earlier) (instructions|prompts?|rules)"
    r"|disregard (all |any |the )?(previous|prior|above|your)"
    r"|(reveal|show|print|repeat) (me )?(your|the) (system )?(prompt|instructions)"
    r"|you are now\b|pretend (to be|you are)|\bjailbreak\b|developer mode|act as (an? )?(unrestricted|dan)\b",
    re.I,
)

OK = Verdict(True)


def looks_like_injection(text: str) -> bool:
    return bool(_INJECTION.search(text))


def check_input(text: str, stt_confidence: float | None, cfg: Config) -> Verdict:
    if len(text) > cfg.max_question_chars:
        return Verdict(False, Status.UNSAFE, "question too long")
    if looks_like_injection(text):
        return Verdict(False, Status.UNSAFE, "prompt-injection pattern in question")
    if len(tokens(text)) < 1:
        return Verdict(False, Status.NO_QUERY, "no content words in transcript")
    if stt_confidence is not None and stt_confidence < cfg.min_stt_confidence:
        return Verdict(False, Status.NO_QUERY, f"low speech-recognition confidence ({stt_confidence:.2f})")
    return OK


def filter_hits(hits):
    """Indirect-injection defence: never let instructions hidden in documents reach the generator."""
    return [h for h in hits if not looks_like_injection(h.chunk.text)]


def check_retrieval(hits, cfg: Config) -> Verdict:
    if not hits:
        return Verdict(False, Status.LOW_CONFIDENCE, "nothing retrieved")
    best_dense = hits[0].dense
    best_cov = max(h.coverage for h in hits)
    if best_dense < cfg.min_dense:
        return Verdict(False, Status.LOW_CONFIDENCE, f"top similarity {best_dense:.2f} < {cfg.min_dense}")
    if best_cov < cfg.min_coverage:
        return Verdict(False, Status.LOW_CONFIDENCE, f"question term coverage {best_cov:.2f} < {cfg.min_coverage}")
    return OK


def grounding_score(answer: str, hits) -> float:
    """Average share of each answer sentence's content terms that occur in the retrieved context."""
    context = set()
    for h in hits:
        context |= set(tokens(h.chunk.text))
    scores = []
    for s in split_sentences(re.sub(r"\[\d+\]", "", answer)):
        terms = set(tokens(s))
        if terms:
            scores.append(len(terms & context) / len(terms))
    return sum(scores) / len(scores) if scores else 0.0


def check_output(answer: str, hits, cfg: Config) -> Verdict:
    g = grounding_score(answer, hits)
    if g < cfg.min_grounding:
        return Verdict(False, Status.UNGROUNDED, f"grounding {g:.2f} < {cfg.min_grounding}")
    return OK
