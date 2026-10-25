"""generate_answer for the eval suite: runs the full voicerag pipeline over the chunks the suite retrieved.

The suite's chunks are indexed into a throwaway voicerag Index, so the input/retrieval/output guardrails and
the optional LLM (OPENAI_API_KEY / ANTHROPIC_API_KEY from .env) are what get graded. `grounded` is True only
when the pipeline answered, False when it refused - that is what the "lying factor" check reads.
"""
from dataclasses import dataclass
import os

from voicerag.chunking import Chunk
from voicerag.config import Config
from voicerag.guardrails import Status
from voicerag.harness import VoiceRAG
from voicerag.index import Index

CFG = Config()


def _make_generator():
    try:
        if os.environ.get("OPENAI_API_KEY"):
            from voicerag.generator import OpenAIGenerator
            return OpenAIGenerator(timeout=30)
        if os.environ.get("ANTHROPIC_API_KEY"):
            from voicerag.generator import ClaudeGenerator
            return ClaudeGenerator(timeout=30)
    except Exception:
        pass
    return None


GEN = _make_generator()


@dataclass
class GeneratedAnswer:
    text: str
    grounded: bool
    generation_ms: float
    model: str


def generate_answer(query: str, results) -> GeneratedAnswer:
    chunks = [Chunk(f"r{i}", str(getattr(r, "source", "") or f"doc{i}"), str(getattr(r, "source", "") or f"doc{i}"),
                    r.text, "eval") for i, r in enumerate(results)]
    label = GEN.name if GEN else "extractive"
    if not chunks:
        return GeneratedAnswer("I couldn't find that in my documents, so I won't guess.", False, 0.0, label)
    idx = Index(CFG)
    idx.build(chunks)
    resp = VoiceRAG(CFG, idx, generator=GEN).ask(query)
    return GeneratedAnswer(resp.answer, resp.status == Status.ANSWERED, resp.total_ms, label)
