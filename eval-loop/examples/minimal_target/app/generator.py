"""Minimal fake generator for testing eval/target.py's interface contract --
no real LLM call, just proves the plumbing (duck-typed context objects in,
duck-typed answer object out) works with zero config.py and zero API key."""
import time
from dataclasses import dataclass


@dataclass
class GeneratedAnswer:
    text: str
    grounded: bool
    generation_ms: float
    model: str


def generate_answer(query: str, results) -> GeneratedAnswer:
    t0 = time.perf_counter()
    if not results:
        return GeneratedAnswer(
            text="The provided documents don't contain information about this.",
            grounded=False,
            generation_ms=0.0,
            model="fake-echo-generator",
        )
    # Not a real LLM -- just echoes the first retrieved chunk's text, proving
    # results[i].text / .source were passed through correctly.
    text = f"(fake answer) {results[0].text[:120]} (source: {results[0].source})"
    return GeneratedAnswer(
        text=text,
        grounded=True,
        generation_ms=(time.perf_counter() - t0) * 1000,
        model="fake-echo-generator",
    )
