from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Config:
    docs_dir: Path = ROOT / "data" / "docs"
    eval_file: Path = ROOT / "data" / "eval.json"

    # chunking
    chunk_strategy: str = "recursive"      # fixed | sentence | recursive | semantic
    chunk_size: int = 420                  # characters
    chunk_overlap: int = 60

    # retrieval
    dim: int = 2048                        # hashed embedding width
    top_k: int = 4
    alpha: float = 0.6                     # weight of dense score vs BM25 in fusion

    # guardrails
    min_dense: float = 0.03                # weak floor: raw cosine overlaps between in/out-of-scope with hashed vectors
    min_coverage: float = 0.25             # idf-weighted share of question terms found in a chunk (OOS<=0.10, in-scope>=0.41)
    min_stt_confidence: float = 0.35
    min_grounding: float = 0.60            # share of answer terms that must appear in context
    max_question_chars: int = 500

    # harness
    retry_attempts: int = 3
    retry_base_delay: float = 0.05         # seconds, doubled each retry (+ jitter)
    latency_budget_ms: float = 200.0
