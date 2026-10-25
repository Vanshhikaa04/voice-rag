"""
Generic, config-driven target adapter for RAG projects that aren't a
Python module you can import in-process -- a Node/TS service, a Python
project behind its own HTTP API, anything reachable only over HTTP. Most
real submissions won't be a bare `app/embedder.py` + `app/generator.py`;
this is the reusable path for those, instead of hand-writing one-off shim
code per project.

This is not a special case bolted onto eval/target.py -- it satisfies the
*exact same* interface every native Python target does (TARGET_INTERFACE.md):
`embed`, `embed_one`, `get_model`, `generate_answer`. Point at it exactly
the way you'd point at any other target module:

    EVAL_EMBEDDER_MODULE=eval.http_target
    EVAL_GENERATOR_MODULE=eval.http_target
    EVAL_HTTP_CONFIG=/path/to/your_target_config.json

No changes to eval/target.py, eval/runner.py, or eval/pipeline.py are
needed -- this module just happens to live inside this suite's own `eval`
package rather than the target project's, so it's importable without
`--rag-root`/`RAG_PROJECT_ROOT` pointing anywhere in particular. (A
`--rag-root`/`RAG_PROJECT_ROOT` is still required by eval/target.py's own
resolution -- it just won't be *used* for anything by this module. Point
it at the target's cloned repo anyway, e.g. so a human reading the report
can see which project was under test.)

## Config file (EVAL_HTTP_CONFIG)

    {
      "generate": {
        "base_url": "http://localhost:8787",
        "path": "/api/query",
        "method": "POST",
        "body": {"query": "{{query}}", "stream": false, "options": {"retrievalOnly": false}},
        "timeout_s": 120,
        "answer_field": "answer",
        "model_field": "model",
        "generation_ms_field": "latency.total",
        "grounded_field": "status",
        "grounded_values": ["answered"]
      },
      "retrieval": {
        "mode": "reembed",
        "model_name": "intfloat/multilingual-e5-small",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "normalize": true
      }
    }

`generate.*_field` values are dotted paths into the JSON response
(`"latency.total"` reads `response["latency"]["total"]`). `body` is
walked recursively; every occurrence of the literal string `"{{query}}"`
is replaced with the real query text, at any depth/key.

`grounded_field` + `grounded_values`: the reliability/"lying factor" check
needs to know whether the target *believed* it had a real answer. Most
projects report this as some kind of status/confidence field already
(GoaRAG's `status: "answered" | "insufficient_context" | "low_confidence"
| "blocked"` is the example this was built against) -- read whichever
field is closest to that and list the values that count as "grounded".
If a project reports this as a plain boolean instead, still set
`grounded_field` to that field's dotted path and `grounded_values` to
`[true]`.

## Retrieval modes

Every real submission's generation gets tested faithfully this way --
`generate_answer` always calls the target's real, deployed endpoint.
Retrieval is the one check with no universal answer, because "give me a
raw embedding vector" is rarely a public endpoint on its own -- three
modes, pick based on what's actually knowable about the target:

  "reembed"  -- load a NAMED public embedding model directly (via
                sentence-transformers, already a dependency of this
                suite) and use its real vectors to build this suite's own
                throwaway index, exactly like a native Python target
                would. Needs the model name + whatever query/passage
                prefix convention it was trained with (check the
                target's own README/.env.example -- get this wrong and
                retrieval degrades silently, it won't error).
                Tests: the embedding model's own retrieval quality.
                Doesn't exercise the target's fusion/rerank layer, if it
                has one -- that only gets tested implicitly, through
                whatever `generate_answer` actually returns.

  "skip"     -- no usable embedding model to reload (custom-trained with
                no model card, embeddings genuinely only reachable
                through a paid third-party API you don't want to call
                per eval query, etc). embed()/embed_one() return
                small fixed-seed random vectors so the pipeline doesn't
                crash, but the retrieval numbers this produces are noise,
                not signal -- this prints a loud warning at runtime for
                exactly that reason. Every other check (faithfulness,
                correctness, reliability, latency) is completely
                unaffected, since none of them depend on embed().

  (a third option, reading ground-truth relevance straight out of a
  target's own retrieval-only response instead of re-embedding anything,
  would be the most faithful -- it'd exercise the target's *real*
  fusion+rerank, not just a reloaded base embedder -- but that's a
  fundamentally different code path than embed()+this suite's own
  throwaway index, and doesn't fit this module's job of satisfying the
  existing interface unmodified. Worth adding as a real eval/checks/
  retrieval.py extension if enough targets want it; out of scope here.)
"""

from __future__ import annotations

import json
import os
import time
import warnings
from functools import lru_cache
from typing import Any

import numpy as np
import requests

_CONFIG_PATH = os.environ.get("EVAL_HTTP_CONFIG")


@lru_cache(maxsize=1)
def _config() -> dict:
    if not _CONFIG_PATH:
        raise RuntimeError(
            "eval.http_target is loaded but EVAL_HTTP_CONFIG isn't set -- point it at your "
            "target's config JSON, e.g.:\n"
            '  $env:EVAL_HTTP_CONFIG = "C:\\path\\to\\goarag_config.json"   # PowerShell\n'
            "  export EVAL_HTTP_CONFIG=/path/to/goarag_config.json          # bash"
        )
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def _get_path(obj: Any, dotted: str) -> Any:
    for part in dotted.split("."):
        if obj is None:
            return None
        obj = obj.get(part) if isinstance(obj, dict) else getattr(obj, part, None)
    return obj


def _fill_template(node: Any, query: str) -> Any:
    if isinstance(node, str):
        return query if node == "{{query}}" else node
    if isinstance(node, dict):
        return {k: _fill_template(v, query) for k, v in node.items()}
    if isinstance(node, list):
        return [_fill_template(v, query) for v in node]
    return node


class Answer:
    def __init__(self, text: str, grounded: bool, generation_ms: float, model: str):
        self.text = text
        self.grounded = grounded
        self.generation_ms = generation_ms
        self.model = model


def get_model():
    """Called once by eval/target.py-adjacent code; only the side effect
    (warming up whatever's needed) matters. For "reembed" mode that's
    loading the sentence-transformers model once so per-call embed_one()
    isn't paying model-load cost per query."""
    cfg = _config().get("retrieval", {"mode": "skip"})
    if cfg.get("mode") == "reembed":
        return _embedder_model(cfg["model_name"])
    return None


@lru_cache(maxsize=4)
def _embedder_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def _reembed(texts: list[str], prefix: str, cfg: dict) -> np.ndarray:
    model = _embedder_model(cfg["model_name"])
    prefixed = [f"{prefix}{t}" for t in texts]
    return np.asarray(model.encode(prefixed, normalize_embeddings=cfg.get("normalize", True)))


_SKIP_WARNED = False


def _skip_vectors(texts: list[str]) -> np.ndarray:
    global _SKIP_WARNED
    if not _SKIP_WARNED:
        warnings.warn(
            "eval.http_target: retrieval mode is \"skip\" -- embed()/embed_one() are returning "
            "meaningless fixed-seed random vectors. The retrieval check's Recall@k/MRR numbers "
            "in this report are noise, not a real signal. Every other check (faithfulness, "
            "correctness, reliability, latency) is unaffected.",
            stacklevel=2,
        )
        _SKIP_WARNED = True
    dim = 16
    return np.stack(
        [np.random.RandomState(abs(hash(t)) % (2**32)).rand(dim).astype(np.float32) for t in texts]
    )


def embed(texts: list[str]) -> np.ndarray:
    cfg = _config().get("retrieval", {"mode": "skip"})
    if cfg.get("mode") == "reembed":
        return _reembed(texts, cfg.get("passage_prefix", ""), cfg)
    return _skip_vectors(texts)


def embed_one(text: str) -> np.ndarray:
    cfg = _config().get("retrieval", {"mode": "skip"})
    if cfg.get("mode") == "reembed":
        return _reembed([text], cfg.get("query_prefix", ""), cfg)[0]
    return _skip_vectors([text])[0]


def generate_answer(query: str, results: list) -> Answer:
    # `results` (this suite's own throwaway-index retrieval) is deliberately
    # unused here -- an HTTP target does its own real retrieval internally
    # as part of answering; this tests its real, deployed answer for this
    # query, not "how it behaves fed someone else's chunks."
    cfg = _config()["generate"]
    body = _fill_template(cfg["body"], query)

    start = time.perf_counter()
    resp = requests.request(
        cfg.get("method", "POST"),
        f"{cfg['base_url']}{cfg['path']}",
        json=body,
        timeout=cfg.get("timeout_s", 120),
    )
    resp.raise_for_status()
    parsed = resp.json()
    elapsed_ms = (time.perf_counter() - start) * 1000

    grounded_value = _get_path(parsed, cfg["grounded_field"])
    return Answer(
        text=_get_path(parsed, cfg["answer_field"]) or "",
        grounded=grounded_value in cfg.get("grounded_values", [True]),
        generation_ms=_get_path(parsed, cfg.get("generation_ms_field", "")) or elapsed_ms,
        model=_get_path(parsed, cfg.get("model_field", "")) or "unknown",
    )
