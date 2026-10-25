# Target interface

This suite evaluates **your own RAG project** — your own embedding model,
your own vector index, your own LLM API key or local SLM. The dataset
(ai4bharat/MSMARCO-XI) is the one fixed thing across everyone who runs
this suite; what varies is the target project under test. This document
is the full, exact contract your project needs to satisfy.

A real, working, tested-as-committed example that satisfies this contract
with the least possible code is in [`examples/minimal_target/`](examples/minimal_target/)
— copy it as your starting point.

## Required

Two Python modules, importable once your project's root is on `sys.path`
(which this suite arranges — nothing to configure for that part). By
default that means `app/embedder.py` and `app/generator.py`, but **this
suite verifies by importing the real module and checking real function
names on it — never by checking for an expected file path** (a filename
check is a proxy that breaks the moment your project is laid out
differently, and this suite shipped exactly that bug once already before
switching to real import-based verification). If your project doesn't
use an `app/` package — a flat `main.py` at the root, for instance — point
at it instead:

```bash
export EVAL_EMBEDDER_MODULE=main      # default: app.embedder
export EVAL_GENERATOR_MODULE=main     # default: app.generator
```

(Same module name twice is fine if both functions live in one file — see
below for a real, tested `main.py`-only example.)

### Embedder module (default `app.embedder`)

```python
def embed(texts: list[str]) -> "array-like, shape (len(texts), dim)": ...
def embed_one(text: str) -> "array-like, shape (dim,)": ...
def get_model(): ...  # called once; only its side effect (loading the model) matters
```

`embed_one`'s return value must support `.reshape(1, -1)` and `.shape[-1]`
(a NumPy array satisfies both trivially). The embedding dimension is
inferred empirically from a real call to `embed_one` — there's no
`EMBEDDING_DIM` config value to declare anywhere.

### Generator module (default `app.generator`)

```python
def generate_answer(query: str, results: list) -> "answer object": ...
```

Each item in `results` is a plain object with `.text: str` and
`.source: str` attributes (this suite builds its own; it does not import
any particular `SearchResult` class from your project, so your function
just needs to read those two attribute names off whatever it's given).

The returned answer object needs:

```
.text: str            -- the generated answer
.grounded: bool        -- did your system believe it had a real answer,
                          as opposed to declining / saying the context
                          doesn't cover the question? This drives the
                          "lying factor" reliability check (see
                          eval/checks/reliability.py) -- get this signal
                          right, since a generator that always reports
                          grounded=True can't ever be caught fabricating.
.generation_ms: float   -- wall-clock time your call took, for the
                          latency report
.model: str             -- a label for the report; any string is fine
```

That's the whole required surface. Nothing about FAISS, HNSW, chunking,
or any particular config file name is required — this suite builds and
chunks its own throwaway evaluation index using its own defaults (see
`eval/index_build.py`) and never touches your production index.

## Optional, with fallbacks

If your project has an `app/config.py`, this suite will read these
specific names from it *if they exist* — none are required, and a missing
one just falls back to a suite-owned default rather than erroring:

| Name | Used for | Fallback if absent |
|---|---|---|
| `GENERATION_BACKEND` | Auto-clamps `--workers` to 1 when it's exactly the string `"local"` (protects a single shared local-GPU model from concurrent-call contention) | No auto-clamp — pass `--workers 1` yourself if you're running a shared local model |
| `LATENCY_BUDGET_MS` | Retrieval latency budget shown in the report | `50` (override via `EVAL_RETRIEVAL_LATENCY_BUDGET_MS`) |
| `GENERATION_MODEL` / `LOCAL_GENERATION_MODEL` | Cosmetic "model" label in the report | `"unknown"` |

## What this suite needs *itself*, independent of your project

The judge (faithfulness + correctness checks) needs its own LLM
credential — `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` in the environment
this suite runs in. This is completely separate from whatever your
`generate_answer()` uses internally; see `eval/judge.py`'s docstring and
the README's "Judge credentials" section. Retrieval, reliability, and
latency checks don't need a judge credential at all.

## Verifying your target actually works

```bash
RAG_PROJECT_ROOT=/path/to/your-project python -m eval.runner --num-answerable 3 --num-unanswerable 3 --workers 1
```

A small, fast, cheap run. If your embedder/generator modules are wired
correctly, you'll get a real report (even a bad one — a low Recall number
from a genuinely weak embedder is a correct result, not a bug). If
something's missing, `eval.target.verify_target()` runs *before* anything
else (before even downloading the dataset) and fails with the exact
missing module or function name — never a bare stack trace — and each
check that needs a judge reports `SKIPPED` with a plain-English reason
(e.g. missing credential) rather than crashing the whole run.

This has been tested for real against three different layouts, not just
designed to work in theory: this suite's original `app/`-package target
project, [`examples/minimal_target/`](examples/minimal_target/) (same
layout, zero real logic), and a flat single-file `main.py` with
`EVAL_EMBEDDER_MODULE=main EVAL_GENERATOR_MODULE=main` and no `app/`
package at all. All three produced real, correctly-varying reports (a
random-embedding target scored `Recall@1: 0.000`; a real fine-tuned model
scored meaningfully higher) rather than the same canned numbers regardless
of what was actually under test.
