# rag-local-eval-loop

An evaluation loop for **your own RAG system** — retrieval quality,
hallucination rate, answer correctness, and a "lying factor" reliability
check, all sampled straight from the [ai4bharat/MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI)
dataset. No hand-written eval queries anywhere in this repo.

The dataset is the one fixed constant across everyone who runs this
suite; the RAG system under test is not — this repo doesn't ship a RAG
system of its own, and it isn't tied to any one project's specific stack
or file layout. It needs exactly two things from your project: an
`embed()`/`embed_one()` function and a `generate_answer()` function,
verified by actually **importing your real modules and checking real
function names on them** — not by checking for an expected file path, so
a flat `main.py` with no `app/` package at all works just as well as an
`app/embedder.py` + `app/generator.py` layout (point at it with
`EVAL_EMBEDDER_MODULE`/`EVAL_GENERATOR_MODULE`). See
[TARGET_INTERFACE.md](TARGET_INTERFACE.md) for the full, minimal contract,
and [`examples/minimal_target/`](examples/minimal_target/) for a real,
tested, working example with no vector database and no LLM API key at
all. It tests the *real* target system in-process — the real embedding
model, the real generation backend, whatever those are for you — not a
reimplementation of its logic.

## Methodology

The check design follows the reference-based vs. reference-free framing
and the LLM-as-a-judge technique from CampusX's *"LLM Eval Methods |
LLM-as-a-Judge | Reference Based Evals Vs Reference Free Evals"*
([video](https://youtu.be/uQFLY8rQVYA)). In honesty: what's confirmed here
is that video's title and the standard industry meaning of those terms —
YouTube's page didn't serve this tool a full transcript, so the five
checks below are this project's own application of that taxonomy to a RAG
system, not a line-by-line reproduction of the video's code. The mapping:

| Check | Kind | Ground truth source |
|---|---|---|
| Retrieval | **Reference-based** | MSMARCO-XI `passages.is_selected` labels |
| Correctness | **Reference-based**, via LLM-as-judge | MSMARCO-XI `Eng_Answer` |
| Faithfulness / hallucination | **Reference-free**, via LLM-as-judge | none shown to the judge — only the retrieved context |
| Reliability / "lying factor" | **Reference-based** (behavioral) | MSMARCO-XI answerable-vs-unanswerable split |
| Latency | Measurement, not a judgment call | — |

`eval/judge.py` implements the two LLM-as-judge calls. They always run
against a fixed OpenAI judge model, independent of whichever generation
backend the target system is using — grading a model with itself is a
known bias risk, so the judge never reuses the exact call that produced
the answer under test.

## Where the data comes from

Every query, candidate passage, and ground-truth answer in this suite
comes from a real MSMARCO-XI validation row — nothing is hand-written.
Verified directly (not assumed) by downloading and inspecting
`validation/hinval.parquet` with `pyarrow`:

- Each row: a query (English + one of 14 Indic languages), 10 candidate
  passages per language, an `is_selected` label per candidate, and
  `Eng_Answer`/`Answer` ground-truth answer text.
- A random 500-row sample split almost exactly 50/50 between rows that
  **have** a relevant candidate (`is_selected` contains a 1, real answer
  text) and rows that **don't** (`is_selected` all zero, `Eng_Answer` is
  the literal string `"No Answer Present."`). This isn't a data quality
  issue — MSMARCO-XI inherits it directly from the original MS MARCO
  dataset, which deliberately includes a large share of unanswerable
  queries.

This suite uses **both** buckets on purpose:

- **Answerable** rows drive the retrieval check (is the right passage
  found?) and the correctness check (does the answer match the reference?).
- **Unanswerable** rows are a built-in negative control: none of their 10
  candidates actually answer the query, so a well-behaved system should
  decline. If it answers anyway, that's a fabrication — see "lying
  factor" below.

`eval/dataset.py` (backed by `eval/msmarco.py`, this suite's own parquet
loader — not borrowed from the target) samples a fixed-seed (`--seed`,
default 42) N of each bucket. `eval/index_build.py` then builds a
**throwaway, in-memory FAISS index** from just the sampled examples'
candidate passages, using this suite's own chunking and HNSW parameters
(not the target's) — this suite never touches your project's actual
production index, and doesn't require you to be using FAISS at all in
your own retrieval path; only `embed()`/`embed_one()` are real calls into
your project.

The index is **mixed-language on purpose**: every candidate passage goes
in in both English and the Indic language, tagged by language. Retrieval
is graded two ways — "cross-lingual" (either language counts as a hit)
and "same-language only". This suite's original target project's
embedding model was specifically fine-tuned for Hindi+English cross-lingual
retrieval on this exact dataset, which is why cross-lingual recall is the
headline metric here — for a target that only handles one language, expect
the two variants to read identically (same-language recall is what
matters for you; cross-lingual just won't show any extra benefit).

Generation and the LLM-judge checks are **English-only**: the ground-truth
answers and the judge prompts are English, so English is the language
those checks can grade correctly without guessing at a Hindi judge
prompt's accuracy. Retrieval is graded bilingually; generation isn't —
see [Scope and limitations](#scope-and-limitations).

## What's checked

Five independent checks, each in `eval/checks/`:

1. **`retrieval.py`** — Recall@1/3/5 and MRR against `is_selected` ground
   truth, both cross-lingual and same-language variants.
2. **`faithfulness.py`** — reference-free LLM-judge: is every claim in the
   answer supported by the context that was actually retrieved for it?
   This is the hallucination measurement. Also computes **self-report
   precision**: of the answers the target system's own `grounded` flag
   marked as confident, how many did the judge independently confirm as
   faithful? (Restricted to `grounded=True` cases — a refusal is trivially
   "faithful" to any context, so including refusals would make the two
   signals disagree on every correct refusal for reasons unrelated to
   self-report accuracy.)
3. **`correctness.py`** — reference-based LLM-judge: does the answer
   convey MSMARCO-XI's actual `Eng_Answer`, for answerable queries only?
4. **`reliability.py`** — the **"lying factor"**: a 2×2 of ground-truth
   answerable-or-not against system-answered-or-not.
   - **False refusal rate**: answerable per the dataset, but the system
     declined. Lost information, not wrong information.
   - **False confidence rate**: unanswerable per the dataset (no candidate
     is relevant), but the system answered anyway. A fabrication with no
     basis in the retrieved evidence — the sharper failure of the two.
5. **`latency.py`** — embed/search/generation timing percentiles, against
   the target's `LATENCY_BUDGET_MS` if it declares one (optional — falls
   back to this suite's own default, `50ms`, otherwise) and this suite's
   own explicitly-labeled `GENERATION_LATENCY_TARGET_MS`.

## Architecture

```
eval/dataset.py      -- sample N answerable + N unanswerable rows from MSMARCO-XI
eval/index_build.py  -- build a throwaway mixed-language FAISS index from their candidates
        |
        v
eval/pipeline.py (Phase A)
  for each example: real retrieval (target's app.embedder + this index)
                     + real generation (target's app.generator.generate_answer)
  parallelized across examples via a thread pool
        |
        v
eval/checks/*.py (Phase B -- five checks, dispatched concurrently)
  retrieval.py   faithfulness.py   correctness.py   reliability.py   latency.py
        |
        v
eval/report.py  -- terminal gap-to-perfect report + results/<timestamp>.json
```

**Parallelism, honestly described**: Phase B runs all five checks as
concurrent futures — this is the "multiple parallel checking mechanisms"
this repo was built around. Two of them (`faithfulness`, `correctness`)
further parallelize their own judge calls across examples internally, and
*that* part is a real wall-clock win: each judge call is a blocking
network request, and network waits release Python's GIL, so multiple
in-flight requests genuinely overlap. `retrieval`, `reliability`, and
`latency` are pure aggregation over numbers Phase A already collected —
running them concurrently doesn't buy real speed, they're dispatched
through the same pool for architectural consistency, not because they're
a bottleneck.

Phase A's parallelism depends on your generation path. If it's a hosted
API call, concurrent workers are a real speedup (same reasoning as the
judge calls above). If it holds one model on one local GPU, concurrent
threads calling into it would contend for the same CUDA device with no
throughput gain and a real risk of GPU memory pressure from multiple
simultaneous KV caches — pass `--workers 1` for that case. If your
project declares `app.config.GENERATION_BACKEND = "local"` (this suite's
original target project's own convention — entirely optional, see
[TARGET_INTERFACE.md](TARGET_INTERFACE.md)), `eval/pipeline.py` detects it
and clamps to 1 worker automatically; otherwise it's on you to pass
`--workers 1` yourself if that applies to your setup.

## Setup

This suite needs your target project's exact runtime (it imports and
executes your embedder/generator modules in-process — same embedding
model, same generation backend, same dependencies those files need).
**Run it with the target project's own virtualenv Python** rather than
creating a new one. See [TARGET_INTERFACE.md](TARGET_INTERFACE.md) for
exactly what your project needs to provide.

### Option A: drop it into your project (simplest)

Copy the `eval/` folder and `run.sh`/`run.ps1` straight into your RAG
project's own root, then just run the script from inside your project —
no flags, no environment variables:

```bash
cp -r eval/ run.sh /path/to/your-project/
cd /path/to/your-project
./run.sh
```

It detects that `eval/` is sitting inside a real project (by actually
importing your embedder/generator modules — not by checking for a
particular filename, see [TARGET_INTERFACE.md](TARGET_INTERFACE.md)) and
just runs. Add flags the same way: `./run.sh --num-answerable 50`.

### Option B: keep it as a separate repo

Clone this repo next to your project instead, and point at it:

```powershell
.\run.ps1                                              # sibling ..\RAG, sensible defaults
.\run.ps1 --num-answerable 50 --num-unanswerable 50
.\run.ps1 --rag-root D:\path\to\your-project
```

```bash
./run.sh
./run.sh --num-answerable 50 --num-unanswerable 50
./run.sh --rag-root /path/to/your-project
```

Both scripts use the same resolution order: `--rag-root` flag →
`RAG_PROJECT_ROOT` env var → the script's own directory (Option A) → a
sibling `../RAG` directory (Option B). They only check that a
*directory* with a `.venv` exists — they deliberately don't check for any
particular file inside it (see [TARGET_INTERFACE.md](TARGET_INTERFACE.md)
for why). The real check — actually importing your embedder/generator
modules and verifying the required functions exist — happens once Python
starts, and fails with the exact missing module or function name if
something's wrong, before this
suite wastes any time downloading the dataset.

### Option C: the target isn't a Python module at all

Options A and B both assume you can `import` the target's embedder and
generator — fine for a Python project, not for a Node/TS service, a Go
binary, or anything else only reachable over its own HTTP API.
`eval/http_target.py` is a generic, config-driven adapter that satisfies
the exact same interface ([TARGET_INTERFACE.md](TARGET_INTERFACE.md)) by
calling a real running service instead of importing real code — same
downstream checks, same report, no changes anywhere else in this suite:

```powershell
$env:EVAL_EMBEDDER_MODULE = "eval.http_target"
$env:EVAL_GENERATOR_MODULE = "eval.http_target"
$env:EVAL_HTTP_CONFIG = "C:\path\to\your_target_config.json"
python -m eval.runner --num-answerable 3 --num-unanswerable 3 --workers 1 --rag-root C:\path\to\the\cloned\target-repo
```

```bash
export EVAL_EMBEDDER_MODULE=eval.http_target
export EVAL_GENERATOR_MODULE=eval.http_target
export EVAL_HTTP_CONFIG=/path/to/your_target_config.json
python -m eval.runner --num-answerable 3 --num-unanswerable 3 --workers 1 --rag-root /path/to/the/cloned/target-repo
```

(`--rag-root` here is only so the report records which project was under
test — this adapter itself doesn't read anything from that directory.)

See `eval/http_target.py`'s own docstring for the full config schema and
the three retrieval-check modes (`reembed`, `skip`, and why "read
ground-truth straight from the target's own API" is deliberately out of
scope for now). [`examples/http_target_configs/goarag.json`](examples/http_target_configs/goarag.json)
is a real config written against an actual Task #2 submission — a
Node/TS project with no Python at all, its embedder unreachable except by
reloading the public model it names in its own docs.

### Manual invocation

Equivalent to what the launcher does, if you'd rather run it yourself:

```bash
# from this repo's root
RAG_PROJECT_ROOT=/path/to/RAG  /path/to/RAG/.venv/bin/python -m eval.runner        # macOS/Linux
$env:RAG_PROJECT_ROOT="D:\path\to\RAG"; D:\path\to\RAG\.venv\Scripts\python.exe -m eval.runner   # PowerShell
```

### Judge credentials

The judge (`eval/judge.py`) auto-detects whichever real credential is
actually present — it doesn't assume OpenAI. Whatever credential (if any)
your own `app.generator.generate_answer()` needs internally is entirely
separate from what the *judge* needs — the judge always needs one of its
own, regardless of what your project's generation path does:

| Env var | Effect |
|---|---|
| `OPENAI_API_KEY` | judge uses OpenAI (`EVAL_JUDGE_MODEL_OPENAI`, default `gpt-5.4-mini`) |
| `ANTHROPIC_API_KEY` (or an `ant auth login` profile) | judge uses Anthropic (`EVAL_JUDGE_MODEL_ANTHROPIC`, default `claude-opus-5`) if no OpenAI key is set |
| `EVAL_JUDGE_PROVIDER=openai\|anthropic\|auto` | force a provider instead of auto-detecting (default `auto`) |

No live Anthropic key was available while building this suite — the
OpenAI path has been run end-to-end repeatedly (see below); the Anthropic
path is written directly from Anthropic's current API docs (verified
`output_config` JSON-schema shape, verified exception classes — including
a real, non-obvious one: the Anthropic SDK raises a bare `TypeError`, not
`AuthenticationError`, when literally no credential resolves at all, which
this suite catches explicitly) and its "no credentials configured" failure
path has been tested for real, but a genuine judge *call* with a live
Anthropic key has not. If you're the first to run it that way and
something's off, that's the part to check first.

Retrieval, reliability, and latency checks need neither key — only
`faithfulness` and `correctness` call the judge.

### CLI options

```
python -m eval.runner
  --num-answerable N     answerable rows to sample (default: 25)
  --num-unanswerable N   unanswerable rows to sample (default: 25)
  --top-k K              results retrieved per query (default: 5)
  --workers N             parallel workers for retrieval+generation (default: 6;
                           auto-clamped to 1 if the target declares
                           app.config.GENERATION_BACKEND == "local" -- optional,
                           set --workers 1 yourself if that applies but isn't declared)
  --judge-workers N       parallel workers per judge check (default: 8)
  --seed N                sampling seed (default: 42)
  --language CODE         MSMARCO-XI language code (default: hin)
  --split NAME             MSMARCO-XI split (default: validation)
  --rag-root PATH          path to the target RAG project
```

## A real run

The output below is real — copied verbatim from an actual run against the
target project (`GENERATION_BACKEND="local"`, `Qwen/Qwen3-0.6B`,
`--num-answerable 15 --num-unanswerable 15 --seed 42`), not
hand-constructed:

```
RAG Local Eval Loop -- results
======================================================================
Target project:     C:\Users\User\Pictures\RAG
Generation backend: local (Qwen/Qwen3-0.6B)
Dataset:            ai4bharat/MSMARCO-XI (hin, validation)
Sample:             15 answerable + 15 unanswerable (seed=42)
Index:              671 chunks (EN+HI) from 30 examples' candidates
top_k:              5

RETRIEVAL  (reference-based -- vs. MSMARCO-XI is_selected labels)
-----------------------------------------------------------------
  15 answerable queries evaluated

  cross-lingual (either language is a hit):
  Recall@1                      0.733  [##################......]  ideal 1.000  -26.7pp short
  Recall@3                      1.000  [########################]  ideal 1.000  PERFECT
  Recall@5                      1.000  [########################]  ideal 1.000  PERFECT
  MRR                           0.844  [####################....]  ideal 1.000  -15.6pp short

FAITHFULNESS / HALLUCINATION  (reference-free -- LLM-as-judge, no ground truth shown to judge)
----------------------------------------------------------------------------------------------
  30 answers evaluated
  Faithful rate                 0.633  [###############.........]  ideal 1.000  -36.7pp short
  Hallucination rate            0.367  [#########...............]  ideal 0.000  +36.7pp over
  Self-report precision         0.476  [###########.............]  ideal 1.000  -52.4pp short

CORRECTNESS  (reference-based -- LLM-as-judge vs. MSMARCO-XI Eng_Answer)
------------------------------------------------------------------------
  15 answerable-query answers evaluated
  Correct rate                  0.667  [################........]  ideal 1.000  -33.3pp short

RELIABILITY / "LYING FACTOR"  (should-answer vs. did-answer)
------------------------------------------------------------
  False refusal rate            0.267  [######..................]  ideal 0.000  +26.7pp over
  False confidence rate         0.667  [################........]  ideal 0.000  +66.7pp over

LATENCY
-------
  stage                 avg      p50      p95      p99   (ms)
  retrieval_total       6.84     6.53     8.69     8.72
  generation          1226.70  1208.13  2312.18  2418.05

  Retrieval  p95 8.69ms vs. 50ms budget  -> PASS
  Generation p95 2312.18ms vs. 1500.0ms target  -> OVER TARGET
```

(Recall@3/@5 both hit the retrieval index's ceiling — perfect — while
Recall@1, faithfulness, correctness, and both reliability rates all show
real, non-trivial gaps. That contrast is the report doing its job: it's
telling you retrieval isn't the bottleneck here, generation reliability
is — a conclusion a single blended "accuracy" number would have hidden.)

This suite has also been run, for real, against a *different* project
with zero shared code — [`examples/minimal_target/`](examples/minimal_target/),
a two-file fake embedder + generator with no config file and no LLM key
at all — specifically to verify the target interface decoupling actually
works and not just that it was designed to. See that example's own
README for the real output from that run.

## This suite already found a real bug

On its first real run against the target project, this suite's own
faithfulness/reliability checks flagged an example where the local model
declined to answer ("The provided documents *do not* contain information
about...") but the target project's own `grounded` flag reported `True`
(confident answer). Tracing it down: the target's refusal detector
(`app/generator.py`'s `_is_grounded()`) only matched the system prompt's
exact literal phrase — `"don't contain information"` — and the smaller
local model doesn't always reproduce that verbatim; it paraphrases
("do not contain information", "doesn't contain information", ...). The
single-string check missed the paraphrase and mislabeled a genuine refusal
as a confident answer — which would have shown up in the target
project's own dashboard, not just this suite's report.

Fixed in the target project by matching a negation + "contain information"
pattern instead of one fixed string (see that project's `app/generator.py`
git history). This is included here as the intended proof of concept: an
eval loop that only ever produces reassuring numbers isn't testing
anything.

## Interpreting results

Every rate metric is reported as **gap-to-perfect**, not gap-to-an-
arbitrary-bar picked by this suite: recall/faithful/correct rates compare
against an ideal of `1.000`; hallucination/false-refusal/false-confidence
rates compare against an ideal of `0.000`. `PERFECT` means the gap rounded
to zero at this sample size — not a guarantee it stays that way at scale.
The one place a real threshold exists is latency: retrieval is checked
against the target's declared budget if it has one (`LATENCY_BUDGET_MS`,
optional — falls back to `50ms`, override via
`EVAL_RETRIEVAL_LATENCY_BUDGET_MS`), and generation against this suite's
own explicitly labeled target (`GENERATION_LATENCY_TARGET_MS`, override
via `EVAL_GENERATION_LATENCY_TARGET_MS`) — those two get PASS/FAIL because
they're the only two with a stated bar to check against.

`results/<timestamp>.json` (gitignored) holds the full report, including
more flagged examples than the terminal shows (5 per check, vs. 3 printed).

## Scope and limitations

- **Generation/judge checks are English-only.** Retrieval is graded
  bilingually (see above), but the system prompt, both judge prompts, and
  MSMARCO-XI's ground-truth answers are all English — extending
  correctness/faithfulness grading to Hindi would need a separately
  designed and verified Hindi judge prompt, not a naive translation of the
  English one. Not implemented here.
- **Judge output isn't perfectly reproducible.** The judge model doesn't
  accept a non-default `temperature` (verified for this project's whole
  generation stack, not assumed), so repeated runs with the same seed can
  see faithfulness/correctness rates shift a few points between runs. The
  dataset sample itself is fully reproducible (fixed-seed); the judge's
  reading of it is not.
- **The local-GPU worker clamp only fires automatically if your project
  declares `app.config.GENERATION_BACKEND == "local"`** — that's this
  suite's original target project's own convention, not a general
  standard (see [TARGET_INTERFACE.md](TARGET_INTERFACE.md)). If your
  generation path holds one model on one GPU under a different config
  name or no config at all, the clamp won't see it — pass `--workers 1`
  yourself. When it does apply, don't route around it by editing
  `eval/pipeline.py`; it's a GPU contention/correctness concern, not a
  conservative default.
- **This suite grades whatever your project's `generate_answer()`
  currently does** — it doesn't run multiple configurations and compare
  them in one invocation. Re-run after changing your own project's
  generation setup to compare.
- **The isolated eval index doesn't match your project's real retrieval
  setup** — chunking, HNSW parameters, and even the choice of FAISS itself
  are this suite's own (see [Where the data comes from](#where-the-data-comes-from)).
  Only `embed()`/`embed_one()` are real calls into your project; if your
  production retrieval differs meaningfully from a plain HNSW index over
  your raw embeddings (e.g. reranking, hybrid search, metadata filtering),
  this suite's retrieval numbers reflect your embedding model's quality,
  not your full retrieval pipeline's.

## License

MIT — see [LICENSE](LICENSE).
