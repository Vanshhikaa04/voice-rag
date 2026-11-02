# NEXUS AI

A workspace intelligence platform: connect your sources, ask in text or voice (English, Hindi, Hinglish), delegate work to
agents, and get notified when something important changes. It is built on the Voice RAG core documented further down.

## Quick start (no services needed)

Needs Python 3.12+ and Node 22+. SQLite, embedded Qdrant and an in-process cache are the defaults, so nothing else has to be installed.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-nexus.txt
cd frontend; npm install; cd ..
copy .env.example .env       # then add OPENAI_API_KEY or ANTHROPIC_API_KEY
.\dev.ps1                    # open http://localhost:3000 and create a workspace
```

With no model key it still works: it indexes files and answers by quoting them, and agents refuse to run with a clear message.

## Production with Docker

Set `NEXUS_SECRET_KEY`, `POSTGRES_PASSWORD` and `NEXUS_PUBLIC_URL` in `.env`, then run `docker compose up --build`.
This uses PostgreSQL, a Qdrant server and Redis. **The API must run with one worker**, because the scheduler and rate limits are in-process.

## Credentials

| Feature | What you need | Without it |
|---|---|---|
| Generated answers, agents, graph, signals | `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` | Quoted answers; those features are off |
| Understanding paraphrases and Hindi | `OPENAI_API_KEY`, or `NEXUS_EMBEDDING_PROVIDER=local` | Keyword-style matching only |
| Speech | `SARVAM_API_KEY`, `ELEVENLABS_API_KEY` + `ELEVENLABS_VOICE_ID`, or `OPENAI_API_KEY` | Local Whisper for input; no spoken replies |
| Websites, public GitHub repos | Nothing | |
| Private GitHub repos, comments | Personal access token, entered in Sources | |
| Slack | Bot token with `channels:read`, `channels:history`, `chat:write`; invite the bot to each channel | |
| Google Drive, Gmail | `NEXUS_GOOGLE_CLIENT_ID` and `NEXUS_GOOGLE_CLIENT_SECRET`, with redirect URI `{NEXUS_PUBLIC_URL}/api/connectors/oauth/callback` | Shown as "Not configured" |
| PostgreSQL as a source | Connection string for a read-only role (admins only) | |

## What is verified

Tested against live services: OpenAI chat, embeddings, Whisper and TTS; public GitHub; website crawling; a real browser including a
microphone turn and barge-in. `tests/test_nexus.py` (75 tests) runs through the real API, SQL and embedded Qdrant with model vendors stubbed.

| Reference measurement (OpenAI, this machine) | Result |
|---|---|
| Retrieval, 18 labelled questions | hit@1 94%, hit@4 100%, MRR 0.97 |
| Latency over 10 mixed queries | P50 2.3 s, P70 2.9 s, P100 7.0 s; first words P50 1.3 s |

Written against provider documentation but **not yet run against the live service**: Slack, Google Drive, Gmail, Sarvam, ElevenLabs, Anthropic,
image understanding, the PostgreSQL source, and the Docker stack with its Postgres, Qdrant server and Redis. The Postgres schema is compile-checked only.
Slack, Gmail and Sarvam have tests against mock responses, which only prove our handling of the API as documented.

## How it works

```
frontend/ Next.js  --/api-->  nexus/ FastAPI
  chat:     guard > rewrite > translate > hybrid search > rerank > relevance gate > stream > verify
  agents:   LangGraph supervisor > workers > verification; side effects only through approvals
  sync:     connector > parse > chunk > embed > Qdrant, then graph and signals
```

| Module | Responsibility |
|---|---|
| `nexus/rag.py` | Answer pipeline, language handling, verification, fallbacks |
| `nexus/vectors.py` | Qdrant dense + sparse search; the permission filter lives here |
| `nexus/llm.py` | Provider gateway, failover, usage and cost metering |
| `nexus/connectors.py` | Website, GitHub, Slack, Drive, Gmail, PostgreSQL, and sync |
| `nexus/agents.py` | Supervisor, agents, templates, documents and slides, approved actions |
| `nexus/graph.py`, `memory.py`, `proactive.py`, `analytics.py`, `voice.py` | One feature each |
| `nexus/security.py` | Auth, encryption of credentials, throttling, SSRF guard |

**Decisions.** Neo4j is not used: the graph needs one-hop lookups, which SQL tables handle without a fourth datastore. Celery is not used: an
in-process scheduler is enough for one server. The Data Analyst applies a declarative plan and never executes model-written code.

**Security model.** Self-signup creates an isolated workspace; joining one needs an admin. Search filters by workspace and visibility inside
`vectors.py`, so agents inherit the requesting user's permissions. Credentials are encrypted at rest. Agents cannot cause side effects:
the Action Agent only creates an approval, and `execute_approval` is the single exit. Sensitive operations are written to an audit log.

## Not built yet

Computer-use agent, cloud storage connectors (S3 and similar), other database engines, Langfuse/OpenTelemetry export, token-by-token TTS
streaming, OCR for scanned PDFs, per-document permissions mirrored from the source system (access is set per connector or document), and
entity merging in the graph ("Atlas" and "Project Atlas" are currently two nodes).

Roadmap detail: [docs/NEXUS_PLAN.md](docs/NEXUS_PLAN.md).

---

# Voice-Enabled RAG (core library)

Speak a question, get a grounded, cited answer, or an honest "I don't know".

```
 mic ─► [1 STT] ─► [2 input guard] ─► [3 retrieve] ─► [4 retrieval guard] ─► [5 generate] ─► [6 output guard] ─► answer
         Whisper     injection,          hybrid          is the evidence       extractive or    is every sentence
         + confidence  low-confidence    dense+BM25      strong enough?        Claude           supported by context?
                       speech            over chunks
 └──────────────── harness: timing per stage · retries · fallbacks · structured RagResponse · never raises ─────────────┘
```

## Run it

```bash
pip install numpy                      # core: everything below works with just this
pip install faster-whisper sounddevice # voice input
pip install anthropic                  # optional: Claude writes the answer (set ANTHROPIC_API_KEY)

python -m voicerag ask "How do I factory reset the hub?"
python -m voicerag voice                       # speak into the mic (Enter, then talk)
python -m voicerag voice --wav question.wav    # or transcribe a file
python -m voicerag --llm ask "..."             # Claude generator, extractive fallback on failure
python -m voicerag bench --rounds 200          # P50/P70/P100 latency
python -m voicerag eval                        # chunking comparison, guardrail accuracy, threshold sweep
python -m unittest discover -s tests
```

## Measured results (this machine: 12-core CPU, no GPU)

| Measurement | Result |
|---|---|
| Text-in → answer-out, 5,200 samples | **P50 0.64 ms · P70 0.71 ms · P100 9.3 ms** (budget 200 ms: PASS) |
| Full voice path incl. Whisper `tiny.en` (3 s questions) | **P50 457 ms · P70 468 ms · P100 513 ms** (budget 200 ms: **FAIL**) |
| Of which Whisper | 455 ms (retrieval + generation + guardrails total < 2 ms) |
| Guardrails on the eval set | 18/18 answerable questions answered, 16/18 judged correct by strict phrase match, **8/8 out-of-scope questions refused** |

**The honest reading:** everything except speech recognition runs in about a millisecond.
On a CPU, Whisper needs ~400 ms for a 3 s clip (its encoder always processes a padded 30 s window), so
a *batch* "record, then transcribe" voice pipeline cannot be under 200 ms here. Ways to actually get there:
a GPU (`device="cuda"`, typically 30-80 ms), a streaming recogniser that transcribes while you are still
speaking so only the final fragment is on the critical path, or measuring the budget from end-of-speech.
The text stages stay well inside budget whichever you choose.

## What each file does, and why

### `text.py`: shared text utilities
Tokenising, stop-words, a crude suffix stemmer (so *resetting* matches *reset*, *devices* matches *device*) and a
sentence splitter that understands markdown bullets and headings. One shared definition of "a term" is
what lets BM25, embeddings, coverage and grounding checks agree with one another.
It also stop-lists *question scaffolding* ("how many", "how long", "work") because those words describe how
you ask, not what you ask about, and would otherwise look like missing evidence.

### `chunking.py`: four strategies, compared with data
Retrieval quality is mostly decided here. A chunk that cuts the answer in half can never be retrieved whole.
- **fixed**: raw 420-character windows. The naive baseline; cuts mid-word and mid-fact.
- **sentence**: packs whole sentences up to the size budget, re-using trailing sentences as overlap.
- **recursive**: tries the biggest natural boundary first (heading → paragraph → line → sentence → word) and only goes finer when a piece is still too big. Respects document structure.
- **semantic**: sentence packing that also starts a new chunk where adjacent sentences stop being similar (a topic shift).
Every chunk is also *contextualised*: the document title is prefixed before embedding, so "reset" in the
Hub guide is understood as a Hub reset.

`python -m voicerag eval` result (18 questions, did the top-4 contain the full evidence phrase):

| strategy | hit@1 | hit@4 | MRR |
|---|---|---|---|
| fixed | 0.72 | 0.94 | 0.83 |
| sentence | 0.83 | 1.00 | 0.92 |
| **recursive** | **0.94** | **1.00** | **0.97** |
| semantic | 0.78 | 1.00 | 0.88 |

Recursive wins on this corpus because the docs are well-structured markdown. Semantic chunking would
benefit more from real embeddings and unstructured prose. Re-run the eval on your own documents before choosing.

### `embeddings.py`: vectors without a model download
`HashingEmbedder` hashes unigrams and bigrams into 2,048 signed buckets, applies sublinear tf × learned IDF,
and L2-normalises. It is deterministic, needs no network, and encodes a query in microseconds, which is what makes the
latency budget achievable. Trade-off: it is *lexical*, so it cannot connect "month" to "30 days". That is
the cause of the two remaining eval misses. `SentenceTransformerEmbedder` has the same interface for true semantic
vectors (costs ~5-20 ms per query on CPU, so still inside the budget).

### `index.py`: hybrid retrieval
Each query is scored two ways. **Dense**: cosine similarity (one matrix-vector product). **BM25**: classic
keyword scoring with an inverted index. Both are max-normalised and fused (`0.6·dense + 0.4·bm25`), because
dense is good at fuzzy matches and BM25 at exact tokens such as `8443` or `Z-Wave`. Each hit also carries a
**coverage** score: the IDF-weighted share of the question's terms that appear in that chunk. Terms that never
occur in the corpus get the *maximum* weight, so asking about "Alexa" or "price" when the docs never mention
them produces very low coverage. That is the main out-of-scope signal.

### `stt.py`: voice in
`record_until_silence` captures the microphone (16 kHz mono) and stops 0.7 s after you finish speaking, using a
simple energy threshold. `WhisperTranscriber` runs `faster-whisper` locally (`tiny.en`, int8, greedy
decoding, warmed up at start) and returns the text **plus a confidence** (exp of mean token log-probability),
which the guardrails use to reject mumbled input. `load_wav` handles files.

### `generator.py`: answers
- **ExtractiveGenerator** (default) scores every sentence in the retrieved chunks by IDF-weighted overlap with the question, keeps the best one or two (a second one only if nearly as relevant), restores reading order, and appends `[n]` citations. It only *quotes*, so it cannot hallucinate, and takes ~0.3 ms.
- **ClaudeGenerator** (`--llm`) writes a short spoken-style answer constrained to numbered passages, told to say "I don't know" when unsure and to treat passages as untrusted data. It costs a network round trip, so it is opt-in and always backed by the extractive fallback.

### `guardrails.py`: knowing when *not* to answer
1. **Input**: rejects empty/garbled transcripts, low STT confidence (< 0.35), over-long input, and prompt-injection phrases ("ignore all previous instructions…").
2. **Retrieval**: refuses if question-term coverage < 0.25 or cosine < 0.03. Chunks that themselves contain instructions are dropped before generation (*indirect* prompt injection via documents).
3. **Output**: a grounding score checks that ≥ 60 % of each answer sentence's content terms exist in the retrieved context. A hallucinating LLM answer fails here and is replaced by the extractive answer.
Each refusal has its own status (`refused_no_query`, `refused_unsafe`, `refused_low_confidence`, `refused_ungrounded`) and the low-confidence reply names the closest topic.

**How thresholds were chosen:** by measurement, not guesswork. On the eval set, out-of-scope questions scored coverage ≤ 0.10 and answerable ones ≥ 0.41, so 0.25 sits in the gap. Cosine similarity overlapped between the two groups (0.11 vs 0.07), so it is only a weak floor. `eval` prints a threshold sweep so you can re-tune. Caveat: this is a small, hand-written eval set; re-calibrate on yours.

### `harness.py`: the runtime around it all
- **Structured I/O**: input is text or audio; output is always a `RagResponse` (status, answer, sources, confidence signals, per-stage timings, degraded flag, trace id) that serialises to JSON.
- **Retries**: `with_retry` gives exponential backoff with jitter on STT (2 attempts) and generation (3 attempts).
- **Error recovery**: generator failure/timeouts → extractive fallback, flagged `degraded=true`; ungrounded LLM output → extractive fallback; any unexpected exception → `status=error` with a friendly message. `ask()` never raises.
- **Observability**: every stage is timed; one JSON log line per request.

### `benchmark.py`: P50 / P70 / P100
Warm-up first, then every eval query (answerable and unanswerable, so refusal paths are timed too) repeated
`--rounds` times. Reports per-stage and total percentiles, and checks the **worst case (P100)** against the budget, not the mean.
Pass `--wav a.wav b.wav` to include real STT.

### `tests/`
11 tests: chunk invariants, answer/refuse/injection/garbled-speech paths, retry behaviour, generator outage → fallback,
hallucinating generator caught, harness never raises, and a quality floor on the eval set.

## Known limits
- Lexical embeddings miss paraphrases (two eval misses: "month" vs "30 days"; one padded answer). Use `SentenceTransformerEmbedder` or `--llm`.
- Mic capture uses a fixed energy threshold; noisy rooms need a real VAD.
- The knowledge base is a small fictional product manual. Put your own `.md` files in `data/docs/` and edit `data/eval.json`.
- No text-to-speech: answers are printed, not spoken (e.g. `pyttsx3` would be the next step).
