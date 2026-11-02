# NEXUS AI: implementation plan

## What exists and how it is treated

| Existing | Decision |
|---|---|
| `voicerag/` chunking, guardrails, generators, text utilities (11 tests) | **Kept and reused** by `nexus/`. Only change: Devanagari support in `text.py`. |
| `server.py`, `web/index.html`, CLI | **Untouched**; keep working as the legacy console. |
| `eval-loop/` | **Untouched**. |

## Architecture

```
frontend/ (Next.js 16)  --/api rewrite-->  nexus/ (FastAPI)
                                             |- auth, workspaces, roles
                                             |- ingest -> chunk -> embed -> Qdrant (dense + sparse)
                                             |- chat: rewrite > retrieve > rerank > gate > stream > verify
                                             |- agents: LangGraph orchestrator + approvals
                                             |- scheduler: connector sync > proactive insights
                                             '- llm: provider-independent gateway, usage metering
Stores: SQL (SQLite or Postgres) | Qdrant (embedded or server) | Redis (optional)
```

## Stack decisions

| Preference | Decision | Why |
|---|---|---|
| PostgreSQL | SQLAlchemy; `DATABASE_URL` selects Postgres, default SQLite | App stays runnable with no services installed. Compose uses Postgres. |
| Qdrant | Used everywhere; embedded mode by default, `QDRANT_URL` for server | One retrieval code path, tested here without Docker. Native dense + sparse RRF. |
| Redis | Optional cache and rate-limit store; in-memory fallback | Nothing requires it for a single process. |
| Neo4j | **Not used**: entity and relation tables in SQL | Queries are 1-2 hop lookups. A fourth datastore is not justified yet. |
| LangGraph | Used for the orchestrator | As requested. |
| Background work | In-process scheduler | No Celery until one process is too small. |

## Status

| Stage | State |
|---|---|
| 1. Foundation: config, DB, LLM gateway, metering | Done, tested live |
| 2. Auth, roles, tenancy, ingestion, core UI | Done, tested live |
| 3. Multimodal parsing, hybrid RAG, citations, voice | Done, tested live on OpenAI. Images and Sarvam/ElevenLabs not yet run live |
| 4. Agents, memory, knowledge graph | Done, tested live |
| 5. Workflows, builder, signals, Research and Developer modes | Done, tested live. Conflict detection is single-neighbour |
| 6. Security, analytics, evaluation, Docker | Done except Docker, which could not be run here |

## Next, in priority order

1. Run the Docker stack; run the Postgres and Qdrant server paths. Add Alembic before the first real deployment: `create_all` cannot alter tables.
2. Live-verify Slack, Google Drive, Gmail and Sarvam with real credentials.
3. Move rate limits and OAuth state to Redis so the API can run more than one worker.
4. Streaming TTS for faster spoken replies; a calibration set of real Hindi and Hinglish questions.
5. Graph entity merging; Organization Digital Twin views on top of the graph.
6. Langfuse and OpenTelemetry export; S3-style cloud storage connector.
7. Computer-use agent, last: it needs a sandbox and its own approval model.

## Honesty rules

- A connector with no credentials shows "not configured"; it never shows sample data.
- No LLM configured means extractive answers (labelled) and agents refuse to run.
- Metrics come only from recorded events. Unmeasured values show "not measured", never zero.
- Unknown model prices show as "unpriced", not $0.
