# Minimal target example

The smallest possible thing this suite can evaluate: two files, no
`app/config.py`, no vector database, no LLM API key, no shared code with
any particular RAG project. `app/embedder.py` returns deterministic
hash-based vectors (no real semantic meaning -- proves the plumbing, not
retrieval quality); `app/generator.py` echoes back the first retrieved
chunk instead of calling an LLM.

This is exactly what got run to verify eval/target.py's interface
decoupling actually works, not just what it was designed to do in theory:

```bash
cd rag-local-eval-loop
RAG_PROJECT_ROOT=./examples/minimal_target python -m eval.runner --num-answerable 3 --num-unanswerable 3
```

Real, verified output from that run: `Recall@1: 0.000` (correct -- random
embeddings carry no real retrieval signal, so this isn't supposed to
score well) and `False confidence rate: 1.000` (also correct -- this fake
generator always answers, so a well-designed reliability check should
flag every unanswerable query it was asked). Both numbers being exactly
what a broken-on-purpose target *should* produce is the actual point: it
shows the checks are measuring something real, not returning canned
numbers regardless of what's under test.

Copy `app/embedder.py` and `app/generator.py` as your starting point --
replace the fake logic with real calls to your own embedding model and
your own LLM API key or local SLM, keeping the same function signatures.
See `../../TARGET_INTERFACE.md` for the full contract these two files
need to satisfy.
