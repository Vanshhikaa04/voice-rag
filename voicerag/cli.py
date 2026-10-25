import argparse
import logging
import os

from . import benchmark as bm
from .config import Config
from .evaluate import compare_chunking, evaluate_guardrails, sweep_thresholds
from .harness import VoiceRAG
from .index import Index


def _table(rows: list[dict]) -> None:
    cols = list(rows[0])
    print("  ".join(f"{c:<14}" for c in cols))
    for r in rows:
        print("  ".join(f"{str(r[c]):<14}" for c in cols))


def build(args, need_stt=False) -> VoiceRAG:
    cfg = Config(chunk_strategy=args.strategy)
    index = Index.from_dir(cfg)
    stt = None
    if need_stt:
        from .stt import WhisperTranscriber
        stt = WhisperTranscriber(args.whisper_model)
    gen = None
    if args.llm:
        from .generator import ClaudeGenerator, OpenAIGenerator
        gen = OpenAIGenerator() if os.environ.get("OPENAI_API_KEY") else ClaudeGenerator()
    return VoiceRAG(cfg, index, stt, gen)


def show(r, verbose) -> None:
    print(f"\n[{r.status.value}] {r.answer}")
    if r.question:
        print(f"  heard : {r.question}")
    if r.sources:
        print("  cites : " + ", ".join(f"[{s['n']}] {s['chunk_id']}" for s in r.sources))
    t = "  ".join(f"{k}={v:.1f}ms" for k, v in r.timings_ms.items())
    print(f"  time  : {r.total_ms:.1f} ms total  ({t})")
    if verbose:
        print(r.to_json())


def main() -> None:
    p = argparse.ArgumentParser(prog="voicerag", description="Voice-enabled RAG")
    p.add_argument("--strategy", default="recursive", choices=["fixed", "sentence", "recursive", "semantic"])
    p.add_argument("--llm", action="store_true", help="use an LLM to write the answer (OPENAI_API_KEY, else ANTHROPIC_API_KEY)")
    p.add_argument("--whisper-model", default="tiny.en")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ask", help="ask a typed question"); s.add_argument("question")
    s = sub.add_parser("voice", help="speak a question (mic, or --wav file)"); s.add_argument("--wav")
    sub.add_parser("repl", help="interactive typed loop")
    s = sub.add_parser("bench", help="P50/P70/P100 latency")
    s.add_argument("--rounds", type=int, default=50); s.add_argument("--wav", nargs="*", help="include STT using these wavs")
    s.add_argument("--out", default="benchmark.json")
    sub.add_parser("eval", help="chunking comparison + guardrail accuracy + threshold sweep")

    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING)

    if args.cmd == "ask":
        show(build(args).ask(args.question), args.verbose)
    elif args.cmd == "repl":
        rag = build(args)
        while (q := input("\nquestion> ").strip()) not in ("", "quit", "exit"):
            show(rag.ask(q), args.verbose)
    elif args.cmd == "voice":
        from .stt import load_wav, record_until_silence
        rag = build(args, need_stt=True)
        if args.wav:
            show(rag.ask(audio=load_wav(args.wav)), args.verbose)
        else:
            while True:
                input("\nPress Enter, then speak (Ctrl+C to quit)... ")
                print("listening...")
                show(rag.ask(audio=record_until_silence()), args.verbose)
    elif args.cmd == "bench":
        rag = build(args, need_stt=bool(args.wav))
        audio = None
        if args.wav:
            from .stt import load_wav
            audio = [load_wav(w) for w in args.wav]
        rep = bm.run(rag.cfg, rag, rounds=args.rounds, audio=audio)
        bm.print_report(rep)
        bm.save(rep, args.out)
    elif args.cmd == "eval":
        cfg = Config(chunk_strategy=args.strategy)
        print("\n== Chunking strategies (retrieval quality) =="); _table(compare_chunking(cfg))
        g = evaluate_guardrails(cfg)
        print("\n== Guardrails (", cfg.chunk_strategy, ") ==")
        for k in ("answerable_answered", "answerable_correct", "unanswerable_refused"):
            print(f"  {k}: {g[k]}")
        for w in g["missed_or_wrong"]: print("  MISS :", w)
        for w in g["wrongly_answered"]: print("  LEAK :", w)
        print("\n== Coverage-threshold sweep =="); _table(sweep_thresholds(cfg))
