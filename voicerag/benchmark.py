"""Latency benchmark: P50 / P70 / P100 per stage over many real queries, repeated for stable tails."""
import json
import time
from collections import defaultdict

import numpy as np

from .evaluate import load_eval
from .harness import VoiceRAG

PCTS = (50, 70, 100)


def run(cfg, rag: VoiceRAG, rounds: int = 50, audio=None, warmup: int = 5) -> dict:
    """audio: optional list of float32 arrays; if given, the speech-to-text stage is included."""
    data = load_eval(cfg)
    queries = [i["q"] for i in data["answerable"]] + data["unanswerable"]
    inputs = [{"audio": a} for a in audio] if audio else [{"text": q} for q in queries]

    for _ in range(warmup):                                  # fill caches / lazy imports
        for kw in inputs:
            rag.ask(**kw)

    stage: dict[str, list[float]] = defaultdict(list)
    total: list[float] = []
    for _ in range(rounds):
        for kw in inputs:
            t0 = time.perf_counter()
            r = rag.ask(**kw)
            total.append((time.perf_counter() - t0) * 1000)
            for k, v in r.timings_ms.items():
                stage[k].append(v)

    def pct(xs):
        return {f"p{p}": round(float(np.percentile(xs, p)), 3) for p in PCTS}

    report = {
        "samples": len(total),
        "queries": len(inputs),
        "includes_stt": bool(audio),
        "budget_ms": cfg.latency_budget_ms,
        "total_ms": pct(total),
        "stages_ms": {k: pct(v) for k, v in stage.items()},
    }
    report["within_budget"] = report["total_ms"]["p100"] <= cfg.latency_budget_ms
    return report


def print_report(rep: dict) -> None:
    print(f"\n{rep['samples']} samples over {rep['queries']} distinct queries "
          f"({'voice, incl. STT' if rep['includes_stt'] else 'text in -> answer out, STT excluded'})")
    print(f"{'stage':<18}{'P50 ms':>10}{'P70 ms':>10}{'P100 ms':>10}")
    print("-" * 48)
    for k, v in rep["stages_ms"].items():
        print(f"{k:<18}{v['p50']:>10.3f}{v['p70']:>10.3f}{v['p100']:>10.3f}")
    t = rep["total_ms"]
    print("-" * 48)
    print(f"{'TOTAL':<18}{t['p50']:>10.3f}{t['p70']:>10.3f}{t['p100']:>10.3f}")
    verdict = "PASS" if rep["within_budget"] else "FAIL"
    print(f"\nBudget {rep['budget_ms']:.0f} ms at P100 (worst case): {verdict}\n")


def save(rep: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
