"""Quality evals: (1) which chunking strategy retrieves best, (2) do the guardrails answer/refuse correctly."""
import dataclasses
import json
import re

from .chunking import STRATEGIES
from .config import Config
from .guardrails import Status
from .harness import VoiceRAG
from .index import Index
from .text import tokens


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def load_eval(cfg: Config) -> dict:
    return json.loads(cfg.eval_file.read_text(encoding="utf-8"))


def compare_chunking(cfg: Config) -> list[dict]:
    """A hit means the retrieved chunks contain the full gold evidence phrase."""
    qa = load_eval(cfg)["answerable"]
    rows = []
    for name in STRATEGIES:
        idx = Index.from_dir(cfg, strategy=name)
        hit1 = hitk = rr = 0.0
        for item in qa:
            ranks = [i for i, h in enumerate(idx.search(item["q"], cfg.top_k)) if _norm(item["expect"]) in _norm(h.chunk.text)]
            if ranks:
                hitk += 1
                hit1 += ranks[0] == 0
                rr += 1 / (ranks[0] + 1)
        n = len(qa)
        lens = [len(c.text) for c in idx.chunks]
        rows.append({"strategy": name, "chunks": len(idx.chunks), "avg_chars": round(sum(lens) / len(lens)),
                     "hit@1": round(hit1 / n, 3), f"hit@{cfg.top_k}": round(hitk / n, 3), "mrr": round(rr / n, 3)})
    return rows


def evaluate_guardrails(cfg: Config, rag: VoiceRAG | None = None) -> dict:
    data = load_eval(cfg)
    rag = rag or VoiceRAG(cfg, Index.from_dir(cfg))
    answered = correct = 0
    wrong = []
    for item in data["answerable"]:
        r = rag.ask(item["q"])
        if r.status == Status.ANSWERED:
            answered += 1
            gold = set(tokens(item["expect"]))
            if len(gold & set(tokens(r.answer))) / max(len(gold), 1) >= 0.6:
                correct += 1
                continue
        wrong.append((item["q"], r.status.value, r.answer[:90], r.reason))
    refused = 0
    leaked = []
    for q in data["unanswerable"]:
        r = rag.ask(q)
        if r.status != Status.ANSWERED:
            refused += 1
        else:
            leaked.append((q, r.answer[:90]))
    na, nu = len(data["answerable"]), len(data["unanswerable"])
    return {"answerable_answered": f"{answered}/{na}", "answerable_correct": f"{correct}/{na}",
            "unanswerable_refused": f"{refused}/{nu}", "missed_or_wrong": wrong, "wrongly_answered": leaked}


def sweep_thresholds(cfg: Config) -> list[dict]:
    """Show the precision/recall trade-off of the retrieval guardrail so thresholds are chosen, not guessed."""
    rows = []
    for cov in (0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8):
        r = evaluate_guardrails(dataclasses.replace(cfg, min_coverage=cov))
        rows.append({"min_coverage": cov, "answered": r["answerable_answered"], "correct": r["answerable_correct"],
                     "refused_unanswerable": r["unanswerable_refused"]})
    return rows
