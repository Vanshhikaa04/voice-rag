"""Metrics computed from recorded events. A value with no data is None, which the UI shows as "not measured"."""
import json
from datetime import timedelta

from sqlalchemy import func
from sqlalchemy.orm import Session

from . import ingest, rag
from .config import ROOT
from .db import AgentRun, Connector, Conversation, Document, EvalRun, Message, QueryEvent, UsageEvent, Workspace, now
from .ingest import Item
from .security import Principal


def percentile(values: list[float], pct: float) -> float | None:
    """Nearest rank, so P100 is the true worst case."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(round(pct / 100 * len(ordered) + 0.4999)) - 1))]


def summary(db: Session, p: Principal, days: int = 30) -> dict:
    ws, since = p.workspace_id, now() - timedelta(days=days)
    qs = db.query(QueryEvent).filter(QueryEvent.workspace_id == ws, QueryEvent.created_at >= since).order_by(QueryEvent.created_at).all()
    latencies = [q.total_ms for q in qs]
    first = [q.first_token_ms for q in qs if q.first_token_ms]
    checked = [q for q in qs if q.verified is not None]
    failed = [q for q in qs if q.status in ("error", "degraded")]
    usage = db.query(UsageEvent).filter(UsageEvent.workspace_id == ws, UsageEvent.created_at >= since).all()
    priced = [u.cost_usd for u in usage if u.cost_usd is not None]
    runs = db.query(AgentRun).filter(AgentRun.workspace_id == ws, AgentRun.started_at >= since).all()
    done = [r for r in runs if r.status in ("succeeded", "failed")]
    evaluation = db.query(EvalRun).filter_by(workspace_id=ws).order_by(EvalRun.created_at.desc()).first()
    votes = db.query(Message.feedback).join(Conversation, Message.conversation_id == Conversation.id).filter(
        Conversation.workspace_id == ws, Message.feedback != 0).all()

    by_day: dict[str, dict] = {}
    for q in qs:
        d = by_day.setdefault(q.created_at.strftime("%Y-%m-%d"), {"day": q.created_at.strftime("%Y-%m-%d"), "queries": 0, "failed": 0})
        d["queries"] += 1
        d["failed"] += q.status in ("error", "degraded")
    by_purpose: dict[str, dict] = {}
    for u in usage:
        b = by_purpose.setdefault(u.purpose.split(":")[0], {"purpose": u.purpose.split(":")[0], "tokens": 0, "cost": 0.0})
        b["tokens"] += u.input_tokens + u.output_tokens
        b["cost"] += u.cost_usd or 0
    status: dict[str, int] = {}
    for q in qs:
        status[q.status] = status.get(q.status, 0) + 1

    return {
        "days": days,
        "sources": db.query(func.count(Connector.id)).filter_by(workspace_id=ws).scalar(),
        "documents": db.query(func.count(Document.id)).filter_by(workspace_id=ws).scalar(),
        "chunks": db.query(func.coalesce(func.sum(Document.chunk_count), 0)).filter_by(workspace_id=ws).scalar(),
        "queries": len(qs),
        "status": status,
        "latency": {"p50": percentile(latencies, 50), "p70": percentile(latencies, 70), "p100": percentile(latencies, 100),
                    "first_token_p50": percentile(first, 50)},
        "faithfulness": sum(q.verified for q in checked) / len(checked) if checked else None,
        "verified_count": len(checked),
        "agents": {"runs": len(runs), "active": len(runs) - len(done),
                   "success_rate": sum(r.status == "succeeded" for r in done) / len(done) if done else None},
        "tokens": {"input": sum(u.input_tokens for u in usage), "output": sum(u.output_tokens for u in usage),
                   "cost_usd": sum(priced) if priced else None, "unpriced_calls": len(usage) - len(priced)},
        "usage_by_purpose": sorted(by_purpose.values(), key=lambda b: -b["tokens"]),
        "feedback": {"up": sum(v[0] > 0 for v in votes), "down": sum(v[0] < 0 for v in votes)},
        "retrieval": {"hit_at_1": evaluation.hit_at_1, "hit_at_k": evaluation.hit_at_k, "mrr": evaluation.mrr, "cases": evaluation.cases,
                      "at": evaluation.created_at.isoformat()} if evaluation else None,
        "daily": [by_day[k] for k in sorted(by_day)],
        "failures": [{"at": q.created_at.isoformat(), "question": q.question, "status": q.status, "error": q.error}
                     for q in reversed(failed[-15:])],
    }


def run_eval(db: Session, p: Principal) -> EvalRun:
    """Measures retrieval on the labelled reference corpus, in a sandbox workspace so user data is untouched."""
    name = f"__eval__{p.workspace_id}"
    sandbox = db.query(Workspace).filter_by(name=name).first()
    if not sandbox:
        sandbox = Workspace(name=name)
        db.add(sandbox)
        db.commit()
    who = Principal(p.id, sandbox.id, "member")
    for path in sorted((ROOT / "data" / "docs").glob("*.md")):
        ingest.upsert(db, sandbox.id, p.id, Item(f"eval:{path.stem}", path.stem, path.read_text(encoding="utf-8")), enrich=False)

    cases = [c for c in json.loads((ROOT / "data" / "eval.json").read_text(encoding="utf-8"))["answerable"]]
    k, hit1, hitk, rr, misses = 4, 0, 0, 0.0, []
    for c in cases:
        hits = rag.retrieve(who, c["q"], k)
        rank = next((i for i, h in enumerate(hits, 1) if c["expect"].lower() in " ".join(h.text.lower().split())), None)
        hit1 += rank == 1
        hitk += rank is not None
        rr += 1 / rank if rank else 0
        if not rank:
            misses.append(c["q"])
    n = len(cases)
    run = EvalRun(workspace_id=p.workspace_id, cases=n, hit_at_1=hit1 / n, hit_at_k=hitk / n, mrr=rr / n, detail={"k": k, "misses": misses})
    db.add(run)
    db.commit()
    return run
