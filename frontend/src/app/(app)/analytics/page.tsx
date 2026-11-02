"use client";

import { Activity, Play } from "lucide-react";
import { useState } from "react";
import { api, ago } from "@/lib/api";
import { Empty, Page, Skeleton, Spinner, useLoad } from "@/components/ui";

type A = {
  sources: number; documents: number; chunks: number; queries: number;
  status: Record<string, number>;
  latency: { p50: number | null; p70: number | null; p100: number | null; first_token_p50: number | null };
  faithfulness: number | null; verified_count: number;
  agents: { runs: number; active: number; success_rate: number | null };
  tokens: { input: number; output: number; cost_usd: number | null; unpriced_calls: number };
  usage_by_purpose: { purpose: string; tokens: number; cost: number }[];
  feedback: { up: number; down: number };
  retrieval: { hit_at_1: number; hit_at_k: number; mrr: number; cases: number; at: string } | null;
  daily: { day: string; queries: number; failed: number }[];
  failures: { at: string; question: string; status: string; error: string }[];
};

const ms = (v: number | null) => (v === null ? null : v >= 1000 ? `${(v / 1000).toFixed(2)}s` : `${Math.round(v)}ms`);
const pct = (v: number | null | undefined) => (v === null || v === undefined ? null : `${Math.round(v * 100)}%`);
const num = (v: number) => new Intl.NumberFormat("en", { notation: v > 99999 ? "compact" : "standard" }).format(v);

function Stat({ label, value, note }: { label: string; value: string | number | null; note?: string }) {
  return (
    <div className="card stat">
      <span>{label}</span>
      {value === null ? <b className="none">Not measured</b> : <b>{value}</b>}
      {note && <span>{note}</span>}
    </div>
  );
}

function Daily({ rows }: { rows: A["daily"] }) {
  const [tip, setTip] = useState<{ i: number; x: number } | null>(null);
  const top = Math.max(...rows.map((r) => r.queries), 1);
  return (
    <div className="bars" role="img" aria-label={`Queries per day, peak ${top}`} onMouseLeave={() => setTip(null)}>
      {rows.map((r, i) => (
        <div key={r.day} className="bar-hit" onMouseEnter={(e) => setTip({ i, x: e.currentTarget.offsetLeft })}>
          <div className="bar" style={{ height: `${(r.queries / top) * 100}%` }} />
        </div>
      ))}
      {tip && <div className="tip" style={{ transform: `translateX(${tip.x}px)` }}>{rows[tip.i].day}: {rows[tip.i].queries} queries{rows[tip.i].failed ? `, ${rows[tip.i].failed} failed` : ""}</div>}
    </div>
  );
}

export default function Analytics() {
  const { data: a, error, reload } = useLoad<A>("/analytics", 15000);
  const [running, setRunning] = useState(false);
  const [evalError, setEvalError] = useState("");

  async function runEval() {
    setRunning(true);
    setEvalError("");
    try { await api("/analytics/eval", { method: "POST" }); await reload(); } catch (e) { setEvalError((e as Error).message); }
    setRunning(false);
  }

  if (error && !a) return <Page title="Analytics"><div className="error">{error}</div></Page>;
  if (!a) return <Page title="Analytics"><Skeleton rows={4} /></Page>;
  const tokens = a.tokens.input + a.tokens.output;
  const top = Math.max(...a.usage_by_purpose.map((u) => u.tokens), 1);

  return (
    <Page title="Analytics" sub="The last 30 days, computed from recorded events. Nothing here is estimated except cost."
      action={<button className="btn" onClick={runEval} disabled={running}>{running ? <Spinner /> : <Play size={13} />} Run retrieval benchmark</button>}>
      {evalError && <div className="error" style={{ marginBottom: 12 }}>{evalError}</div>}
      <div className="grid-4">
        <Stat label="Connected sources" value={a.sources} />
        <Stat label="Indexed records" value={num(a.documents)} note={`${num(a.chunks)} passages`} />
        <Stat label="Queries" value={num(a.queries)} />
        <Stat label="Agent runs" value={a.agents.runs} note={a.agents.active ? `${a.agents.active} running` : undefined} />
      </div>

      <section className="section"><h2>Latency</h2>
        <div className="grid-4">
          <Stat label="P50" value={ms(a.latency.p50)} />
          <Stat label="P70" value={ms(a.latency.p70)} />
          <Stat label="P100" value={ms(a.latency.p100)} note="Slowest query" />
          <Stat label="First words, P50" value={ms(a.latency.first_token_p50)} />
        </div>
      </section>

      <section className="section"><h2>Quality</h2>
        <div className="grid-4">
          <Stat label="Answer faithfulness" value={pct(a.faithfulness)} note={a.verified_count ? `${a.verified_count} answers checked` : undefined} />
          <Stat label="Retrieval hit rate" value={pct(a.retrieval?.hit_at_k)} note={a.retrieval ? `Top 4, ${a.retrieval.cases} labelled cases, ${ago(a.retrieval.at)}` : "Run the benchmark"} />
          <Stat label="First-result accuracy" value={pct(a.retrieval?.hit_at_1)} note={a.retrieval ? `MRR ${a.retrieval.mrr.toFixed(2)}` : undefined} />
          <Stat label="Agent success" value={pct(a.agents.success_rate)} />
        </div>
      </section>

      <section className="section grid-2">
        <div className="card">
          <h2 style={{ marginBottom: 12 }}>Queries per day</h2>
          {a.daily.length ? <Daily rows={a.daily} /> : <Empty icon={Activity} title="No queries yet" />}
        </div>
        <div className="card">
          <div className="spread" style={{ marginBottom: 12 }}>
            <h2>Tokens by purpose</h2>
            <span className="mono dim">{a.tokens.cost_usd === null ? "cost not measured" : `$${a.tokens.cost_usd.toFixed(4)}`}</span>
          </div>
          {!a.usage_by_purpose.length ? <Empty icon={Activity} title="No model usage yet" /> : a.usage_by_purpose.map((u) => (
            <div className="hbar" key={u.purpose}>
              <span className="clip muted">{u.purpose}</span>
              <div className="track"><div className="fill" style={{ width: `${(u.tokens / top) * 100}%` }} /></div>
              <span className="mono dim" style={{ textAlign: "right" }}>{num(u.tokens)}</span>
            </div>
          ))}
          {tokens > 0 && <p className="dim" style={{ marginTop: 8 }}>{num(tokens)} tokens in total{a.tokens.unpriced_calls ? `; ${a.tokens.unpriced_calls} calls have no known price and are excluded from cost` : ""}.</p>}
        </div>
      </section>

      <section className="section">
        <h2>Outcomes</h2>
        <div className="row" style={{ flexWrap: "wrap", gap: 6 }}>
          {Object.entries(a.status).map(([k, v]) => <span className="tag" key={k}>{k.replaceAll("_", " ")}: {v}</span>)}
          <span className="tag">helpful votes: {a.feedback.up}</span><span className="tag">unhelpful votes: {a.feedback.down}</span>
        </div>
      </section>

      <section className="section">
        <h2>Failures</h2>
        {!a.failures.length ? <p className="dim">None recorded.</p> : (
          <div className="card table-wrap" style={{ padding: 4 }}>
            <table>
              <thead><tr><th>When</th><th>Query</th><th>Status</th><th>Error</th></tr></thead>
              <tbody>{a.failures.map((f, i) => <tr key={i}><td className="dim">{ago(f.at)}</td><td>{f.question}</td><td><span className="tag bad">{f.status}</span></td><td className="dim">{f.error}</td></tr>)}</tbody>
            </table>
          </div>
        )}
      </section>
    </Page>
  );
}
