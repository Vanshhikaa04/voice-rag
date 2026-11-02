"use client";

import { Bot, Check, X } from "lucide-react";
import { useState } from "react";
import { api, ago } from "@/lib/api";
import { Empty, NoModelNote, Page, Run, RunView, Skeleton, agentName, useLoad } from "@/components/ui";

type Approval = { id: string; action: string; payload: Record<string, unknown>; status: string; outcome: string; created_at: string };
const ACTION: Record<string, string> = { "gmail.create_draft": "Create a Gmail draft", "slack.post": "Post to Slack", "github.comment": "Comment on GitHub" };

function Pending({ a, onDone }: { a: Approval; onDone: () => void }) {
  const [error, setError] = useState("");
  const decide = (approve: boolean) => api(`/approvals/${a.id}`, { body: { approve } }).then(onDone).catch((e) => setError(e.message));
  return (
    <div className="card stack" style={{ borderColor: "var(--warn)" }}>
      <div className="spread"><b style={{ fontWeight: 600 }}>{ACTION[a.action] ?? a.action}</b><span className="dim">{ago(a.created_at)}</span></div>
      <div className="result">{Object.entries(a.payload).map(([k, v]) => `${k}: ${v}`).join("\n")}</div>
      {error && <div className="error">{error}</div>}
      <div className="row">
        <button className="btn primary" onClick={() => decide(true)}><Check size={14} /> Approve</button>
        <button className="btn" onClick={() => decide(false)}><X size={14} /> Reject</button>
      </div>
    </div>
  );
}

export default function Agents() {
  const runs = useLoad<Run[]>("/runs", 2500);
  const approvals = useLoad<Approval[]>("/approvals", 4000);
  const catalog = useLoad<{ agents: Record<string, string> }>("/agents");
  const [goal, setGoal] = useState("");
  const [error, setError] = useState("");
  const [open, setOpen] = useState<string | null>(null);
  const pending = approvals.data?.filter((a) => a.status === "pending") ?? [];

  async function start(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    try {
      const r = await api<Run>("/runs", { body: { goal } });
      setGoal("");
      setOpen(r.id);
      runs.reload();
    } catch (err) {
      setError((err as Error).message);
    }
  }

  return (
    <Page title="Agents" sub="Give a goal. The supervisor picks the agents it needs, and nothing leaves NEXUS without your approval.">
      <NoModelNote />
      <form className="row" onSubmit={start} style={{ marginBottom: 8, marginTop: 8 }}>
        <input className="input" placeholder="Investigate why signups dropped and draft an update for the team" value={goal} onChange={(e) => setGoal(e.target.value)} aria-label="Goal" />
        <button className="btn primary" disabled={!goal.trim()}>Start</button>
      </form>
      {error && <div className="error" role="alert">{error}</div>}
      <div className="row" style={{ flexWrap: "wrap", gap: 6 }}>
        {catalog.data && [...Object.entries(catalog.data.agents)].map(([k, v]) => <span key={k} className="tag" title={v}>{agentName(k)}</span>)}
        <span className="tag" title="Checks results against evidence">Verification</span>
      </div>

      {pending.length > 0 && (
        <section className="section">
          <h2>Waiting for your approval</h2>
          <div className="stack">{pending.map((a) => <Pending key={a.id} a={a} onDone={approvals.reload} />)}</div>
        </section>
      )}

      <section className="section">
        <h2>Activity</h2>
        {!runs.data ? <Skeleton /> : !runs.data.length ? <Empty icon={Bot} title="No runs yet">Start one above, or from Workflows.</Empty> : (
          <div className="stack" style={{ gap: 8 }}>
            {runs.data.map((r) => (
              <div className="card" key={r.id}>
                <button className="btn ghost" style={{ all: "unset", cursor: "pointer", display: "block", width: "100%" }} onClick={() => setOpen(open === r.id ? null : r.id)} aria-expanded={open === r.id}>
                  <div className="clip">{r.goal}</div>
                  {open !== r.id && <div className="dim">{r.status === "failed" ? "Failed" : r.status === "succeeded" ? "Finished" : "Working"} · {r.steps?.map((s) => agentName(s.agent)).join(" → ")} · {ago(r.started_at)}</div>}
                </button>
                {open === r.id && <div style={{ marginTop: 10 }}><RunView run={r} /></div>}
              </div>
            ))}
          </div>
        )}
      </section>

      {approvals.data && approvals.data.some((a) => a.status !== "pending") && (
        <section className="section">
          <h2>Decisions</h2>
          <div className="card table-wrap" style={{ padding: 4 }}>
            <table><tbody>
              {approvals.data.filter((a) => a.status !== "pending").map((a) => (
                <tr key={a.id}><td>{ACTION[a.action] ?? a.action}</td><td><span className={`tag ${a.status === "executed" ? "good" : a.status === "failed" ? "bad" : ""}`}>{a.status}</span></td><td className="dim">{a.outcome}</td></tr>
              ))}
            </tbody></table>
          </div>
        </section>
      )}
    </Page>
  );
}
