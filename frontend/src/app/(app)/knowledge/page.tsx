"use client";

import { Brain, Network, Search, Trash2 } from "lucide-react";
import { useState } from "react";
import { api, ago } from "@/lib/api";
import { Empty, Page, Skeleton, Spinner, useLoad } from "@/components/ui";
import { Graph, GraphData, TYPE_COLOR } from "@/components/graph";

type Hit = { document_id: string; title: string; url: string; text: string; relevance: number };
type Memory = { id: string; content: string; source: string; created_at: string };

function Finder() {
  const [q, setQ] = useState("");
  const [hits, setHits] = useState<Hit[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  async function go(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError("");
    try { setHits(await api<Hit[]>("/search", { body: { query: q } })); } catch (err) { setError((err as Error).message); }
    setBusy(false);
  }
  return (
    <>
      <form className="row" onSubmit={go}>
        <input className="input" value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search across every source" aria-label="Search" />
        <button className="btn primary icon" aria-label="Search" disabled={!q.trim() || busy}>{busy ? <Spinner /> : <Search size={15} />}</button>
      </form>
      {error && <div className="error" style={{ marginTop: 8 }}>{error}</div>}
      {hits && (
        <div className="sources">
          {!hits.length && <Empty icon={Search} title="No matches" />}
          {hits.map((h, i) => (
            <div className="source" key={i}>
              <div className="spread"><span className="clip">{h.url ? <a href={h.url} target="_blank" rel="noreferrer">{h.title}</a> : h.title}</span><span className="dim mono">{h.relevance.toFixed(2)}</span></div>
              <p>{h.text}</p>
            </div>
          ))}
        </div>
      )}
    </>
  );
}

function Memories() {
  const m = useLoad<Memory[]>("/memories");
  const [text, setText] = useState("");
  return (
    <div className="stack">
      <form className="row" onSubmit={(e) => { e.preventDefault(); api("/memories", { body: { content: text } }).then(() => { setText(""); m.reload(); }); }}>
        <input className="input" value={text} onChange={(e) => setText(e.target.value)} placeholder="Add something NEXUS should remember" aria-label="New memory" />
        <button className="btn" disabled={!text.trim()}>Save</button>
      </form>
      {!m.data ? <Skeleton rows={2} /> : !m.data.length ? <Empty icon={Brain} title="No memories">Say "remember that..." in chat, or add one here. Only you can see these.</Empty> : (
        <div className="card table-wrap" style={{ padding: 4 }}>
          <table><tbody>
            {m.data.map((x) => (
              <tr key={x.id}>
                <td>{x.content}</td>
                <td className="dim" style={{ whiteSpace: "nowrap" }}>{x.source === "extracted" ? "learned" : "saved"} {ago(x.created_at)}</td>
                <td style={{ width: 36 }}><button className="btn ghost icon" aria-label="Forget this" onClick={() => api(`/memories/${x.id}`, { method: "DELETE" }).then(m.reload)}><Trash2 size={13} /></button></td>
              </tr>
            ))}
          </tbody></table>
        </div>
      )}
    </div>
  );
}

export default function Knowledge() {
  const graph = useLoad<GraphData>("/graph");
  const [hidden, setHidden] = useState(() => new Set(["document"]));
  const toggle = (t: string) => setHidden((h) => { const s = new Set(h); if (!s.delete(t)) s.add(t); return s; });
  const counts: Record<string, number> = {};
  graph.data?.nodes.forEach((n) => (counts[n.type] = (counts[n.type] ?? 0) + 1));
  return (
    <Page title="Explorer" sub="Search everything, and see how people, projects and decisions connect.">
      <Finder />
      <section className="section">
        <h2>Knowledge graph</h2>
        {!graph.data ? <Skeleton /> : !graph.data.nodes.length ? <Empty icon={Network} title="No connections yet">The graph builds itself as documents are indexed.</Empty> : (
          <>
            <Graph data={graph.data} hidden={hidden} />
            <div className="legend">
              {Object.entries(TYPE_COLOR).filter(([t]) => counts[t]).map(([t, c]) => (
                <button key={t} aria-pressed={!hidden.has(t)} onClick={() => toggle(t)} style={{ opacity: hidden.has(t) ? 0.45 : 1 }}><i style={{ background: c }} />{t} ({counts[t]})</button>
              ))}
              <span className="dim">Hover an entity to see its connections. Click a type to show or hide it.</span>
            </div>
          </>
        )}
      </section>
      <section className="section">
        <h2>Your memory</h2>
        <Memories />
      </section>
    </Page>
  );
}
