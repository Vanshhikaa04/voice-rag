"use client";

import { useState } from "react";
import { api, ago } from "@/lib/api";
import { Page, Skeleton, useLoad } from "@/components/ui";

type User = { id: string; email: string; name: string; role: string; active: boolean };
type System = {
  llm: { provider: string; model: string }[]; embedder: { provider: string; model: string; semantic: boolean }; reranker: string; database: string;
  vector_store: string; cache: string; stt: string; tts: string | null; scheduler: { enabled: boolean; interval_minutes: number }; keys: Record<string, boolean>;
};
type Audit = { id: string; at: string; user: string; action: string; target: string };

function Row({ k, v, warn }: { k: string; v: string; warn?: boolean }) {
  return <tr><td className="muted">{k}</td><td className="mono" style={{ color: warn ? "var(--warn)" : undefined }}>{v}</td></tr>;
}

function Team({ me }: { me: string }) {
  const users = useLoad<User[]>("/users");
  const [form, setForm] = useState({ name: "", email: "", password: "", role: "member" });
  const [error, setError] = useState("");
  const change = (id: string, body: object) => api(`/users/${id}`, { method: "PATCH", body }).then(users.reload).catch((e) => setError(e.message));
  return (
    <section className="section">
      <h2>Team</h2>
      <div className="card table-wrap" style={{ padding: 4 }}>
        <table>
          <tbody>
            {users.data?.map((u) => (
              <tr key={u.id} style={{ opacity: u.active ? 1 : 0.5 }}>
                <td>{u.name}<div className="dim">{u.email}</div></td>
                <td>{u.id === me ? <span className="tag">{u.role}</span> : (
                  <select className="input" style={{ width: 110, height: 30 }} value={u.role} aria-label={`Role of ${u.email}`} onChange={(e) => change(u.id, { role: e.target.value })}>
                    <option value="member">member</option><option value="admin">admin</option></select>)}</td>
                <td style={{ textAlign: "right" }}>{u.id !== me && <button className="btn ghost" onClick={() => change(u.id, { active: !u.active })}>{u.active ? "Deactivate" : "Reactivate"}</button>}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <form className="card stack" style={{ marginTop: 10 }} onSubmit={(e) => {
        e.preventDefault();
        setError("");
        api("/users", { body: form }).then(() => { setForm({ name: "", email: "", password: "", role: "member" }); users.reload(); }).catch((err) => setError(err.message));
      }}>
        <h2>Add a teammate</h2>
        <div className="grid-3">
          <label className="field">Name<input className="input" required value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} /></label>
          <label className="field">Email<input className="input" type="email" required value={form.email} onChange={(e) => setForm({ ...form, email: e.target.value })} /></label>
          <label className="field">Temporary password<input className="input" type="password" minLength={10} required autoComplete="new-password" value={form.password} onChange={(e) => setForm({ ...form, password: e.target.value })} /></label>
        </div>
        {error && <div className="error" role="alert">{error}</div>}
        <button className="btn" style={{ justifySelf: "start" }}>Add</button>
      </form>
    </section>
  );
}

function AuditTrail() {
  const log = useLoad<Audit[]>("/audit");
  return (
    <section className="section">
      <h2>Audit trail</h2>
      {!log.data ? <Skeleton rows={2} /> : !log.data.length ? <p className="dim">Nothing recorded yet.</p> : (
        <div className="card table-wrap" style={{ padding: 4, maxHeight: 340, overflowY: "auto" }}>
          <table><tbody>
            {log.data.map((a) => <tr key={a.id}><td className="dim" style={{ whiteSpace: "nowrap" }}>{ago(a.at)}</td><td className="dim">{a.user}</td><td className="mono" style={{ fontSize: 12.5 }}>{a.action}</td><td className="dim">{a.target}</td></tr>)}
          </tbody></table>
        </div>
      )}
    </section>
  );
}

export default function Settings() {
  const me = useLoad<User & { workspace: string }>("/auth/me");
  const sys = useLoad<System>("/system");
  const s = sys.data;
  return (
    <Page title="Settings" sub={me.data ? `${me.data.workspace} · ${me.data.email}` : undefined}>
      <h2 style={{ marginBottom: 10 }}>What is running</h2>
      {!s ? <Skeleton /> : (
        <div className="card table-wrap" style={{ padding: 4 }}>
          <table><tbody>
            <Row k="Language model" v={s.llm.map((l) => `${l.provider} ${l.model}`).join(", ") || "none: answers are quoted, agents are off"} warn={!s.llm.length} />
            <Row k="Embeddings" v={`${s.embedder.model}${s.embedder.semantic ? "" : " (keyword only: no paraphrase or cross-language matching)"}`} warn={!s.embedder.semantic} />
            <Row k="Reranker" v={s.reranker} />
            <Row k="Database" v={s.database} />
            <Row k="Vector store" v={s.vector_store} />
            <Row k="Cache" v={s.cache} />
            <Row k="Speech to text" v={s.stt} />
            <Row k="Text to speech" v={s.tts ?? "none"} warn={!s.tts} />
            <Row k="Automatic sync" v={s.scheduler.enabled ? `every ${s.scheduler.interval_minutes} minutes` : "off"} />
            <Row k="Google sign-in app" v={s.keys.google_oauth ? "configured" : "not configured"} warn={!s.keys.google_oauth} />
          </tbody></table>
        </div>
      )}
      {me.data?.role === "admin" && <><Team me={me.data.id} /><AuditTrail /></>}
    </Page>
  );
}
