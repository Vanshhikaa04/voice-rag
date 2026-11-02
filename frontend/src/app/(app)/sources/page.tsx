"use client";

import { FileText,Globe, Lock, RefreshCw, Trash2, Upload, Users, X } from "lucide-react";
import { useRef, useState } from "react";
import { api, ago } from "@/lib/api";
import { Doc, Empty, Page, Skeleton, Spinner, useLoad } from "@/components/ui";

type Field = { name: string; label: string; secret: boolean };
type Kind = { kind: string; label: string; fields: Field[]; oauth: boolean; missing: string[]; admin_only: boolean };
type Conn = { id: string; kind: string; name: string; status: string; last_error: string; last_synced_at: string | null; documents: number; visibility: string };

const STATUS: Record<string, [string, string]> = {
  idle: ["good", "Connected"], syncing: ["live", "Syncing"], error: ["bad", "Sync failed"], needs_auth: ["warn", "Reconnect needed"],
};

function Add({ kind, onClose, onDone }: { kind: Kind; onClose: () => void; onDone: () => void }) {
  const [values, setValues] = useState<Record<string, string>>({});
  const [name, setName] = useState(kind.label);
  const [visibility, setVisibility] = useState("workspace");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError("");
    const config: Record<string, string> = {}, secrets: Record<string, string> = {};
    kind.fields.forEach((f) => { if (values[f.name]) (f.secret ? secrets : config)[f.name] = values[f.name]; });
    try {
      await api("/connectors", { body: { kind: kind.kind, name, config, secrets, visibility } });
      onDone();
    } catch (err) {
      setError((err as Error).message);
      setBusy(false);
    }
  }

  return (
    <div className="scrim" onClick={onClose}>
      <form className="dialog" onClick={(e) => e.stopPropagation()} onSubmit={submit}>
        <div className="spread"><h2>Connect {kind.label}</h2><button type="button" className="btn ghost icon" aria-label="Close" onClick={onClose}><X size={16} /></button></div>
        <label className="field">Name<input className="input" value={name} required onChange={(e) => setName(e.target.value)} /></label>
        {kind.fields.map((f) => (
          <label className="field" key={f.name}>{f.label}
            <input className="input" type={f.secret ? "password" : "text"} autoComplete="off" value={values[f.name] ?? ""} onChange={(e) => setValues({ ...values, [f.name]: e.target.value })} />
          </label>
        ))}
        <label className="field">Who can search this
          <select className="input" value={visibility} onChange={(e) => setVisibility(e.target.value)}>
            <option value="workspace">Everyone in the workspace</option>
            <option value="private">Only me</option>
          </select>
        </label>
        {kind.fields.some((f) => f.secret) && <p className="dim">Credentials are encrypted at rest and never shown again.</p>}
        {error && <div className="error" role="alert">{error}</div>}
        <button className="btn primary" disabled={busy}>{busy ? <Spinner /> : "Connect and sync"}</button>
      </form>
    </div>
  );
}

function Uploads({ onDone }: { onDone: () => void }) {
  const [over, setOver] = useState(false);
  const [note, setNote] = useState("");
  const [fails, setFails] = useState<string[]>([]);
  const [priv, setPriv] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  async function send(files: FileList | null) {
    if (!files?.length) return;
    const failed: string[] = [];
    let i = 0;
    for (const f of Array.from(files)) {
      setNote(`Processing ${++i} of ${files.length}: ${f.name}`);
      const form = new FormData();
      form.append("file", f);
      form.append("visibility", priv ? "private" : "workspace");
      try { await api("/documents", { form }); } catch (e) { failed.push(`${f.name}: ${(e as Error).message}`); }
      onDone();
    }
    setNote("");
    setFails(failed);
  }

  return (
    <div className="stack">
      <label className={`drop ${over ? "over" : ""}`} onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)}
        onDrop={(e) => { e.preventDefault(); setOver(false); send(e.dataTransfer.files); }}>
        <input ref={input} type="file" multiple hidden onChange={(e) => { send(e.target.files); e.target.value = ""; }} />
        {note ? <span className="row" style={{ justifyContent: "center" }}><Spinner /> <span className="clip">{note}</span></span> : (
          <>
            <Upload size={18} style={{ marginBottom: 4 }} />
            <div>Drop files or click to upload</div>
            <div className="dim">PDF, DOCX, PPTX, XLSX, CSV, images, audio, video, code</div>
          </>
        )}
      </label>
      <label className="row dim"><input type="checkbox" checked={priv} onChange={(e) => setPriv(e.target.checked)} /> Keep uploads private to me</label>
      {fails.map((f) => <div key={f} className="error">{f}</div>)}
    </div>
  );
}

export default function Sources() {
  const catalog = useLoad<Kind[]>("/connectors/catalog");
  const conns = useLoad<Conn[]>("/connectors", 4000);
  const docs = useLoad<Doc[]>("/documents", 8000);
  const [adding, setAdding] = useState<Kind | null>(null);
  const [error, setError] = useState("");

  async function start(k: Kind) {
    setError("");
    if (!k.oauth) return setAdding(k);
    try {
      const { url } = await api<{ url: string }>(`/connectors/oauth/${k.kind}/start`);
      window.location.href = url;
    } catch (e) {
      setError((e as Error).message);
    }
  }
  const act = (fn: Promise<unknown>) => fn.then(() => { conns.reload(); docs.reload(); }).catch((e) => setError(e.message));

  return (
    <Page title="Sources" sub="Everything NEXUS can read. Connected sources stay in sync automatically.">
      {error && <div className="error" style={{ marginBottom: 12 }} role="alert">{error}</div>}
      <div className="grid-3">
        {catalog.data?.map((k) => (
          <div className="card" key={k.kind}>
            <div className="spread"><b style={{ fontWeight: 600 }}>{k.label}</b>
              {k.missing.length > 0 ? <span className="tag">Not configured</span> : null}</div>
            {k.missing.length > 0
              ? <p className="dim" style={{ margin: "6px 0 0", overflowWrap: "anywhere" }}>Server needs {k.missing.join(" and ")}</p>
              : <button className="btn" style={{ marginTop: 10 }} onClick={() => start(k)}>Connect</button>}
          </div>
        ))}
        {!catalog.data && <Skeleton rows={1} />}
      </div>

      <section className="section">
        <h2>Connected</h2>
        {!conns.data ? <Skeleton rows={2} /> : !conns.data.length ? <Empty icon={Globe} title="No sources connected">Connect one above, or upload files below.</Empty> : (
          <div className="stack" style={{ gap: 8 }}>
            {conns.data.map((c) => {
              const [tone, label] = STATUS[c.status] ?? ["", c.status];
              return (
                <div className="card" key={c.id}>
                  <div className="spread">
                    <div className="grow">
                      <div className="row"><b className="clip" style={{ fontWeight: 600 }}>{c.name}</b><span className={`tag ${tone}`}>{c.status === "syncing" && <Spinner size={10} />}{label}</span></div>
                      <div className="dim">{c.documents} {c.documents === 1 ? "item" : "items"} · synced {ago(c.last_synced_at)}{c.visibility === "private" && " · private"}</div>
                    </div>
                    <button className="btn ghost icon" aria-label={`Sync ${c.name}`} disabled={c.status === "syncing"} onClick={() => act(api(`/connectors/${c.id}/sync`, { method: "POST" }))}><RefreshCw size={14} /></button>
                    <button className="btn ghost icon" aria-label={`Remove ${c.name}`} onClick={() => confirm(`Remove ${c.name} and its ${c.documents} indexed items?`) && act(api(`/connectors/${c.id}`, { method: "DELETE" }))}><Trash2 size={14} /></button>
                  </div>
                  {c.last_error && <div className="error" style={{ marginTop: 8 }}>{c.last_error}</div>}
                </div>
              );
            })}
          </div>
        )}
      </section>

      <section className="section">
        <h2>Files</h2>
        <Uploads onDone={docs.reload} />
        <div style={{ marginTop: 12 }}>
          {!docs.data ? <Skeleton rows={2} /> : !docs.data.length ? <Empty icon={FileText} title="Nothing indexed yet" /> : (
            <div className="card table-wrap" style={{ padding: 4 }}>
              <table>
                <thead><tr><th>Name</th><th>Type</th><th>Passages</th><th>Updated</th><th>Access</th><th /></tr></thead>
                <tbody>
                  {docs.data.slice(0, 100).map((d) => (
                    <tr key={d.id}>
                      <td style={{ maxWidth: 340 }}><div className="clip">{d.url ? <a href={d.url} target="_blank" rel="noreferrer">{d.title}</a> : d.title}</div></td>
                      <td className="dim">{d.kind}</td>
                      <td className="mono dim">{d.chunks}</td>
                      <td className="dim">{ago(d.updated_at)}</td>
                      <td>
                        <button className="btn ghost" style={{ height: 26, padding: "0 6px" }} title="Change access"
                          onClick={() => act(api(`/documents/${d.id}`, { method: "PATCH", body: { visibility: d.visibility === "private" ? "workspace" : "private" } }))}>
                          {d.visibility === "private" ? <><Lock size={12} /> Private</> : <><Users size={12} /> Workspace</>}
                        </button>
                      </td>
                      <td><button className="btn ghost icon" aria-label={`Delete ${d.title}`} onClick={() => confirm(`Delete ${d.title}?`) && act(api(`/documents/${d.id}`, { method: "DELETE" }))}><Trash2 size={13} /></button></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </section>
      {adding && <Add kind={adding} onClose={() => setAdding(null)} onDone={() => { setAdding(null); conns.reload(); }} />}
    </Page>
  );
}
