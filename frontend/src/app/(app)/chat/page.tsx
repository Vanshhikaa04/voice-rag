"use client";

import { ArrowUp, Plus, ShieldAlert, ShieldCheck, Square, ThumbsDown, ThumbsUp, Trash2 } from "lucide-react";
import { Fragment, useEffect, useRef, useState } from "react";
import { api, streamChat, type DoneMeta, type Source } from "@/lib/api";
import { useLoad } from "@/components/ui";

type Msg = { id?: string; role: "user" | "assistant"; content: string; meta?: Partial<DoneMeta>; feedback?: number; note?: string };

/** Renders [n] as a citation chip, without interpreting anything else in the text. */
function Cited({ text }: { text: string }) {
  return <>{text.split(/(\[\d+\])/).map((part, i) => /^\[\d+\]$/.test(part) ? <span key={i} className="cite">{part.slice(1, -1)}</span> : <Fragment key={i}>{part}</Fragment>)}</>;
}

function Status({ meta }: { meta: Partial<DoneMeta> }) {
  if (meta.verified === true) return <span className="tag good"><ShieldCheck size={12} /> Verified against sources</span>;
  if (meta.status === "refused_ungrounded") return <span className="tag warn"><ShieldAlert size={12} /> Unsupported claim removed</span>;
  if (meta.status === "degraded") return <span className="tag warn">Model unavailable: quoted from sources</span>;
  if (meta.generator === "extractive") return <span className="tag">Quoted from sources</span>;
  if (meta.verified === null && meta.generator === "llm") return <span className="tag warn">Not verified</span>;
  return null;
}

function SourceList({ sources }: { sources: Source[] }) {
  return (
    <div className="sources">
      {sources.map((s) => {
        const body = <><div className="clip"><span className="cite">{s.n}</span> {s.title}</div><p>{s.snippet}</p></>;
        return s.url ? <a key={s.n} className="source" href={s.url} target="_blank" rel="noreferrer">{body}</a> : <div key={s.n} className="source">{body}</div>;
      })}
    </div>
  );
}

function Assistant({ m, onVote }: { m: Msg; onVote: (v: number) => void }) {
  return (
    <div className="msg-ai">
      <div className="body"><Cited text={m.content} /></div>
      {m.note && <p className="dim" style={{ marginTop: 6 }}>{m.note}</p>}
      {m.meta?.sources && m.meta.sources.length > 0 && <SourceList sources={m.meta.sources} />}
      {m.meta?.status && (
        <div className="msg-meta">
          <Status meta={m.meta} />
          {m.id && (
            <>
              <button className="btn ghost icon" style={{ height: 26, width: 26 }} aria-label="Good answer" aria-pressed={m.feedback === 1} onClick={() => onVote(m.feedback === 1 ? 0 : 1)}><ThumbsUp size={13} fill={m.feedback === 1 ? "currentColor" : "none"} /></button>
              <button className="btn ghost icon" style={{ height: 26, width: 26 }} aria-label="Bad answer" aria-pressed={m.feedback === -1} onClick={() => onVote(m.feedback === -1 ? 0 : -1)}><ThumbsDown size={13} fill={m.feedback === -1 ? "currentColor" : "none"} /></button>
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default function Chat() {
  const convs = useLoad<{ id: string; title: string }[]>("/conversations");
  const docs = useLoad<unknown[]>("/documents");
  const [active, setActive] = useState<string | null>(null);
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [stage, setStage] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const abort = useRef<AbortController | null>(null);
  const end = useRef<HTMLDivElement>(null);

  useEffect(() => { end.current?.scrollIntoView({ block: "end" }); }, [msgs, stage]);

  async function open(id: string) {
    const c = await api(`/conversations/${id}`);
    setActive(id);
    setMsgs(c.messages);
    setError("");
  }

  async function send(text: string) {
    text = text.trim();
    if (!text || busy) return;
    setInput("");
    setBusy(true);
    setError("");
    setMsgs((m) => [...m, { role: "user", content: text }, { role: "assistant", content: "" }]);
    const patch = (fn: (a: Msg) => Msg) => setMsgs((all) => [...all.slice(0, -1), fn(all[all.length - 1])]);
    const ctl = new AbortController();
    abort.current = ctl;
    try {
      await streamChat({ message: text, conversation_id: active }, (e) => {
        if (e.event === "start") setActive(e.conversation_id);
        if (e.event === "status") setStage(e.stage);
        if (e.event === "token") { setStage(""); patch((a) => ({ ...a, content: a.content + e.text })); }
        if (e.event === "replace") { setStage(""); patch((a) => ({ ...a, content: e.text, note: e.reason })); }
        if (e.event === "done") { setStage(""); setBusy(false); patch((a) => ({ ...a, id: e.message_id, meta: e })); }
      }, ctl.signal);
    } catch (e) {
      if ((e as Error).name !== "AbortError") setError((e as Error).message);
    } finally {
      setStage("");
      setBusy(false);
      convs.reload();
    }
  }

  function vote(i: number, value: number) {
    const m = msgs[i];
    api(`/messages/${m.id}/feedback`, { body: { value } });
    setMsgs(msgs.map((x, j) => (j === i ? { ...x, feedback: value } : x)));
  }

  const noDocs = docs.data && docs.data.length === 0;

  return (
    <div className="chat">
      <aside className="convs">
        <button className="btn" onClick={() => { setActive(null); setMsgs([]); }} style={{ marginBottom: 8 }}><Plus size={14} /> New chat</button>
        {convs.data?.map((c) => (
          <div key={c.id} className={`conv ${c.id === active ? "on" : ""}`} role="button" tabIndex={0} onClick={() => open(c.id)} onKeyDown={(e) => e.key === "Enter" && open(c.id)}>
            <span className="clip grow">{c.title}</span>
            <button className="btn ghost icon del" style={{ height: 22, width: 22 }} aria-label="Delete conversation" onClick={(e) => {
              e.stopPropagation();
              api(`/conversations/${c.id}`, { method: "DELETE" }).then(() => { if (c.id === active) { setActive(null); setMsgs([]); } convs.reload(); });
            }}><Trash2 size={12} /></button>
          </div>
        ))}
      </aside>
      <div className="thread-wrap">
        <div className="thread">
          <div className="thread-in">
            {!msgs.length && (
              <div className="hero">
                <h1>What do you need to know?</h1>
                <p className="muted" style={{ marginTop: 6 }}>
                  {noDocs ? <>Answers come from your sources, and you have none yet. <a href="/sources">Add some first</a>.</> : "Ask across everything you have connected, in English, Hindi or Hinglish."}
                </p>
                {!noDocs && (
                  <div className="prompts">
                    {["Summarise what changed recently", "What decisions have we made?", "Who is working on what?", "Hamare pending tasks kya hain?"].map((p) => <button key={p} onClick={() => send(p)}>{p}</button>)}
                  </div>
                )}
              </div>
            )}
            {msgs.map((m, i) => m.role === "user"
              ? <div key={i} className="msg-user">{m.content}</div>
              : (m.content || !busy) ? <Assistant key={i} m={m} onVote={(v) => vote(i, v)} /> : null)}
            {stage && <div className="stage"><i /> {stage}</div>}
            {error && <div className="error" role="alert">{error}</div>}
            <div ref={end} />
          </div>
        </div>
        <div className="composer">
          <form className="composer-in" onSubmit={(e) => { e.preventDefault(); send(input); }}>
            <textarea rows={1} value={input} placeholder="Ask anything" aria-label="Message" maxLength={2000} onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(input); } }} />
            {busy
              ? <button type="button" className="btn icon" aria-label="Stop" onClick={() => abort.current?.abort()}><Square size={13} /></button>
              : <button className="btn primary icon" aria-label="Send" disabled={!input.trim()}><ArrowUp size={16} /></button>}
          </form>
        </div>
      </div>
    </div>
  );
}
