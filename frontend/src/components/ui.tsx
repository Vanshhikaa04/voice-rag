"use client";

import { Download, FileText, Loader2, Sparkles } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { api, ago } from "@/lib/api";

/** Loads on mount and on an interval; keeps the last good data through a failed refresh. */
export function useLoad<T>(path: string | null, every = 0) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState("");
  const pathRef = useRef(path);
  pathRef.current = path;
  const reload = useCallback(async () => {
    if (!pathRef.current) return;
    try {
      setData(await api<T>(pathRef.current));
      setError("");
    } catch (e) {
      setError((e as Error).message);
    }
  }, []);
  useEffect(() => {
    reload();
    if (!every) return;
    const t = setInterval(() => { if (!document.hidden) reload(); }, every);
    return () => clearInterval(t);
  }, [reload, every, path]);
  return { data, error, reload, setData };
}

export function Empty({ icon: Icon, title, children }: { icon: React.ComponentType<{ size?: number }>; title: string; children?: React.ReactNode }) {
  return <div className="empty"><Icon size={22} /><strong>{title}</strong>{children}</div>;
}

export function Skeleton({ rows = 3 }: { rows?: number }) {
  return <div className="stack">{Array.from({ length: rows }, (_, i) => <div key={i} className="skeleton" />)}</div>;
}

export function Spinner({ size = 14 }: { size?: number }) {
  return <Loader2 size={size} className="spin" />;
}

export function Page({ title, sub, action, children }: { title: string; sub?: string; action?: React.ReactNode; children: React.ReactNode }) {
  return (
    <div className="page">
      <header className="page-head">
        <div><h1>{title}</h1>{sub && <p>{sub}</p>}</div>
        {action}
      </header>
      {children}
    </div>
  );
}

const VERDICT: Record<string, [string, string]> = {
  verified: ["good", "Verified against sources"],
  partial: ["warn", "Partly verified"],
  unsupported: ["bad", "Not supported by sources"],
  unchecked: ["", "No sources to check"],
};
const AGENT_NAMES: Record<string, string> = {
  supervisor: "Supervisor", research: "Research", data_analyst: "Data Analyst", coding: "Coding", meeting: "Meeting",
  planning: "Planning", action: "Action", verification: "Verification",
};
export const agentName = (a: string) => AGENT_NAMES[a] ?? a;

export type Run = {
  id: string; workflow: string; goal: string; status: string; verdict: string; result: string; error: string; has_artifact: boolean;
  started_at: string; finished_at: string | null;
  steps?: { id: string; agent: string; task: string; output: string; status: string; ms: number }[];
};

export function RunView({ run, compact }: { run: Run; compact?: boolean }) {
  const [open, setOpen] = useState<string | null>(null);
  const active = run.status === "running" || run.status === "queued";
  const verdict = VERDICT[run.verdict];
  return (
    <div className="stack">
      <div className="row" style={{ flexWrap: "wrap" }}>
        {active && <span className="tag live"><Spinner size={11} /> Working</span>}
        {run.status === "succeeded" && <span className="tag good">Finished</span>}
        {run.status === "failed" && <span className="tag bad">Failed</span>}
        {verdict && <span className={`tag ${verdict[0]}`}>{verdict[1]}</span>}
        <span className="dim">{ago(run.started_at)}</span>
      </div>
      {!!run.steps?.length && (
        <div>
          {run.steps.map((s) => (
            <div key={s.id} className="step" data-s={s.status}>
              <span className="dot" />
              <button className="btn ghost" style={{ height: 24, padding: "0 6px", margin: "0 0 0 -6px" }} onClick={() => setOpen(open === s.id ? null : s.id)}>
                <b style={{ color: "var(--ink)", fontWeight: 500 }}>{agentName(s.agent)}</b>
                {s.status !== "running" && <span className="dim mono">{(s.ms / 1000).toFixed(1)}s</span>}
              </button>
              {open === s.id && <div className="output">{s.output || s.task}</div>}
            </div>
          ))}
        </div>
      )}
      {run.status === "failed" && <div className="error">{run.error}</div>}
      {run.result && !compact && <div className="result">{run.result}</div>}
      {run.status === "succeeded" && run.error && <p className="dim">{run.error}</p>}
      {run.has_artifact && <a className="btn" href={`/api/runs/${run.id}/artifact`} style={{ justifySelf: "start" }}><Download size={14} /> Download file</a>}
    </div>
  );
}

/** Starts a run, then polls it until finished. Shared by every workspace that delegates to agents. */
export function useRun() {
  const [run, setRun] = useState<Run | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const timer = useRef<ReturnType<typeof setInterval>>(undefined);
  useEffect(() => () => clearInterval(timer.current), []);
  const start = useCallback(async (body: Record<string, unknown>) => {
    setBusy(true);
    setError("");
    setRun(null);
    clearInterval(timer.current);
    try {
      const r = await api<Run>("/runs", { body });
      setRun(r);
      timer.current = setInterval(async () => {
        const next = await api<Run>(`/runs/${r.id}`).catch(() => null);
        if (!next) return;
        setRun(next);
        if (next.status === "succeeded" || next.status === "failed") {
          clearInterval(timer.current);
          setBusy(false);
        }
      }, 900);
    } catch (e) {
      setError((e as Error).message);
      setBusy(false);
    }
  }, []);
  return { run, error, busy, start };
}

export type Doc = { id: string; title: string; kind: string; visibility: string; chunks: number; url: string; updated_at: string; connector_id: string | null };

export function DocPicker({ value, onChange, filter, max = 6 }: { value: string[]; onChange: (v: string[]) => void; filter?: (d: Doc) => boolean; max?: number }) {
  const { data } = useLoad<Doc[]>("/documents");
  const docs = (data ?? []).filter(filter ?? (() => true));
  if (!data) return <Skeleton rows={1} />;
  if (!docs.length) return <Empty icon={FileText} title="Nothing suitable yet"><a href="/sources">Add documents under Sources</a></Empty>;
  return (
    <div className="stack" style={{ gap: 4, maxHeight: 220, overflowY: "auto" }}>
      {docs.map((d) => (
        <label key={d.id} className="row" style={{ cursor: "pointer" }}>
          <input type="checkbox" checked={value.includes(d.id)} disabled={!value.includes(d.id) && value.length >= max}
            onChange={(e) => onChange(e.target.checked ? [...value, d.id] : value.filter((x) => x !== d.id))} />
          <span className="clip">{d.title}</span>
        </label>
      ))}
    </div>
  );
}

export function NoModelNote() {
  const { data } = useLoad<{ available: boolean }>("/agents");
  if (!data || data.available) return null;
  return <div className="error"><Sparkles size={13} /> Agents need a language model. Set OPENAI_API_KEY or ANTHROPIC_API_KEY on the server.</div>;
}
