"use client";

import { ArrowRight, Plus, Trash2, Workflow as Icon, X } from "lucide-react";
import { useState } from "react";
import { api } from "@/lib/api";
import { DocPicker, Empty, NoModelNote, Page, RunView, agentName, useLoad, useRun } from "@/components/ui";

type Step = { agent: string; task: string };
type Flow = { id: string; name: string; steps: Step[]; output: string; mine?: boolean; label?: string };
type Catalog = { agents: Record<string, string>; templates: (Flow & { label: string })[] };

function Launcher({ flow, onClose }: { flow: Flow; onClose: () => void }) {
  const { run, error, busy, start } = useRun();
  const [goal, setGoal] = useState("");
  const [docs, setDocs] = useState<string[]>([]);
  const [transcript, setTranscript] = useState("");
  const needsDocs = flow.steps.some((s) => s.agent === "data_analyst");
  const meeting = flow.steps.some((s) => s.agent === "meeting");
  return (
    <div className="scrim" onClick={onClose}>
      <form className="dialog" style={{ width: "min(620px, 100%)" }} onClick={(e) => e.stopPropagation()} onSubmit={(e) => {
        e.preventDefault();
        start({ goal, workflow: flow.id, document_ids: docs, transcript });
      }}>
        <div className="spread"><h2>{flow.label ?? flow.name}</h2><button type="button" className="btn ghost icon" aria-label="Close" onClick={onClose}><X size={16} /></button></div>
        <label className="field">What should it work on?<textarea className="input" rows={2} value={goal} required onChange={(e) => setGoal(e.target.value)} /></label>
        {(needsDocs || meeting) && <div className="field"><span className="muted">{needsDocs ? "Dataset" : "Recording or transcript file"}</span>
          <DocPicker value={docs} onChange={setDocs} max={needsDocs ? 1 : 6} filter={needsDocs ? (d) => /\.(csv|xlsx)$/i.test(d.title) : undefined} /></div>}
        {meeting && <label className="field">Or paste a transcript<textarea className="input" rows={4} value={transcript} onChange={(e) => setTranscript(e.target.value)} /></label>}
        {error && <div className="error" role="alert">{error}</div>}
        <button className="btn primary" disabled={busy || !goal.trim()}>{busy ? "Running" : "Run"}</button>
        {run && <RunView run={run} />}
      </form>
    </div>
  );
}

function Builder({ agents, onSaved }: { agents: Record<string, string>; onSaved: () => void }) {
  const [name, setName] = useState("");
  const [output, setOutput] = useState("text");
  const [steps, setSteps] = useState<Step[]>([{ agent: "research", task: "Gather the facts about {input}" }]);
  const [error, setError] = useState("");
  const put = (i: number, s: Partial<Step>) => setSteps(steps.map((x, j) => (j === i ? { ...x, ...s } : x)));
  async function save(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    try {
      await api("/workflows", { body: { name, steps, output } });
      setName("");
      onSaved();
    } catch (err) {
      setError((err as Error).message);
    }
  }
  return (
    <form className="card stack" onSubmit={save}>
      <div className="grid-2">
        <label className="field">Name<input className="input" value={name} required onChange={(e) => setName(e.target.value)} placeholder="Weekly project update" /></label>
        <label className="field">Result<select className="input" value={output} onChange={(e) => setOutput(e.target.value)}>
          <option value="text">Text</option><option value="docx">Word document</option><option value="pptx">Presentation</option></select></label>
      </div>
      {steps.map((s, i) => (
        <div className="row" key={i} style={{ alignItems: "flex-start" }}>
          <span className="mono dim" style={{ paddingTop: 8, width: 16 }}>{i + 1}</span>
          <select className="input" style={{ width: 150 }} value={s.agent} aria-label={`Step ${i + 1} agent`} onChange={(e) => put(i, { agent: e.target.value })}>
            {Object.keys(agents).map((a) => <option key={a} value={a}>{agentName(a)}</option>)}
          </select>
          <input className="input grow" value={s.task} required aria-label={`Step ${i + 1} task`} onChange={(e) => put(i, { task: e.target.value })} />
          <button type="button" className="btn ghost icon" aria-label="Remove step" disabled={steps.length === 1} onClick={() => setSteps(steps.filter((_, j) => j !== i))}><Trash2 size={14} /></button>
        </div>
      ))}
      <p className="dim">{"{input}"} becomes what you type when running it. Each step sees the output of the steps before it. Results are checked by the Verification agent.</p>
      {error && <div className="error" role="alert">{error}</div>}
      <div className="row">
        <button type="button" className="btn" disabled={steps.length >= 8} onClick={() => setSteps([...steps, { agent: "planning", task: "" }])}><Plus size={14} /> Add step</button>
        <button className="btn primary" disabled={!name.trim()}>Save workflow</button>
      </div>
    </form>
  );
}

export default function Workflows() {
  const catalog = useLoad<Catalog>("/agents");
  const saved = useLoad<Flow[]>("/workflows");
  const [running, setRunning] = useState<Flow | null>(null);

  const Card = ({ f, extra }: { f: Flow; extra?: React.ReactNode }) => (
    <div className="card stack" style={{ gap: 8 }}>
      <div className="spread"><b style={{ fontWeight: 600 }}>{f.label ?? f.name}</b>{extra}</div>
      <div className="dim row" style={{ gap: 4, flexWrap: "wrap" }}>
        {f.steps.map((s, i) => <span key={i} className="row" style={{ gap: 4 }}>{i > 0 && <ArrowRight size={11} />}{agentName(s.agent)}</span>)}
        {f.output !== "text" && <span className="tag" style={{ marginLeft: 4 }}>.{f.output}</span>}
      </div>
      <button className="btn" onClick={() => setRunning(f)} style={{ justifySelf: "start" }}>Run</button>
    </div>
  );

  return (
    <Page title="Workflows" sub="Repeatable work, run by agents. Anything sensitive waits for approval.">
      <NoModelNote />
      <div className="grid-3" style={{ marginTop: 8 }}>{catalog.data?.templates.map((t) => <Card key={t.id} f={t} />)}</div>
      <section className="section">
        <h2>Your workflows</h2>
        {saved.data && !saved.data.length && <Empty icon={Icon} title="None saved yet">Build your first one below.</Empty>}
        <div className="grid-3">
          {saved.data?.map((w) => <Card key={w.id} f={w} extra={w.mine && (
            <button className="btn ghost icon" style={{ height: 26, width: 26 }} aria-label={`Delete ${w.name}`} onClick={() => api(`/workflows/${w.id}`, { method: "DELETE" }).then(saved.reload)}><Trash2 size={13} /></button>
          )} />)}
        </div>
      </section>
      <section className="section">
        <h2>Build a workflow</h2>
        {catalog.data && <Builder agents={catalog.data.agents} onSaved={saved.reload} />}
      </section>
      {running && <Launcher flow={running} onClose={() => setRunning(null)} />}
    </Page>
  );
}
