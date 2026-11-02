"use client";

import { useState } from "react";
import { NoModelNote, Page, RunView, useLoad, useRun } from "@/components/ui";

type Conn = { kind: string; config: { repo?: string }; name: string };

export default function Developer() {
  const { run, error, busy, start } = useRun();
  const conns = useLoad<Conn[]>("/connectors");
  const repos = (conns.data ?? []).filter((c) => c.kind === "github" && c.config.repo).map((c) => c.config.repo!);
  const [mode, setMode] = useState<"issue" | "code_review">("issue");
  const [repo, setRepo] = useState("");
  const [issue, setIssue] = useState("");
  const [notes, setNotes] = useState("");
  const chosen = repo || repos[0] || "";

  function submit(e: React.FormEvent) {
    e.preventDefault();
    if (mode === "issue") start({ goal: notes || `Investigate issue #${issue}`, workflow: "issue", repo: chosen, issue: Number(issue) });
    else start({ goal: notes, workflow: "code_review", repo: chosen });
  }

  return (
    <Page title="Developer" sub="Investigate issues and review code using your indexed repositories.">
      <NoModelNote />
      <form className="card stack" style={{ marginTop: 8 }} onSubmit={submit}>
        <div className="seg" style={{ justifySelf: "start" }}>
          <button type="button" className={mode === "issue" ? "on" : ""} onClick={() => setMode("issue")}>Investigate issue</button>
          <button type="button" className={mode === "code_review" ? "on" : ""} onClick={() => setMode("code_review")}>Review code</button>
        </div>
        <div className="grid-2">
          <label className="field">Repository
            {repos.length
              ? <select className="input" value={chosen} onChange={(e) => setRepo(e.target.value)}>{repos.map((r) => <option key={r}>{r}</option>)}</select>
              : <input className="input" placeholder="owner/name" value={repo} onChange={(e) => setRepo(e.target.value)} />}
            {conns.data && !repos.length && <span className="dim"><a href="/sources">Connect it under Sources</a> so its code can be searched.</span>}
          </label>
          {mode === "issue" && <label className="field">Issue number<input className="input" inputMode="numeric" pattern="[0-9]+" required value={issue} onChange={(e) => setIssue(e.target.value)} /></label>}
        </div>
        <label className="field">{mode === "issue" ? "Anything to focus on" : "What should be reviewed?"}
          <textarea className="input" rows={2} value={notes} required={mode === "code_review"} onChange={(e) => setNotes(e.target.value)} placeholder={mode === "code_review" ? "The authentication middleware" : ""} /></label>
        <button className="btn primary" style={{ justifySelf: "start" }} disabled={busy}>{busy ? "Working" : mode === "issue" ? "Investigate" : "Review"}</button>
      </form>
      {error && <div className="error" role="alert" style={{ marginTop: 12 }}>{error}</div>}
      {run && <section className="section card"><RunView run={run} /></section>}
    </Page>
  );
}
