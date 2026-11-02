"use client";

import { useState } from "react";
import { DocPicker, NoModelNote, Page, RunView, useRun } from "@/components/ui";

export default function Research() {
  const { run, error, busy, start } = useRun();
  const [topic, setTopic] = useState("");
  const [docs, setDocs] = useState<string[]>([]);
  return (
    <Page title="Research" sub="Compare papers and reports. Get findings, contradictions and open questions, with citations.">
      <NoModelNote />
      <form className="card stack" style={{ marginTop: 8 }} onSubmit={(e) => { e.preventDefault(); start({ goal: topic, workflow: "research", document_ids: docs }); }}>
        <label className="field">Research question<input className="input" value={topic} required onChange={(e) => setTopic(e.target.value)} placeholder="How do these papers evaluate retrieval quality?" /></label>
        <div className="field"><span className="muted">Papers to compare (up to 6)</span><DocPicker value={docs} onChange={setDocs} /></div>
        <button className="btn primary" style={{ justifySelf: "start" }} disabled={busy || !topic.trim()}>{busy ? "Researching" : "Write report"}</button>
      </form>
      {error && <div className="error" role="alert" style={{ marginTop: 12 }}>{error}</div>}
      {run && <section className="section card"><RunView run={run} /></section>}
    </Page>
  );
}
