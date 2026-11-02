"use client";

import { useState } from "react";
import { api } from "@/lib/api";

export default function Login() {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [form, setForm] = useState({ email: "", name: "", password: "", workspace: "" });
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const set = (k: keyof typeof form) => (e: React.ChangeEvent<HTMLInputElement>) => setForm({ ...form, [k]: e.target.value });

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      const body = mode === "login" ? { email: form.email, password: form.password } : { ...form, workspace: form.workspace || "My workspace" };
      await api(`/auth/${mode}`, { body });
      window.location.href = "/chat";
    } catch (err) {
      setError((err as Error).message);
      setBusy(false);
    }
  }

  return (
    <main className="auth">
      <form className="auth-box card stack" onSubmit={submit}>
        <div className="brand"><span className="brand-mark" />NEXUS</div>
        {mode === "register" && (
          <>
            <label className="field">Your name<input className="input" required value={form.name} onChange={set("name")} autoComplete="name" /></label>
            <label className="field">Workspace<input className="input" placeholder="My workspace" value={form.workspace} onChange={set("workspace")} /></label>
          </>
        )}
        <label className="field">Email<input className="input" type="email" required value={form.email} onChange={set("email")} autoComplete="email" /></label>
        <label className="field">
          Password
          <input className="input" type="password" required minLength={mode === "register" ? 10 : 1} value={form.password} onChange={set("password")}
            autoComplete={mode === "login" ? "current-password" : "new-password"} />
          {mode === "register" && <span className="dim">At least 10 characters</span>}
        </label>
        {error && <div className="error" role="alert">{error}</div>}
        <button className="btn primary" disabled={busy}>{busy ? "One moment" : mode === "login" ? "Sign in" : "Create workspace"}</button>
        <button type="button" className="btn ghost" onClick={() => { setMode(mode === "login" ? "register" : "login"); setError(""); }}>
          {mode === "login" ? "New here? Create a workspace" : "Already have an account? Sign in"}
        </button>
      </form>
    </main>
  );
}
