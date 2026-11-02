"use client";

import { Mic, Square, Volume2 } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { api, streamChat } from "@/lib/api";

type State = "idle" | "listening" | "thinking" | "speaking";
const LABEL: Record<State, string> = { idle: "Tap to talk", listening: "Listening. Tap when done", thinking: "Thinking", speaking: "Speaking. Tap to interrupt" };
const LANGS = [["", "Auto"], ["en", "English"], ["hi", "हिन्दी"]] as const;
const SILENCE_MS = 1100;

export default function Voice() {
  const [state, setState] = useState<State>("idle");
  const [lang, setLang] = useState("");
  const [turns, setTurns] = useState<{ role: string; text: string }[]>([]);
  const [error, setError] = useState("");
  const orb = useRef<HTMLButtonElement>(null);
  const rec = useRef<MediaRecorder | null>(null);
  const audio = useRef<HTMLAudioElement | null>(null);
  const conv = useRef<string | null>(null);
  const ctl = useRef<AbortController | null>(null);
  const stopMeter = useRef<() => void>(() => {});
  const log = useRef<HTMLDivElement>(null);

  useEffect(() => { log.current?.scrollTo(0, log.current.scrollHeight); }, [turns]);
  useEffect(() => () => { rec.current?.state === "recording" && rec.current.stop(); audio.current?.pause(); ctl.current?.abort(); stopMeter.current(); }, []);

  /** Drives the orb from mic level and ends the turn after a pause in speech. */
  function meter(stream: MediaStream) {
    const ac = new AudioContext();
    const an = ac.createAnalyser();
    ac.createMediaStreamSource(stream).connect(an);
    const buf = new Uint8Array(an.fftSize);
    let raf = 0, heard = false, quietSince = 0;
    const tick = () => {
      an.getByteTimeDomainData(buf);
      const rms = Math.sqrt(buf.reduce((s, v) => s + ((v - 128) / 128) ** 2, 0) / buf.length);
      orb.current?.style.setProperty("--level", String(Math.min(rms * 5, 1)));
      const now = performance.now();
      if (rms > 0.03) { heard = true; quietSince = 0; }
      else if (heard) {
        quietSince ||= now;
        if (now - quietSince > SILENCE_MS) { rec.current?.stop(); return; }
      }
      raf = requestAnimationFrame(tick);
    };
    tick();
    stopMeter.current = () => { cancelAnimationFrame(raf); ac.close(); orb.current?.style.setProperty("--level", "0"); };
  }

  async function listen() {
    setError("");
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch {
      setError("Microphone access was blocked. Allow it in your browser and try again.");
      return;
    }
    const chunks: Blob[] = [];
    const r = new MediaRecorder(stream);
    rec.current = r;
    r.ondataavailable = (e) => chunks.push(e.data);
    r.onstop = () => {
      stopMeter.current();
      stream.getTracks().forEach((t) => t.stop());
      process(new Blob(chunks, { type: r.mimeType }));
    };
    r.start();
    meter(stream);
    setState("listening");
  }

  async function process(blob: Blob) {
    setState("thinking");
    try {
      const form = new FormData();
      form.append("file", blob, "speech.webm");
      form.append("language", lang);
      const heard = await api<{ text: string }>("/voice/transcribe", { form });
      if (!heard.text.trim()) { setError("I didn't catch that. Try again."); setState("idle"); return; }
      setTurns((t) => [...t, { role: "you", text: heard.text }, { role: "nexus", text: "" }]);
      const patch = (fn: (s: string) => string) => setTurns((t) => [...t.slice(0, -1), { role: "nexus", text: fn(t[t.length - 1].text) }]);
      let answer = "";
      ctl.current = new AbortController();
      await streamChat({ message: heard.text, conversation_id: conv.current, channel: "voice" }, (e) => {
        if (e.event === "start") conv.current = e.conversation_id;
        if (e.event === "token") { answer += e.text; patch((s) => s + e.text); }
        if (e.event === "replace") { answer = e.text; patch(() => e.text); }
      }, ctl.current.signal);
      await say(answer.replace(/\[\d+\]/g, ""));
    } catch (e) {
      if ((e as Error).name !== "AbortError") setError((e as Error).message);
      setState("idle");
    }
  }

  async function say(text: string) {
    const r = await fetch("/api/voice/speak", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text: text.slice(0, 2400) }) });
    if (!r.ok) {
      setError("Couldn't play the reply aloud, so it is shown as text.");
      setState("idle");
      return;
    }
    const url = URL.createObjectURL(await r.blob());
    const a = new Audio(url);
    audio.current = a;
    a.onended = () => { URL.revokeObjectURL(url); setState("idle"); };
    setState("speaking");
    await a.play();
  }

  function tap() {
    if (state === "idle") listen();
    else if (state === "listening") rec.current?.stop();
    else if (state === "speaking") { audio.current?.pause(); listen(); }      // barge in; the conversation carries on
    else if (state === "thinking") { ctl.current?.abort(); setState("idle"); }
  }

  return (
    <div className="voice">
      <div className="seg" role="group" aria-label="Language">
        {LANGS.map(([v, l]) => <button key={v} className={lang === v ? "on" : ""} aria-pressed={lang === v} onClick={() => setLang(v)}>{l}</button>)}
      </div>
      <button ref={orb} className="orb" data-state={state} onClick={tap} aria-label={LABEL[state]}>
        {state === "speaking" ? <Volume2 size={38} /> : state === "thinking" ? <Square size={30} /> : <Mic size={38} />}
      </button>
      <p className="muted" aria-live="polite">{LABEL[state]}</p>
      {error && <div className="error" role="alert">{error}</div>}
      {turns.length > 0 && (
        <div className="transcript" ref={log}>
          {turns.map((t, i) => <div key={i}><span className="dim">{t.role === "you" ? "You" : "Nexus"}</span><div>{t.text.replace(/\s*\[\d+\]/g, "")}</div></div>)}
        </div>
      )}
    </div>
  );
}
