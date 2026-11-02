"use client";

import { Activity, Bell, BookOpen, Bot, CircleAlert, CalendarClock, Code2, Database, GitBranch, Gavel, LogOut, Menu, MessageSquare, Mic, Network, Settings, TrendingUp, Workflow, X } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { api, ago } from "@/lib/api";
import { Empty, useLoad } from "@/components/ui";

const NAV = [
  ["Work", [["/chat", "Chat", MessageSquare], ["/voice", "Voice", Mic], ["/agents", "Agents", Bot], ["/workflows", "Workflows", Workflow]]],
  ["Modes", [["/research", "Research", BookOpen], ["/developer", "Developer", Code2]]],
  ["Knowledge", [["/sources", "Sources", Database], ["/knowledge", "Explorer", Network]]],
  ["Admin", [["/analytics", "Analytics", Activity], ["/settings", "Settings", Settings]]],
] as const;

type Insight = { id: string; kind: string; title: string; detail: string; due: string; read: boolean; created_at: string };
const KIND: Record<string, [React.ComponentType<{ size?: number }>, string]> = {
  deadline: [CalendarClock, "Deadline"], decision: [Gavel, "Decision"], task: [GitBranch, "Open task"], update: [TrendingUp, "Update"],
  anomaly: [CircleAlert, "Unusual activity"], conflict: [CircleAlert, "Conflict"],
};

function Signals({ onClose, items, reload }: { onClose: () => void; items: Insight[]; reload: () => void }) {
  return (
    <>
      <div className="scrim" onClick={onClose} style={{ zIndex: 29 }} />
      <aside className="drawer" aria-label="Signals">
        <div className="spread"><h2>Signals</h2><button className="btn ghost icon" onClick={onClose} aria-label="Close"><X size={16} /></button></div>
        <p className="dim">Noticed in your connected sources after each sync.</p>
        {!items.length && <Empty icon={Bell} title="All quiet">Signals appear when connected sources change.</Empty>}
        {items.map((i) => {
          const [Icon, label] = KIND[i.kind] ?? [Bell, i.kind];
          return (
            <button key={i.id} className="card" style={{ textAlign: "left", cursor: "pointer", opacity: i.read ? 0.6 : 1 }}
              onClick={() => api(`/insights/${i.id}/read`, { method: "POST" }).then(reload)}>
              <div className="row dim"><Icon size={13} /> {label}{i.due && ` · ${i.due}`}<span style={{ marginLeft: "auto" }}>{ago(i.created_at)}</span></div>
              <div style={{ marginTop: 4 }}>{i.title}</div>
              {i.detail && <p className="dim">{i.detail}</p>}
            </button>
          );
        })}
      </aside>
    </>
  );
}

export default function AppLayout({ children }: { children: React.ReactNode }) {
  const path = usePathname();
  const [menu, setMenu] = useState(false);
  const [signals, setSignals] = useState(false);
  const me = useLoad<{ name: string; workspace: string }>("/auth/me");
  const insights = useLoad<Insight[]>("/insights", 30000);
  const approvals = useLoad<{ status: string }[]>("/approvals", 15000);
  const unread = insights.data?.filter((i) => !i.read).length ?? 0;
  const waiting = approvals.data?.filter((a) => a.status === "pending").length ?? 0;

  return (
    <div className="shell">
      <div className="mobile-bar">
        <div className="brand"><span className="brand-mark" />NEXUS</div>
        <button className="btn ghost icon" aria-label="Menu" onClick={() => setMenu(true)}><Menu size={18} /></button>
      </div>
      <nav className={`side ${menu ? "open" : ""}`} onClick={() => setMenu(false)}>
        <div className="brand"><span className="brand-mark" />NEXUS</div>
        {NAV.map(([group, items]) => (
          <div key={group}>
            <div className="nav-group">{group}</div>
            {items.map(([href, label, Icon]) => (
              <Link key={href} href={href} className={`nav ${path.startsWith(href) ? "on" : ""}`}>
                <Icon size={15} /> {label}
                {href === "/agents" && waiting > 0 && <span className="count" title="Waiting for your approval">{waiting}</span>}
              </Link>
            ))}
          </div>
        ))}
        <div className="side-foot">
          <button className="nav" onClick={() => setSignals(true)}><Bell size={15} /> Signals {unread > 0 && <span className="count">{unread}</span>}</button>
          <button className="nav" onClick={() => api("/auth/logout", { method: "POST" }).then(() => (window.location.href = "/login"))}>
            <LogOut size={15} /> <span className="clip">{me.data ? `${me.data.name}` : "Sign out"}</span>
          </button>
        </div>
      </nav>
      <main className="main" key={path}>{children}</main>
      {signals && <Signals items={insights.data ?? []} reload={insights.reload} onClose={() => setSignals(false)} />}
    </div>
  );
}
