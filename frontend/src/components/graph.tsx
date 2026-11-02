"use client";

import { useEffect, useMemo, useRef, useState } from "react";

export type GraphData = { nodes: { id: string; name: string; type: string; degree: number }[]; edges: { source: string; target: string; label: string }[] };

// Validated for dark surfaces with the dataviz palette checker. The order follows the type, never its rank.
export const TYPE_COLOR: Record<string, string> = {
  project: "#84a326", person: "#4d8fd6", decision: "#d1702c", technology: "#239a84", organization: "#8a78e6", document: "#6b7480",
};

type P = { x: number; y: number; vx: number; vy: number };
const W = 900, H = 480, PAD = 46, TICKS = 260, MAX_NODES = 70;

/** Force layout fitted to the frame afterwards, so nothing piles up against the edges. Stops once settled. */
export function Graph({ data, hidden }: { data: GraphData; hidden: Set<string> }) {
  const pos = useRef<Record<string, P>>({});
  const [, frame] = useState(0);
  const [hover, setHover] = useState<string | null>(null);

  const { nodes, edges } = useMemo(() => {
    const nodes = data.nodes.filter((n) => !hidden.has(n.type)).sort((a, b) => b.degree - a.degree).slice(0, MAX_NODES);
    const ids = new Set(nodes.map((n) => n.id));
    return { nodes, edges: data.edges.filter((e) => ids.has(e.source) && ids.has(e.target)) };
  }, [data, hidden]);

  const near = useMemo(() => {
    const s = new Set<string>();
    if (hover) { s.add(hover); edges.forEach((e) => { if (e.source === hover) s.add(e.target); if (e.target === hover) s.add(e.source); }); }
    return s;
  }, [hover, edges]);

  useEffect(() => {
    const P = pos.current;
    nodes.forEach((n, i) => {
      const a = i * 2.4;                         // golden-angle spiral: no two nodes start in the same spot
      const r = 30 * Math.sqrt(i + 1);
      P[n.id] ??= { x: Math.cos(a) * r, y: Math.sin(a) * r, vx: 0, vy: 0 };
    });
    let tick = 0, raf = 0;
    const step = () => {
      const heat = 1 - tick / TICKS;
      for (let i = 0; i < nodes.length; i++) for (let j = i + 1; j < nodes.length; j++) {
        const a = P[nodes[i].id], b = P[nodes[j].id];
        const dx = a.x - b.x || 0.1, dy = a.y - b.y || 0.1, d2 = Math.max(dx * dx + dy * dy, 25), f = 900 / d2;
        a.vx += dx * f * 0.05; a.vy += dy * f * 0.05; b.vx -= dx * f * 0.05; b.vy -= dy * f * 0.05;
      }
      edges.forEach((e) => {
        const a = P[e.source], b = P[e.target];
        const dx = b.x - a.x, dy = b.y - a.y, d = Math.hypot(dx, dy) || 1, f = (d - 70) * 0.03;
        a.vx += dx / d * f; a.vy += dy / d * f; b.vx -= dx / d * f; b.vy -= dy / d * f;
      });
      nodes.forEach((n) => {
        const p = P[n.id];
        p.vx -= p.x * 0.012; p.vy -= p.y * 0.02;
        p.x += p.vx * heat; p.y += p.vy * heat;
        p.vx *= 0.82; p.vy *= 0.82;
      });
      frame((f) => f + 1);
      if (++tick < TICKS) raf = requestAnimationFrame(step);
    };
    step();
    return () => cancelAnimationFrame(raf);
  }, [nodes, edges]);

  const placed = nodes.map((n) => pos.current[n.id]).filter(Boolean);
  const xs = placed.map((p) => p.x), ys = placed.map((p) => p.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
  const scale = Math.min((W - PAD * 2) / Math.max(maxX - minX, 1), (H - PAD * 2) / Math.max(maxY - minY, 1), 1.6);
  const fit = (p: P) => ({ x: W / 2 + (p.x - (minX + maxX) / 2) * scale, y: H / 2 + (p.y - (minY + maxY) / 2) * scale });

  return (
    <svg className="graph" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Knowledge graph with ${nodes.length} entities`}>
      {edges.map((e, i) => {
        const pa = pos.current[e.source], pb = pos.current[e.target];
        if (!pa || !pb) return null;
        const a = fit(pa), b = fit(pb);
        const lit = hover && (e.source === hover || e.target === hover);
        return (
          <g key={i} opacity={hover && !lit ? 0.08 : 1}>
            <line x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke={lit ? "#aab2bd" : "#2c333d"} strokeWidth={lit ? 1.5 : 1} />
            {lit && e.label !== "mentions" && <text x={(a.x + b.x) / 2} y={(a.y + b.y) / 2 - 4} fill="#aab2bd" fontSize="11" textAnchor="middle" paintOrder="stroke" stroke="#12151a" strokeWidth="3">{e.label}</text>}
          </g>
        );
      })}
      {nodes.map((n) => {
        const p = pos.current[n.id];
        if (!p) return null;
        const { x, y } = fit(p);
        const r = 5 + Math.min(n.degree, 8) * 0.8;
        const labelled = hover ? near.has(n.id) : n.degree >= 3 || nodes.length <= 24;
        return (
          <g key={n.id} transform={`translate(${x},${y})`} opacity={hover && !near.has(n.id) ? 0.18 : 1}
            onPointerEnter={() => setHover(n.id)} onPointerLeave={() => setHover(null)}>
            <circle r={r + 8} fill="transparent" />
            <circle r={r} fill={TYPE_COLOR[n.type] ?? "#6b7480"} stroke="#12151a" strokeWidth="2" />
            {labelled && <text y={r + 13} fontSize="11.5" fill="#eef0f3" textAnchor="middle" paintOrder="stroke" stroke="#12151a" strokeWidth="3">{n.name.length > 26 ? n.name.slice(0, 25) + "…" : n.name}</text>}
          </g>
        );
      })}
    </svg>
  );
}
