// MAP · CLUSTERS: every vault strategy as a node in its archetype's cluster. A node sits closer to its cluster's core
// the higher it scores, is sized by its trades and ringed when it passed the held-out test; clusters slowly orbit.
// Scroll zooms around the cursor, dragging empty space pans, dragging a cluster's core moves it, clicking a core opens
// its callout (count, share, best, worst, averages, dominant market), clicking a node opens the strategy.
// One canvas, one rAF loop paced by the FX tier (FULL ~60 fps, LITE ~20 fps, OFF / reduced motion: still).
// idea: research MAP (RF 442-444, 475-482, GNU 34-40); the drawing and code are ARQVIO's.

import { useEffect, useMemo, useRef, useState } from "react";
import type { StrategyRow } from "../api/ws";
import { cssVar, FAMILY_NAMES, fitCanvas, fmt, shortId } from "../ws/util";
import { familyColour, heldOutPositive } from "./StrategyMap";

// what a family trades like, in a trader's words (the engine's name stays in the callout)
const ARCHETYPE: Record<string, string> = {
  MA_CROSS: "TREND FOLLOWER", RSI_REVERSION: "MEAN REVERTER", STOCH_CROSS: "OSCILLATOR", MACD_CROSS: "MOMENTUM",
  DONCHIAN_BREAKOUT: "BREAKOUT HUNTER", SIGNALS: "MULTI SIGNAL",
};
const archetype = (f: string) => ARCHETYPE[f] ?? f;

interface Node { r: StrategyRow; ang: number; dist: number; size: number }
interface Cluster { fam: string; colour: string; rows: StrategyRow[]; nodes: Node[]; x: number; y: number; radius: number }

function hash(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619);
  return (h >>> 0) / 4294967296;
}

function callout(c: Cluster, total: number) {
  const rets = c.rows.map((r) => r.total_return ?? 0);
  const dds = c.rows.map((r) => r.max_dd ?? 0);
  const best = [...c.rows].sort((a, b) => b.score - a.score)[0];
  const worst = [...c.rows].sort((a, b) => a.score - b.score)[0];
  const bySym = new Map<string, number>();
  c.rows.forEach((r) => bySym.set(`${r.symbol} ${r.timeframe}`, (bySym.get(`${r.symbol} ${r.timeframe}`) ?? 0) + 1));
  const dom = [...bySym.entries()].sort((a, b) => b[1] - a[1])[0];
  const avg = (xs: number[]) => xs.reduce((s, x) => s + x, 0) / Math.max(1, xs.length);
  return { count: c.rows.length, share: c.rows.length / Math.max(1, total), best, worst, avgRet: avg(rets), avgDd: avg(dds),
    avgScore: avg(c.rows.map((r) => r.score)), dominant: dom ? `${dom[0]} (${dom[1]})` : "—" };
}

export function NeuralMap({ rows, onOpen }: { rows: StrategyRow[]; onOpen: (id: string) => void }) {
  const ref = useRef<HTMLCanvasElement | null>(null);
  const [picked, setPicked] = useState<string | null>(null);
  const [tip, setTip] = useState<{ x: number; y: number; r: StrategyRow } | null>(null);
  const fams = useMemo(() => [...new Set(rows.map((r) => r.family))].sort(), [rows]);
  // cluster positions are kept across data refreshes, and wherever the owner dragged them
  const placed = useRef<Record<string, { x: number; y: number }>>({});
  const view = useRef({ zoom: 1, px: 0, py: 0 });
  const fitted = useRef(false);
  const redraw = useRef<(() => void) | null>(null);  // FX OFF / reduced motion: one frame per change
  useEffect(() => { fitted.current = false; }, [rows.length]);  // a new filter: fit the new picture once
  // the loop reads these through refs, so a parent re-render never restarts the orbit
  const openRef = useRef(onOpen);
  openRef.current = onOpen;
  const pickedRef = useRef(picked);
  pickedRef.current = picked;

  const clusters: Cluster[] = useMemo(() => {
    const groups = fams.map((f) => rows.filter((r) => r.family === f));
    const maxTrades = Math.max(1, ...rows.map((r) => r.n_trades));
    const radii = groups.map((g) => 50 + 11 * Math.sqrt(g.length));
    // homes on a ring, then pushed apart until no two clusters overlap (dragged clusters stay where they were put)
    const ring = Math.max(260, radii.reduce((s, r) => s + r, 0) / Math.PI);
    const pos = fams.map((fam, i) => {
      const a = (i / Math.max(1, fams.length)) * Math.PI * 2 - Math.PI / 2;
      return { ...(placed.current[fam] ?? { x: Math.cos(a) * ring * 1.25, y: Math.sin(a) * ring * 0.8 }), fixed: !!placed.current[fam] };
    });
    for (let it = 0; it < 120; it++) {
      let moved = false;
      for (let i = 0; i < pos.length; i++) {
        for (let j = i + 1; j < pos.length; j++) {
          const dx = pos[j].x - pos[i].x;
          const dy = pos[j].y - pos[i].y;
          const d = Math.hypot(dx, dy) || 1;
          const need = radii[i] + radii[j] + 50;
          if (d >= need) continue;
          const push = (need - d) / 2;
          const ux = dx / d;
          const uy = dy / d;
          if (!pos[i].fixed) { pos[i].x -= ux * push; pos[i].y -= uy * push; }
          if (!pos[j].fixed) { pos[j].x += ux * push; pos[j].y += uy * push; }
          moved = true;
        }
      }
      if (!moved) break;
    }
    return fams.map((fam, i) => {
      const g = groups[i];
      const radius = radii[i];
      const home = pos[i];
      return {
        fam, colour: familyColour(fam), rows: g, radius, x: home.x, y: home.y,
        nodes: g.map((r) => ({ r, ang: hash(r.id) * Math.PI * 2, dist: 0.15 + 0.85 * (1 - Math.min(1, Math.max(0, r.score / 100))),
          size: 1.6 + 3.2 * Math.sqrt(r.n_trades / maxTrades) })),
      };
    });
  }, [rows, fams]);
  const data = useRef(clusters);
  data.current = clusters;

  useEffect(() => {
    const c = ref.current;
    if (!c) return undefined;
    const fx = document.documentElement.dataset.fx ?? "FULL";
    const reduced = typeof window.matchMedia === "function" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const still = fx === "OFF" || reduced;
    const frameMs = fx === "FULL" ? 15 : 48;
    let raf = 0;
    let last = 0;
    const t0 = performance.now();
    const mouse = { x: 0, y: 0 };
    // sx/sy follow the mouse for the deltas; ox/oy keep the press point, so a slow drag still counts as a drag
    const drag: { mode: "pan" | "cluster" | null; fam?: string; sx: number; sy: number; ox: number; oy: number; moved: boolean } =
      { mode: null, sx: 0, sy: 0, ox: 0, oy: 0, moved: false };

    const toWorld = (sx: number, sy: number, w: number, h: number) => {
      const v = view.current;
      return { x: (sx - w / 2 - v.px) / v.zoom, y: (sy - h / 2 - v.py) / v.zoom };
    };
    const nodePos = (cl: Cluster, n: Node, t: number) => {
      const spin = still ? 0 : t * 0.04 * (0.6 + n.dist);
      const a = n.ang + spin;
      return { x: cl.x + Math.cos(a) * n.dist * cl.radius, y: cl.y + Math.sin(a) * n.dist * cl.radius * 0.9 };
    };

    const draw = (now: number) => {
      const fit = fitCanvas(c);
      if (!fit) return;
      const { ctx, w, h } = fit;
      const t = (now - t0) / 1000;
      const v = view.current;
      if (!fitted.current && data.current.length) {  // first frame: every cluster on screen, legend and help clear
        const xs = data.current.flatMap((cl) => [cl.x - cl.radius, cl.x + cl.radius]);
        const ys = data.current.flatMap((cl) => [cl.y - cl.radius - 30, cl.y + cl.radius]);
        const bw = Math.max(...xs) - Math.min(...xs);
        const bh = Math.max(...ys) - Math.min(...ys);
        v.zoom = Math.min(1.4, Math.max(0.3, Math.min((w - 80) / bw, (h - 80) / bh)));
        v.px = -((Math.max(...xs) + Math.min(...xs)) / 2) * v.zoom;
        v.py = -((Math.max(...ys) + Math.min(...ys)) / 2) * v.zoom;
        fitted.current = true;
      }
      ctx.clearRect(0, 0, w, h);
      // parallax: the background grid drifts against the mouse
      const red = cssVar("--brand-red", "#ff1744");
      ctx.save();
      ctx.strokeStyle = "rgba(255,23,68,0.05)";
      ctx.lineWidth = 1;
      const cell = 66 * v.zoom;
      const ox = ((v.px - (mouse.x - w / 2) * 0.02) % cell + cell) % cell;
      const oy = ((v.py - (mouse.y - h / 2) * 0.02) % cell + cell) % cell;
      for (let x = ox; x < w; x += cell) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke(); }
      for (let y = oy; y < h; y += cell) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }
      ctx.restore();
      ctx.save();
      ctx.translate(w / 2 + v.px, h / 2 + v.py);
      ctx.scale(v.zoom, v.zoom);
      for (const cl of data.current) {
        // spokes (one path per cluster), then nodes, then the core and its label
        const pts = cl.nodes.map((n) => nodePos(cl, n, t));
        ctx.globalAlpha = 0.12;
        ctx.strokeStyle = cl.colour;
        ctx.lineWidth = 0.6 / v.zoom;
        ctx.beginPath();
        for (const p of pts) {
          ctx.moveTo(cl.x, cl.y);
          ctx.lineTo(p.x, p.y);
        }
        ctx.stroke();
        for (let k = 0; k < cl.nodes.length; k++) {
          const n = cl.nodes[k];
          const p = pts[k];
          const ret = n.r.total_return ?? 0;
          ctx.globalAlpha = 0.45 + 0.5 * Math.min(1, Math.max(0, ret));
          ctx.fillStyle = cl.colour;
          ctx.beginPath();
          ctx.arc(p.x, p.y, n.size, 0, Math.PI * 2);
          ctx.fill();
          if (heldOutPositive(n.r)) {
            ctx.globalAlpha = 0.9;
            ctx.strokeStyle = "#3FBF7F";
            ctx.lineWidth = 1 / v.zoom;
            ctx.beginPath();
            ctx.arc(p.x, p.y, n.size + 1.6, 0, Math.PI * 2);
            ctx.stroke();
          }
        }
        ctx.globalAlpha = 1;
        const on = pickedRef.current === cl.fam;
        ctx.fillStyle = on ? red : cl.colour;
        ctx.beginPath();
        ctx.arc(cl.x, cl.y, 4 + (on ? 2 : 0), 0, Math.PI * 2);
        ctx.fill();
        ctx.font = `700 ${11 / Math.max(0.6, v.zoom)}px Consolas, monospace`;
        ctx.textAlign = "center";
        ctx.fillStyle = "#eef1f5";
        ctx.fillText(archetype(cl.fam), cl.x, cl.y - cl.radius - 14);
        ctx.fillStyle = cl.colour;
        ctx.fillText(String(cl.rows.length), cl.x, cl.y - cl.radius - 1);
      }
      ctx.restore();
    };

    const hit = (sx: number, sy: number) => {
      const r = c.getBoundingClientRect();
      const wp = toWorld(sx, sy, r.width, r.height);
      const t = (performance.now() - t0) / 1000;
      for (const cl of data.current) {
        if (Math.hypot(wp.x - cl.x, wp.y - cl.y) < 14 / view.current.zoom) return { cluster: cl };
      }
      let best: { node: Node; d: number } | null = null;
      for (const cl of data.current) {
        for (const n of cl.nodes) {
          const p = nodePos(cl, n, t);
          const d = Math.hypot(wp.x - p.x, wp.y - p.y);
          if (d < n.size + 4 / view.current.zoom && (!best || d < best.d)) best = { node: n, d };
        }
      }
      return best ? { node: best.node } : {};
    };
    const local = (e: MouseEvent) => {
      const r = c.getBoundingClientRect();
      return { x: e.clientX - r.left, y: e.clientY - r.top };
    };
    const onDown = (e: MouseEvent) => {
      const p = local(e);
      const h = hit(p.x, p.y);
      drag.sx = drag.ox = p.x;
      drag.sy = drag.oy = p.y;
      drag.moved = false;
      drag.mode = h.cluster ? "cluster" : "pan";
      drag.fam = h.cluster?.fam;
    };
    const onMove = (e: MouseEvent) => {
      const p = local(e);
      mouse.x = p.x;
      mouse.y = p.y;
      if (drag.mode) {
        const dx = p.x - drag.sx;
        const dy = p.y - drag.sy;
        if (Math.abs(p.x - drag.ox) + Math.abs(p.y - drag.oy) > 3) drag.moved = true;
        drag.sx = p.x;
        drag.sy = p.y;
        if (drag.mode === "pan") {
          view.current.px += dx;
          view.current.py += dy;
        } else {
          const cl = data.current.find((x) => x.fam === drag.fam);
          if (cl) {
            cl.x += dx / view.current.zoom;
            cl.y += dy / view.current.zoom;
            placed.current[cl.fam] = { x: cl.x, y: cl.y };
          }
        }
        setTip(null);
      } else if (e.target === c) {  // hover only over the canvas itself, never through the legend or callout
        const h = hit(p.x, p.y);
        setTip(h.node ? { x: p.x, y: p.y, r: h.node.r } : null);
        c.style.cursor = h.cluster ? "grab" : h.node ? "pointer" : "default";
      } else {
        setTip(null);
        return;
      }
      if (still) draw(performance.now());
    };
    const onUp = (e: MouseEvent) => {
      if (!drag.mode) return;  // the press did not start on the canvas
      const p = local(e);
      if (!drag.moved) {
        const h = hit(p.x, p.y);
        if (h.cluster) setPicked((x) => (x === h.cluster!.fam ? null : h.cluster!.fam));
        else if (h.node) openRef.current(h.node.r.id);
        else setPicked(null);
      }
      drag.mode = null;
      if (still) draw(performance.now());
    };
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const p = local(e);
      const r = c.getBoundingClientRect();
      const v = view.current;
      const before = toWorld(p.x, p.y, r.width, r.height);
      v.zoom = Math.min(4, Math.max(0.3, v.zoom * (e.deltaY < 0 ? 1.12 : 1 / 1.12)));
      // keep the point under the cursor where it was
      v.px = p.x - r.width / 2 - before.x * v.zoom;
      v.py = p.y - r.height / 2 - before.y * v.zoom;
      if (still) draw(performance.now());
    };
    c.addEventListener("mousedown", onDown);
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    c.addEventListener("wheel", onWheel, { passive: false });
    const loop = (now: number) => {
      if (now - last >= frameMs) {
        last = now;
        draw(now);
      }
      raf = requestAnimationFrame(loop);
    };
    if (still) {
      redraw.current = () => draw(performance.now());
      draw(performance.now());
    } else raf = requestAnimationFrame(loop);
    const onResize = () => draw(performance.now());
    window.addEventListener("resize", onResize);
    return () => {
      cancelAnimationFrame(raf);
      c.removeEventListener("mousedown", onDown);
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
      c.removeEventListener("wheel", onWheel);
      window.removeEventListener("resize", onResize);
      redraw.current = null;
    };
  }, []);
  useEffect(() => redraw.current?.(), [clusters, picked]);

  const pickedCluster = clusters.find((c) => c.fam === picked);
  const info = useMemo(() => (pickedCluster ? callout(pickedCluster, rows.length) : null), [pickedCluster, rows.length]);
  return (
    <div className="nmap">
      <canvas ref={ref} className="nmap-canvas" role="img" aria-label={`${rows.length} strategies in ${clusters.length} archetype clusters`} />
      <div className="nmap-legend dk-panel">
        <div className="dk-ph"><span className="dk-pt">Archetypes</span><span className="dk-ps">{clusters.length}</span></div>
        <div className="dk-pb">
          {clusters.map((c) => (
            <button key={c.fam} type="button" className={`nmap-leg${picked === c.fam ? " on" : ""}`} onClick={() => setPicked(picked === c.fam ? null : c.fam)}>
              <i style={{ background: c.colour }} /><span>{archetype(c.fam)}</span><b>{c.rows.length}</b>
            </button>
          ))}
        </div>
      </div>
      {info && pickedCluster && (
        <div className="nmap-call dk-panel" role="dialog" aria-label={`${archetype(pickedCluster.fam)} cluster`}>
          <div className="dk-ph"><span className="dk-pt">{archetype(pickedCluster.fam)}</span><span className="dk-ps">{FAMILY_NAMES[pickedCluster.fam] ?? pickedCluster.fam}</span>
            <span className="sp" /><button type="button" className="lnk" onClick={() => setPicked(null)}>CLOSE ✕</button></div>
          <div className="dk-pb kv-list">
            <div><span>STRATEGIES</span><b>{info.count} · {fmt.pct(info.share, 0)} of the view</b></div>
            <div><span>BEST</span><b className="pos">{info.best ? `#${shortId(info.best.id)} S ${info.best.score.toFixed(0)} · ${fmt.pct(info.best.total_return)}` : "—"}</b></div>
            <div><span>WORST</span><b className="neg">{info.worst ? `#${shortId(info.worst.id)} S ${info.worst.score.toFixed(0)} · ${fmt.pct(info.worst.total_return)}` : "—"}</b></div>
            <div><span>AVG RETURN</span><b>{fmt.pct(info.avgRet)}</b></div>
            <div><span>AVG MAX DD</span><b className="neg">{fmt.pct(-Math.abs(info.avgDd))}</b></div>
            <div><span>AVG SCORE</span><b className="acc">{info.avgScore.toFixed(1)}</b></div>
            <div><span>DOMINANT MARKET</span><b>{info.dominant}</b></div>
          </div>
          {info.best && <div className="dk-pb"><button type="button" className="rb rb-sm" onClick={() => onOpen(info.best!.id)}>OPEN THE BEST</button></div>}
        </div>
      )}
      {tip && (
        <div className="nmap-tip" style={{ left: tip.x + 14, top: tip.y + 10 }}>
          <b>#{shortId(tip.r.id)}</b> {tip.r.symbol} {tip.r.timeframe}<br />
          S {tip.r.score.toFixed(0)} · {fmt.pct(tip.r.total_return)} · DD {fmt.pct(tip.r.max_dd)} · {tip.r.n_trades} trades
        </div>
      )}
      <div className="nmap-help">SCROLL = ZOOM · DRAG = PAN · CLICK CLUSTER = INSPECT · DRAG CLUSTER = MOVE · CLICK DOT = OPEN</div>
    </div>
  );
}
