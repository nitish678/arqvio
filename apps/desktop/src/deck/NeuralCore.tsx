// COMMAND's centre piece: the population as a cloud of points. Idle, it drifts loosely; breeding, it gathers into a
// rotating sphere whose fill is the share of the budget tested and whose survivors (alive) glow brighter. The drawing
// is a picture of the engine's numbers -- tested of budget, alive, generation -- not a separate source of facts.
// idea: research §8 (one canvas, DPR cap, FPS paced by a clock); the drawing itself is ARQVIO's.

import { useEffect, useRef } from "react";
import { cssVar, fitCanvas } from "../ws/util";

const POINTS_FULL = 420;
const POINTS_LITE = 160;

interface P { th: number; ph: number; r: number; seed: number }

function makePoints(n: number): P[] {
  // a Fibonacci sphere: even coverage without clumps
  const out: P[] = [];
  const golden = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < n; i++) {
    const y = 1 - (i / (n - 1)) * 2;
    out.push({ th: golden * i, ph: Math.acos(y), r: 0.82 + Math.random() * 0.3, seed: Math.random() });
  }
  return out;
}

export function NeuralCore({ fraction, alive, running, paused }: {
  fraction: number | null;
  alive: number;
  running: boolean;
  paused: boolean;
}) {
  const ref = useRef<HTMLCanvasElement | null>(null);
  const state = useRef({ fraction, alive, running, paused });
  state.current = { fraction, alive, running, paused };
  const still = useRef<(() => void) | null>(null);
  useEffect(() => still.current?.(), [fraction, alive, running, paused]);  // FX OFF: one frame per new number

  useEffect(() => {
    const c = ref.current;
    if (!c) return undefined;
    const fx = document.documentElement.dataset.fx ?? "FULL";
    const reduced = typeof window.matchMedia === "function" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const pts = makePoints(fx === "FULL" ? POINTS_FULL : POINTS_LITE);
    const frameMs = fx === "FULL" ? 15 : 48;  // the FX tiers' clock (FULL ~60 fps, LITE ~20 fps)
    let raf = 0;
    let last = 0;
    let gather = state.current.running ? 1 : 0;
    const t0 = performance.now();

    const draw = (now: number) => {
      const fit = fitCanvas(c);
      if (!fit) return;
      const { ctx, w, h } = fit;
      const s = state.current;
      const t = (now - t0) / 1000;
      gather += ((s.running ? 1 : 0) - gather) * 0.04;
      ctx.clearRect(0, 0, w, h);
      const red = cssVar("--brand-red", "#ff1744");
      const R = Math.min(w, h) * 0.34;
      const cx = w / 2;
      const cy = h / 2;
      const spin = s.paused ? 0 : t * (0.12 + 0.25 * gather);
      const filled = Math.floor((s.fraction ?? 0) * pts.length);
      const lit = Math.min(pts.length, Math.max(0, Math.round(s.alive)));

      // the ring: the share of the budget tested, as an arc
      ctx.lineWidth = 1;
      ctx.strokeStyle = "rgba(255,255,255,0.06)";
      ctx.beginPath();
      ctx.arc(cx, cy, R * 1.32, 0, Math.PI * 2);
      ctx.stroke();
      if (s.fraction != null && s.fraction > 0) {
        ctx.strokeStyle = red;
        ctx.globalAlpha = 0.75;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(cx, cy, R * 1.32, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * s.fraction);
        ctx.stroke();
        ctx.globalAlpha = 1;
      }

      for (let i = 0; i < pts.length; i++) {
        const p = pts[i];
        const th = p.th + spin;
        // idle: points loosen into a drifting cloud; breeding: they sit on the sphere
        const loose = 1 + (1 - gather) * (0.55 + 0.45 * Math.sin(t * 0.3 + p.seed * 9));
        const rr = R * p.r * loose;
        const x3 = rr * Math.sin(p.ph) * Math.cos(th);
        const y3 = rr * Math.cos(p.ph);
        const z3 = rr * Math.sin(p.ph) * Math.sin(th);
        const depth = (z3 / (R * 1.6) + 1) / 2;  // 0 back .. 1 front
        const x = cx + x3;
        const y = cy + y3 * 0.92 + (1 - gather) * Math.sin(t * 0.5 + p.seed * 20) * 6;
        const tested = i < filled;
        const survivor = i < lit;
        // idle points are a dim crimson dust; tested ones brighten, survivors glow white-hot
        const a = (survivor ? 1 : tested ? 0.8 : 0.32 + 0.25 * p.seed) * (0.35 + 0.65 * depth);
        const size = (survivor ? 2.4 : tested ? 1.8 : 1.4) * (0.6 + 0.7 * depth);
        ctx.globalAlpha = a;
        ctx.fillStyle = survivor ? "#ffe3e8" : red;
        ctx.beginPath();
        ctx.arc(x, y, size, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.globalAlpha = 1;
    };

    const loop = (now: number) => {
      if (now - last >= frameMs) {
        last = now;
        draw(now);
      }
      raf = requestAnimationFrame(loop);
    };
    if (fx === "OFF" || reduced) {
      const once = () => {
        gather = state.current.running ? 1 : 0;
        draw(performance.now());
      };
      still.current = once;
      once();
      window.addEventListener("resize", once);
      return () => {
        still.current = null;
        window.removeEventListener("resize", once);
      };
    }
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, []);

  return <canvas ref={ref} className="neural-canvas" aria-hidden="true" />;
}
