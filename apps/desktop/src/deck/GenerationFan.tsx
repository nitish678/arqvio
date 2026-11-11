// The generation fan: every best survivor's equity curve on the training bars, replayed left to right behind a
// crimson scan line each time a new generation lands, then held. Below it, the legend of the best bots.
// The curves are the engine's (live.json, engine/research/search/live_view.py); only the replay is staged.
// idea: research §7.3 (CAMPAIGN generations); drawing and code are ARQVIO's.

import { useEffect, useRef } from "react";
import type { LiveView } from "../api/ws";
import { cssVar, fitCanvas, fmt } from "../ws/util";

export const FAN_COLOURS = ["#ff1744", "#ff9100", "#ffd740", "#ff4081", "#b388ff", "#18ffff", "#ff6e40", "#f50057",
  "#ea80fc", "#69f0ae", "#ffab40", "#ff5252"];

const SWEEP_MS = { FULL: 4200, LITE: 2400 } as const;

export function GenerationFan({ view, compact = false }: { view: LiveView | null; compact?: boolean }) {
  const ref = useRef<HTMLCanvasElement | null>(null);
  const data = useRef(view);
  data.current = view;
  const stamp = view?.live ? `${view.task?.campaign_id}/${view.live.generation}/${view.live.utc}` : "";
  const sweepStart = useRef(0);
  useEffect(() => {
    sweepStart.current = performance.now();
  }, [stamp]);

  useEffect(() => {
    const c = ref.current;
    if (!c) return undefined;
    const fx = (document.documentElement.dataset.fx ?? "FULL") as "FULL" | "LITE" | "OFF";
    const reduced = typeof window.matchMedia === "function" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const still = fx === "OFF" || reduced;
    const frameMs = fx === "FULL" ? 15 : 48;
    let raf = 0;
    let last = 0;

    const draw = (now: number) => {
      const fit = fitCanvas(c);
      if (!fit) return;
      const { ctx, w, h } = fit;
      ctx.clearRect(0, 0, w, h);
      const lv = data.current?.live;
      const bots = (lv?.bots ?? []).filter((b) => b.curve && b.curve.length > 1);
      const red = cssVar("--brand-red", "#ff1744");
      const padL = 14;
      const padR = 44;
      const padT = 22;
      const padB = 14;
      const W = w - padL - padR;
      const H = h - padT - padB;
      let lo = 0;
      let hi = 0;
      bots.forEach((b) => b.curve.forEach((v) => { lo = Math.min(lo, v); hi = Math.max(hi, v); }));
      if (hi - lo < 1) hi = lo + 1;
      const y = (v: number) => padT + (1 - (v - lo) / (hi - lo)) * H;

      // the zero line and the scale (max, 0, min) on the right
      ctx.strokeStyle = "rgba(255,255,255,0.08)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(padL, y(0));
      ctx.lineTo(padL + W, y(0));
      ctx.stroke();
      ctx.font = "10px Consolas, monospace";
      ctx.textAlign = "left";
      ctx.fillStyle = "#4fdc9c";
      ctx.fillText(`+${hi.toFixed(0)}%`, padL + W + 6, padT + 8);
      ctx.fillStyle = "rgba(255,255,255,0.45)";
      ctx.fillText("0%", padL + W + 6, y(0) + 3);
      if (lo < -0.5) {
        ctx.fillStyle = red;
        ctx.fillText(`${lo.toFixed(0)}%`, padL + W + 6, padT + H);
      }
      if (lv) {
        ctx.fillStyle = "rgba(255,255,255,0.85)";
        ctx.fillText(`GEN ${lv.generation} · ${lv.survivors} survivors`, padL + 4, 12);
      }
      if (!bots.length) return;

      const sweep = still ? 1 : Math.min(1, (now - sweepStart.current) / SWEEP_MS[fx === "LITE" ? "LITE" : "FULL"]);
      const scanX = padL + W * sweep;
      ctx.save();
      ctx.beginPath();
      ctx.rect(0, 0, scanX, h);
      ctx.clip();
      // weaker bots first, so the best draw on top
      for (let i = bots.length - 1; i >= 0; i--) {
        const b = bots[i];
        const n = b.curve.length;
        ctx.strokeStyle = FAN_COLOURS[i % FAN_COLOURS.length];
        ctx.globalAlpha = i < 12 ? 0.9 : 0.45;
        ctx.lineWidth = i < 3 ? 1.6 : 1;
        ctx.beginPath();
        b.curve.forEach((v, k) => {
          const x = padL + (k / (n - 1)) * W;
          if (k === 0) ctx.moveTo(x, y(v));
          else ctx.lineTo(x, y(v));
        });
        ctx.stroke();
      }
      ctx.restore();
      ctx.globalAlpha = 1;

      // the scan line: a glowing band trailing a bright edge
      const band = Math.min(90, W * 0.12);
      const g = ctx.createLinearGradient(scanX - band, 0, scanX, 0);
      g.addColorStop(0, "rgba(255,23,68,0)");
      g.addColorStop(1, "rgba(255,23,68,0.28)");
      ctx.fillStyle = g;
      ctx.fillRect(scanX - band, padT - 8, band, H + 16);
      ctx.fillStyle = "#ff6f8a";
      ctx.shadowColor = red;
      ctx.shadowBlur = 14;
      ctx.fillRect(scanX - 1.5, padT - 8, 3, H + 16);
      ctx.shadowBlur = 0;
    };

    const loop = (now: number) => {
      if (now - last >= frameMs) {
        last = now;
        draw(now);
      }
      raf = requestAnimationFrame(loop);
    };
    if (still) {
      const once = () => draw(performance.now());
      once();
      const t = setInterval(once, 1000);  // new data still shows, without motion
      window.addEventListener("resize", once);
      return () => {
        clearInterval(t);
        window.removeEventListener("resize", once);
      };
    }
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, []);

  const lv = view?.live;
  const sym = view?.task ? `${view.task.symbol} ${view.task.timeframe}` : "";
  return (
    <div className={`fan${compact ? " compact" : ""}`}>
      <canvas ref={ref} className="fan-canvas" role="img"
        aria-label={lv ? `generation ${lv.generation}: ${lv.bots.length} survivor curves` : "no generation yet"} />
      {lv && lv.bots.length > 0 && (
        <div className="fan-legend">
          {lv.bots.slice(0, compact ? 6 : 15).map((b, i) => (
            <span key={b.id} title={`fitness ${fmt.num(b.fit)} · max DD ${b.dd}%`}>
              <i style={{ background: FAN_COLOURS[i % FAN_COLOURS.length] }} />{sym} <b>#{b.id}</b>
              <em className={b.ret >= 0 ? "pos" : "neg"}>{b.ret >= 0 ? "+" : ""}{b.ret.toFixed(1)}%</em>
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
