// The workstation's three drawings, each from real values only: the campaign ring (strategies tested of the budget),
// the equity curve (the inspector's re-run) and the strategy topology (the genes the engine actually uses).
// Canvases are drawn at the device pixel ratio so lines and text stay sharp; motion follows the FX setting.

import { useEffect, useRef } from "react";
import { cssVar, fitCanvas, fmt, FAMILY_SHORT, motionOn } from "./util";

function useAnimation(draw: (t: number) => void, animate: boolean, deps: unknown[]) {
  const ref = useRef<HTMLCanvasElement | null>(null);
  useEffect(() => {
    let raf = 0;
    const start = performance.now();
    const frame = () => {
      draw((performance.now() - start) / 1000);
      if (animate && motionOn()) raf = requestAnimationFrame(frame);
    };
    frame();
    const onResize = () => draw((performance.now() - start) / 1000);
    window.addEventListener("resize", onResize);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", onResize);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [animate, ...deps]);
  return ref;
}

const RED = "255,59,77";
const WHITE = "233,236,241";

// ---- campaign ring --------------------------------------------------------------------------------------------------------

export function CoreRing({ fraction, generation, generations, running }: {
  fraction: number | null; generation: number | null; generations: number | null; running: boolean;
}) {
  const ref = useAnimation((t) => {
    const c = ref.current;
    const fit = c && fitCanvas(c);
    if (!fit) return;
    const { ctx: x, w, h } = fit;
    x.clearRect(0, 0, w, h);
    const cx = w / 2;
    const cy = h / 2;
    const R = Math.min(w, h) / 2 - 4;
    const rot = running ? t * 0.12 : 0;
    x.lineWidth = 1;
    x.strokeStyle = `rgba(${WHITE},.10)`;
    x.beginPath();
    x.arc(cx, cy, R, 0, Math.PI * 2);
    x.stroke();
    x.beginPath();
    for (let i = 0; i < 60; i++) {
      const a = rot + (i / 60) * Math.PI * 2;
      const inner = i % 5 === 0 ? R - 7 : R - 4;
      x.moveTo(cx + Math.cos(a) * inner, cy + Math.sin(a) * inner);
      x.lineTo(cx + Math.cos(a) * R, cy + Math.sin(a) * R);
    }
    x.strokeStyle = `rgba(${WHITE},.22)`;
    x.stroke();
    // the budget arc: strategies tested of the campaign's budget
    const r2 = R * 0.78;
    x.lineWidth = 6;
    x.strokeStyle = `rgba(${WHITE},.07)`;
    x.beginPath();
    x.arc(cx, cy, r2, 0, Math.PI * 2);
    x.stroke();
    if (fraction != null && fraction > 0) {
      x.strokeStyle = `rgba(${RED},.95)`;
      x.beginPath();
      x.arc(cx, cy, r2, -Math.PI / 2, -Math.PI / 2 + Math.min(1, fraction) * Math.PI * 2);
      x.stroke();
    }
    // generations: one segment each
    if (generations && generations > 0) {
      const r3 = R * 0.58;
      const gap = generations > 30 ? 0.02 : 0.06;
      x.lineWidth = 3;
      for (let g = 0; g < generations; g++) {
        const s = -Math.PI / 2 + (g / generations) * Math.PI * 2 + gap / 2;
        const e = s + (Math.PI * 2) / generations - gap;
        x.strokeStyle = generation != null && g <= generation ? `rgba(${RED},.75)` : `rgba(${WHITE},.10)`;
        x.beginPath();
        x.arc(cx, cy, r3, s, e);
        x.stroke();
      }
    }
    if (running) {
      x.lineWidth = 1;
      x.setLineDash([3, 5]);
      x.strokeStyle = `rgba(${WHITE},.18)`;
      x.beginPath();
      x.arc(cx, cy, R * 0.42, -t * 0.4, -t * 0.4 + Math.PI * 1.5);
      x.stroke();
      x.setLineDash([]);
    }
    x.fillStyle = cssVar("--text-num", "#fff");
    x.textAlign = "center";
    x.textBaseline = "middle";
    x.font = "600 18px Cascadia Mono, Consolas, monospace";
    x.fillText(fraction == null ? "—" : `${Math.round(Math.min(1, fraction) * 100)}%`, cx, cy - 4);
    x.font = "600 9px Cascadia Mono, Consolas, monospace";
    x.fillStyle = cssVar("--text-2", "#a4acbb");
    x.fillText(generation != null && generations ? `GEN ${generation + 1}/${generations}` : "IDLE", cx, cy + 14);
  }, running, [fraction, generation, generations, running]);
  return <canvas ref={ref} role="img" aria-label={fraction == null ? "No campaign running" :
    `Campaign ${Math.round(fraction * 100)}% of its budget tested`} />;
}

// ---- equity curve ---------------------------------------------------------------------------------------------------------

export function EquityChart({ curve, initial }: { curve: [number, number][]; initial: number }) {
  const hover = useRef<number | null>(null);
  const ref = useAnimation(() => {
    const c = ref.current;
    const fit = c && fitCanvas(c);
    if (!fit) return;
    const { ctx: x, w, h } = fit;
    x.clearRect(0, 0, w, h);
    if (curve.length < 2) return;
    const padL = 48;
    const padR = 8;
    const padT = 8;
    const padB = 18;
    const ys = curve.map((p) => p[1]);
    let lo = Math.min(initial, ...ys);
    let hi = Math.max(initial, ...ys);
    if (hi - lo < 1e-9) {
      hi += 1;
      lo -= 1;
    }
    const X = (i: number) => padL + (i / (curve.length - 1)) * (w - padL - padR);
    const Y = (v: number) => padT + (1 - (v - lo) / (hi - lo)) * (h - padT - padB);
    const up = ys[ys.length - 1] >= initial;
    const line = up ? cssVar("--sem-survivor", "#4FDC9C") : cssVar("--sem-refused", "#FF6B76");
    x.font = "500 10px Cascadia Mono, Consolas, monospace";
    x.fillStyle = cssVar("--text-2", "#a4acbb");
    x.textAlign = "right";
    x.textBaseline = "middle";
    const marks: number[] = [];
    for (const v of [initial, hi, lo]) {  // the start line first; a label too close to one already drawn is skipped
      if (marks.some((m) => Math.abs(Y(m) - Y(v)) < 12)) continue;
      marks.push(v);
      x.fillText(Math.round(v).toLocaleString("en-US"), padL - 6, Y(v));
    }
    x.strokeStyle = `rgba(${WHITE},.14)`;
    x.setLineDash([3, 4]);
    x.beginPath();
    x.moveTo(padL, Math.round(Y(initial)) + 0.5);
    x.lineTo(w - padR, Math.round(Y(initial)) + 0.5);
    x.stroke();
    x.setLineDash([]);
    const grad = x.createLinearGradient(0, padT, 0, h - padB);
    grad.addColorStop(0, up ? "rgba(79,220,156,.22)" : "rgba(255,107,118,.22)");
    grad.addColorStop(1, "rgba(0,0,0,0)");
    x.beginPath();
    x.moveTo(X(0), Y(ys[0]));
    ys.forEach((v, i) => x.lineTo(X(i), Y(v)));
    x.lineTo(X(ys.length - 1), h - padB);
    x.lineTo(X(0), h - padB);
    x.closePath();
    x.fillStyle = grad;
    x.fill();
    x.beginPath();
    ys.forEach((v, i) => (i ? x.lineTo(X(i), Y(v)) : x.moveTo(X(i), Y(v))));
    x.strokeStyle = line;
    x.lineWidth = 1.5;
    x.stroke();
    x.textAlign = "left";
    x.textBaseline = "alphabetic";
    const year = (s: number) => new Date(s * 1000).toISOString().slice(0, 7);
    x.fillText(year(curve[0][0]), padL, h - 4);
    x.textAlign = "right";
    x.fillText(year(curve[curve.length - 1][0]), w - padR, h - 4);
    const hi_ = hover.current;
    if (hi_ != null && hi_ >= 0 && hi_ < curve.length) {
      const px = X(hi_);
      x.strokeStyle = `rgba(${WHITE},.35)`;
      x.lineWidth = 1;
      x.beginPath();
      x.moveTo(Math.round(px) + 0.5, padT);
      x.lineTo(Math.round(px) + 0.5, h - padB);
      x.stroke();
      x.fillStyle = line;
      x.beginPath();
      x.arc(px, Y(ys[hi_]), 3, 0, Math.PI * 2);
      x.fill();
      const label = `${new Date(curve[hi_][0] * 1000).toISOString().slice(0, 10)}  ${fmt.money(ys[hi_])}`;
      x.font = "600 10px Cascadia Mono, Consolas, monospace";
      const tw = x.measureText(label).width + 10;
      const bx = Math.min(Math.max(padL, px - tw / 2), w - padR - tw);
      x.fillStyle = cssVar("--s-3", "#161B26");
      x.fillRect(bx, padT, tw, 16);
      x.fillStyle = cssVar("--text-1", "#EEF1F5");
      x.textAlign = "left";
      x.textBaseline = "middle";
      x.fillText(label, bx + 5, padT + 8);
    }
  }, false, [curve, initial]);
  const onMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const f = (e.clientX - r.left - 48) / Math.max(1, r.width - 56);
    hover.current = Math.round(Math.min(1, Math.max(0, f)) * (curve.length - 1));
    window.dispatchEvent(new Event("resize"));
  };
  const onLeave = () => {
    hover.current = null;
    window.dispatchEvent(new Event("resize"));
  };
  return <canvas ref={ref} className="chart" onMouseMove={onMove} onMouseLeave={onLeave} role="img"
    aria-label={`Equity curve from ${fmt.money(initial)} to ${fmt.money(curve[curve.length - 1]?.[1])}`} />;
}

// ---- strategy topology ----------------------------------------------------------------------------------------------------

export interface TopoNode {
  name: string;
  lines: string[];
}

/** The strategy's anatomy from its DNA: only genes the backtest engine reads (family genes, risk and stops, entry
 *  gates) and the markets it was bred and re-tested on. */
export function topologyOf(dna: Record<string, unknown>, symbol: string, timeframe: string, cross: string): TopoNode[] {
  const n = (k: string, d = 0) => (typeof dna[k] === "number" ? (dna[k] as number).toFixed(d) : "—");
  const fam = String(dna.signal ?? "");
  // engine/research/search/real_backtest.py: the MA pair comes from the lookback gene, the lot from risk_pct
  const look = Number(dna.max_feature_lookback ?? 0);
  const maFast = Math.max(5, Math.trunc(look / 4));
  const maSlow = Math.max(maFast + 5, Math.trunc(look));
  const lot = Math.max(0.01, Math.min(1.0, Number(dna.risk_pct ?? 0) * 10));
  const signal: Record<string, string[]> = {
    MA_CROSS: [`MA ${maFast} / ${maSlow}`],
    RSI_REVERSION: [`RSI ${n("rsi_period")}`, `${n("rsi_oversold", 1)} / ${n("rsi_overbought", 1)}`,
      dna.rsi_exit_midline ? "exit at 50" : "exit at the opposite band"],
    STOCH_CROSS: [`%K ${n("stoch_k_period")} %D ${n("stoch_d_period")} sm ${n("stoch_smoothing")}`,
      `${n("stoch_oversold", 1)} / ${n("stoch_overbought", 1)}`],
    MACD_CROSS: [`${n("macd_fast")} / ${n("macd_slow")} / ${n("macd_signal_period")}`],
    DONCHIAN_BREAKOUT: [`channel ${n("donchian_period")} bars`],
  };
  const gates: string[] = [];
  if (dna.session) gates.push(`session ${String(dna.session)}`);
  if (typeof dna.trend_ema === "number" && dna.trend_ema > 0) gates.push(`trend EMA ${dna.trend_ema}`);
  // v4 switches (engine/research/backtest/switch_gates.py)
  const BIAS: Record<string, string> = { sma: "bias SMA 100", momentum: "bias momentum 20", donchian_mid: "bias channel mid 20" };
  const FILT: Record<string, string> = { no_open_friday: "no Friday openings", volatility: "calm volatility", rsi: "RSI 30–70" };
  for (const b of (dna.bias_switches as string[] | undefined) ?? []) gates.push(BIAS[b] ?? `bias ${b}`);
  for (const f of (dna.filter_switches as string[] | undefined) ?? []) gates.push(FILT[f] ?? `filter ${f}`);
  for (const m of (dna.mgmt_switches as string[] | undefined) ?? []) gates.push(m === "breakeven" ? "breakeven at 1R" : `management ${m}`);
  return [
    { name: "MARKET", lines: [`${symbol} ${timeframe}`, `re-tested: ${cross}`] },
    { name: "SIGNAL", lines: signal[fam] ?? [fam] },
    { name: "GATES", lines: gates.length ? gates : ["none (any hour)"] },
    { name: "SIZE", lines: [`${lot.toFixed(2)} lot / trade`, "on a 10,000 test account"] },
    { name: "EXITS", lines: [`SL ${n("sl_atr", 2)}×ATR`, `TP ${n("tp_atr", 2)}×ATR`] },
  ];
}

export function StrategyTopology({ family, nodes }: { family: string; nodes: TopoNode[] }) {
  const ref = useAnimation((t) => {
    const c = ref.current;
    const fit = c && fitCanvas(c);
    if (!fit) return;
    const { ctx: x, w, h } = fit;
    x.clearRect(0, 0, w, h);
    const cx = w / 2;
    const cy = h / 2 + 12;
    const rx = w / 2 - 64;
    const ry = h / 2 - 52;
    const pts = nodes.map((nd, i) => {
      const a = -Math.PI / 2 + (i / nodes.length) * Math.PI * 2;
      return { nd, x: cx + Math.cos(a) * rx, y: cy + Math.sin(a) * ry };
    });
    x.lineWidth = 1;
    pts.forEach((p, i) => {
      const q = pts[(i + 1) % pts.length];
      x.strokeStyle = `rgba(${WHITE},.07)`;
      x.beginPath();
      x.moveTo(p.x, p.y);
      x.lineTo(q.x, q.y);
      x.stroke();
      x.strokeStyle = `rgba(${RED},.40)`;
      x.beginPath();
      x.moveTo(cx, cy);
      x.lineTo(p.x, p.y);
      x.stroke();
      if (motionOn()) {
        const ph = (t * 0.45 + i * 0.2) % 1;
        x.fillStyle = `rgba(${RED},.95)`;
        x.beginPath();
        x.arc(cx + (p.x - cx) * ph, cy + (p.y - cy) * ph, 1.8, 0, Math.PI * 2);
        x.fill();
      }
    });
    const text1 = cssVar("--text-1", "#EEF1F5");
    const text2 = cssVar("--text-2", "#A4ACBB");
    const bg = cssVar("--s-1", "#0B0E14");
    pts.forEach((p) => {
      x.fillStyle = bg;
      x.strokeStyle = `rgba(${RED},.85)`;
      x.beginPath();
      x.arc(p.x, p.y, 5, 0, Math.PI * 2);
      x.fill();
      x.stroke();
      x.textAlign = "center";
      x.font = "700 9px Cascadia Mono, Consolas, monospace";
      x.fillStyle = `rgb(${RED})`;
      const above = p.y < cy;
      const ty = above ? p.y - 12 : p.y + 16;
      x.textBaseline = "middle";
      x.fillText(p.nd.name, p.x, ty);
      x.font = "500 10px Cascadia Mono, Consolas, monospace";
      p.nd.lines.slice(0, 2).forEach((ln, j) => {
        x.fillStyle = j === 0 ? text1 : text2;
        x.fillText(ln, p.x, above ? ty - 12 * (p.nd.lines.slice(0, 2).length - j) : ty + 12 * (j + 1));
      });
    });
    x.fillStyle = bg;
    x.strokeStyle = `rgba(${RED},.9)`;
    x.lineWidth = 1.5;
    x.beginPath();
    x.arc(cx, cy, 22, 0, Math.PI * 2);
    x.fill();
    x.stroke();
    x.fillStyle = text1;
    x.font = "700 10px Cascadia Mono, Consolas, monospace";
    x.textAlign = "center";
    x.textBaseline = "middle";
    x.fillText(FAMILY_SHORT[family] ?? family.slice(0, 5), cx, cy + 0.5);
  }, true, [family, JSON.stringify(nodes)]);
  return <canvas ref={ref} className="topo" role="img"
    aria-label={`Strategy topology: ${nodes.map((nd) => `${nd.name} ${nd.lines.join(", ")}`).join("; ")}`} />;
}
