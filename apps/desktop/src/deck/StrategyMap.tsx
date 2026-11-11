// MAP (DECK): every vault strategy as a point -- Sharpe across, max drawdown down, colour by type, size by trades,
// a ring when it passed the held-out test. Copies are drawn once. Click a point to inspect it.

import { useMemo, useState } from "react";
import type { StrategyRow } from "../api/ws";
import { FAMILY_NAMES, fmt, shortId } from "../ws/util";

// A family keeps one colour in every view and under every filter: its place in this fixed order.
const FAMILY_ORDER = ["MA_CROSS", "RSI_REVERSION", "STOCH_CROSS", "MACD_CROSS", "DONCHIAN_BREAKOUT", "SIGNALS"];
export function familyColour(f: string): string {
  const i = FAMILY_ORDER.indexOf(f);
  return MAP_COLOURS[(i < 0 ? FAMILY_ORDER.length + (f.length % 3) : i) % MAP_COLOURS.length];
}
/** The ring both MAP views draw: the held-out test ran and made money. */
export const heldOutPositive = (r: StrategyRow) => r.held_out?.state === "DONE" && (r.held_out?.return ?? 0) > 0;

export const MAP_COLOURS = ["#FF3B4D", "#C3C8D0", "#4FA3FF", "#E0B341", "#3FBF7F", "#B07CFF", "#FF8A3D"];
const W = 900;
const H = 520;
const PAD = 46;

export function StrategyMap({ rows, onOpen }: { rows: StrategyRow[]; onOpen: (id: string) => void }) {
  const [hover, setHover] = useState<StrategyRow | null>(null);
  const pts = useMemo(() => rows.filter((r) => !r.copy_of && r.sharpe != null && r.max_dd != null), [rows]);
  const fams = useMemo(() => [...new Set(pts.map((r) => r.family))].sort(), [pts]);
  if (!pts.length) {
    return (
      <section className="panel grow"><div className="panel-h">Map</div>
        <div className="empty"><h3>No strategies yet</h3><p>The map draws the vault. Run a campaign from COMMAND first.</p></div>
      </section>
    );
  }
  const xs = pts.map((r) => r.sharpe as number);
  const ys = pts.map((r) => r.max_dd as number);
  const x0 = Math.min(0, ...xs), x1 = Math.max(...xs) * 1.05 || 1;
  const y1 = Math.max(...ys) * 1.05 || 0.1;
  const X = (v: number) => PAD + ((v - x0) / (x1 - x0)) * (W - 2 * PAD);
  const Y = (v: number) => PAD + (v / y1) * (H - 2 * PAD);
  const maxTrades = Math.max(1, ...pts.map((r) => r.n_trades));
  const ticks = (a: number, b: number) => Array.from({ length: 5 }, (_, i) => a + ((b - a) * i) / 4);
  return (
    <section className="panel grow" aria-label="Strategy map">
      <div className="panel-h">Map <small>{pts.length} strategies · Sharpe across, max drawdown down · ring = held-out positive</small></div>
      <div className="panel-b map-wrap">
        <svg viewBox={`0 0 ${W} ${H}`} className="map-svg" role="img" aria-label="Strategies by Sharpe and drawdown">
          {ticks(x0, x1).map((t) => <g key={`x${t}`}><line x1={X(t)} x2={X(t)} y1={PAD} y2={H - PAD} className="map-grid" /><text x={X(t)} y={H - PAD + 16} className="map-tick" textAnchor="middle">{t.toFixed(1)}</text></g>)}
          {ticks(0, y1).map((t) => <g key={`y${t}`}><line x1={PAD} x2={W - PAD} y1={Y(t)} y2={Y(t)} className="map-grid" /><text x={PAD - 6} y={Y(t) + 4} className="map-tick" textAnchor="end">{(t * 100).toFixed(0)}%</text></g>)}
          <text x={W / 2} y={H - 8} className="map-axis" textAnchor="middle">Sharpe (per year)</text>
          <text x={14} y={H / 2} className="map-axis" textAnchor="middle" transform={`rotate(-90 14 ${H / 2})`}>max drawdown</text>
          {pts.map((r) => {
            const c = familyColour(r.family);
            const rad = 3 + 7 * Math.sqrt(r.n_trades / maxTrades);
            const ho = r.held_out?.state === "DONE" && (r.held_out.return ?? 0) > 0;
            return (
              <g key={r.id} onClick={() => onOpen(r.id)} onMouseEnter={() => setHover(r)} onMouseLeave={() => setHover(null)} style={{ cursor: "pointer" }}>
                {ho && <circle cx={X(r.sharpe as number)} cy={Y(r.max_dd as number)} r={rad + 3} fill="none" stroke="#3FBF7F" strokeWidth={1.5} />}
                <circle cx={X(r.sharpe as number)} cy={Y(r.max_dd as number)} r={rad} fill={c} fillOpacity={r.tier === "CERTIFIED" ? 1 : 0.7} />
              </g>
            );
          })}
        </svg>
        <div className="map-side">
          {fams.map((f) => <div key={f} className="row" style={{ gap: 6 }}><span className="map-dot" style={{ background: familyColour(f) }} />{FAMILY_NAMES[f] ?? f} <span className="dim">{pts.filter((r) => r.family === f).length}</span></div>)}
          {hover && (
            <div className="map-card">
              <b>{hover.symbol} {hover.timeframe} · {shortId(hover.id)}</b>
              <div>{FAMILY_NAMES[hover.family] ?? hover.family} · grade {hover.grade} · {hover.tier}</div>
              <div>Sharpe {fmt.num(hover.sharpe)} · DD {fmt.pct(hover.max_dd)} · {hover.n_trades} trades</div>
              <div>return {fmt.pct(hover.total_return)}{hover.held_out?.state === "DONE" ? ` · held-out ${fmt.pct(hover.held_out.return)}` : ""}</div>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
