// MAP: the vault in two pictures -- CLUSTERS (the neural map by archetype) and SCATTER (Sharpe across, drawdown down),
// with one VIEW filter for both: a symbol and a timeframe. Copies are drawn once. The view is remembered.
// idea: research MAP (VIEW filter by symbol / TF, N STRATEGIES); layout ARQVIO's.

import { useMemo, useState } from "react";
import type { StrategyRow } from "../api/ws";
import { NeuralMap } from "./NeuralMap";
import { StrategyMap } from "./StrategyMap";

const KEY = "arqvio.map.view";
type View = { mode: "clusters" | "scatter"; symbol: string; tf: string };

function read(): View {
  try {
    return { mode: "clusters", symbol: "", tf: "", ...JSON.parse(localStorage.getItem(KEY) ?? "{}") };
  } catch {
    return { mode: "clusters", symbol: "", tf: "" };
  }
}

export function MapPage({ rows, onOpen }: { rows: StrategyRow[]; onOpen: (id: string) => void }) {
  const [view, setViewState] = useState<View>(read);
  const setView = (p: Partial<View>) => setViewState((v) => {
    const n = { ...v, ...p };
    try {
      localStorage.setItem(KEY, JSON.stringify(n));
    } catch {
      /* per-viewer convenience only */
    }
    return n;
  });
  const lead = useMemo(() => rows.filter((r) => !r.copy_of), [rows]);
  const symbols = useMemo(() => [...new Set(lead.map((r) => r.symbol))].sort(), [lead]);
  const tfs = useMemo(() => [...new Set(lead.map((r) => r.timeframe))].sort(), [lead]);
  // a remembered symbol or timeframe that left the vault counts as ALL (the select shows ALL too)
  const symbol = symbols.includes(view.symbol) ? view.symbol : "";
  const tf = tfs.includes(view.tf) ? view.tf : "";
  const mode = view.mode === "scatter" ? "scatter" : "clusters";
  const shown = useMemo(() => lead.filter((r) => (!symbol || r.symbol === symbol) && (!tf || r.timeframe === tf)),
    [lead, symbol, tf]);
  return (
    <section className="dk-panel mapp" aria-label="Map">
      <div className="mapp-bar">
        <span className="dk-pt">View</span>
        <select className="select" value={symbol} onChange={(e) => setView({ symbol: e.target.value })} aria-label="Symbol">
          <option value="">ALL SYMBOLS</option>{symbols.map((s) => <option key={s} value={s}>{s}</option>)}</select>
        <select className="select" value={tf} onChange={(e) => setView({ tf: e.target.value })} aria-label="Timeframe">
          <option value="">ALL TIMEFRAMES</option>{tfs.map((s) => <option key={s} value={s}>{s}</option>)}</select>
        <span className="mapp-count"><b>{shown.length}</b> STRATEGIES</span>
        <span className="sp" />
        <span className="chips">
          <button type="button" className="chip" aria-pressed={mode === "clusters"} onClick={() => setView({ mode: "clusters" })}>CLUSTERS</button>
          <button type="button" className="chip" aria-pressed={mode === "scatter"} onClick={() => setView({ mode: "scatter" })}>SCATTER</button>
        </span>
      </div>
      <div className={`mapp-body ${mode}`}>
        {!shown.length ? <div className="dk-empty tall">{lead.length ? "NO STRATEGY FOR THIS VIEW" : "The map draws the vault. Run a campaign first."}</div>
          : mode === "clusters" ? <NeuralMap rows={shown} onOpen={onOpen} /> : <StrategyMap rows={shown} onOpen={onOpen} />}
      </div>
    </section>
  );
}
