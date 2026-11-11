// The trade tape: the backtest trades of the generation's best survivors (newest exit first), each with its side,
// P&L as % of the 10,000 test account and how it closed. Simulated trades on history, never live ones; the header
// says so. New rows fade in at the top; the list drifts slowly while a campaign runs.
// idea: research §7.3 (LIVE TRADE TAPE); code and copy are ARQVIO's.

import type { LiveView } from "../api/ws";

export function TradeTape({ view, running }: { view: LiveView | null; running: boolean }) {
  const lv = view?.live;
  const sym = view?.task ? `${view.task.symbol} ${view.task.timeframe}` : "";
  const rows = lv?.tape ?? [];
  return (
    <section className="dk-panel tape" aria-label="Trade tape">
      <div className="dk-ph"><span className="dk-pt">Trade tape</span>
        <span className="dk-ps" title="backtest trades of the best survivors, on the training bars">
          {lv ? `GEN ${lv.generation} · ${rows.length} backtest trades` : "awaiting the first generation"}</span></div>
      <div className={`tape-body${running && rows.length > 14 ? " drift" : ""}`}>
        {rows.length === 0 && <div className="dk-empty">{lv ? "NO KERNEL TRADE LOG FOR THESE BOTS" : "TRADES APPEAR AFTER THE FIRST GENERATION"}</div>}
        <ul key={`${lv?.generation}-${lv?.utc}`}>
          {rows.map((t, i) => (
            <li key={`${t.bot}-${t.bar}-${i}`} style={{ animationDelay: `${Math.min(i, 20) * 40}ms` }}>
              <span className={t.dir > 0 ? "up" : "down"}>{t.dir > 0 ? "▲" : "▼"}</span>
              <b>#{t.bot}</b>
              <span className="dim">{sym}</span>
              <span className={t.pct >= 0 ? "pos" : "neg"}>{t.pct >= 0 ? "+" : ""}{t.pct.toFixed(2)}%</span>
              <small>{t.reason}</small>
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}
