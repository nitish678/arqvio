// Live performance (DECK LIVE): each bot's closed trades from its terminal's deal history against the band its own
// backtest predicts for that many trades (engine/research/deploy/live_perf.py), and ARQVIO magics without a bot.

import { useEffect, useState } from "react";
import { api, type LivePerf as LP } from "../api/ws";
import { errText, fmt } from "../ws/util";

const TONE: Record<string, string> = { HOLDING: "ok", AHEAD: "ok", WEAKENING: "warn" };

export function LivePerf({ only }: { only?: Set<string> } = {}) {
  const [d, setD] = useState<LP | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    let alive = true;
    const load = () => api.livePerf().then((x) => alive && (setD(x), setErr(null))).catch((e) => alive && setErr(errText(e)));
    void load();
    const t = setInterval(load, 60_000);
    return () => { alive = false; clearInterval(t); };
  }, []);
  return (
    <section className="panel" aria-label="Live performance">
      <div className="panel-h">Live vs backtest <small>closed trades from MetaTrader 5 · band = p5–p95 of the backtest for the same number of trades · judged from {d?.min_trades ?? 15} trades</small></div>
      <div className="panel-b">
        {err && <div className="err">{err}</div>}
        {d?.notes.map((n) => <div key={n} className="hint">{n}</div>)}
        {!d && !err && <div className="hint">Reading the deal history…</div>}
        {d && only && !d.bots.some((b) => only.has(b.bot)) && <div className="hint">None of this portfolio's strategies is live yet.</div>}
        {d && d.bots.length > 0 && (!only || d.bots.some((b) => only.has(b.bot))) && (
          <table className="vt">
            <thead><tr><th className="l">Bot</th><th>Trades</th><th>Net</th><th>Win rate</th><th>Band p5 · p50 · p95</th><th className="l">Verdict</th><th className="l">Incubation</th><th className="l">Last trade</th></tr></thead>
            <tbody>
              {d.bots.filter((b) => !only || only.has(b.bot)).map((b) => (
                <tr key={b.bot}>
                  <td className="l" title={b.bot}>{b.bot.replace(/^arqvio_/, "")}</td>
                  <td>{b.trades ?? "—"}</td>
                  <td className={(b.net ?? 0) < 0 ? "neg" : "pos"}>{b.net != null ? fmt.money(b.net) : "—"}</td>
                  <td>{b.win_rate != null ? fmt.pct(b.win_rate) : "—"}</td>
                  <td>{b.band ? `${fmt.money(b.band.p5)} · ${fmt.money(b.band.p50)} · ${fmt.money(b.band.p95)}` : "—"}</td>
                  <td className="l"><span className={`tag ${TONE[b.verdict] ?? ""}`}>{b.verdict}</span></td>
                  <td className="l" title={b.incubation ? b.incubation.checks.map((c) => `${c.passed ? "PASS" : "HELD"} ${c.check} (${c.need})`).join("\n") + `\n${b.incubation.note}` : undefined}>
                    {!b.incubation ? "—" : b.incubation.next_lot_scale == null ? "full size"
                      : b.incubation.ready ? <span className="tag ok">ready for {fmt.pct(b.incubation.next_lot_scale)}</span>
                        : `${fmt.pct(b.incubation.lot_scale)} · ${b.incubation.checks.filter((c) => c.passed).length}/${b.incubation.checks.length} checks`}
                  </td>
                  <td className="l dim">{b.last_trade ? fmt.time(b.last_trade) : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {d && d.orphans.length > 0 && (
          <div className="err" style={{ marginTop: 8 }}>Orphan ARQVIO trades (magic in ARQVIO's range, no bot in this list):{" "}
            {d.orphans.map((o) => `${o.symbol} #${o.magic} (${o.deals} deals, ${o.open_positions} open)`).join("; ")}</div>
        )}
        {d && d.orphans.length === 0 && d.bots.length > 0 && <div className="hint" style={{ marginTop: 6 }}>No orphan ARQVIO trades.</div>}
      </div>
    </section>
  );
}
