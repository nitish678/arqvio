// The inspector: the selected strategy's numbers (from the vault and from a trade-by-trade re-run on its own data),
// its equity curve, its topology, and the tests it passed or failed.

import { useEffect, useState } from "react";
import { type InspectMonth, api, type Inspect, type StrategyDetail } from "../api/ws";
import { EquityChart, StrategyTopology, topologyOf } from "./Canvas";
import { errText, familyName, fmt, shortId } from "./util";

function Stat({ k, v, cls, title }: { k: string; v: string; cls?: string; title?: string }) {
  return <div className="stat" title={title}><span>{k}</span><b className={cls}>{v}</b></div>;
}

function gateValue(v: unknown): string {
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (typeof v === "number") return Math.abs(v) < 10 ? v.toFixed(3) : v.toFixed(0);
  if (Array.isArray(v)) return v.map((x) => (typeof x === "number" ? x.toFixed(2) : String(x))).join(" · ");
  return v == null ? "—" : String(v);
}

export function Inspector({ id, onDeploy, onDelete, onChanged }: {
  id: string | null; onDeploy: () => void; onDelete: () => void; onChanged: () => void;
}) {
  const [detail, setDetail] = useState<StrategyDetail | null>(null);
  const [ins, setIns] = useState<Inspect | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [insErr, setInsErr] = useState<string | null>(null);
  const [showGates, setShowGates] = useState(false);
  const [robBusy, setRobBusy] = useState(false);

  useEffect(() => {
    setDetail(null);
    setIns(null);
    setErr(null);
    setInsErr(null);
    if (!id) return undefined;
    let alive = true;
    api.strategy(id).then((d) => alive && setDetail(d)).catch((e) => alive && setErr(errText(e)));
    api.inspect(id).then((d) => alive && setIns(d)).catch((e) => alive && setInsErr(errText(e)));
    return () => {
      alive = false;
    };
  }, [id]);

  useEffect(() => {
    if (!id || detail?.robustness_state !== "RUNNING") return undefined;
    const t = setInterval(() => {
      api.strategy(id).then((d) => {
        setDetail(d);
        if (d.robustness_state !== "RUNNING") onChanged();
      }).catch(() => undefined);
    }, 3000);
    return () => clearInterval(t);
  }, [id, detail?.robustness_state, onChanged]);

  if (!id) {
    return (
      <section className="panel grow" aria-label="Inspector">
        <div className="panel-h">Inspector</div>
        <div className="empty"><h3>Select a strategy</h3><p>Click a row in the vault to see its numbers, equity curve and topology.</p></div>
      </section>
    );
  }
  const m = (detail?.metrics ?? {}) as Record<string, number | null>;
  const s = ins?.stats;
  const passed = detail?.gates.filter((g) => g.passed === true).length ?? 0;
  const judged = detail?.gates.filter((g) => g.passed !== null).length ?? 0;
  const parity = async () => {
    setRobBusy(true);
    try {
      await api.parity(id);
      setDetail(await api.strategy(id));
    } catch (e) {
      setErr(errText(e));
    } finally {
      setRobBusy(false);
    }
  };
  const certify = async () => {
    setRobBusy(true);
    try {
      await api.certify(id);
      setDetail(await api.strategy(id));
    } catch (e) {
      setErr(errText(e));
    } finally {
      setRobBusy(false);
    }
  };
  const robustness = async () => {
    setRobBusy(true);
    try {
      await api.robustness(id);
      setDetail(await api.strategy(id));
    } catch (e) {
      setErr(errText(e));
    } finally {
      setRobBusy(false);
    }
  };

  return (
    <>
      <section className="panel" aria-label="Inspector">
        <div className="panel-h">
          Inspector <small>{shortId(id)}</small>
          <span className="sp" />
          {detail && <span className={`tag ${detail.tier === "CERTIFIED" ? "ok" : ""}`}>{detail.engine === "v1" ? "V1 · " : ""}{detail.tier}</span>}
        </div>
        <div className="panel-b" style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          {err && <div className="err">{err}</div>}
          {detail && (
            <div className="headline">
              <div>
                <div className="label" style={{ marginBottom: 6 }}>{detail.symbol} {detail.timeframe} · {familyName(detail.family)}</div>
                <div className="big">{fmt.pct(m.total_return)}<small>net return</small></div>
              </div>
              <div style={{ textAlign: "right" }}>
                <span className={`grade ${detail.grade}`} style={{ minWidth: 34, height: 30, fontSize: 15 }}>{detail.grade}</span>
                <div className="hint" style={{ marginTop: 4 }}>{fmt.num(detail.score, 1)} / {detail.score_max}</div>
              </div>
            </div>
          )}
          {detail?.address && (
            <div className="hint" title={detail.address.campaign_dir ?? undefined}>
              source {detail.address.campaign_id ?? "—"} · {detail.address.file} · bars {detail.address.dataset_id ?? "—"} ·
              {" "}{detail.address.cost_model} spread cost · as of {detail.address.as_of?.slice(0, 16).replace("T", " ") ?? "—"} UTC
            </div>
          )}
          <div className="stats">
            <Stat k="Net profit" v={s ? fmt.money(s.net_profit) : "…"} cls={s && s.net_profit < 0 ? "neg" : "pos"} title="On a 10,000 test account, costs included" />
            <Stat k="Max drawdown" v={fmt.pct(s?.max_drawdown ?? m.max_dd)} cls="neg" />
            <Stat k="Trades" v={fmt.int(s?.trades ?? m.n_trades)} />
            <Stat k="Win rate" v={s ? fmt.pct(s.win_rate) : "…"} />
            <Stat k="Profit factor" v={s ? fmt.num(s.profit_factor) : "…"} />
            <Stat k="Sharpe" v={fmt.num(m.sharpe)} title="Per year" />
            <Stat k="Avg win" v={s ? fmt.money(s.avg_win) : "…"} cls="pos" />
            <Stat k="Avg loss" v={s ? fmt.money(s.avg_loss) : "…"} cls="neg" />
            <Stat k="Long / short" v={s ? `${s.long_trades} / ${s.short_trades}` : "…"} />
            <Stat k="Avg hold" v={s ? fmt.hours(s.avg_hold_hours) : "…"} />
            <Stat k="Win / loss streak" v={s ? `${s.max_win_streak} / ${s.max_loss_streak}` : "…"} />
            <Stat k="DSR" v={fmt.num(m.dsr_final, 3)} title="Deflated Sharpe ratio: the edge after counting every strategy tried" />
            <Stat k="2nd market" v={fmt.num(m.cross_sharpe)} cls={m.cross_sharpe != null && m.cross_sharpe < 0 ? "neg" : ""} title={`Sharpe on ${detail?.cross_cell ?? "the second market"}`} />
            <Stat k="Beats random" v={fmt.num(m.sham_margin)} title="Sharpe margin over the same trades at random entry times" />
          </div>
          {insErr && <div className="err">Trade statistics: {insErr}</div>}
          {!s && !insErr && <div className="hint">Re-running the strategy on its own data for the trade-by-trade numbers…</div>}
        </div>
      </section>

      <section className="panel" aria-label="Equity curve">
        <div className="panel-h">Equity <small>{ins ? `${fmt.int(ins.bars)} bars · ${ins.first_bar.slice(0, 10)} → ${ins.last_bar.slice(0, 10)}` : ""}</small>
          <span className="sp" />{ins && <span className={`tag ${ins.matches_vault ? "ok" : "warn"}`} title={ins.engine}>{ins.matches_vault ? "matches breeding" : "differs"}</span>}
        </div>
        <div className="panel-b">{s ? <EquityChart curve={s.curve} initial={s.initial_equity} /> : <div className="chart" />}</div>
      </section>

      {ins?.monthly && ins.monthly.length > 0 && <MonthlyPanel months={ins.monthly} gate={ins.spread_gate} />}

      {detail && (
        <section className="panel" aria-label="Strategy topology">
          <div className="panel-h">Topology <small>what the strategy does</small></div>
          <div className="panel-b" style={{ paddingTop: 4, paddingBottom: 4 }}>
            <StrategyTopology family={detail.family} nodes={topologyOf(detail.dna, detail.symbol, detail.timeframe, detail.cross_cell)} />
          </div>
        </section>
      )}

      {detail && (
        <section className="panel" aria-label="Tests">
          <div className="panel-h">
            Tests <small>{passed} of {judged} passed{judged < detail.gates.length ? ` · ${detail.gates.length - judged} need the robustness check` : ""}</small>
            <span className="sp" />
            <button type="button" className="btn btn-link" onClick={() => setShowGates(!showGates)}>{showGates ? "Hide" : "Show"}</button>
          </div>
          {showGates && (
            <div className="panel-b genes" style={{ gridTemplateColumns: "1fr auto auto" }}>
              {detail.gates.map((g) => (
                <span key={g.gate} style={{ display: "contents" }}>
                  <span title={`need ${g.need}`}>{g.gate}</span>
                  <b>{gateValue(g.value)}</b>
                  <b className={g.passed === true ? "pos" : g.passed === false ? "neg" : "dim"}>{g.passed === true ? "PASS" : g.passed === false ? "FAIL" : "—"}</b>
                </span>
              ))}
            </div>
          )}
          <div className="dialog-f" style={{ justifyContent: "space-between" }}>
            <button type="button" className="btn" onClick={() => void robustness()}
              disabled={robBusy || detail.robustness_state === "RUNNING"}
              title="Re-runs, parameter nudges, time segments and a bootstrap: needed for CERTIFIED">
              {detail.robustness_state === "RUNNING" ? "Checking…" : detail.robustness_state === "DONE" ? "Re-check robustness" : "Check robustness"}
            </button>
            <button type="button" className="btn" onClick={() => void parity()}
              disabled={robBusy || detail.parity?.state === "RUNNING"}
              title={detail.parity?.state === "DONE" ? detail.parity.levels?.map((l) => `${l.passed ? "PASS" : "FAIL"} ${l.level}: ${l.summary}`).join(" | ")
                : detail.parity?.reason ?? "Run its EA in the Strategy Tester of your TEST terminal against the engine (grade A-F); never on a terminal with bots"}>
              {detail.parity?.state === "RUNNING" ? "Parity test…" : detail.parity?.state === "DONE" ? `Parity ${detail.parity.grade} · re-test` : "Parity test"}
            </button>
            {detail.engine === "v1" && (
              <button type="button" className="btn" onClick={() => void certify()}
                disabled={robBusy || detail.certify?.state === "RUNNING"}
                title="Engine V2's validation on this V1 strategy: its bred window, the second market, the sham and the DSR against its own campaign's trials">
                {detail.certify?.state === "RUNNING" ? "Certifying…" : detail.certify?.state === "DONE" ? "Re-certify (V2)" : "Certify (V2)"}
              </button>
            )}
            <span className="row">
              <button type="button" className="btn" onClick={onDelete}>Delete</button>
              <button type="button" className="btn btn-primary" onClick={onDeploy}>Deploy</button>
            </span>
          </div>
        </section>
      )}
    </>
  );
}

/** Profit and trades per month, with the bars the spread gate would refuse: a flat stretch of the equity curve shows
 *  whether the strategy stopped trading (and why) or traded and lost (ARQVIO_V4.md 10.1-10.2). */
function MonthlyPanel({ months, gate }: { months: InspectMonth[]; gate?: { mode: string; detail: string } }) {
  const top = Math.max(1, ...months.map((m) => Math.abs(m.pnl)));
  const total = months.reduce((a, m) => a + m.pnl, 0);
  const best = months.reduce((a, m) => (m.pnl > a.pnl ? m : a), months[0]);
  const share = total > 0 ? best.pnl / total : null;
  return (
    <section className="panel" aria-label="Monthly profit">
      <div className="panel-h">Monthly <small>{share != null ? `best month ${best.month}: ${fmt.pct(share)} of the profit` : "no net profit"}
        {gate ? ` · spread gate: ${gate.mode === "off" ? gate.detail : gate.detail}` : ""}</small></div>
      <div className="panel-b months">
        {months.map((m) => (
          <div key={m.month} className="month" title={`${m.month}: ${m.trades} trades, ${fmt.money(m.pnl)}; spread gate would refuse entries on ${m.gate_bars} of ${m.bars} bars`}>
            <span className="m-k">{m.month.slice(2)}</span>
            <span className="m-bar"><span className={m.pnl >= 0 ? "pos-bg" : "neg-bg"} style={{ width: `${(Math.abs(m.pnl) / top) * 100}%` }} /></span>
            <span className={`m-v ${m.pnl < 0 ? "neg" : "pos"}`}>{fmt.money(m.pnl)}</span>
            <span className="m-t dim">{m.trades} tr</span>
            <span className={`m-g ${m.gate_bars / Math.max(1, m.bars) > 0.5 ? "warn" : "dim"}`}>{m.gate_bars ? `${Math.round((m.gate_bars / Math.max(1, m.bars)) * 100)}% gated` : ""}</span>
          </div>
        ))}
      </div>
    </section>
  );
}
