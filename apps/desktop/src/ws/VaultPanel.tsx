// The vault: every strategy that survived, best first; the latest launch's; and the bots running in MetaTrader 5.

import { useMemo, useState } from "react";
import type { Bot, DeployOverview, LaunchState, StrategyRow, VaultOverview } from "../api/ws";
import { FAMILY_NAMES, FAMILY_SHORT, fmt, shortId } from "./util";
import { VaultTools } from "./VaultTools";

type SortKey = "grade" | "market" | "family" | "n_trades" | "total_return" | "max_dd" | "sharpe" | "dsr_final" | "cross_sharpe" | "score" | "held_out";

const COLS: { key: SortKey; label: string; left?: boolean; title: string }[] = [
  { key: "grade", label: "Grade", left: true, title: "A–F from the score" },
  { key: "market", label: "Market", left: true, title: "The market and timeframe it was bred on" },
  { key: "family", label: "Type", left: true, title: "Strategy family" },
  { key: "n_trades", label: "Trades", title: "Trades in the backtest" },
  { key: "total_return", label: "Return", title: "Net return over the whole history (costs included)" },
  { key: "max_dd", label: "Max DD", title: "Largest fall from a peak" },
  { key: "sharpe", label: "Sharpe", title: "Risk-adjusted return, per year" },
  { key: "dsr_final", label: "DSR", title: "Deflated Sharpe: the chance the edge is real after the many strategies tried" },
  { key: "cross_sharpe", label: "2nd mkt", title: "Sharpe of the same strategy on the second (unseen) market" },
  { key: "held_out", label: "Held-out", title: "Return on the last 30% of the window, which breeding never saw (trades in brackets); — = bred before v4" },
  { key: "score", label: "Score", title: "Score v2, 0–100: a part counts once its check ran (robustness, held-out); /N says out of how much" },
];

function val(r: StrategyRow, k: SortKey): number | string {
  if (k === "market") return `${r.symbol} ${r.timeframe}`;
  if (k === "family") return r.family;
  if (k === "grade") return r.score;
  if (k === "held_out") return r.held_out?.state === "DONE" && r.held_out.return != null ? r.held_out.return : -Infinity;
  const v = r[k];
  return typeof v === "number" ? v : -Infinity;
}

export function BotsTable({ dep, onStage, onRemove }: { dep: DeployOverview; onStage: (b: Bot, s: number) => void; onRemove: (b: Bot) => void }) {
  const active = dep.bots.filter((b) => b.status !== "REMOVED");
  if (!active.length) {
    return <div className="empty"><h3>No bots yet</h3><p>Pick a strategy in the vault and press Deploy: ARQVIO compiles it into
      your MetaTrader 5 and opens it on its chart. It keeps running when ARQVIO is closed.</p></div>;
  }
  return (
    <div className="vt-wrap">
      <table className="vt">
        <thead><tr><th className="l">Bot</th><th className="l">Terminal</th><th>Lot size</th><th className="l">Heartbeat</th><th>Position</th><th>Day P&amp;L</th><th></th></tr></thead>
        <tbody>
          {active.map((b) => {
            const hb = b.heartbeat;
            const state = hb ? hb.state : "NOT SEEN";
            return (
              <tr key={b.bot}>
                <td className="l" title={b.bot}>{b.symbol} {b.timeframe} · {shortId(b.source.vault_id)}</td>
                <td className="l dim" title={b.terminal ?? ""}>{b.terminal ? b.terminal.replace(/\\terminal64\.exe$/i, "") : "—"}</td>
                <td>
                  <span className="row" style={{ justifyContent: "flex-end", gap: 4, flexWrap: "nowrap" }}>
                    <button type="button" className="btn btn-sm" disabled={b.stage <= 0} onClick={() => onStage(b, b.stage - 1)} aria-label="Smaller lot">−</button>
                    <span className="num" style={{ width: 44, textAlign: "center" }}>{Math.round(b.lot_scale * 100)}%</span>
                    <button type="button" className="btn btn-sm" disabled={b.stage >= dep.incubation_stages.length - 1} onClick={() => onStage(b, b.stage + 1)} aria-label="Larger lot">+</button>
                  </span>
                </td>
                <td className="l">
                  <span className={`tag ${state === "LIVE" ? "ok" : state === "SILENT" ? "warn" : ""}`}>{state}</span>
                  {hb?.age_s != null && <span className="dim"> {fmt.duration(hb.age_s)} ago</span>}
                  {hb?.status ? <span className="dim" title={hb.reason ?? ""}> · {hb.status}</span>
                    : hb?.standby ? <span className="warn" title={hb.standby}> · standby</span> : null}
                </td>
                <td>{hb?.position ? `${hb.position.dir > 0 ? "BUY" : "SELL"} ${hb.position.volume.toFixed(2)} (${fmt.signed(hb.position.profit)})` : "—"}</td>
                <td className={hb?.day_pnl_pct != null && hb.day_pnl_pct < 0 ? "neg" : ""}>{hb?.day_pnl_pct != null ? `${fmt.signed(hb.day_pnl_pct)}%` : "—"}</td>
                <td><button type="button" className="btn btn-sm" onClick={() => onRemove(b)}>Remove</button></td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export function VaultPanel({ vault, vaultError, launch, dep, selected, onSelect, onDeploy, onDelete, onStage, onRemoveBot, onStopAll, onChanged }: {
  vault: VaultOverview | null;
  vaultError: string | null;
  launch: LaunchState | null;
  dep: DeployOverview | null;
  selected: string | null;
  onSelect: (id: string) => void;
  onDeploy: () => void;
  onDelete: () => void;
  onStage: (b: Bot, stage: number) => void;
  onRemoveBot: (b: Bot) => void;
  onStopAll: (stop: boolean) => void;
  onChanged?: () => void;
}) {
  const [tab, setTab] = useState<"all" | "latest" | "deployed">("all");
  const [market, setMarket] = useState("");
  const [showCopies, setShowCopies] = useState(false);
  const [family, setFamily] = useState("");
  const [sort, setSort] = useState<{ key: SortKey; desc: boolean }>({ key: "score", desc: true });
  const rows = vault?.strategies ?? [];
  const latestIds = useMemo(() => new Set((launch?.tasks ?? []).filter((t) => t.kind === "breed" && t.run).map((t) => t.cell)), [launch]);
  const markets = useMemo(() => [...new Set(rows.map((r) => `${r.symbol} ${r.timeframe}`))].sort(), [rows]);
  const families = useMemo(() => [...new Set(rows.map((r) => r.family))].sort(), [rows]);
  const shown = useMemo(() => {
    let out = rows;
    if (tab === "latest") {
      const since = launch?.utc ?? "";
      out = out.filter((r) => latestIds.has(r.cell) && r.added_utc >= since);
    }
    if (market) out = out.filter((r) => `${r.symbol} ${r.timeframe}` === market);
    if (family) out = out.filter((r) => r.family === family);
    if (!showCopies) out = out.filter((r) => !r.copy_of);
    const k = sort.key;
    return [...out].sort((a, b) => {
      const x = val(a, k);
      const y = val(b, k);
      const c = typeof x === "number" && typeof y === "number" ? x - y : String(x).localeCompare(String(y));
      return sort.desc ? -c : c;
    });
  }, [rows, tab, market, family, sort, latestIds, launch, showCopies]);
  const copyCount = rows.filter((r) => r.copy_of).length;
  const liveCount = (dep?.bots ?? []).filter((b) => b.status !== "REMOVED").length;

  return (
    <section className="panel grow" aria-label="Vault">
      <div className="panel-h">
        Vault <small>{rows.length} strateg{rows.length === 1 ? "y" : "ies"}{vault?.counts?.CERTIFIED ? ` · ${vault.counts.CERTIFIED} certified` : ""}</small>
        <div className="tabs" role="tablist">
          <button type="button" role="tab" className="tab" aria-selected={tab === "all"} onClick={() => setTab("all")}>All</button>
          <button type="button" role="tab" className="tab" aria-selected={tab === "latest"} onClick={() => setTab("latest")}>Latest</button>
          <button type="button" role="tab" className="tab" aria-selected={tab === "deployed"} onClick={() => setTab("deployed")}>Deployed ({liveCount})</button>
        </div>
        <span className="sp" />
        {tab !== "deployed" ? (
          <>
            <button type="button" className="btn" disabled={!selected} onClick={onDelete}>Delete</button>
            <button type="button" className="btn btn-primary" disabled={!selected} onClick={onDeploy}>Deploy</button>
          </>
        ) : (
          dep && <button type="button" className={`btn${dep.stop_all ? " btn-primary" : ""}`} onClick={() => onStopAll(!dep.stop_all)}
            title={dep.stop_all ? "Let every ARQVIO bot trade again" : "Every ARQVIO bot closes its own positions and stands by"}>
            {dep.stop_all ? "Resume all bots" : "Stop all bots"}
          </button>
        )}
      </div>
      {tab !== "deployed" && rows.length > 0 && (
        <div className="toolbar">
          <select className="select" value={market} onChange={(e) => setMarket(e.target.value)} aria-label="Filter by market">
            <option value="">All markets</option>
            {markets.map((m) => <option key={m} value={m}>{m}</option>)}
          </select>
          <select className="select" value={family} onChange={(e) => setFamily(e.target.value)} aria-label="Filter by type">
            <option value="">All types</option>
            {families.map((f) => <option key={f} value={f}>{FAMILY_NAMES[f] ?? f}</option>)}
          </select>
          {copyCount > 0 && (
            <label className="row" style={{ gap: 6 }} title="Copies made exactly the same trades as a strategy above them">
              <input type="checkbox" checked={showCopies} onChange={(e) => setShowCopies(e.target.checked)} />
              <span>show {copyCount} cop{copyCount === 1 ? "y" : "ies"}</span>
            </label>
          )}
          <span className="hint">{shown.length} shown · click a column to sort · click a row to inspect</span>
          <span className="sp" />
          <VaultTools onChanged={onChanged ?? (() => undefined)} />
        </div>
      )}
      {vaultError && <div className="err" style={{ margin: 10 }}>{vaultError}</div>}
      {tab === "deployed" ? (
        dep ? <BotsTable dep={dep} onStage={onStage} onRemove={onRemoveBot} /> : <div className="empty"><p>Reading the bots…</p></div>
      ) : shown.length === 0 ? (
        <div className="empty">
          <h3>{rows.length ? "Nothing matches" : "No strategies yet"}</h3>
          <p>{rows.length ? "Clear the filters to see every strategy." : "Pick markets and timeframes above and press LAUNCH. Strategies that survive the tests land here, best first."}</p>
        </div>
      ) : (
        <div className="vt-wrap">
          <table className="vt">
            <thead>
              <tr>
                <th className="l" style={{ width: 70 }}>ID</th>
                {COLS.map((c) => (
                  <th key={c.key} className={c.left ? "l" : ""} title={c.title}
                    onClick={() => setSort((s) => ({ key: c.key, desc: s.key === c.key ? !s.desc : true }))}
                    aria-sort={sort.key === c.key ? (sort.desc ? "descending" : "ascending") : "none"}>
                    {c.label}{sort.key === c.key ? (sort.desc ? " ▾" : " ▴") : ""}
                  </th>
                ))}
                <th className="l">Tier</th>
              </tr>
            </thead>
            <tbody>
              {shown.map((r) => (
                <tr key={r.id} aria-selected={r.id === selected} onClick={() => onSelect(r.id)}>
                  <td className="l dim">{shortId(r.id)}{(r.copies ?? 1) > 1 && !r.copy_of ? <span className="tag" title={`${(r.copies ?? 1) - 1} other strateg${(r.copies ?? 1) === 2 ? "y" : "ies"} made exactly these trades`}> ×{r.copies}</span> : r.copy_of ? <span className="dim" title={`same trades as ${shortId(r.copy_of)}`}> copy</span> : null}</td>
                  <td className="l"><span className={`grade ${r.grade}`}>{r.grade}</span></td>
                  <td className="l">{r.symbol} {r.timeframe}</td>
                  <td className="l" title={FAMILY_NAMES[r.family] ?? r.family}>{FAMILY_SHORT[r.family] ?? r.family}</td>
                  <td>{fmt.int(r.n_trades)}</td>
                  <td className={r.total_return != null && r.total_return < 0 ? "neg" : "pos"}>{fmt.pct(r.total_return)}</td>
                  <td>{fmt.pct(r.max_dd)}</td>
                  <td>{fmt.num(r.sharpe)}</td>
                  <td>{fmt.num(r.dsr_final, 3)}</td>
                  <td className={r.cross_sharpe != null && r.cross_sharpe < 0 ? "neg" : ""}>{fmt.num(r.cross_sharpe)}</td>
                  <td className={r.held_out?.return != null && r.held_out.return < 0 ? "neg" : ""}
                    title={r.held_out?.state === "DONE" ? `${r.held_out.trades} held-out trades, Sharpe ${fmt.num(r.held_out.sharpe)}` : "no held-out test for this strategy"}>
                    {r.held_out?.state === "DONE" ? <>{fmt.pct(r.held_out.return)} <span className="dim">({r.held_out.trades})</span></> : "—"}</td>
                  <td>{fmt.num(r.score, 1)}<span className="dim">/{r.score_max}</span></td>
                  <td className="l"><span className={`tag ${r.tier === "CERTIFIED" ? "ok" : ""}`}
                    title={r.engine === "v1" ? "Bred by engine V1: its gates passed, V2's validation not run yet (Certify)" : undefined}>
                    {r.engine === "v1" ? "V1 · " : ""}{r.tier}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
