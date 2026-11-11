// PORTFOLIOS (DECK): the deployed bots (or the vault's best) judged together -- return correlation, diversity by axis,
// grades with their published bars, the combined equity and plain suggestions. Every number is the engine's
// (engine/research/vault/portfolio_view.py); a pair with too little shared history shows no number.

import { useEffect, useState } from "react";
import { api, type PortfolioView, type TrueSim } from "../api/ws";
import { EquityChart } from "../ws/Canvas";
import { errText, FAMILY_NAMES, fmt, shortId } from "../ws/util";

const AXES = ["correlation", "market", "currency", "type", "timeframe", "session"];

function cellColour(v: number | null): string | undefined {
  if (v == null) return undefined;
  const a = Math.min(0.85, Math.abs(v));
  return v >= 0 ? `rgba(255,59,77,${a})` : `rgba(80,160,255,${a})`;
}

export function Portfolio({ onOpen, fixed }: { onOpen: (id: string) => void; fixed?: string }) {
  const [picked, setSource] = useState<"deployed" | "top">("deployed");
  const source = fixed ?? picked;
  const [view, setView] = useState<PortfolioView | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [sim, setSim] = useState<TrueSim | null>(null);
  const [simBusy, setSimBusy] = useState(false);
  const runSim = () => {
    setSimBusy(true);
    setSim(null);
    api.truesim(source).then(setSim).catch((e) => setErr(errText(e))).finally(() => setSimBusy(false));
  };

  useEffect(() => {
    let alive = true;
    setBusy(true);
    setErr(null);
    api.portfolio(source).then((v) => alive && setView(v)).catch((e) => alive && setErr(errText(e))).finally(() => alive && setBusy(false));
    return () => {
      alive = false;
    };
  }, [source]);

  const members = view?.members ?? [];
  return (
    <section className="panel grow" aria-label="Portfolio">
      <div className="panel-h">Portfolio
        {!fixed && <div className="tabs" role="tablist">
          <button type="button" role="tab" className="tab" aria-selected={source === "deployed"} onClick={() => setSource("deployed")}>Deployed bots</button>
          <button type="button" role="tab" className="tab" aria-selected={source === "top"} onClick={() => setSource("top")}>Vault top 8</button>
        </div>}
        <small>{busy ? "running each member on its own data…" : view?.shared_days ? `${view.shared_days} shared days` : ""}</small>
      </div>
      {err && <div className="err" style={{ margin: 10 }}>{err}</div>}
      {view && !view.overall && !busy && <div className="empty"><h3>Not enough members</h3><p>{view.note ?? "A portfolio needs at least two strategies."}</p></div>}
      {view?.overall && (
        <div className="panel-b pf-grid">
          <div className="pf-grades">
            <div className="ins-tile pf-overall"><span className="label">Diversity grade</span><b className={`grade-big g-${view.overall}`}>{view.overall}</b><small>the worst axis</small></div>
            {AXES.map((a) => (
              <div key={a} className="ins-tile"><span className="label">{a}</span><b className={`g-${view.grades?.[a] ?? "x"}`}>{view.grades?.[a] ?? "—"}</b>
                <small>{a === "correlation" ? (view.worst_pair ? `worst pair ${fmt.num(view.worst_pair.corr)}` : "no pair shares 30 days")
                  : view.axes?.[a]?.largest ? `${view.axes[a].largest} ${fmt.pct(view.axes[a].share)}` : ""}</small></div>
            ))}
          </div>

          <div className="ins-box">
            <div className="label">Return correlation <span className="dim">daily returns on shared days · blank: under {view.bars?.min_shared_days} shared days</span></div>
            <table className="vt pf-heat">
              <thead><tr><th />{members.map((m) => <th key={m.id} title={m.id}>{shortId(m.id)}</th>)}</tr></thead>
              <tbody>
                {members.map((m, i) => (
                  <tr key={m.id}>
                    <td className="l" title={m.id} onClick={() => onOpen(m.id)} style={{ cursor: "pointer" }}>{m.symbol} {m.timeframe} <span className="dim">{shortId(m.id)}</span></td>
                    {members.map((n, j) => {
                      const v = view.corr?.[i]?.[j] ?? null;
                      return <td key={n.id} style={{ background: i === j ? undefined : cellColour(v) }}>{i === j ? "·" : v == null ? "" : v.toFixed(2)}</td>;
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="hint" style={{ marginTop: 6 }}>red: move together · blue: move opposite · pale: decorrelated</div>
          </div>

          <div className="ins-box">
            <div className="label">By axis <span className="dim">the largest label's share; grade A ≤ 40%, B ≤ 55%, C ≤ 70%, D ≤ 85%</span></div>
            {Object.entries(view.axes ?? {}).map(([a, r]) => (
              <div key={a} className="ins-bar"><span>{a}</span>
                <span className="m-bar"><span className="neg-bg" style={{ width: `${(r.share ?? 0) * 100}%` }} /></span>
                <b className={`g-${r.grade ?? "x"}`}>{r.grade ?? "—"}</b>
                <span className="dim" title={Object.entries(r.counts).map(([k, v]) => `${a === "type" ? FAMILY_NAMES[k] ?? k : k}: ${v}`).join(", ")}>{r.largest}</span></div>
            ))}
            {!!view.tips?.length && (
              <>
                <div className="label" style={{ marginTop: 12 }}>Suggestions</div>
                <ul className="pf-tips">{view.tips.map((t) => <li key={t}>{t}</li>)}</ul>
              </>
            )}
          </div>

          <div className="ins-box pf-equity">
            <div className="label">Combined equity <span className="dim">equal weight, on the {view.shared_days} days every member has</span></div>
            {view.combined && view.combined.length ? <EquityChart curve={view.combined} initial={10_000} /> : <div className="hint">The members share fewer than {view.bars?.min_shared_days} days: no combined curve.</div>}
          </div>

          <div className="ins-box pf-equity">
            <div className="label">TRUE SIM <span className="dim">every member on ONE 10,000 account: one cash ledger, risk caps that refuse trades (H1 members)</span></div>
            {!sim && <button type="button" className="btn btn-sm" disabled={simBusy} onClick={runSim}
              title="Runs every member through the slow engine on one account; takes minutes">{simBusy ? "Running the members on one account…" : "Run TRUE SIM"}</button>}
            {sim?.state === "DONE" && sim.curve && (
              <>
                <EquityChart curve={sim.curve.map((p) => [p.t, p.equity] as [number, number])} initial={sim.e0 ?? 10_000} />
                <div className="hint">together {fmt.pct(sim.return ?? 0)} · max drawdown {fmt.pct(sim.max_dd ?? 0)} · {sim.refused_trades} trade(s) refused by the caps
                  · {sim.conflicts} opposite same-symbol position(s){sim.skipped?.length ? ` · skipped: ${sim.skipped.map((x) => shortId(x.id)).join(", ")}` : ""}</div>
              </>
            )}
            {sim?.state === "EMPTY" && <div className="hint">{sim.note ?? "No member could run on one account (H1 members only)."}</div>}
          </div>
        </div>
      )}
    </section>
  );
}
