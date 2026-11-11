// SETTINGS (DECK, ARQVIO_V4.md 12.3): this computer's display preferences, and -- read-only -- the rules the engine
// costs and judges every strategy by, exactly as the engine reports them (/api/v2/app/engine-settings).

import { useState } from "react";
import { api, type EngineSettings } from "../api/ws";
import { type Prefs, SettingsForm } from "../ws/Dialogs";
import { errText, usePoll } from "../ws/util";

const pct = (x: number) => `${Math.round(x * 100)}%`;

export function V1Rules({ s }: { s: EngineSettings }) {
  const v = s.v1;
  if (!v) return null;
  const r = v.presets.research, q = v.presets.quick;
  const rows: [string, string][] = [
    ["Breeding", `${r.proving_grounds} proving-ground strategies, TRIBE A + TRIBE B x ${r.generations} generations of ${r.tribe_size}, Final War (Quick: ${q.proving_grounds}, ${q.generations} x ${q.tribe_size})`],
    ["Culls", `drawdown above ${pct(r.drawdown_limit)} (never back), low fitness, stagnation -> revolution`],
    ["Held-out", `at least ${r.held_out_min_trades} trades on the 30% breeding never saw`],
    ["Random", `beat ${pct(r.random_beat_min)} of ${r.random_n} random strategies on the held-out bars`],
    ["Reshuffled", `unluckiest 1 in 20 of ${r.shuffle_n} reshuffled runs under ${pct(r.shuffle_dd_max)} drawdown`],
    ["Nudged", `${pct(r.nudge_hold_min)} of nudged settings still make money`],
    ["Floors", `drawdown <= ${pct(r.floor_max_dd)}, profit factor >= ${r.floor_pf}, trades >= ${r.floor_trades}`],
    ["Score", Object.entries(v.score_weights).map(([k, w]) => `${k.replace(/_/g, " ")} ${w}`).join(" · ") + ` · GOOD at ${v.good_at}`],
  ];
  return (
    <table className="vt">
      <tbody>{rows.map(([k, x]) => <tr key={k}><td className="l dim" style={{ width: 150 }}>{k}</td><td className="l">{x}</td></tr>)}</tbody>
    </table>
  );
}

export function Rules({ s }: { s: EngineSettings }) {
  const c = s.costs, g = s.certified;
  const rows: [string, string][] = [
    ["Spread cost", c.spread_cost_mode === "bar" ? "each fill pays its signal bar's own spread (hour average where unknown)"
      : "each fill pays the hour's average spread"],
    ["Spread gate", c.spread_gate_mode === "regime"
      ? `refuse an entry above ${c.spread_regime_multiple}x the median spread of the last ${c.spread_regime_days} days`
      : `refuse an entry above the p${c.max_spread_gate_percentile} spread`],
    ["Commission", `${c.commission_per_lot_per_side} per lot per side (account currency)`],
    ["Trades", `at least ${g.min_trades}`],
    ["Drawdown", `at most ${pct(g.max_dd)}; Monte Carlo p95 at most ${pct(g.mc_dd_p95_max)}`],
    ["Neighbours", `${pct(g.neighbour_share_min)} make money; ${pct(g.plateau_min)} keep half the Sharpe`],
    ["Random entries", `beats ${pct(g.beat_random_min)} of 100 random-entry twins; bootstrap P(Sharpe <= 0) at most ${pct(g.bootstrap_p_max)}`],
    ["Profit spread", `no month above ${pct(g.max_month_share)} of the profit; a position in ${pct(g.min_months_in_market)} of months`],
    ["Held-out", `at least ${g.min_held_out_trades} trades on bars breeding never saw, and a positive return`],
    ["Grades", s.grades.map((x) => `${x.grade} >= ${x.min}`).join(" · ") + " · else F"],
  ];
  return (
    <table className="vt">
      <tbody>{rows.map(([k, v]) => <tr key={k}><td className="l dim" style={{ width: 150 }}>{k}</td><td className="l">{v}</td></tr>)}</tbody>
    </table>
  );
}

/** Diagnostics and the support report (research APP-8): this machine's numbers and one-press COPY. */
export function Support() {
  const d = usePoll(() => api.diagnostics(), 60_000);
  const [copied, setCopied] = useState<string | null>(null);
  const copy = async () => {
    try {
      const r = await api.supportReport();
      await navigator.clipboard.writeText(r.text);
      setCopied("Copied: paste it to whoever helps you. Long numbers (account logins) are masked.");
    } catch (e) {
      setCopied(errText(e));
    }
  };
  const x = d.data;
  return (
    <section className="panel" style={{ flex: "1 1 320px", maxWidth: 460 }} aria-label="Support">
      <div className="panel-h">Support <small>this computer, read every minute</small></div>
      <div className="panel-b" style={{ display: "flex", flexDirection: "column", gap: 10 }}>
        {x ? (
          <table className="vt">
            <tbody>
              <tr><td className="l dim">ARQVIO</td><td className="l">{x.version ?? "—"} · Python {x.python}</td></tr>
              <tr><td className="l dim">System</td><td className="l">{x.os} · {x.cores ?? "—"} cores</td></tr>
              <tr><td className="l dim">Memory</td><td className="l">{x.memory.available_gb ?? "—"} GB free of {x.memory.total_gb ?? "—"} GB</td></tr>
              <tr><td className="l dim">Disk</td><td className="l">{x.disk.free_gb ?? "—"} GB free ({x.disk.path})</td></tr>
              <tr><td className="l dim">Vault</td><td className="l">{String(x.vault.strategies ?? x.vault.error ?? "—")} strategies</td></tr>
              <tr><td className="l dim">Problems</td><td className="l">{x.recent_problems.length ? `${x.recent_problems.length} recent warnings / errors` : "none recently"}</td></tr>
            </tbody>
          </table>
        ) : d.load.state === "error" ? <span className="warn">{errText(d.load.error)}</span> : <span className="dim">reading…</span>}
        <div className="row" style={{ gap: 8 }}>
          <button type="button" className="btn" onClick={() => void copy()}>Copy support report</button>
        </div>
        {copied && <span className="hint">{copied}</span>}
      </div>
    </section>
  );
}

export function Settings({ prefs, setPrefs, home }: { prefs: Prefs; setPrefs: (p: Prefs) => void; home: string | null }) {
  const eng = usePoll(() => api.engineSettings(), 60_000);
  return (
    <div className="row" style={{ alignItems: "flex-start", gap: 12, flexWrap: "wrap" }}>
      <section className="panel" style={{ flex: "1 1 320px", maxWidth: 460 }} aria-label="This computer">
        <div className="panel-h">This computer</div>
        <div className="panel-b" style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <SettingsForm prefs={prefs} setPrefs={setPrefs} home={home} />
        </div>
      </section>
      <Support />
      <section className="panel" style={{ flex: "2 1 480px" }} aria-label="Engine rules">
        <div className="panel-h">Engine rules <small>CERTIFIED needs every gate; read from the engine, not editable here</small></div>
        <div className="panel-b">
          {eng.data ? <><Rules s={eng.data} /><div className="label" style={{ margin: "14px 0 6px" }}>Engine V1 · UNCERTIFIED until Certify (V2)</div><V1Rules s={eng.data} /></>
            : eng.load.state === "error" ? <span className="warn">{errText(eng.load.error)}</span>
            : <span className="dim">reading the engine…</span>}
        </div>
      </section>
    </div>
  );
}
