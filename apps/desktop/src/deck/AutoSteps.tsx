// One automation step's settings card (CAMPAIGN, PORTFOLIO, PARITY, PURGE), research field by field with its
// defaults; REPLACE and REASSESS show why they are not built yet. Blank numbers mean "no limit" / "the launch page".
// idea: research automation editor (RF 46645-47455); layout and copy are ARQVIO's.

import type { AutomationList, AutoStep, NamedPortfolio } from "../api/ws";

const POWER = [["", "INHERIT"], ["eco", "ECO"], ["medium", "MEDIUM"], ["high", "HIGH"], ["extreme", "MAX"]] as const;
const DEPTH = [["", "INHERIT"], ["QUICK", "QUICK"], ["STANDARD", "STANDARD"], ["DEEP", "DEEP"]] as const;
const HISTORY = [["", "INHERIT"], ["FOCUSED", "FOCUSED ≈3Y"], ["DEEP", "DEEP ≈6Y"], ["BOTH", "BOTH"]] as const;
const ENGINE = [["", "INHERIT"], ["v1", "V1"], ["standard", "V2"]] as const;

export const KIND_TEXT: Record<string, string> = {
  campaign: "CAMPAIGN", portfolio: "PORTFOLIO", parity: "PARITY", purge: "PURGE", replace: "REPLACE", reassess: "REASSESS",
};
export const KIND_HINT: Record<string, string> = {
  campaign: "Breed new strategies on the markets you pick",
  portfolio: "Fill a portfolio with strategies that trade differently",
  parity: "Test how closely a portfolio's backtests match MetaTrader",
  purge: "Send weak strategies to the recycle bin. Portfolio members are never touched",
  replace: "Swap a portfolio's live bots for new strategies",
  reassess: "Check strategies on new data since they were trained",
};

function Chips<T extends string>({ items, value, onPick, multi }: {
  items: readonly (readonly [T, string])[];
  value: T | T[];
  onPick: (v: T) => void;
  multi?: boolean;
}) {
  const on = (v: T) => (multi ? (value as T[]).includes(v) : value === v);
  return <span className="chips">{items.map(([v, l]) => <button key={v} type="button" className="chip" aria-pressed={on(v)} onClick={() => onPick(v)}>{l}</button>)}</span>;
}

function Num({ label, value, onChange, min, max, step, hint }: {
  label: string; value: unknown; onChange: (v: number | null) => void; min?: number; max?: number; step?: number; hint?: string;
}) {
  return (
    <label className="af" title={hint}>
      <span>{label}</span>
      <input type="number" value={value == null ? "" : String(value)} min={min} max={max} step={step ?? 1}
        placeholder="—" onChange={(e) => onChange(e.target.value === "" ? null : Number(e.target.value))} />
    </label>
  );
}

const toggle = <T,>(xs: T[], x: T) => (xs.includes(x) ? xs.filter((y) => y !== x) : [...xs, x]);

export function StepCard({ step, set, meta, portfolios, hasPortfolioStep }: {
  step: AutoStep;
  set: (s: AutoStep) => void;
  meta: AutomationList;
  portfolios: NamedPortfolio[];
  hasPortfolioStep: boolean;
}) {
  const s = step as Record<string, unknown>;
  const up = (patch: Record<string, unknown>) => set({ ...step, ...patch } as AutoStep);
  const crit = (s.criteria as Record<string, number | null> | undefined) ?? {};
  const setCrit = (k: string, v: number | null) => up({ criteria: { ...crit, [k]: v } });
  const ov = (s.overrides as Record<string, string> | undefined) ?? {};
  const setOv = (k: string, v: string) => up({ overrides: { ...ov, [k]: v || undefined } });

  if (step.kind === "campaign") {
    const groups = (s.preset_groups as string[]) ?? [];
    const tfs = (s.timeframes as string[]) ?? [];
    const changed = Object.values(ov).filter(Boolean).length;
    return (
      <div className="as-body">
        <div className="as-row"><span className="k">MARKET GROUPS</span><span className="chips">
          {meta.groups.map((g) => (
            <button key={g.id} type="button" className="chip" aria-pressed={groups.includes(g.id)} disabled={g.count === 0}
              title={g.count === 0 ? `NO ${g.label} MARKETS ON YOUR BROKER` : g.count == null ? "open MetaTrader 5 to count them" : `${g.count} on your broker`}
              onClick={() => up({ preset_groups: toggle(groups, g.id) })}>{g.label}{g.count != null ? ` ${g.count}` : ""}</button>
          ))}</span></div>
        <div className="as-row"><span className="k">EXTRA MARKETS</span>
          <input className="as-text" placeholder="e.g. US500, BTCUSD" value={((s.extra_symbols as string[]) ?? []).join(", ")}
            onChange={(e) => up({ extra_symbols: e.target.value.split(/[,\s]+/).map((x) => x.trim().toUpperCase()).filter(Boolean) })} /></div>
        <div className="as-row"><span className="k">TIMEFRAMES</span>
          <Chips items={meta.timeframes.map((t) => [t, t] as const)} value={tfs} multi onPick={(t) => up({ timeframes: toggle(tfs, t) })} /></div>
        <div className="as-row"><span className="k">DEAD MARKETS</span>
          <Chips items={[["skip", "SKIP THEM"], ["all", "BREED EVERYTHING"]] as const} value={s.skip_graveyards === false ? "all" : "skip"}
            onPick={(v) => up({ skip_graveyards: v === "skip" })} />
          {s.skip_graveyards !== false && <>
            <Num label="NO SURVIVORS FOR (RUNS)" value={s.graveyard_runs} min={1} max={10} onChange={(v) => up({ graveyard_runs: v ?? 2 })} />
            <Num label="TRY AGAIN AFTER (RUNS)" value={s.graveyard_retest_every} min={2} max={50} onChange={(v) => up({ graveyard_retest_every: v ?? 4 })}
              hint="or after 30 days, whichever comes first; never a permanent blacklist" />
          </>}</div>
        <div className="as-sub">CAMPAIGN SETTINGS <small>{changed ? `${changed} CHANGED` : "SAME AS YOUR LAUNCH PAGE"} · blank = your last launch ({meta.launch_defaults.years}y · {meta.launch_defaults.depth} · {meta.launch_defaults.power})</small></div>
        <div className="as-row"><span className="k">POWER</span><Chips items={POWER} value={(ov.power ?? "") as never} onPick={(v) => setOv("power", v)} /></div>
        <div className="as-row"><span className="k">DEPTH</span><Chips items={DEPTH} value={(ov.depth ?? "") as never} onPick={(v) => setOv("depth", v)} /></div>
        <div className="as-row"><span className="k">HISTORY</span><Chips items={HISTORY} value={(ov.history ?? "") as never} onPick={(v) => setOv("history", v)} /></div>
        <div className="as-row"><span className="k">ENGINE</span><Chips items={ENGINE} value={(ov.engine ?? "") as never} onPick={(v) => setOv("engine", v)} /></div>
      </div>
    );
  }
  if (step.kind === "portfolio") {
    const corr = Number(s.max_avg_corr ?? 0.35);
    const how = corr === 0.35 && Number(s.max_per_symbol) === 4 ? "spread" : corr >= 0.99 ? "best" : "custom";
    return (
      <div className="as-body">
        <div className="as-row"><span className="k">PORTFOLIO</span>
          <input className="as-text" list="as-portfolios" placeholder="e.g. Auto Best 20" value={String(s.name ?? "")}
            onChange={(e) => up({ name: e.target.value })} />
          <datalist id="as-portfolios">{portfolios.map((p) => <option key={p.id} value={p.name} />)}</datalist>
          <small className="dim">{portfolios.some((p) => p.name.toLowerCase() === String(s.name ?? "").toLowerCase()) ? "existing portfolio: adds only" : "a new portfolio is created"}</small></div>
        <div className="as-row"><span className="k">MODE</span>
          <Chips items={[["top_up", "TOP UP"], ["re_pick", "RE-PICK"]] as const} value={(s.mode as never) ?? "top_up"} onPick={(v) => up({ mode: v })} />
          <small className="dim">{s.mode === "re_pick" ? "rebuild every run; live bots keep their seats" : "fill empty seats; over the limit, remove what it added (lowest score first)"}</small></div>
        <div className="as-row"><span className="k">HOW TO PICK</span>
          <Chips items={[["spread", "SPREAD THE RISK"], ["best", "JUST BEST SCORES"], ["custom", "CUSTOM"]] as const} value={how}
            onPick={(v) => v === "spread" ? up({ max_avg_corr: 0.35, max_per_symbol: 4 }) : v === "best" ? up({ max_avg_corr: 0.99, max_per_symbol: 4 }) : up({ max_avg_corr: 0.5 })} /></div>
        <div className="as-grid">
          <Num label="TEAM SIZE" value={s.target_size} min={1} max={200} onChange={(v) => up({ target_size: v ?? 30 })} />
          <Num label="MIN SCORE" value={s.min_score} min={0} max={100} onChange={(v) => up({ min_score: v ?? 0 })} />
          <Num label="MAX PER MARKET" value={s.max_per_symbol} min={0} max={50} onChange={(v) => up({ max_per_symbol: v ?? 0 })} hint="0 = off" />
          <Num label="MAX PER TIMEFRAME" value={s.max_per_tf} min={0} max={50} onChange={(v) => up({ max_per_tf: v ?? 0 })} hint="per market and timeframe; 0 = off" />
          <Num label="MAX AGE (DAYS)" value={s.max_age_days} min={0} onChange={(v) => up({ max_age_days: v ?? 0 })} hint="0 = any age" />
          <label className="af"><span>SIMILARITY LIMIT {corr.toFixed(2)}</span>
            <input type="range" min={0.1} max={0.99} step={0.01} value={corr} onChange={(e) => up({ max_avg_corr: Number(e.target.value) })}
              title="skip a pick whose daily returns correlate above this with any member" /></label>
        </div>
        <div className="as-row"><span className="k">POOL</span>
          <Chips items={[["run", "THIS RUN"], ["vault", "WHOLE VAULT"]] as const} value={s.use_run_pool === false ? "vault" : "run"}
            onPick={(v) => up({ use_run_pool: v === "run" })} /><small className="dim">THIS RUN = this run's campaign output (the whole vault when no campaign step comes first)</small></div>
        <div className="as-sub">STRATEGY CRITERIA <small>blank = no limit</small></div>
        <div className="as-grid">
          <Num label="MIN TRADES" value={crit.min_trades} min={0} onChange={(v) => setCrit("min_trades", v)} />
          <Num label="MAX DD %" value={crit.max_dd} min={0} step={0.5} onChange={(v) => setCrit("max_dd", v)} />
          <Num label="MIN RETURN %" value={crit.min_return} step={1} onChange={(v) => setCrit("min_return", v)} />
          <Num label="MIN OOS RETURN %" value={crit.min_oos_return} step={1} onChange={(v) => setCrit("min_oos_return", v)} />
        </div>
        <div className="as-row"><span className="k">DEPLOY NEW BOTS</span><span className="chip ro">OFF</span>
          <small className="dim">putting bots live from an automation waits for your OK (it trades your account)</small></div>
      </div>
    );
  }
  if (step.kind === "parity") {
    return (
      <div className="as-body">
        <div className="as-row"><span className="k">PASS MARK</span>
          <Chips items={[["A", "A ONLY"], ["B", "B OR BETTER"], ["C", "C OR BETTER"]] as const} value={(s.pass_mark as never) ?? "B"} onPick={(v) => up({ pass_mark: v })} /></div>
        <div className="as-grid"><Num label="TESTS PER RUN" value={s.tests_per_run} min={1} max={100} onChange={(v) => up({ tests_per_run: v ?? 10 })} hint="about 2 minutes each in a test MetaTrader 5" /></div>
        {!hasPortfolioStep && (
          <div className="as-row"><span className="k">PORTFOLIO</span>
            <select className="select" value={String(s.portfolio ?? "")} onChange={(e) => up({ portfolio: e.target.value })}>
              <option value="">pick a portfolio</option>
              {portfolios.map((p) => <option key={p.id} value={p.id}>{p.name} ({p.members.length})</option>)}
            </select></div>
        )}
        <div className="dk-note">Tests this run's additions first, then members never tested. A bot below the pass mark never goes live. A MetaTrader 5 that runs bots is never used for a test.</div>
      </div>
    );
  }
  if (step.kind === "purge") {
    return (
      <div className="as-body">
        <div className="as-grid">
          <Num label="MIN SCORE" value={crit.min_score} min={0} max={100} onChange={(v) => setCrit("min_score", v)} />
          <Num label="MIN TRADES" value={crit.min_trades} min={0} onChange={(v) => setCrit("min_trades", v)} />
          <Num label="MAX DD %" value={crit.max_dd} min={0} step={0.5} onChange={(v) => setCrit("max_dd", v)} />
          <Num label="MIN RETURN %" value={crit.min_return} step={1} onChange={(v) => setCrit("min_return", v)} />
          <Num label="MIN OOS RETURN %" value={crit.min_oos_return} step={1} onChange={(v) => setCrit("min_oos_return", v)} />
          <Num label="MAX AGE (DAYS)" value={crit.max_age_days} min={0} onChange={(v) => setCrit("max_age_days", v)} />
        </div>
        <div className="dk-note">All blank = built-in criteria (min 30 trades, max 35 % drawdown, no loss). Sends to the recycle bin for 30 days; portfolio members and live bots are never touched.</div>
      </div>
    );
  }
  return <div className="as-body"><div className="dk-note bad">{meta.planned_why[step.kind] ?? "not built yet"}</div></div>;
}
