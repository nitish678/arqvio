// The campaign panel: which MetaTrader 5 the data comes from, which markets and timeframes, how deep. Tabs keep the
// rest out of the way: Advanced (what each depth means, history limits, the data folder) and Gene pool (strategy
// families and entry gates).

import { useMemo, useState } from "react";
import { api } from "../api/ws";
import type { Depth, GenePoolDoc, LaunchOverview, Power, SearchEngine, SymbolsOverview, Terminal, TerminalsOverview } from "../api/ws";
import { Layer } from "./Dialogs";
import { FAMILY_NAMES, fmt } from "./util";

export const TIMEFRAMES = ["M5", "M15", "M30", "H1", "H4", "D1"] as const;
const YEARS = [1, 3, 6, 10];
const DEPTH_TEXT: Record<Depth, string> = { QUICK: "Quick", STANDARD: "Standard", DEEP: "Deep" };
const POWER_TEXT: Record<Power, string> = { eco: "Eco", medium: "Medium", high: "High", extreme: "Max" };
const ENGINE_TEXT: Record<SearchEngine, string> = { standard: "V2", v1: "V1" };
const ENGINE_HINT: Record<SearchEngine, string> = {
  standard: "V2: V1's breeding plus strict validation (deflated Sharpe, PBO, sham and canary tests, held-out); only V2 can certify a strategy",
  v1: "V1: tribal breeding (16 seeds, 3,000 proving-ground strategies, 2 tribes x 20 generations, Final War); its strategies stay UNCERTIFIED until V2 certifies them",
};
const FAMILY_TEXT: Record<string, string> = {
  MA_CROSS: "trend: a fast average crosses a slow one",
  RSI_REVERSION: "reversal: RSI leaves an oversold / overbought band",
  STOCH_CROSS: "reversal: stochastic %K crosses %D at the extremes",
  MACD_CROSS: "momentum: MACD crosses its signal line",
  DONCHIAN_BREAKOUT: "breakout: price leaves its N-bar channel",
};

export interface CampaignForm {
  symbols: string[];
  timeframes: string[];
  years: number;
  depth: Depth;
  power: Power;
  engine?: SearchEngine;
}

export function terminalLabel(t: Terminal): string {
  const a = t.account;
  const where = t.install_dir;
  if (a.state === "OK") {
    return `${a.company ?? ""} · ${a.server ?? ""} · ${a.mode ?? ""} · ${fmt.money(a.balance, a.currency)}  (${where})`;
  }
  return `${where} · ${a.state === "CLOSED" ? "not running" : a.state.toLowerCase()}${t.remembered_server ? ` · last ${t.remembered_server}` : ""}`;
}

function BrowseDialog({ symbols, picked, onClose, onToggle }: {
  symbols: SymbolsOverview; picked: string[]; onClose: () => void; onToggle: (s: string) => void;
}) {
  const [q, setQ] = useState("");
  const groups = useMemo(() => {
    const qq = q.trim().toLowerCase();
    const rows = symbols.symbols.filter((s) => !qq || s.name.toLowerCase().includes(qq) || s.description.toLowerCase().includes(qq));
    const by: Record<string, typeof rows> = {};
    for (const r of rows) {
      const g = r.path.split("\\").slice(0, -1).join(" / ") || "Other";
      (by[g] ??= []).push(r);
    }
    return Object.entries(by).sort(([a], [b]) => a.localeCompare(b));
  }, [symbols, q]);
  return (
    <Layer>
      <div className="scrim" role="dialog" aria-modal="true" aria-label="All symbols" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
        <section className="panel dialog">
          <div className="panel-h">All symbols <small>{symbols.symbols.length} offered by this terminal · {picked.length} picked</small></div>
          <div className="panel-b">
            <input className="input" style={{ width: "100%", marginBottom: 10 }} placeholder="Search: EURUSD, gold, US30…"
              value={q} onChange={(e) => setQ(e.target.value)} autoFocus aria-label="Search symbols" />
            <div className="browse-list">
              {groups.map(([g, rows]) => (
                <div key={g} style={{ display: "contents" }}>
                  <div className="browse-group label">{g}</div>
                  {rows.map((r) => {
                    const on = picked.includes(r.name);
                    return (
                      <label key={r.name} className={`asset${on ? " on" : ""}`} title={r.description}>
                        <input type="checkbox" checked={on} onChange={() => onToggle(r.name)} />
                        {r.name}
                      </label>
                    );
                  })}
                </div>
              ))}
              {groups.length === 0 && <div className="hint">No symbol matches “{q}”.</div>}
            </div>
          </div>
          <div className="dialog-f"><button type="button" className="btn btn-primary" onClick={onClose}>Done</button></div>
        </section>
      </div>
    </Layer>
  );
}

function Seg<T extends string | number>({ items, value, onPick, label, disabled, render, multi }: {
  items: readonly T[]; value: T | T[]; onPick: (v: T) => void; label: string; disabled?: boolean;
  render?: (v: T) => string; multi?: boolean;
}) {
  return (
    <div className="seg" role={multi ? "group" : "radiogroup"} aria-label={label}>
      {items.map((v) => {
        const on = Array.isArray(value) ? value.includes(v) : value === v;
        return (
          <button key={String(v)} type="button" aria-pressed={on} disabled={disabled} onClick={() => onPick(v)}>
            {render ? render(v) : String(v)}
          </button>
        );
      })}
    </div>
  );
}

export function CampaignPanel({ terms, termId, onTerm, onRefreshTerms, symbols, symbolsError, form, setForm, ov, busy,
  pool, poolError, onPatchPool, hidePool = false }: {
  terms: TerminalsOverview | null;
  termId: string | null;
  onTerm: (id: string) => void;
  onRefreshTerms: () => void;
  symbols: SymbolsOverview | null;
  symbolsError: string | null;
  form: CampaignForm;
  setForm: (f: CampaignForm) => void;
  ov: LaunchOverview | null;
  busy: boolean;
  pool: GenePoolDoc | null;
  poolError: string | null;
  onPatchPool: (body: Record<string, unknown>) => Promise<void>;
  /** DECK shows the gene pool on its own BREEDING tab */
  hidePool?: boolean;
}) {
  const [tab, setTab] = useState<"new" | "advanced" | "pool">("new");
  const [browse, setBrowse] = useState(false);
  const term = terms?.terminals.find((t) => t.id === termId) ?? null;
  const offered = new Set(symbols?.symbols.map((s) => s.name) ?? []);
  const toggleSym = (s: string) => setForm({ ...form, symbols: form.symbols.includes(s) ? form.symbols.filter((x) => x !== s)
    : [...form.symbols, s] });
  const toggleTf = (tf: string) => setForm({ ...form, timeframes: form.timeframes.includes(tf)
    ? form.timeframes.filter((x) => x !== tf) : [...form.timeframes, tf] });
  const mains = symbols?.main_pairs ?? [];
  const mainNames = new Set(mains.map((m) => m.symbol).filter(Boolean) as string[]);
  const extra = form.symbols.filter((s) => !mainNames.has(s));
  const markets = form.symbols.length * form.timeframes.length;

  return (
    <section className="panel" aria-label="Campaign" style={{ flex: "0 0 auto" }}>
      <div className="panel-h">
        Campaign
        <div className="tabs" role="tablist">
          {(hidePool ? (["new", "advanced"] as const) : (["new", "advanced", "pool"] as const)).map((k) => (
            <button key={k} type="button" role="tab" className="tab" aria-selected={tab === k} onClick={() => setTab(k)}>
              {k === "new" ? "New campaign" : k === "advanced" ? "Advanced" : "Gene pool"}
            </button>
          ))}
        </div>
        <span className="sp" />
        {busy && <span className="tag red">running · settings locked</span>}
      </div>

      {tab === "new" && (
        <div className="panel-b" style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <div className="row" style={{ flexWrap: "nowrap" }}>
            <span className="label" style={{ width: 92 }}>MetaTrader 5</span>
            {terms && terms.terminals.length > 0 ? (
              <select className="select" style={{ flex: "1 1 auto" }} value={termId ?? ""} disabled={busy}
                onChange={(e) => onTerm(e.target.value)} aria-label="MetaTrader 5 terminal for the data">
                {!termId && <option value="">choose a terminal</option>}
                {terms.terminals.map((t) => <option key={t.id} value={t.id}>{terminalLabel(t)}</option>)}
              </select>
            ) : (
              <span className="hint" style={{ flex: "1 1 auto" }}>{terms ? "No MetaTrader 5 found on this machine." : "Looking for MetaTrader 5…"}</span>
            )}
            <button type="button" className="icon-btn" onClick={onRefreshTerms} title="Look again for terminals and accounts" aria-label="Re-check MetaTrader 5">
              <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M13.5 8a5.5 5.5 0 1 1-1.6-3.9M13.5 2.5v3h-3" /></svg>
            </button>
          </div>
          {term && term.account.state !== "OK" && (
            <div className="note">{term.account.state === "CLOSED"
              ? `Open MetaTrader 5 (${term.path}) and log in to your account, then press the re-check button.`
              : term.account.detail}</div>
          )}
          {terms && terms.terminals.length === 0 && (
            <div className="note">Install MetaTrader 5 from your broker, log in, then press the re-check button. ARQVIO reads market
              data from it and installs your strategies into it.</div>
          )}

          <div>
            <div className="row" style={{ marginBottom: 8 }}>
              <span className="label">Markets</span>
              <span className="hint">{form.symbols.length} picked</span>
              <span style={{ flex: 1 }} />
              <button type="button" className="btn btn-link" onClick={() => setBrowse(true)} disabled={!symbols || busy}>
                Browse all symbols
              </button>
            </div>
            {symbolsError && <div className="err">{symbolsError}</div>}
            <div className="assets">
              {(mains.length ? mains : []).map((m) => {
                const name = m.symbol;
                const on = !!name && form.symbols.includes(name);
                return (
                  <label key={m.pair} className={`asset${on ? " on" : ""}${name ? "" : " off"}`}
                    title={name ? `${m.pair} on this broker: ${name}` : `${m.pair}: not offered by this terminal`}>
                    <input type="checkbox" checked={on} disabled={!name || busy} onChange={() => name && toggleSym(name)} />
                    {m.pair}
                    {name && name !== m.pair && <small>{name}</small>}
                  </label>
                );
              })}
              {extra.map((s) => (
                <label key={s} className="asset on" title={offered.has(s) ? s : `${s}: not offered by this terminal`}>
                  <input type="checkbox" checked disabled={busy} onChange={() => toggleSym(s)} />
                  {s}
                </label>
              ))}
              {!symbols && !symbolsError && <span className="hint">The markets appear when MetaTrader 5 is running.</span>}
            </div>
          </div>

          <div className="cols">
            <div className="field">
              <span className="label">Timeframes</span>
              <Seg items={TIMEFRAMES} value={form.timeframes} onPick={toggleTf} label="Timeframes" disabled={busy} multi />
            </div>
            <div className="field">
              <span className="label">History</span>
              <Seg items={YEARS} value={form.years} onPick={(y) => setForm({ ...form, years: y })} label="History in years"
                disabled={busy} render={(y) => `${y}y`} />
            </div>
            <div className="field">
              <span className="label">Engine</span>
              <Seg items={["standard", "v1"] as SearchEngine[]} value={form.engine ?? "standard"} label="Breeding engine"
                disabled={busy} onPick={(e) => setForm({ ...form, engine: e })} render={(e) => ENGINE_TEXT[e]} />
            </div>
            <div className="field">
              <span className="label">Depth</span>
              <Seg items={["QUICK", "STANDARD", "DEEP"] as Depth[]} value={form.depth} label="Search depth" disabled={busy}
                onPick={(d) => setForm({ ...form, depth: d })} render={(d) => DEPTH_TEXT[d]} />
            </div>
            <div className="field">
              <span className="label">Power</span>
              <Seg items={["eco", "medium", "high", "extreme"] as Power[]} value={form.power} label="CPU power" disabled={busy}
                onPick={(p) => setForm({ ...form, power: p })} render={(p) => POWER_TEXT[p]} />
            </div>
          </div>
          <div className="hint">
            {markets ? `${markets} campaign${markets === 1 ? "" : "s"}` : "Pick at least one market and one timeframe"}
            {ov && markets ? ` · ${DEPTH_TEXT[form.depth]}: up to ${fmt.int(ov.depths[form.depth].budget)} strategies tested each` : ""}
            {ov && markets ? ` · ${ov.power[form.power]} of ${ov.logical_cores} CPU cores` : ""}
            {markets ? ((form.engine ?? "standard") === "v1" ? " · " + ENGINE_HINT.v1 : " · finalists are re-tested on a second market") : ""}
          </div>
        </div>
      )}

      {tab === "advanced" && (
        <div className="panel-b" style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          {ov && (
            <table className="vt" style={{ maxWidth: 560 }}>
              <thead><tr><th className="l">Depth</th><th>Population</th><th>Generations</th><th>Tested (max)</th><th>Finalists</th></tr></thead>
              <tbody>
                {(Object.keys(ov.depths) as Depth[]).map((d) => (
                  <tr key={d} aria-selected={form.depth === d} onClick={() => !busy && setForm({ ...form, depth: d })}>
                    <td className="l">{DEPTH_TEXT[d]}</td><td>{ov.depths[d].pop}</td><td>{ov.depths[d].gen}</td>
                    <td>{fmt.int(ov.depths[d].budget)}</td><td>{ov.depths[d].final_n}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          <div className="note">
            Every campaign first runs a short safety check on a copy of the data, then the genetic search. The finalists
            are re-tested on another market of the same timeframe (when you pick one market, ARQVIO reads a main pair for
            this), and only strategies that stay alive go to the vault.
          </div>
          <div className="note">A timeframe gets at least the history a strategy needs (2,000 bars: about 1.6 years of H4, 9.6 years of D1) even when History asks for less; a symbol with less history in your terminal gets what it has.</div>
          {ov && <div className="hint">Data folder: <span className="mono">{ov.home}</span></div>}
        </div>
      )}

      {tab === "pool" && (
        <div className="panel-b" style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          {poolError && <div className="err">{poolError}</div>}
          {pool ? (
            <>
              <div className="row" style={{ gap: 8, alignItems: "center" }}>
                <div className="label" style={{ flex: 1 }}>Strategy families the breeder may use</div>
                <button type="button" className="btn btn-sm" disabled={busy}
                  title="Weigh the families by what survives in your vault; half of every draw stays uniform as the control"
                  onClick={() => void api.smartPriors().then((s) => onPatchPool(s.patch))}>Smart priors</button>
              </div>
              <div className="assets" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(240px, 1fr))" }}>
                {Object.entries(pool.families).map(([f, v]) => (
                  <label key={f} className={`asset${v.enabled ? " on" : ""}`} style={{ height: "auto", padding: "8px 10px" }}
                    title={f}>
                    <input type="checkbox" checked={v.enabled} disabled={busy}
                      onChange={() => void onPatchPool({ families: { [f]: { enabled: !v.enabled } } })} />
                    <span style={{ display: "flex", flexDirection: "column", gap: 3 }}>
                      <span>{FAMILY_NAMES[f] ?? f}</span>
                      <small style={{ margin: 0, fontFamily: "var(--font-ui)", fontSize: 11 }}>{FAMILY_TEXT[f] ?? ""}</small>
                    </span>
                  </label>
                ))}
              </div>
              <div className="label">Entry gates (genes the breeder can switch on)</div>
              <div className="cols">
                <div className="field">
                  <span className="hint">Session window: open only in Asia, London, New York or London+NY hours (UTC)</span>
                  <Seg items={["Off", "On"]} value={pool.gates?.session ? "On" : "Off"} label="Session window gate" disabled={busy}
                    onPick={(v) => void onPatchPool({ gates: { session: v === "On" } })} />
                </div>
                <div className="field">
                  <span className="hint">Trend EMA: open only on the side of the 50 / 100 / 200 EMA</span>
                  <Seg items={["Off", "On"]} value={pool.gates?.trend_ema ? "On" : "Off"} label="Trend EMA gate" disabled={busy}
                    onPick={(v) => void onPatchPool({ gates: { trend_ema: v === "On" } })} />
                </div>
                <div className="field">
                  <span className="hint">Bias switches: side from SMA(100), 20-bar momentum or the 20-bar channel middle</span>
                  <Seg items={["Off", "On"]} value={pool.gates?.bias_switches ? "On" : "Off"} label="Bias switches" disabled={busy}
                    onPick={(v) => void onPatchPool({ gates: { bias_switches: v === "On" } })} />
                </div>
                <div className="field">
                  <span className="hint">Filter switches: no Friday openings, a calm-volatility band, RSI(14) not at an extreme</span>
                  <Seg items={["Off", "On"]} value={pool.gates?.filter_switches ? "On" : "Off"} label="Filter switches" disabled={busy}
                    onPick={(v) => void onPatchPool({ gates: { filter_switches: v === "On" } })} />
                </div>
                <div className="field">
                  <span className="hint">Trade management: breakeven moves the stop to the entry once a close is a full stop ahead</span>
                  <Seg items={["Off", "On"]} value={pool.gates?.mgmt_switches ? "On" : "Off"} label="Management switches" disabled={busy}
                    onPick={(v) => void onPatchPool({ gates: { mgmt_switches: v === "On" } })} />
                </div>
              </div>
              <div className="hint">Pool <span className="mono">{pool.hash}</span> · saved at once; it cannot change while a campaign runs.</div>
            </>
          ) : !poolError && <div className="hint">Reading the gene pool…</div>}
        </div>
      )}

      {browse && symbols && <BrowseDialog symbols={symbols} picked={form.symbols} onClose={() => setBrowse(false)} onToggle={toggleSym} />}
    </section>
  );
}
