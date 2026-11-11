// COMMAND: the whole engine on one screen. Left: what to breed (targets), what the vault holds, the best genomes.
// Centre: the neural core and the run's numbers. Right: the deployed bots, the generations of this run, the live feed.
// Every number is the engine's; a panel with nothing to show says what would fill it.
// idea: research §7.3 "Deck home" anatomy (vault counters, neural core, portfolio, genome synthesis, live feed);
// the arrangement around ARQVIO's pipeline and all copy are ARQVIO's.

import { useEffect, useMemo, useState } from "react";
import { api, type DeployOverview, type LaunchOverview, type StrategyRow, type SymbolsOverview } from "../api/ws";
import type { CampaignForm } from "../ws/CampaignPanel";
import { TIMEFRAMES } from "../ws/CampaignPanel";
import { LogTerminal } from "../ws/LeftRail";
import { FAMILY_SHORT, familyName, fmt, shortId } from "../ws/util";
import { bestOf, clock, type GenPoint, launchNow, useLiveView, useNow } from "./launchInfo";
import { GenerationFan } from "./GenerationFan";
import { NeuralCore } from "./NeuralCore";

const ENGINES = [["standard", "V2"], ["v1", "V1"]] as const;
const POWERS = [["eco", "ECO"], ["medium", "MEDIUM"], ["high", "HIGH"], ["extreme", "MAX"]] as const;
const DEPTHS = [["QUICK", "QUICK"], ["STANDARD", "STANDARD"], ["DEEP", "DEEP"]] as const;
const YEARS = [1, 3, 6, 10];

function Chip({ on, disabled, onClick, children, title }: { on: boolean; disabled?: boolean; onClick: () => void;
  children: React.ReactNode; title?: string }) {
  return <button type="button" className="chip" aria-pressed={on} disabled={disabled} onClick={onClick} title={title}>{children}</button>;
}

function Panel({ title, sub, right, className, children }: { title: string; sub?: React.ReactNode; right?: React.ReactNode;
  className?: string; children: React.ReactNode }) {
  return (
    <section className={`dk-panel ${className ?? ""}`} aria-label={title}>
      <div className="dk-ph"><span className="dk-pt">{title}</span>{sub && <span className="dk-ps">{sub}</span>}<span className="sp" />{right}</div>
      <div className="dk-pb">{children}</div>
    </section>
  );
}

export function Targets({ symbols, form, setForm, busy, ov, onMore }: {
  symbols: SymbolsOverview | null;
  form: CampaignForm;
  setForm: (f: CampaignForm) => void;
  busy: boolean;
  ov: LaunchOverview | null;
  onMore: () => void;
}) {
  const mains = symbols?.main_pairs ?? [];
  const mainNames = new Set(mains.map((m) => m.symbol).filter(Boolean) as string[]);
  const extra = form.symbols.filter((s) => !mainNames.has(s));
  const tog = (xs: string[], x: string) => (xs.includes(x) ? xs.filter((y) => y !== x) : [...xs, x]);
  const runs = form.symbols.length * form.timeframes.length;
  const engine = form.engine ?? "standard";
  if (busy && ov?.launch) {
    const r = ov.launch.request;
    return (
      <Panel title="Targets" sub="running now">
        <div className="kv-rows">
          <div><span className="k">SYMBOLS</span><span className="chips">{r.symbols.map((s) => <span key={s} className="chip ro">{s}</span>)}</span></div>
          <div><span className="k">TIMEFRAMES</span><span className="chips">{r.timeframes.map((s) => <span key={s} className="chip ro">{s}</span>)}</span></div>
          <div><span className="k">DEPTH · POWER</span><span className="chips"><span className="chip ro">{r.depth}</span><span className="chip ro">{r.power.toUpperCase()}</span></span></div>
          <div><span className="k">HISTORY</span><span className="chips"><span className="chip ro">{r.years}Y</span></span></div>
        </div>
        <div className="dk-note">Targets are locked while the campaign runs. STOP it to change them, or QUEUE the next set-up.</div>
      </Panel>
    );
  }
  return (
    <Panel title="Targets" sub={`scope · ${runs} run${runs === 1 ? "" : "s"} · ${form.symbols.length} market${form.symbols.length === 1 ? "" : "s"} × ${form.timeframes.length} timeframe${form.timeframes.length === 1 ? "" : "s"}`}
      right={<button type="button" className="lnk" onClick={onMore}>ALL SETTINGS ›</button>}>
      <div className="kv-rows">
        <div><span className="k">SYMBOLS</span><span className="chips">
          {mains.map((m) => {
            const name = m.symbol;
            return <Chip key={m.pair} on={!!name && form.symbols.includes(name)} disabled={!name}
              title={name ? `${m.pair} on this broker: ${name}` : `${m.pair}: not offered by this terminal`}
              onClick={() => name && setForm({ ...form, symbols: tog(form.symbols, name) })}>{m.pair}</Chip>;
          })}
          {extra.map((s) => <Chip key={s} on onClick={() => setForm({ ...form, symbols: tog(form.symbols, s) })}>{s}</Chip>)}
          {!symbols && <span className="dim">markets appear when MetaTrader 5 is running</span>}
        </span></div>
        <div><span className="k">TIMEFRAMES</span><span className="chips">
          {TIMEFRAMES.map((tf) => <Chip key={tf} on={form.timeframes.includes(tf)} onClick={() => setForm({ ...form, timeframes: tog(form.timeframes, tf) })}>{tf}</Chip>)}
        </span></div>
        <div><span className="k">ENGINE</span><span className="chips">
          {ENGINES.map(([k, l]) => <Chip key={k} on={engine === k} onClick={() => setForm({ ...form, engine: k })}
            title={k === "v1" ? "V1: tribal breeding; its strategies stay UNCERTIFIED until V2 certifies them" : "V2: V1's breeding plus strict validation; only V2 certifies"}>{l}</Chip>)}
        </span></div>
        <div><span className="k">POWER</span><span className="chips">
          {POWERS.map(([k, l]) => <Chip key={k} on={form.power === k} onClick={() => setForm({ ...form, power: k })}
            title={ov ? `${ov.power[k]} of ${ov.logical_cores} CPU cores` : undefined}>{l}</Chip>)}
        </span></div>
        <div><span className="k">DEPTH</span><span className="chips">
          {DEPTHS.map(([k, l]) => <Chip key={k} on={form.depth === k} onClick={() => setForm({ ...form, depth: k })}
            title={ov ? `up to ${fmt.int(ov.depths[k].budget)} strategies tested per run` : undefined}>{l}</Chip>)}
        </span></div>
        <div><span className="k">HISTORY</span><span className="chips">
          {YEARS.map((y) => <Chip key={y} on={form.years === y} onClick={() => setForm({ ...form, years: y })}>{`${y}Y`}</Chip>)}
        </span></div>
      </div>
    </Panel>
  );
}

export function VaultSummary({ rows, onOpen, onAll }: { rows: StrategyRow[]; onOpen: (id: string) => void; onAll: () => void }) {
  const top = useMemo(() => [...rows].filter((r) => !r.copy_of).sort((a, b) => b.score - a.score).slice(0, 8), [rows]);
  const avg = rows.length ? rows.reduce((s, r) => s + r.score, 0) / rows.length : null;
  const bySym = new Map<string, number>();
  rows.forEach((r) => bySym.set(r.symbol, (bySym.get(r.symbol) ?? 0) + 1));
  const topSym = [...bySym.entries()].sort((a, b) => b[1] - a[1])[0]?.[0] ?? "—";
  const cert = rows.filter((r) => r.tier === "CERTIFIED").length;
  return (
    <Panel title="Strategy vault" sub="best by score" right={<button type="button" className="lnk" onClick={onAll}>OPEN VAULT ›</button>}>
      <div className="vs-head">
        <div className="vs-big"><b>{fmt.int(rows.length)}</b><span>STRATEGIES BRED</span></div>
        <div className="vs-kv"><span>VAULT AVG</span><b>{avg == null ? "—" : `${avg.toFixed(1)}`}</b></div>
        <div className="vs-kv"><span>TOP SYMBOL</span><b>{topSym}</b></div>
        <div className="vs-kv"><span>CERTIFIED</span><b>{cert}</b></div>
      </div>
      {top.length ? (
        <table className="dk-t">
          <thead><tr><th className="l">STRATEGY</th><th>SCORE</th><th>RET</th><th>DD</th><th>TRADES</th></tr></thead>
          <tbody>
            {top.map((r) => (
              <tr key={r.id} onClick={() => onOpen(r.id)} title={`${familyName(r.family)} · ${r.id}`}>
                <td className="l"><b>{r.symbol} {r.timeframe}</b><small>{FAMILY_SHORT[r.family] ?? r.family} · #{shortId(r.id)}{r.engine === "v1" ? " · V1" : ""}</small></td>
                <td className="acc">S {r.score.toFixed(0)}</td>
                <td className={(r.total_return ?? 0) >= 0 ? "pos" : "neg"}>{fmt.pct(r.total_return)}</td>
                <td className="neg">{fmt.pct(r.max_dd == null ? null : -Math.abs(r.max_dd))}</td>
                <td>{fmt.int(r.n_trades)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : <div className="dk-empty">NO STRATEGIES YET · RUN A CAMPAIGN</div>}
    </Panel>
  );
}

// the genome classes, coloured by role (signals, bias, filters, management, exits)
type Gene = { cls: "sig" | "bias" | "filt" | "mgmt" | "exit"; text: string };

/** Only genes the backtest engine reads (same rule as the inspector's topology): unread DNA fields are not shown. */
export function genesOf(dna: Record<string, unknown>): Gene[] {
  const g: Gene[] = [];
  const list = (k: string) => (Array.isArray(dna[k]) ? (dna[k] as string[]) : []);
  g.push({ cls: "sig", text: `sig_${String(dna.signal ?? "?").toLowerCase()}` });
  list("signal_switches").forEach((s) => g.push({ cls: "sig", text: `sig_${s}` }));
  list("bias_switches").forEach((s) => g.push({ cls: "bias", text: `bias_${s}` }));
  if (typeof dna.trend_ema === "number" && dna.trend_ema > 0) g.push({ cls: "bias", text: `bias_ema${dna.trend_ema}` });
  list("filter_switches").forEach((s) => g.push({ cls: "filt", text: `filt_${s}` }));
  if (dna.session) g.push({ cls: "filt", text: `filt_session_${String(dna.session).toLowerCase()}` });
  list("mgmt_switches").forEach((s) => g.push({ cls: "mgmt", text: `mgmt_${s}` }));
  if (typeof dna.sl_atr === "number" && typeof dna.tp_atr === "number") g.push({ cls: "exit", text: `sl ${dna.sl_atr.toFixed(1)}×atr · tp ${dna.tp_atr.toFixed(1)}×atr` });
  return g;
}

function Helix() {
  // decoration only: a double strand drawn once, no data
  const dots = Array.from({ length: 16 }, (_, i) => i);
  return (
    <svg className="helix" viewBox="0 0 60 200" aria-hidden="true">
      {dots.map((i) => {
        const y = 8 + i * 12;
        const x = 30 + Math.sin(i * 0.7) * 20;
        const x2 = 30 - Math.sin(i * 0.7) * 20;
        return <g key={i}><line x1={x} y1={y} x2={x2} y2={y} className="hx-l" /><circle cx={x} cy={y} r="3.2" className="hx-a" /><circle cx={x2} cy={y} r="3.2" className="hx-b" /></g>;
      })}
    </svg>
  );
}

export function GenomeSynthesis({ rows, onOpen }: { rows: StrategyRow[]; onOpen: (id: string) => void }) {
  const top = useMemo(() => [...rows].filter((r) => !r.copy_of).sort((a, b) => b.score - a.score).slice(0, 5), [rows]);
  const ids = top.map((r) => r.id).join(",");
  const [dna, setDna] = useState<Record<string, Gene[]>>({});
  useEffect(() => {
    let alive = true;
    top.forEach((r) => {
      if (dna[r.id]) return;
      api.strategy(r.id).then((d) => alive && setDna((m) => ({ ...m, [r.id]: genesOf(d.dna) }))).catch(() => undefined);
    });
    return () => {
      alive = false;
    };
  }, [ids]);  // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <Panel title="Genome synthesis" sub="best vaulted genomes">
      <div className="gs">
        <Helix />
        <div className="gs-list">
          {top.length === 0 && <div className="dk-empty">GENOMES APPEAR WHEN THE VAULT HAS STRATEGIES</div>}
          {top.map((r, i) => (
            <button type="button" key={r.id} className="gs-row" onClick={() => onOpen(r.id)}>
              <div className="gs-top">
                <span className="gs-n">{String(i + 1).padStart(2, "0")}</span>
                <b>#{shortId(r.id)}</b><span>{r.symbol} {r.timeframe}</span>
                <span className="sp" />
                <span className="acc">S {r.score.toFixed(0)}</span>
                <span className={(r.total_return ?? 0) >= 0 ? "pos" : "neg"}>{fmt.pct(r.total_return)}</span>
              </div>
              <div className="gs-genes">
                {(dna[r.id] ?? []).map((g, k) => <span key={k} className={`gn ${g.cls}`}>{g.text}</span>)}
                {!dna[r.id] && <span className="dim">reading DNA…</span>}
              </div>
            </button>
          ))}
        </div>
      </div>
    </Panel>
  );
}

export function GenChart({ points, height = 120 }: { points: GenPoint[]; height?: number }) {
  if (points.length < 2) return null;
  const W = 400;
  const H = height;
  const best = points.map((p) => p.best ?? 0);
  const lo = Math.min(...best);
  const hi = Math.max(...best);
  const maxAlive = Math.max(1, ...points.map((p) => p.alive));
  const x = (i: number) => 6 + (i / (points.length - 1)) * (W - 12);
  const y = (v: number) => H - 14 - ((v - lo) / (hi - lo || 1)) * (H - 28);
  const path = best.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  const bw = Math.max(2, (W - 12) / points.length - 2);
  return (
    <svg className="genchart" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" role="img"
      aria-label={`best ${fmt.num(best[best.length - 1])} after ${points.length} generations`}>
      {points.map((p, i) => {
        const h = (p.alive / maxAlive) * (H * 0.45);
        return <rect key={i} x={x(i) - bw / 2} y={H - 10 - h} width={bw} height={h} className="gc-bar" />;
      })}
      <path d={path} className="gc-line" />
      {best.map((v, i) => <circle key={i} cx={x(i)} cy={y(v)} r="2.2" className="gc-dot" />)}
    </svg>
  );
}

function Generations({ points, ov }: { points: GenPoint[]; ov: LaunchOverview | null }) {
  const now = launchNow(ov);
  const v1 = now.prog?.engine === "v1";
  const view = useLiveView(now.busy);
  const lv = view?.live;
  return (
    <Panel title="Campaign generations" sub={lv ? `GEN ${lv.generation} · ${lv.survivors} survivors` : now.busy ? `live · ${v1 ? "top fitness" : "best Sharpe"} (line), alive (bars)` : "idle"}>
      {lv && lv.bots.length ? <GenerationFan view={view} compact /> : points.length >= 2 ? (
        <>
          <GenChart points={points} />
          <div className="gc-legend"><span>G{points[0].generation + 1}</span><span className="sp" />
            <span>{v1 ? "fitness" : "Sharpe"} {fmt.num(points[points.length - 1].best)} · alive {points[points.length - 1].alive}</span>
            <span className="sp" /><span>G{points[points.length - 1].generation + 1}</span></div>
          <div className="dk-note">Generations seen while this window was open.</div>
        </>
      ) : now.busy ? (
        <div className="dk-empty"><span className="spin" />{now.phase} · THE FIRST GENERATION DRAWS HERE</div>
      ) : <div className="dk-empty">NO BOTS YET · RUN A CAMPAIGN</div>}
    </Panel>
  );
}

function Deployed({ dep, rows, onOpen, onAll }: { dep: DeployOverview | null; rows: StrategyRow[]; onOpen: (id: string) => void;
  onAll: () => void }) {
  const bots = (dep?.bots ?? []).filter((b) => b.status !== "REMOVED");
  const byId = new Map(rows.map((r) => [r.id, r]));
  const bySym = new Map<string, number>();
  bots.forEach((b) => bySym.set(b.symbol, (bySym.get(b.symbol) ?? 0) + 1));
  const live = bots.filter((b) => b.heartbeat?.state === "LIVE").length;
  return (
    <Panel title="Portfolios" sub={dep ? `${bots.length} deployed · ${live} with a fresh heartbeat` : "reading…"}
      right={<button type="button" className="lnk" onClick={onAll}>OPEN ›</button>}>
      {dep?.stop_all && <div className="dk-note bad">STOP ALL is on: every ARQVIO bot stands by.</div>}
      {bots.length ? (
        <>
          <div className="exp">
            {[...bySym.entries()].sort((a, b) => b[1] - a[1]).map(([s, n]) => (
              <div key={s} className="exp-row"><span>{s}</span><span className="exp-bar"><i style={{ transform: `scaleX(${n / bots.length})` }} /></span>
                <b>{n}</b><small>{Math.round((n / bots.length) * 100)}%</small></div>
            ))}
          </div>
          <table className="dk-t">
            <thead><tr><th className="l">BOT</th><th>STAGE</th><th>SCORE</th><th>RET</th><th className="l">STATE</th></tr></thead>
            <tbody>
              {bots.slice(0, 6).map((b) => {
                const r = byId.get(b.source.vault_id);
                return (
                  <tr key={b.bot} onClick={() => r && onOpen(r.id)}>
                    <td className="l"><b>{b.symbol} {b.timeframe}</b><small>magic {b.magic}</small></td>
                    <td>{b.stage}</td>
                    <td className="acc">{r ? `S ${r.score.toFixed(0)}` : "—"}</td>
                    <td className={(r?.total_return ?? 0) >= 0 ? "pos" : "neg"}>{r ? fmt.pct(r.total_return) : "—"}</td>
                    <td className="l"><span className={`led ${b.heartbeat?.state === "LIVE" ? "ok" : "bad"}`} /> {b.heartbeat?.state ?? b.status}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </>
      ) : <div className="dk-empty">NO BOTS DEPLOYED · PICK ONE IN THE VAULT AND DEPLOY</div>}
    </Panel>
  );
}

export function CoreStats({ ov, rows, nowMs }: { ov: LaunchOverview | null; rows: StrategyRow[]; nowMs: number }) {
  const n = launchNow(ov, nowMs);
  const v1 = n.prog?.engine === "v1";
  const since = n.st ? Date.parse(n.st.utc) : null;
  const fresh = since ? rows.filter((r) => Date.parse(r.added_utc) >= since) : [];
  const bestRet = fresh.length ? Math.max(...fresh.map((r) => r.total_return ?? -Infinity)) : null;
  const d = new Date(nowMs);
  const tiles: [string, string, string?][] = [
    ["TASKS", n.st ? `${n.done} / ${n.total}` : "— / —"],
    ["SURVIVORS", n.prog ? fmt.int(n.prog.alive) : "—"],
    [v1 ? "TOP FITNESS" : "BEST SHARPE", bestOf(n.prog)],
    ["TESTED", n.prog ? `${fmt.int(n.prog.trials)} / ${fmt.int(n.prog.budget)}` : "—"],
    ["VAULTED", n.st ? fmt.int(n.st.vaulted) : "—", n.st && n.st.vaulted > 0 ? "pos" : undefined],
    ["ELAPSED", clock(n.elapsedS)],
    ["ETA REMAINING", n.etaS == null ? "--:--:--" : clock(n.etaS), "acc"],
    ["ACTIVE TASK", n.cur ? `${n.cur.symbol} ${n.cur.timeframe}` : "—"],
    ["TOTAL BRED", fmt.int(rows.length)],
    ["BEST RETURN", bestRet == null || !Number.isFinite(bestRet) ? "—" : fmt.pct(bestRet), bestRet != null && bestRet >= 0 ? "pos" : "neg"],
    ["PHASE", n.phase],
    ["SYSTEM", `${d.toLocaleTimeString("en-GB", { hour12: false })}`],
  ];
  return (
    <div className="core-stats">
      {tiles.map(([k, v, cls]) => <div key={k} className="cs"><span>{k}</span><b className={cls}>{v}</b></div>)}
    </div>
  );
}

export function Command({ ov, rows, dep, symbols, form, setForm, points, onOpen, onTab }: {
  ov: LaunchOverview | null;
  rows: StrategyRow[];
  dep: DeployOverview | null;
  symbols: SymbolsOverview | null;
  form: CampaignForm;
  setForm: (f: CampaignForm) => void;
  points: GenPoint[];
  onOpen: (id: string) => void;
  onTab: (t: "SETTINGS" | "VAULT" | "PORTFOLIOS" | "CAMPAIGN") => void;
}) {
  const nowMs = useNow();
  const n = launchNow(ov, nowMs);
  const engineTag = n.cur?.engine === "v1" || (!n.busy && form.engine === "v1") ? "V1" : "V2";
  const gen = n.prog ? `${n.prog.generation + 1}` : "--";
  const gens = n.prog ? `${n.prog.generations}` : "--";
  return (
    <div className="cmd">
      <div className="cmd-l">
        <Targets symbols={symbols} form={form} setForm={setForm} busy={n.busy} ov={ov} onMore={() => onTab("SETTINGS")} />
        <VaultSummary rows={rows} onOpen={onOpen} onAll={() => onTab("VAULT")} />
        <GenomeSynthesis rows={rows} onOpen={onOpen} />
      </div>
      <section className="dk-panel cmd-c" aria-label="Neural core">
        <div className="dk-ph"><span className="dk-pt">Neural <em>core</em></span><span className="tag-eng">ENGINE {engineTag}</span><span className="sp" />
          <span className="dk-ps">{n.phase}</span></div>
        <div className="core-stage">
          <NeuralCore fraction={n.fraction} alive={n.prog?.alive ?? 0} running={n.busy && !n.paused} paused={n.paused} />
          <div className="core-mid">
            <div className="core-gen"><b>{gen}</b><span>/{gens}</span></div>
            <div className="core-lbl">GENERATION</div>
            <div className="core-sub">{n.busy ? `${n.cur ? `${n.cur.symbol}_${n.cur.timeframe}` : n.st?.id ?? ""}` : "AWAITING CAMPAIGN"}</div>
            {n.busy && n.fraction != null && <div className="core-pct">{Math.round(n.fraction * 100)}% of the budget tested</div>}
          </div>
        </div>
        <CoreStats ov={ov} rows={rows} nowMs={nowMs} />
      </section>
      <div className="cmd-r">
        <Deployed dep={dep} rows={rows} onOpen={onOpen} onAll={() => onTab("PORTFOLIOS")} />
        <Generations points={points} ov={ov} />
        <LogTerminal title="Live engine feed" compact />
      </div>
    </div>
  );
}
