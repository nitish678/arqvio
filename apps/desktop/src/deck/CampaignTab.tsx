// CAMPAIGN: the running (or last) launch in detail. Left: its progress and the live feed. Centre: the generations and
// every task of the launch with its state. Right: the strategies it has added so far. Bottom: the run's clock.
// idea: research §7.3 (CAMPAIGN console: active campaign, generations, live feed, generated strategies).

import type { LaunchOverview, StrategyRow } from "../api/ws";
import { CoreRing } from "../ws/Canvas";
import { LogTerminal } from "../ws/LeftRail";
import { FAMILY_SHORT, fmt, shortId } from "../ws/util";
import { GenChart } from "./Command";
import { GenerationFan } from "./GenerationFan";
import { bestOf, clock, type GenPoint, launchNow, useLiveView, useNow } from "./launchInfo";
import { TradeTape } from "./TradeTape";

export function Bar({ k, v, frac }: { k: string; v: string; frac: number | null }) {
  return (
    <div className="pbar">
      <div className="pbar-k"><span>{k}</span><b>{v}</b></div>
      <div className="pbar-t"><i style={{ transform: `scaleX(${Math.max(0, Math.min(1, frac ?? 0))})` }} /></div>
    </div>
  );
}

const STATE_CLS: Record<string, string> = { DONE: "pos", RUNNING: "acc", CHECKING: "acc", PAUSED: "warn", FAILED: "neg",
  STOPPED: "warn", REFUSED: "neg", WAITING: "dim" };

export function CampaignTab({ ov, rows, points, onOpen }: {
  ov: LaunchOverview | null;
  rows: StrategyRow[];
  points: GenPoint[];
  onOpen: (id: string) => void;
}) {
  const nowMs = useNow();
  const n = launchNow(ov, nowMs);
  const st = n.st;
  const since = st ? Date.parse(st.utc) : null;
  const added = since ? rows.filter((r) => Date.parse(r.added_utc) >= since).sort((a, b) => b.score - a.score) : [];
  const v1 = n.prog?.engine === "v1";
  const tf = n.total ? n.done / n.total : null;
  const view = useLiveView(n.busy);
  const lv = view?.live ?? null;
  return (
    <div className="cmp">
      <div className="cmp-l">
        <section className="dk-panel" aria-label="Active campaign">
          <div className="dk-ph"><span className="dk-pt">Active campaign</span><span className="sp" /><span className="dk-ps">{n.phase}</span></div>
          <div className="dk-pb">
            {st?.detail && <p role="status" className={`dk-note ${["BLOCKED_DATA", "FAILED", "PARTIAL_DATA"].includes(st.state) ? "bad" : ""}`}>
              {st.state} · {st.detail}
            </p>}
            <div className="cmp-ring">
              <CoreRing fraction={n.fraction} generation={n.prog?.generation ?? null} generations={n.prog?.generations ?? null}
                running={n.busy && !n.paused} />
              <div className="cmp-ring-kv">
                <div><span>GENERATION</span><b>{n.prog ? `${n.prog.generation + 1} / ${n.prog.generations}` : "-- / --"}</b></div>
                <div><span>PHASE</span><b>{n.phase}</b></div>
                <div><span>NOW</span><b title={n.what}>{n.what}</b></div>
              </div>
            </div>
            <Bar k="CAMPAIGN" v={tf == null ? "0%" : `${Math.round(tf * 100)}%`} frac={tf} />
            <Bar k="MARKET" v={n.prog ? `${fmt.int(n.prog.trials)} / ${fmt.int(n.prog.budget)}` : "—"} frac={n.fraction} />
            <Bar k="GENERATION" v={n.prog ? `${n.prog.generation + 1} / ${n.prog.generations}` : "—"}
              frac={n.prog ? (n.prog.generation + 1) / Math.max(1, n.prog.generations) : null} />
            <div className="kv-line"><span>{v1 ? "TOP FITNESS" : "BEST SHARPE"}</span><b>{bestOf(n.prog)}</b></div>
            <div className="kv-line"><span>TASKS DONE</span><b>{st ? `${n.done} / ${n.total}` : "0 / 0"}</b></div>
            <div className="kv-line"><span>TARGETS</span><span className="chips">
              {st ? st.request.symbols.flatMap((s) => st.request.timeframes.map((t) => <span key={s + t} className="chip ro">{s} · {t}</span>))
                : <span className="dim">no launch yet</span>}</span></div>
          </div>
        </section>
        <LogTerminal title="Live engine feed" compact />
      </div>

      <div className="cmp-c">
        <section className="dk-panel grow" aria-label="Campaign generations">
          <div className="dk-ph"><span className="dk-pt">Campaign generations</span>
            <span className="dk-ps">{lv ? `GEN ${lv.generation} · ${lv.survivors} survivors · ${lv.phase}` : v1 ? "top fitness (line) · alive (bars)" : "best Sharpe (line) · alive (bars)"}</span></div>
          <div className="dk-pb fan-pb">
            {lv && lv.bots.length ? <GenerationFan view={view} />
              : points.length >= 2 ? <GenChart points={points} height={220} />
                : <div className="dk-empty tall">{n.busy ? <><span className="spin" />{n.phase} · VALIDATING</> : "NO CAMPAIGN RUNNING · PRESS LAUNCH"}</div>}
            <details className="tasks-fold">
              <summary>TASKS {st ? `${n.done} / ${n.total}` : ""}</summary>
            <table className="dk-t" style={{ marginTop: 8 }}>
              <thead><tr><th className="l">TASK</th><th className="l">MARKET</th><th className="l">STATE</th><th className="l">DETAIL</th><th>ADDED</th></tr></thead>
              <tbody>
                {(st?.tasks ?? []).map((t, i) => (
                  <tr key={i} className="ro">
                    <td className="l">{t.kind === "data" ? "DATA" : t.role === "cross" ? "CROSS" : t.engine === "v1" ? "BREED V1" : "BREED V2"}</td>
                    <td className="l"><b>{t.symbol} {t.timeframe}</b></td>
                    <td className={`l ${STATE_CLS[t.state] ?? ""}`}>{t.state}</td>
                    <td className="l dim" title={t.detail ?? ""}>{t.detail ?? (t.progress ? `G${t.progress.generation + 1}/${t.progress.generations} · ${fmt.int(t.progress.trials)} tested · ${t.progress.alive} alive` : "")}</td>
                    <td>{t.added ?? "—"}</td>
                  </tr>
                ))}
                {!st && <tr className="ro"><td className="l dim" colSpan={5}>The tasks of the next launch appear here.</td></tr>}
              </tbody>
            </table>
            </details>
          </div>
        </section>
        <div className="cmp-insight">
          {[
            ["ASSETS", st ? `${st.request.symbols.length}` : "—"],
            ["RUNS", st ? `${st.request.symbols.length * st.request.timeframes.length}` : "—"],
            ["SURVIVORS", n.prog ? `${n.prog.alive}` : "—"],
            ["VAULTED", st ? `${st.vaulted}` : "—"],
            ["DEPTH", st ? st.request.depth : "—"],
            ["POWER", st ? st.request.power.toUpperCase() : "—"],
          ].map(([k, v]) => <div key={k} className="cs"><span>{k}</span><b>{v}</b></div>)}
        </div>
      </div>

      <div className="cmp-r">
      <TradeTape view={view} running={n.busy && !n.paused} />
      <section className="dk-panel gen-list" aria-label="Generated strategies">
        <div className="dk-ph"><span className="dk-pt">Generated strategies</span><span className="dk-ps">{added.length ? `${added.length} this launch` : ""}</span></div>
        <div className="dk-pb scroll">
          {added.length ? (
            <table className="dk-t">
              <thead><tr><th className="l">STRATEGY</th><th>SCORE</th><th>RET</th><th>DD</th></tr></thead>
              <tbody>
                {added.map((r) => (
                  <tr key={r.id} onClick={() => onOpen(r.id)}>
                    <td className="l"><b>{r.symbol} {r.timeframe}</b><small>{FAMILY_SHORT[r.family] ?? r.family} · #{shortId(r.id)}</small></td>
                    <td className="acc">S {r.score.toFixed(0)}</td>
                    <td className={(r.total_return ?? 0) >= 0 ? "pos" : "neg"}>{fmt.pct(r.total_return)}</td>
                    <td className="neg">{fmt.pct(r.max_dd == null ? null : -Math.abs(r.max_dd))}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : <div className="dk-empty">No strategies generated yet</div>}
        </div>
      </section>
      </div>

      <div className="cmp-foot">
        {[
          ["ELAPSED", clock(n.elapsedS), ""],
          ["ETA REMAINING", n.etaS == null ? "--:--:--" : clock(n.etaS), "acc"],
          ["ACTIVE TASK", n.cur ? `${n.cur.symbol} [${n.cur.timeframe}]` : "—", ""],
          [v1 ? "TOP FITNESS" : "BEST SHARPE", bestOf(n.prog), ""],
          ["PHASE", n.phase, ""],
        ].map(([k, v, c]) => <div key={k} className="dk-panel stat"><span>{k}</span><b className={c}>{v}</b></div>)}
      </div>
    </div>
  );
}
