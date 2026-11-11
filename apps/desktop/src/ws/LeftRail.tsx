// The left rail: LAUNCH / PAUSE / STOP, the campaign ring with its numbers, and the one log (every engine step, in
// order, no source picker).

import { useEffect, useRef, useState } from "react";
import { api, type LaunchOverview, type LogLine } from "../api/ws";
import { CoreRing } from "./Canvas";
import { errText, fmt } from "./util";

const MAX_LINES = 2000;

export function LogTerminal({ title = "Log", compact = false }: { title?: string; compact?: boolean } = {}) {
  const [lines, setLines] = useState<LogLine[]>([]);
  const [error, setError] = useState<string | null>(null);
  const last = useRef(0);
  const box = useRef<HTMLDivElement | null>(null);
  const stick = useRef(true);

  useEffect(() => {
    let alive = true;
    let inFlight = false;
    const tick = async () => {
      if (inFlight) return;  // a slow answer is never asked for twice: the same lines would land twice
      inFlight = true;
      try {
        const r = await api.log(last.current);
        if (!alive) return;
        setError(null);
        const fresh = r.lines.filter((l) => l.seq > last.current);
        if (fresh.length) {
          last.current = fresh[fresh.length - 1].seq;
          setLines((prev) => [...prev, ...fresh].slice(-MAX_LINES));
        }
      } catch (e) {
        if (alive) setError(errText(e));
      } finally {
        inFlight = false;
      }
    };
    void tick();
    const t = setInterval(() => void tick(), 1500);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, []);

  useEffect(() => {
    const el = box.current;
    if (el && stick.current) el.scrollTop = el.scrollHeight;
  }, [lines]);

  const onScroll = () => {
    const el = box.current;
    if (el) stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 24;
  };

  const copy = () => {
    const text = lines.map((l) => `${l.utc} ${l.level} ${l.src} ${l.msg}`).join("\n");
    void navigator.clipboard?.writeText(text);
  };

  return (
    <section className={compact ? "dk-panel grow feed" : "panel grow"} aria-label={title}>
      <div className={compact ? "dk-ph" : "panel-h"}>
        {compact ? <span className="dk-pt">{title}</span> : title} <small>{lines.length ? `${lines.length} lines` : ""}</small>
        <span className="sp" />
        <button type="button" className="btn btn-sm" onClick={copy} disabled={!lines.length}>Copy</button>
      </div>
      <div className="term" ref={box} onScroll={onScroll} role="log" aria-live="polite">
        {lines.length === 0 && !error && (
          <div className="dim">ARQVIO is ready. Pick markets and timeframes, then press LAUNCH. Every step the engine
            takes appears here.</div>
        )}
        {lines.map((l) => (
          <div key={l.seq} className={`term-line ${l.level}`}>
            <span className="t">{fmt.time(l.utc)}</span>
            <span className="m"><span className="s">{l.src}</span> {l.msg}</span>
          </div>
        ))}
        {error && <div className="term-line ERROR"><span className="t">--:--</span><span className="m">log unreadable: {error}</span></div>}
        <span className="term-cursor" aria-hidden="true" />
      </div>
    </section>
  );
}

export function LeftRail({ ov, canLaunch, launchWhy, onLaunch, onPause, onResume, onStop, onSkip, acting }: {
  ov: LaunchOverview | null;
  canLaunch: boolean;
  launchWhy: string | null;
  onLaunch: () => void;
  onPause: () => void;
  onResume: () => void;
  onStop: () => void;
  onSkip?: () => void;
  acting: boolean;
}) {
  const st = ov?.launch ?? null;
  const busy = !!ov?.busy;
  const cur = busy ? st?.tasks.find((t) => ["RUNNING", "PAUSED", "CHECKING"].includes(t.state)) ?? st?.tasks.find((t) =>
    t.kind === "data" && t.state === "RUNNING") : undefined;
  const prog = cur?.progress;
  const fraction = busy && prog ? prog.trials / Math.max(1, prog.budget) : busy ? 0 : null;
  let eta: string | null = null;
  if (busy && prog && cur?.started_utc && prog.trials > 0) {
    const elapsed = (Date.parse(prog.utc) - Date.parse(cur.started_utc)) / 1000;
    const left = elapsed / prog.trials * Math.max(0, prog.budget - prog.trials);
    if (Number.isFinite(left)) eta = `~${fmt.duration(left)}`;  // an estimate from the pace so far
  }
  const breedLeft = busy ? st?.tasks.filter((t) => t.kind === "breed" && t.state === "WAITING").length ?? 0 : 0;
  const what = !busy ? (st ? `${st.state.toLowerCase()} · ${st.vaulted} added` : "idle")
    : cur ? `${cur.kind === "data" ? "data" : cur.state === "CHECKING" ? "check" : cur.state === "PAUSED" ? "paused" : "breeding"} · ${cur.symbol} ${cur.timeframe}`
      : "starting";

  return (
    <aside className="ws-left" aria-label="Controls and log">
      <section className="panel">
        <div className="controls">
          {!busy ? (
            <button type="button" className="btn btn-primary btn-big" onClick={onLaunch} disabled={!canLaunch || acting}
              title={launchWhy ?? "Read the data, then breed and test strategies on every chosen market"}>
              Launch
            </button>
          ) : (
            <>
              {ov?.paused ? (
                <button type="button" className="btn btn-primary btn-big" onClick={onResume} disabled={acting}>Resume</button>
              ) : (
                <button type="button" className="btn btn-big" onClick={onPause} disabled={acting}
                  title="Stop the running campaign at its checkpoint; resume later">Pause</button>
              )}
            </>
          )}
          {busy && cur?.engine === "v1" && onSkip && (
            <button type="button" className="btn" onClick={onSkip} disabled={acting}
              title="Engine V1: finish this generation, then go straight to the Final War and the vault gates">Skip</button>
          )}
          <button type="button" className="btn" onClick={onStop} disabled={!busy || acting}
            title="End this launch: the running campaign stops at its checkpoint, the rest is skipped">Stop</button>
          <span className="hint" style={{ alignSelf: "center", fontSize: 11 }}>{busy ? "running" : launchWhy ?? "ready"}</span>
        </div>
      </section>
      <section className="panel" aria-label="Status">
        <div className="panel-h">Status</div>
        <div className="core">
          <CoreRing fraction={fraction} generation={prog?.generation ?? null} generations={prog?.generations ?? null}
            running={busy && !ov?.paused} />
          <div className="core-kv">
            <div>now <b title={what}>{what}</b></div>
            <div>tested <b>{prog ? `${fmt.int(prog.trials)} / ${fmt.int(prog.budget)}` : "—"}</b></div>
            {prog?.engine === "v1"
              ? <div>top fitness <b>{fmt.num(prog.top_fitness ?? 0)}</b></div>
              : <div>best Sharpe <b>{prog && prog.best_sharpe !== null ? fmt.num(prog.best_sharpe) : "—"}</b></div>}
            <div>alive <b>{prog ? fmt.int(prog.alive) : "—"}</b></div>
            <div>queue <b>{busy ? `${breedLeft} more` : "—"}</b></div>
            <div title="estimated from the pace so far">time left <b>{eta ?? "—"}</b></div>
          </div>
        </div>
      </section>
      <LogTerminal />
    </aside>
  );
}
