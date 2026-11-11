// What the running launch is doing right now, read once from /api/v2/app/launch and shared by the header, COMMAND and
// CAMPAIGN: the active task, its progress, the time it has run and a time-left estimate from the pace so far.

import { useEffect, useRef, useState } from "react";
import { api, type LaunchOverview, type LaunchState, type LaunchTask, type LiveView, type Progress } from "../api/ws";
import { fmt } from "../ws/util";

export interface LaunchNow {
  st: LaunchState | null;
  busy: boolean;
  paused: boolean;
  cur: LaunchTask | undefined;
  prog: Progress | undefined;
  /** strategies tested of the task's budget, 0..1; null when nothing runs */
  fraction: number | null;
  done: number;
  total: number;
  elapsedS: number | null;
  etaS: number | null;
  /** one short line: what the engine is doing */
  what: string;
  phase: string;
}

const ACTIVE = ["RUNNING", "PAUSED", "CHECKING"];

export function launchNow(ov: LaunchOverview | null, now = Date.now()): LaunchNow {
  const st = ov?.launch ?? null;
  const busy = !!ov?.busy;
  const paused = !!ov?.paused;
  const cur = busy ? st?.tasks.find((t) => ACTIVE.includes(t.state)) ?? st?.tasks.find((t) =>
    t.kind === "data" && t.state === "RUNNING") : undefined;
  const prog = cur?.progress;
  const fraction = busy && prog ? Math.min(1, prog.trials / Math.max(1, prog.budget)) : busy ? 0 : null;
  const tasks = st?.tasks ?? [];
  const done = tasks.filter((t) => !["WAITING", ...ACTIVE].includes(t.state)).length;
  const end = st?.finished_utc ? Date.parse(st.finished_utc) : now;
  const elapsedS = st ? Math.max(0, (busy ? now : end) - Date.parse(st.utc)) / 1000 : null;
  let etaS: number | null = null;
  if (busy && prog && cur?.started_utc && prog.trials > 0) {
    const ran = (Date.parse(prog.utc) - Date.parse(cur.started_utc)) / 1000;
    const left = ran / prog.trials * Math.max(0, prog.budget - prog.trials);
    if (Number.isFinite(left)) etaS = left;  // this task only: an estimate from its pace so far
  }
  const what = !busy ? (st ? `${st.state.toLowerCase()} · ${st.vaulted} added` : "idle")
    : cur ? `${cur.kind === "data" ? "reading data" : cur.state === "CHECKING" ? "safety check" : cur.state === "PAUSED" ? "paused" : "breeding"} · ${cur.symbol} ${cur.timeframe}`
      : "starting";
  const phase = !busy ? (st ? st.state : "STANDBY")
    : paused || cur?.state === "PAUSED" ? "PAUSED"
      : !cur ? "STARTING"
        : cur.kind === "data" ? "DATA"
          : cur.state === "CHECKING" ? "SAFETY CHECK"
            // engine V1 reports progress from its first generation on; before that it runs the proving grounds
            : cur.engine === "v1" ? (prog ? "TRIBES" : "PROVING GROUNDS")
              : "BREEDING";
  return { st, busy, paused, cur, prog, fraction, done, total: tasks.length, elapsedS, etaS, what, phase };
}

export const clock = (s: number | null) => {
  if (s == null || !Number.isFinite(s)) return "--:--:--";
  const t = Math.round(s);
  return [Math.floor(t / 3600), Math.floor(t / 60) % 60, t % 60].map((n) => String(n).padStart(2, "0")).join(":");
};

export const bestOf = (p: Progress | undefined) =>
  p ? (p.engine === "v1" ? fmt.num(p.top_fitness ?? null) : fmt.num(p.best_sharpe)) : "—";

export interface GenPoint {
  generation: number;
  trials: number;
  alive: number;
  best: number | null;
}

/** The generations seen while this window was open, per launch task (the engine reports only the latest one). */
export function useGenerations(ov: LaunchOverview | null): { key: string | null; points: GenPoint[] } {
  const now = launchNow(ov);
  const key = now.st && now.cur ? `${now.st.id}/${now.cur.cell}` : null;
  const [hist, setHist] = useState<{ key: string | null; points: GenPoint[] }>({ key: null, points: [] });
  const lastKey = useRef<string | null>(null);
  const p = now.prog;
  useEffect(() => {
    if (!key || !p) return;
    const best = p.engine === "v1" ? p.top_fitness ?? null : p.best_sharpe;
    setHist((h) => {
      const base = h.key === key ? h.points : [];
      const pt = { generation: p.generation, trials: p.trials, alive: p.alive, best };
      const prev = base[base.length - 1];
      if (prev && prev.generation === pt.generation) return { key, points: [...base.slice(0, -1), pt] };
      return { key, points: [...base, pt].slice(-200) };
    });
    lastKey.current = key;
  }, [key, p?.generation, p?.trials, p?.alive, p?.best_sharpe, p?.top_fitness, p?.engine]);  // eslint-disable-line react-hooks/exhaustive-deps
  return hist.key === key ? hist : { key, points: [] };
}

/** A ticking clock for elapsed time and the SYSTEM tile (one interval, one second). */
export function useNow(ms = 1000): number {
  const [t, setT] = useState(() => Date.now());
  useEffect(() => {
    const i = setInterval(() => setT(Date.now()), ms);
    return () => clearInterval(i);
  }, [ms]);
  return t;
}

/** live.json of the breeding task (GET /launch/live): every 3 s while a campaign runs, every 20 s otherwise. */
export function useLiveView(busy: boolean): LiveView | null {
  const [view, setView] = useState<LiveView | null>(null);
  useEffect(() => {
    let alive = true;
    const tick = () => api.launchLive().then((v) => alive && setView(v)).catch(() => undefined);
    void tick();
    const t = setInterval(tick, busy ? 3000 : 20_000);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, [busy]);
  return view;
}
