// Small shared pieces of the workstation: polling, formatting, plain-language names, local preferences.

import { useCallback, useEffect, useRef, useState } from "react";

export type Load<T> = { state: "loading" } | { state: "ok"; data: T } | { state: "error"; error: string };

/** Read ``fn`` now and every ``everyMs`` (0 = once); ``reload`` reads again at once. Keeps the last good answer
 *  visible while a later read fails (the error is still reported). */
export function usePoll<T>(fn: () => Promise<T>, everyMs: number, deps: unknown[] = []) {
  const [load, setLoad] = useState<Load<T>>({ state: "loading" });
  const fnRef = useRef(fn);
  fnRef.current = fn;
  const alive = useRef(true);
  const reload = useCallback(async () => {
    try {
      const data = await fnRef.current();
      if (alive.current) setLoad({ state: "ok", data });
    } catch (e) {
      if (alive.current) setLoad((prev) => (prev.state === "ok" ? prev : { state: "error", error: errText(e) }));
    }
  }, []);
  useEffect(() => {
    alive.current = true;
    void reload();
    if (!everyMs) return () => { alive.current = false; };
    const t = setInterval(() => void reload(), everyMs);
    return () => {
      alive.current = false;
      clearInterval(t);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [everyMs, reload, ...deps]);
  return { load, reload, data: load.state === "ok" ? load.data : null };
}

export function errText(e: unknown): string {
  if (e instanceof Error) return e.message || e.name;
  return String(e);
}

export const fmt = {
  pct: (v: number | null | undefined, digits = 1) => (v == null || !Number.isFinite(v) ? "—" : `${(v * 100).toFixed(digits)}%`),
  num: (v: number | null | undefined, digits = 2) => (v == null || !Number.isFinite(v) ? "—" : v.toFixed(digits)),
  int: (v: number | null | undefined) => (v == null || !Number.isFinite(v) ? "—" : Math.round(v).toLocaleString("en-US")),
  money: (v: number | null | undefined, ccy = "") =>
    v == null || !Number.isFinite(v) ? "—" : `${v.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}${ccy ? ` ${ccy}` : ""}`,
  signed: (v: number | null | undefined, digits = 2) =>
    v == null || !Number.isFinite(v) ? "—" : `${v > 0 ? "+" : ""}${v.toFixed(digits)}`,
  time: (utc: string) => {
    const d = new Date(utc);
    return Number.isNaN(d.getTime()) ? utc : d.toLocaleTimeString("en-GB", { hour12: false });
  },
  hours: (h: number | null | undefined) => (h == null ? "—" : h >= 48 ? `${(h / 24).toFixed(1)} d` : `${h.toFixed(1)} h`),
  duration: (s: number) => {
    if (!Number.isFinite(s) || s < 0) return "—";
    if (s < 90) return `${Math.round(s)} s`;
    if (s < 5400) return `${Math.round(s / 60)} min`;
    return `${(s / 3600).toFixed(1)} h`;
  },
};

/** The families as a trader would call them (the engine's names stay in the tooltip). */
export const FAMILY_NAMES: Record<string, string> = {
  MA_CROSS: "Moving-average cross",
  RSI_REVERSION: "RSI reversion",
  STOCH_CROSS: "Stochastic cross",
  MACD_CROSS: "MACD cross",
  DONCHIAN_BREAKOUT: "Donchian breakout",
};

export const FAMILY_SHORT: Record<string, string> = {
  MA_CROSS: "MA",
  RSI_REVERSION: "RSI",
  STOCH_CROSS: "STOCH",
  MACD_CROSS: "MACD",
  DONCHIAN_BREAKOUT: "DONCH",
};

export const familyName = (f: string) => FAMILY_NAMES[f] ?? f;

export function shortId(id: string): string {
  return id.split("@")[0].slice(0, 8).toUpperCase();
}

// ---- local preferences (per viewer; the app works without them) ---------------------------------------------------------

const PREFS = "arqvio.ws.v3";

export function readPrefs<T extends object>(fallback: T): T {
  try {
    const raw = localStorage.getItem(PREFS);
    return raw ? { ...fallback, ...(JSON.parse(raw) as Partial<T>) } : fallback;
  } catch {
    return fallback;
  }
}

export function writePrefs(p: object): void {
  try {
    localStorage.setItem(PREFS, JSON.stringify(p));
  } catch {
    /* storage unavailable: preferences simply do not persist */
  }
}

/** Device-pixel-ratio aware canvas sizing: draw at the backing-store resolution so lines stay sharp. */
export function fitCanvas(c: HTMLCanvasElement): { ctx: CanvasRenderingContext2D; w: number; h: number } | null {
  const ctx = c.getContext("2d");
  if (!ctx) return null;
  const r = c.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 3);
  const w = Math.max(1, Math.round(r.width));
  const h = Math.max(1, Math.round(r.height));
  if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
    c.width = Math.round(w * dpr);
    c.height = Math.round(h * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}

export function cssVar(name: string, fallback: string): string {
  if (typeof window === "undefined" || !document.documentElement) return fallback;
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

export function motionOn(): boolean {
  const fx = document.documentElement.dataset.fx;
  if (fx === "OFF" || fx === "LITE") return false;
  return !(typeof window.matchMedia === "function" && window.matchMedia("(prefers-reduced-motion: reduce)").matches);
}
