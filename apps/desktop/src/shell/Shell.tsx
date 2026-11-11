// The app's opening and closing (ARQVIO_V4.md section 12.2): a boot screen that prints the start-up as a log -- the
// desktop shell, the engine process, then every check the engine ran (/api/v2/app/boot), grouped by phase -- and the
// close dialog when a campaign is still running. Outside the desktop shell (tests, a browser) the Tauri calls are
// skipped and the boot screen only waits for the engine.

import { createContext, type ReactNode, useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError, type BootCheck } from "../api/ws";
import { ENGINE_HTTP_ORIGIN } from "../config/engine";
import { BrandMark } from "../ui/BrandMark";
import { Layer } from "../ws/Dialogs";

const IN_TAURI = typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
const BOOT_GIVE_UP_S = 90;

type ShellBoot = { engine: string; detail: string; ctrl_shift: boolean };

/** Ctrl+Shift was held while ARQVIO started (the desktop shell read it; false outside it). */
export const CtrlShiftAtStart = createContext(false);

async function tauriInvoke<T>(cmd: string, args?: Record<string, unknown>): Promise<T | null> {
  if (!IN_TAURI) return null;
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<T>(cmd, args);
}

type BootLine = { key: string; t: number; group: string; check: string; ok: boolean | null; detail: string; kind: "sys" | "check" | "done" };

const GROUP_LABEL: Record<string, string> = { shell: "SHELL", engine: "ENGINE", environment: "ENVIRON", library: "LIBRARY", session: "SESSION" };
const PHASE: Record<string, string> = {
  shell: "STARTING THE ENGINE", engine: "BOOTING THE ENGINE", environment: "SETTING UP THE ENVIRONMENT",
  library: "LOADING THE LIBRARY", session: "RESTORING THE SESSION",
};
// The checks arrive together in one answer; they are printed one after another, in the order the engine ran them,
// so the start-up reads as the sequence it was. Only the pace is staged -- every line's text is the engine's.
const REVEAL_MS = 90;

function stamp(t: number): string {
  return `+${(t / 1000).toFixed(2).padStart(6, " ")}s`;
}

function BootScreen({ checks, shell, waited, failed, onRetry, started }: {
  checks: BootCheck[] | null;
  shell: ShellBoot | null;
  waited: number;
  failed: string | null;
  onRetry: () => void;
  started: number;
}) {
  const [shown, setShown] = useState(0);
  const [arrived, setArrived] = useState<number | null>(null);
  useEffect(() => {
    setShown(0);
    if (!checks) return undefined;
    setArrived(performance.now() - started);
    let n = 0;
    const t = setInterval(() => {
      n += 1;
      setShown(n);
      if (n >= checks.length) clearInterval(t);
    }, REVEAL_MS);
    return () => clearInterval(t);
  }, [checks, started]);

  const lines: BootLine[] = [
    { key: "ui", t: 0, group: "shell", check: "interface", ok: null, detail: IN_TAURI ? "desktop window ready" : "browser preview (no desktop shell)", kind: "sys" },
  ];
  if (shell?.detail) lines.push({ key: "sh", t: 0, group: "shell", check: "engine process", ok: null, detail: shell.detail, kind: "sys" });
  if (!checks) {
    lines.push({ key: "wait", t: waited * 1000, group: "engine", check: "engine", ok: failed ? false : null,
      detail: failed ?? `waiting for the engine to answer on ${ENGINE_HTTP_ORIGIN.replace("http://", "")} (${waited} s)`, kind: "check" });
  } else {
    const base = arrived ?? 0;
    let at = base;
    checks.slice(0, shown).forEach((c, i) => {
      at += c.ms;
      lines.push({ key: `c${i}`, t: at + i * REVEAL_MS, group: c.group ?? "engine", check: c.check, ok: c.ok, detail: c.detail, kind: "check" });
    });
    if (shown >= checks.length) {
      const warn = checks.filter((c) => !c.ok).length;
      lines.push({ key: "done", t: at + checks.length * REVEAL_MS, group: "session", check: "ready", ok: null, kind: "done",
        detail: warn ? `ready with ${warn} warning${warn > 1 ? "s" : ""} -- opening ARQVIO` : "all systems ready -- opening ARQVIO" });
    }
  }
  const total = checks ? checks.length : 1;
  const frac = failed ? 1 : checks ? 0.25 + 0.75 * (shown / total) : Math.min(0.22, (waited / 30) * 0.22);
  const last = lines[lines.length - 1];
  const phase = failed ? "START-UP FAILED" : last.kind === "done" ? "READY" : PHASE[last.group] ?? "BOOTING";
  let prevGroup = "";
  return (
    <div className="boot" role="status" aria-label="ARQVIO is starting">
      <div className="boot-grid" aria-hidden="true" />
      <div className="boot-card">
        <header className="boot-head">
          <div className="boot-mark"><BrandMark /></div>
          <div>
            <div className="boot-title">ARQVIO</div>
            <div className="boot-sub">STRATEGY BREEDING ENGINE <span className="dim">// SYSTEM BOOT</span></div>
          </div>
          <div className={`boot-phase ${failed ? "bad" : last.kind === "done" ? "ok" : ""}`}>{phase}</div>
        </header>
        <ul className="boot-lines">
          {lines.map((l) => {
            const head = l.group !== prevGroup;
            prevGroup = l.group;
            return [
              head && <li key={`h-${l.key}`} className="boot-group" aria-hidden="true">{`── ${GROUP_LABEL[l.group] ?? l.group.toUpperCase()}`}</li>,
              <li key={l.key} className={l.kind === "check" ? (l.ok === false ? "bad" : l.ok ? "ok" : "wait") : l.kind}>
                <span className="boot-t">{stamp(l.t)}</span>
                <span className="boot-k">{l.check}</span>
                <span className="boot-v">{l.detail}</span>
                <span className="boot-s" aria-hidden="true">{l.kind !== "check" ? "" : l.ok === false ? "FAIL" : l.ok ? " OK " : " .. "}</span>
              </li>,
            ];
          })}
          {!failed && last.kind !== "done" && <li className="boot-cursor" aria-hidden="true"><span className="boot-t" /><span className="caret" /></li>}
        </ul>
        <div className="boot-foot">
          <div className="boot-bar"><span style={{ transform: `scaleX(${frac})` }} /></div>
          <span className="boot-pct">{Math.round(frac * 100)}%</span>
        </div>
        {failed && (
          <div className="row" style={{ gap: 8, marginTop: 12, alignItems: "center" }}>
            <button className="btn btn-primary" onClick={onRetry}>Retry</button>
            <span className="hint">The engine's own log: ARQVIO/logs/engine.stderr.log in your user folder.</span>
          </div>
        )}
      </div>
    </div>
  );
}

function CloseDialog({ label, onStop, onKeep, onCancel, busy }: {
  label: string;
  onStop: () => void;
  onKeep: () => void;
  onCancel: () => void;
  busy: boolean;
}) {
  const keep = useRef<HTMLButtonElement>(null);
  useEffect(() => keep.current?.focus(), []);
  return (
    <Layer>
      <div className="scrim" role="dialog" aria-modal="true" aria-label="A campaign is still running">
        <section className="panel dialog" style={{ maxWidth: 560 }}>
          <div className="panel-h">ARQVIO — campaign running</div>
          <div className="panel-b" style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            <div>A campaign is still running ({label}).</div>
            <div className="hint">Keep it running and close the window: reopen ARQVIO any time and it reattaches to this campaign.
              Stop it now: the campaign ends and the engine shuts down.</div>
            <div className="row" style={{ gap: 8, justifyContent: "flex-end" }}>
              <button className="btn" onClick={onCancel} disabled={busy}>Cancel</button>
              <button className="btn" onClick={onStop} disabled={busy}>Stop it now</button>
              <button ref={keep} className="btn btn-primary" onClick={onKeep} disabled={busy}>Keep running in background</button>
            </div>
          </div>
        </section>
      </div>
    </Layer>
  );
}

export function Shell({ children }: { children: ReactNode }) {
  const [checks, setChecks] = useState<BootCheck[] | null>(null);
  const [shell, setShell] = useState<ShellBoot | null>(null);
  const [waited, setWaited] = useState(0);
  const [failed, setFailed] = useState<string | null>(null);
  const [ready, setReady] = useState(false);
  const [closing, setClosing] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const attempt = useRef(0);
  const [bootStart, setBootStart] = useState(() => performance.now());

  const boot = useCallback(() => {
    const my = ++attempt.current;
    setFailed(null);
    setChecks(null);
    setBootStart(performance.now());
    const started = Date.now();
    const tick = async () => {
      if (my !== attempt.current) return;
      const s = await tauriInvoke<ShellBoot>("boot_info").catch(() => null);
      if (s) setShell(s);
      if (s?.engine === "failed") return setFailed(s.detail);
      try {
        const b = await api.boot();
        setChecks(b.checks);
        // the lines print one by one (REVEAL_MS each), then the last line stays readable for a moment
        setTimeout(() => my === attempt.current && setReady(true), b.checks.length * REVEAL_MS + 900);
      } catch (e) {
        const secs = Math.round((Date.now() - started) / 1000);
        setWaited(secs);
        if (secs >= BOOT_GIVE_UP_S) return setFailed(e instanceof ApiError ? e.detail : `the engine did not answer in ${BOOT_GIVE_UP_S} s`);
        setTimeout(tick, 1000);
      }
    };
    void tick();
  }, []);

  useEffect(boot, [boot]);

  useEffect(() => {
    if (!IN_TAURI) return;
    let off: (() => void) | undefined;
    void import("@tauri-apps/api/event").then(({ listen }) =>
      listen("close-requested", async () => {
        let label: string | null = null;
        try {
          const st = await api.launchState();
          if (st.busy) label = st.launch?.id ?? "campaign";
        } catch {
          label = null; // no engine to ask: nothing of ours is running
        }
        if (label) setClosing(label);
        else await tauriInvoke("app_exit", { stop: true });
      }).then((u) => { off = u; }));
    return () => off?.();
  }, []);

  const exit = async (stop: boolean) => {
    setBusy(true);
    if (stop) await api.stop().catch(() => undefined);
    await tauriInvoke("app_exit", { stop });
  };

  return (
    <>
      {ready ? <CtrlShiftAtStart.Provider value={!!shell?.ctrl_shift}>{children}</CtrlShiftAtStart.Provider> : <BootScreen checks={checks} shell={shell} waited={waited} failed={failed} onRetry={boot} started={bootStart} />}
      {closing && <CloseDialog label={closing} busy={busy} onCancel={() => setClosing(null)} onStop={() => void exit(true)}
        onKeep={() => void exit(false)} />}
    </>
  );
}
