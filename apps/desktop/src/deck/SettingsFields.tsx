// SETTINGS fields the engine applies (engine/core/app_settings.py), one panel each: TRAINING (split + OOS trades),
// ENGINE V1 (blank = the depth's preset), CONFIG ECHO, PURGE CRITERIA, AUTOMATIC PURGE, OPEN AT LAUNCH, the bug report,
// and the right rail's RUN STATUS and QUEUE. Chips save on click, numbers on leaving the field (or Enter).
// idea: research SETTINGS tabs (RF 54202-54673); only fields ARQVIO's engine reads are offered.

import { createContext, useContext, useEffect, useState } from "react";
import { api, type AppSettings, type ConfigEcho, type LaunchOverview, type QueueItem } from "../api/ws";
import { readChoice, writeChoice } from "../shell/Chooser";
import { errText, fmt } from "../ws/util";
import { ConfirmDialog } from "../ws/Dialogs";
import { Bar } from "./CampaignTab";
import { bestOf, clock, launchNow, useNow } from "./launchInfo";

export { Version as SettingsVersion };

type Patch = (p: Record<string, unknown>) => Promise<void>;

// After every save attempt (Version) the field shows what the engine holds: a blank that saved a default, or a
// value the engine refused, never stays on screen as if it were stored.
const Version = createContext(0);

function Num({ label, value, onSave, step, hint, placeholder, min, max }: {
  label: string; value: number | null | undefined; onSave: (v: number | null) => void; step?: number; hint?: string;
  placeholder?: string; min?: number; max?: number;
}) {
  const version = useContext(Version);
  const [text, setText] = useState(value == null ? "" : String(value));
  useEffect(() => setText(value == null ? "" : String(value)), [value, version]);
  const commit = () => {
    const v = text.trim() === "" ? null : Number(text);
    if (v !== value && (v === null || Number.isFinite(v))) onSave(v);
  };
  return (
    <label className="af" title={hint}>
      <span>{label}</span>
      <input type="number" step={step ?? 1} min={min} max={max} value={text} placeholder={placeholder ?? "—"} onChange={(e) => setText(e.target.value)}
        onBlur={commit} onKeyDown={(e) => e.key === "Enter" && (e.target as HTMLInputElement).blur()} />
    </label>
  );
}

function Panel({ title, sub, children }: { title: string; sub?: string; children: React.ReactNode }) {
  return (
    <section className="dk-panel" aria-label={title}>
      <div className="dk-ph"><span className="dk-pt">{title}</span>{sub && <span className="dk-ps">{sub}</span>}</div>
      <div className="dk-pb sf-body">{children}</div>
    </section>
  );
}

export function TrainingPanel({ s, splits, patch }: { s: AppSettings; splits: number[]; patch: Patch }) {
  return (
    <Panel title="Training" sub="how much of the window breeding sees">
      <div className="as-row"><span className="k">TRAINING SPLIT</span><span className="chips">
        {splits.map((x) => (
          <button key={x} type="button" className="chip" aria-pressed={s.campaign.train_split === x}
            onClick={() => void patch({ campaign: { train_split: x } })}>{Math.round(x * 100)}/{Math.round((1 - x) * 100)}{x === 0.7 ? " (RECOMMENDED)" : ""}</button>
        ))}</span></div>
      <div className="as-grid">
        <Num label="OOS TRADES (MIN)" value={s.campaign.held_out_min_trades} min={5} max={200} placeholder="15" hint="fewer trades on the held-out bars judge nothing: the strategy is refused"
          onSave={(v) => void patch({ campaign: { held_out_min_trades: v ?? 15 } })} />
      </div>
      <div className="dk-note">Breeding never sees the held-out share; every strategy is tested on it after breeding. Applies to the next launch.</div>
    </Panel>
  );
}

// [key, label, step, hint, min, max] -- the ranges are engine/core/app_settings.py RANGES
const V1_FIELDS: [keyof AppSettings["v1"], string, number, string, number, number][] = [
  ["proving_grounds", "PG CANDIDATES", 1, "strategies tested before the tribes form", 100, 20000],
  ["generations", "TRIBE GENS", 1, "generations per tribe", 1, 100],
  ["tribe_size", "TRIBE SIZE", 1, "survivors kept in each tribe", 4, 200],
  ["stagnation_after", "STAGNATION", 1, "generations without a better best before a revolution", 1, 50],
  ["drawdown_limit", "DD LIMIT", 0.01, "a strategy above this drawdown is culled (0.30 = 30 %)", 0.05, 0.9],
  ["risk_pct", "RISK", 0.001, "risk per trade in the backtest (0.01 = 1 %)", 0.001, 0.05],
];

export function V1Panel({ s, patch }: { s: AppSettings; patch: Patch }) {
  return (
    <Panel title="Engine V1 parameters" sub="blank = the depth's preset">
      <div className="as-grid">
        {V1_FIELDS.map(([k, label, step, hint, min, max]) => (
          <Num key={k} label={label} value={s.v1[k]} step={step} hint={hint} min={min} max={max} placeholder="PRESET"
            onSave={(v) => void patch({ v1: { [k]: v } })} />
        ))}
      </div>
      <button type="button" className="rb rb-sm" onClick={() => void patch({ v1: Object.fromEntries(V1_FIELDS.map(([k]) => [k, null])) })}>BACK TO THE PRESETS</button>
    </Panel>
  );
}

export function EchoPanel({ depth, engine, version }: { depth: string; engine: string; version: number }) {
  const [e, setE] = useState<ConfigEcho | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    api.configEcho(depth, engine).then((x) => { setE(x); setErr(null); }).catch((x) => setErr(errText(x)));
  }, [depth, engine, version]);
  return (
    <Panel title="Config echo" sub="what the next launch resolves to">
      {err && <div className="dk-note bad">{err}</div>}
      {e?.error && <div className="dk-note bad">{e.error}</div>}
      {e && <pre className="sf-echo">{e.line}</pre>}
    </Panel>
  );
}

const PURGE: [string, string, number][] = [["min_score", "MIN SCORE", 1], ["min_trades", "MIN TRADES", 1], ["max_dd", "MAX DD %", 0.5],
  ["min_return", "MIN RETURN %", 1], ["min_oos_return", "MIN OOS RETURN %", 1], ["max_age_days", "MAX AGE (DAYS)", 1]];

export function PurgePanel({ s, patch }: { s: AppSettings; patch: Patch }) {
  const c = s.purge.criteria;
  return (
    <Panel title="Purge criteria" sub="used by PURGE and by automations whose PURGE step is blank">
      <div className="as-grid">
        {PURGE.map(([k, label, step]) => (
          <Num key={k} label={label} value={c[k] ?? null} step={step} onSave={(v) => {
            const next: Record<string, number> = { ...c };
            if (v == null) delete next[k];
            else next[k] = v;
            void patch({ purge: { criteria: next } });
          }} />
        ))}
      </div>
      <div className="dk-note">All blank = built-in criteria (30 trades, 35 % drawdown, no loss). Portfolio members and live bots are never purged.</div>
    </Panel>
  );
}

export function AutoPurgePanel({ s, patch }: { s: AppSettings; patch: Patch }) {
  const a = s.auto_purge;
  return (
    <Panel title="Automatic purge" sub="a scheduled vault sweep">
      <div className="as-row"><span className="k">AUTO PURGE</span><span className="chips">
        <button type="button" className="chip" aria-pressed={a.enabled} onClick={() => void patch({ auto_purge: { enabled: true } })}>ON</button>
        <button type="button" className="chip" aria-pressed={!a.enabled} onClick={() => void patch({ auto_purge: { enabled: false } })}>OFF</button>
      </span></div>
      <div className="as-grid">
        <Num label="RUN EVERY (HOURS)" value={a.every_hours} min={1} max={720} placeholder="24" onSave={(v) => void patch({ auto_purge: { every_hours: v ?? 24 } })} />
        <Num label="ONLY IF OLDER THAN (DAYS)" value={a.older_than_days} min={0} max={3650} placeholder="45" onSave={(v) => void patch({ auto_purge: { older_than_days: v ?? 45 } })} />
        <Num label="VAULT CAP" value={a.cap} min={100} max={100000} placeholder="3000" onSave={(v) => void patch({ auto_purge: { cap: v ?? 3000 } })}
          hint="above this many strategies, the weakest older than 7 days go to the recycle bin" />
      </div>
      <div className="dk-note">Uses the purge criteria above. Paused while a campaign runs; the first sweep waits one interval after switching on. Restorable for 30 days.</div>
    </Panel>
  );
}

export function InterfacePanel() {
  const [c, setC] = useState(readChoice);
  const pick = (v: "ask" | "deck" | "classic") => {
    const next = v === "ask" ? { layout: c?.layout ?? "deck", always: false } : { layout: v, always: true };
    writeChoice(next);
    setC(next);
  };
  const now = !c || !c.always ? "ask" : c.layout;
  return (
    <Panel title="Interface" sub="hold Ctrl+Shift while ARQVIO starts to choose again">
      <div className="as-row"><span className="k">OPEN AT LAUNCH</span><span className="chips">
        {([["ask", "ASK ME"], ["deck", "DECK"], ["classic", "CLASSIC"]] as const).map(([v, l]) => (
          <button key={v} type="button" className="chip" aria-pressed={now === v} onClick={() => pick(v)}>{l}</button>
        ))}</span></div>
      <div className="dk-note">Privacy: ARQVIO sends nothing over the internet. It talks only to its own engine on this computer and to your MetaTrader 5.</div>
    </Panel>
  );
}

export function BugReportPanel() {
  const [what, setWhat] = useState("");
  const [contact, setContact] = useState("");
  const [note, setNote] = useState<{ text: string; bad?: boolean } | null>(null);
  const save = async () => {
    let r: { path: string; text: string };
    try {
      r = await api.bugReport(what, contact);
    } catch (e) {
      setNote({ text: errText(e), bad: true });
      return;
    }
    setWhat("");
    try {
      await navigator.clipboard?.writeText(r.text);
      setNote({ text: `Saved to ${r.path} and copied: paste it to whoever helps you.` });
    } catch {
      setNote({ text: `Saved to ${r.path}. Copying failed: open that file to send it.` });
    }
  };
  const copyAll = async () => {
    try {
      const [rep, log] = await Promise.all([api.supportReport(), api.log(0)]);
      const tail = log.lines.slice(-400).map((l) => `${l.utc} ${l.level} ${l.src} ${l.msg}`).join("\n");
      await navigator.clipboard?.writeText(`${rep.text}\n\nENGINE LOG (last ${Math.min(400, log.lines.length)} lines)\n${tail}`);
      setNote({ text: "Support report and the engine log copied." });
    } catch (e) {
      setNote({ text: errText(e), bad: true });
    }
  };
  return (
    <Panel title="Bug report" sub="saved on this computer; ARQVIO sends nothing">
      <label className="af"><span>WHAT HAPPENED?</span>
        <textarea className="as-text sf-area" rows={5} value={what} onChange={(e) => setWhat(e.target.value)} placeholder="What you did, what you expected, what happened" /></label>
      <label className="af"><span>CONTACT (DISCORD / EMAIL, OPTIONAL)</span>
        <input className="as-text" value={contact} onChange={(e) => setContact(e.target.value)} /></label>
      <div className="as-row">
        <button type="button" className="rb rb-sm rb-main" disabled={what.trim().length < 3} onClick={() => void save()}>SAVE BUG REPORT</button>
        <button type="button" className="rb rb-sm" onClick={() => void copyAll()}>COPY ALL</button>
      </div>
      {note && <div className={`dk-note ${note.bad ? "bad" : ""}`}>{note.text}</div>}
    </Panel>
  );
}

export function RunStatus({ ov }: { ov: LaunchOverview | null }) {
  const nowMs = useNow();
  const n = launchNow(ov, nowMs);
  const v1 = n.prog?.engine === "v1";
  const tf = n.total ? n.done / n.total : null;
  return (
    <section className="dk-panel" aria-label="Run status">
      <div className="dk-ph"><span className="dk-pt">Run status</span></div>
      <div className="dk-pb kv-list">
        <div><span>STATE</span><b className={n.busy ? "pos" : ""}>{n.busy ? (n.paused ? "PAUSED" : "RUNNING") : n.st ? n.st.state : "IDLE"}</b></div>
        <div><span>CAMPAIGN</span><b className="acc">{n.st?.id ?? "—"}</b></div>
        <div><span>TASK</span><b>{n.cur ? `${n.cur.symbol} [${n.cur.timeframe}]` : "—"}</b></div>
        <div><span>PHASE</span><b>{n.phase}</b></div>
        <div><span>GENERATION</span><b>{n.prog ? `${n.prog.generation + 1} / ${n.prog.generations}` : "—"}</b></div>
        <div><span>SURVIVORS · VAULTED</span><b>{n.prog ? n.prog.alive : "—"} · {n.st?.vaulted ?? "—"}</b></div>
        <div><span>{v1 ? "TOP FITNESS" : "BEST SHARPE"}</span><b>{bestOf(n.prog)}</b></div>
        <div><span>ELAPSED / ETA</span><b>{clock(n.elapsedS)} / {n.etaS == null ? "--:--:--" : clock(n.etaS)}</b></div>
      </div>
      <div className="dk-pb">
        <Bar k="CAMPAIGN" v={tf == null ? "—" : `${Math.round(tf * 100)}%`} frac={tf} />
        <Bar k="MARKET" v={n.fraction == null ? "—" : `${Math.round(n.fraction * 100)}%`} frac={n.fraction} />
        <Bar k="GENERATION" v={n.prog ? `${n.prog.generation + 1} / ${n.prog.generations}` : "—"}
          frac={n.prog ? (n.prog.generation + 1) / Math.max(1, n.prog.generations) : null} />
        <div className="dk-note">Run controls (LAUNCH, STOP, SKIP, PAUSE) are in the header of every tab.</div>
      </div>
    </section>
  );
}

export function QueuePanel() {
  const [items, setItems] = useState<QueueItem[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [ask, setAsk] = useState(false);
  const load = () => api.queue().then((q) => { setItems(q.items); setErr(null); }).catch((e) => setErr(errText(e)));
  useEffect(() => {
    void load();
    const t = setInterval(load, 10_000);
    return () => clearInterval(t);
  }, []);
  return (
    <section className="dk-panel" aria-label="Queue">
      <div className="dk-ph"><span className="dk-pt">Queue</span><span className="dk-ps">{items.length} waiting</span><span className="sp" />
        <button type="button" className="lnk" disabled={!items.length} onClick={() => setAsk(true)}>CLEAR ALL</button></div>
      <div className="dk-pb">
        {err && <div className="dk-note bad">{err}</div>}
        {!items.length && <div className="dk-note">Queue empty. Add a set-up with + QUEUE in the header; it starts when the engine is free and survives a restart.</div>}
        {items.map((q) => <div key={q.id} className="dk-note">{fmt.time(q.utc)} · {q.request.symbols.join(", ")} · {q.request.timeframes.join(", ")} · {q.request.depth}</div>)}
      </div>
      {ask && <ConfirmDialog title={`Clear ${items.length} waiting launch${items.length === 1 ? "" : "es"}?`} confirm="CLEAR ALL"
        body="Every waiting launch leaves the queue. The running launch is not touched."
        onCancel={() => setAsk(false)} onConfirm={() => { setAsk(false); void api.clearQueue().then(load).catch((e) => setErr(errText(e))); }} />}
    </section>
  );
}
