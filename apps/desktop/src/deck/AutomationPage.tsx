// AUTOMATION: named automations that run a chain of steps on a schedule. NOW (the run in progress), the list (one
// row each: ON/OFF, name, interval, steps, status, RUN NOW, settings, delete), the editor under a row, and the
// ACTIVITY log by day. The old launch queue stays on the right. Every state is the engine's.
// idea: research automation page (RF 45578-47957); layout and copy are ARQVIO's.

import { useCallback, useEffect, useMemo, useState } from "react";
import { api, type AfterStep, type AutomationDraft, type AutomationList, type AutomationRec, type AutomationState,
  type AutoStep, type StepKind } from "../api/ws";
import { ConfirmDialog } from "../ws/Dialogs";
import { errText, fmt } from "../ws/util";
import { Automation as LaunchQueue } from "./Automation";
import { KIND_HINT, KIND_TEXT, StepCard } from "./AutoSteps";

const UNITS = [["h", "HOURS", 1], ["d", "DAYS", 24], ["w", "WEEKS", 168]] as const;

function unitOf(hours: number): { n: number; u: "h" | "d" | "w" } {
  if (hours > 0 && hours % 168 === 0) return { n: hours / 168, u: "w" };
  if (hours > 24 && hours % 24 === 0) return { n: hours / 24, u: "d" };
  return { n: hours || 24, u: "h" };
}

function Rings({ steps, at, progress }: { steps: AutoStep[]; at: number | null; progress?: { done: number; total: number } | null }) {
  return (
    <span className="au-rings">
      {steps.map((s, i) => {
        const frac = i === at && progress ? progress.done / Math.max(1, progress.total) : 0;
        const st = at == null ? "" : i < at ? "done" : i === at ? "now" : "";
        return (
          <span key={i} className={`au-ring ${st}`} title={`${KIND_TEXT[s.kind]} · ${KIND_HINT[s.kind]}`}
            style={st === "now" ? { background: `conic-gradient(var(--brand-red) ${frac * 360}deg, transparent 0)` } : undefined}>
            <b>{KIND_TEXT[s.kind][0]}</b>
          </span>
        );
      })}
    </span>
  );
}

function statusOf(a: AutomationRec, st: AutomationState): [string, string, string] {
  const r = st.running;
  if (r && r.automation_id === a.id) return [r.phase === "INTERRUPTED" ? "INTERRUPTED" : `STEP ${r.step_idx + 1}`, `OF ${r.steps_total}`, "acc"];
  if (!a.enabled) return ["OFF", "", "dim"];
  if (a.run_now) return ["QUEUED", "RUN NOW", "warn"];
  if (a.history[0] && !a.history[0].ok && !a.history[0].paused) return ["LAST RUN", "FAILED", "neg"];
  if (a.next_at) {
    const t = Date.parse(a.next_at);
    return [t <= Date.now() ? "DUE" : fmt.time(a.next_at), t <= Date.now() ? "WAITS ITS TURN" : "NEXT RUN", ""];
  }
  return ["—", "", "dim"];
}

function Editor({ draft, setDraft, meta, onClose, onSaved, onDelete }: {
  draft: AutomationDraft;
  setDraft: (d: AutomationDraft) => void;
  meta: AutomationList;
  onClose: () => void;
  onSaved: (note: string) => void;
  onDelete: (() => void) | null;
}) {
  const [errs, setErrs] = useState<string[]>([]);
  const [lines, setLines] = useState<string[] | null>(null);
  const [busy, setBusy] = useState(false);
  const { n, u } = unitOf(draft.every_hours);
  const nonstop = draft.every_hours === 0;
  const hasPf = draft.steps.some((s) => s.kind === "portfolio");
  const setStep = (i: number, s: AutoStep) => setDraft({ ...draft, steps: draft.steps.map((x, k) => (k === i ? s : x)) });
  const move = (i: number, d: number) => {
    const j = i + d;
    if (j < 0 || j >= draft.steps.length) return;
    const st = [...draft.steps];
    [st[i], st[j]] = [st[j], st[i]];
    setDraft({ ...draft, steps: st });
  };
  const add = (k: StepKind) => setDraft({ ...draft, steps: [...draft.steps, { ...meta.defaults[k] }] });
  const save = async () => {
    setBusy(true);
    try {
      const r = await api.automationSave(draft);
      setErrs([]);
      onSaved(r.note);
    } catch (e) {
      setErrs(errText(e).split("; "));
    } finally {
      setBusy(false);
    }
  };
  const preview = async () => {
    setBusy(true);
    try {
      const r = await api.automationPreview(draft);
      setLines([...r.lines, r.note]);
    } catch (e) {
      setLines([errText(e)]);
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="au-ed">
      <div className="au-ed-h">
        <span className="dk-pt">{draft.id ? "Settings" : "New automation"}</span>
        <input className="as-text au-name" placeholder="NAME IT" value={draft.name} onChange={(e) => setDraft({ ...draft, name: e.target.value })} />
        <span className="k">RUNS</span>
        <span className="chips">
          <button type="button" className="chip" aria-pressed={nonstop} title="Starts again as soon as the last run finishes"
            onClick={() => setDraft({ ...draft, every_hours: 0 })}>NON-STOP</button>
          <button type="button" className="chip" aria-pressed={!nonstop} title="Runs once every so many hours, days or weeks"
            onClick={() => setDraft({ ...draft, every_hours: nonstop ? 24 : draft.every_hours })}>EVERY</button>
        </span>
        {!nonstop && <>
          <input className="as-num" type="number" min={1} value={n}
            onChange={(e) => setDraft({ ...draft, every_hours: Math.max(0, Number(e.target.value || 0)) * UNITS.find((x) => x[0] === u)![2] })} />
          <span className="chips">{UNITS.map(([k, l, mult]) => <button key={k} type="button" className="chip" aria-pressed={u === k}
            onClick={() => setDraft({ ...draft, every_hours: n * mult })}>{l}</button>)}</span>
        </>}
        {meta.terminals.length > 1 && (
          <select className="select" value={draft.terminal} onChange={(e) => setDraft({ ...draft, terminal: e.target.value })} title="TERMINAL: the MetaTrader 5 its campaigns read data from">
            <option value="AUTO">TERMINAL: AUTO</option>
            {meta.terminals.map((t) => <option key={t.id} value={t.id}>{t.server ?? t.path}</option>)}
          </select>
        )}
        <span className="sp" />
        {onDelete ? <button type="button" className="rb rb-sm" onClick={onDelete}>DELETE</button>
          : <button type="button" className="rb rb-sm" onClick={onClose}>DISCARD</button>}
        <button type="button" className="rb rb-sm" disabled={busy} onClick={() => void preview()} title="Shows what each step would do right now. Nothing starts.">PREVIEW</button>
        <button type="button" className="rb rb-sm" onClick={onClose}>CLOSE</button>
        <button type="button" className="rb rb-sm rb-main" disabled={busy} onClick={() => void save()}>SAVE</button>
      </div>
      {errs.length > 0 && <ul className="au-errs">{errs.map((e) => <li key={e}>{e}</li>)}</ul>}
      {lines && <div className="au-preview"><b>PREVIEW · WHAT THIS AUTOMATION WOULD DO RIGHT NOW</b>{lines.map((l, i) => <div key={i}>{l}</div>)}</div>}
      <div className="au-track"><Rings steps={draft.steps} at={null} /></div>
      {draft.steps.map((s, i) => (
        <section key={i} className="as-card">
          <div className="as-h"><b>STEP {i + 1} OF {draft.steps.length} · {KIND_TEXT[s.kind]}</b><small>{KIND_HINT[s.kind]}</small><span className="sp" />
            <button type="button" className="lnk" onClick={() => move(i, -1)} disabled={i === 0} aria-label="Move up">◀</button>
            <button type="button" className="lnk" onClick={() => move(i, 1)} disabled={i === draft.steps.length - 1} aria-label="Move down">▶</button>
            <button type="button" className="lnk" onClick={() => setDraft({ ...draft, steps: draft.steps.filter((_, k) => k !== i) })}>REMOVE STEP</button></div>
          <StepCard step={s} set={(x) => setStep(i, x)} meta={meta} portfolios={meta.portfolios} hasPortfolioStep={hasPf} />
        </section>
      ))}
      <div className="au-add"><span className="k">ADD A STEP</span>
        {(["campaign", "portfolio", "parity", "purge", "replace", "reassess"] as StepKind[]).map((k) => {
          const built = meta.built.includes(k);
          const dup = k === "parity" && draft.steps.some((s) => s.kind === "parity");
          return <button key={k} type="button" className="chip" disabled={!built || dup}
            title={built ? KIND_HINT[k] : `not built yet: ${meta.planned_why[k]}`} onClick={() => add(k)}>+ {KIND_TEXT[k]}</button>;
        })}
      </div>
    </div>
  );
}

function Activity({ rows }: { rows: AutomationList["log"] }) {
  const groups = useMemo(() => {
    const out: { day: string; rows: AutomationList["log"] }[] = [];
    const today = new Date().toDateString();
    const yest = new Date(Date.now() - 86_400_000).toDateString();
    for (const r of rows) {
      const d = new Date(r.ts);
      const label = d.toDateString() === today ? "TODAY" : d.toDateString() === yest ? "YESTERDAY"
        : d.toLocaleDateString("en-GB", { weekday: "short", day: "numeric", month: "short" }).toUpperCase();
      if (!out.length || out[out.length - 1].day !== label) out.push({ day: label, rows: [] });
      out[out.length - 1].rows.push(r);
    }
    return out;
  }, [rows]);
  const copy = () => void navigator.clipboard?.writeText(rows.map((r) =>
    `${r.ts.slice(0, 16).replace("T", " ")}  ${r.name}  ${r.status === "ok" ? "OK" : r.status.toUpperCase()}  ${r.detail}`).join("\n"));
  return (
    <section className="dk-panel au-act" aria-label="Activity">
      <div className="dk-ph"><span className="dk-pt">Activity</span><span className="sp" />
        <button type="button" className="lnk" disabled={!rows.length} onClick={copy}>COPY</button></div>
      <div className="dk-pb scroll">
        {!rows.length && <div className="dk-empty">WHAT AUTOMATIONS DO SHOWS UP HERE</div>}
        {groups.map((g) => (
          <div key={g.day}>
            <div className="au-day">{g.day}</div>
            {g.rows.map((r, i) => (
              <div key={i} className="au-ev">
                <span className={`au-dot ${r.status}`} /><b>{r.name}</b><span className="dim">{fmt.time(r.ts).slice(0, 5)}</span>
                <div className={r.status === "failed" ? "neg" : "dim"}>{r.status === "failed" ? "FAILED · " : ""}{r.detail}</div>
              </div>
            ))}
          </div>
        ))}
      </div>
    </section>
  );
}

export function AutomationPage({ canQueue, queueWhy, onQueueCurrent }: {
  canQueue: boolean;
  queueWhy: string | null;
  onQueueCurrent: (after: AfterStep[]) => Promise<void>;
}) {
  const [meta, setMeta] = useState<AutomationList | null>(null);
  const [st, setSt] = useState<AutomationState>({});
  const [err, setErr] = useState<string | null>(null);
  const [open, setOpen] = useState<string | null>(null);  // automation id, or "new"
  const [draft, setDraft] = useState<AutomationDraft | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  const [del, setDel] = useState<AutomationRec | null>(null);
  const load = useCallback(async () => {
    try {
      setMeta(await api.automations());
      setErr(null);
    } catch (e) {
      setErr(errText(e));
    }
  }, []);
  useEffect(() => {
    void load();
    const a = setInterval(() => void load(), 20_000);
    const tick = () => api.automationState().then(setSt).catch(() => undefined);
    void tick();
    const b = setInterval(tick, 5000);
    return () => {
      clearInterval(a);
      clearInterval(b);
    };
  }, [load]);
  const act = async (fn: () => Promise<{ toast?: string } | unknown>, ok?: string) => {
    try {
      const r = await fn() as { toast?: string };
      setToast(r?.toast ?? ok ?? null);
      void load();
      void api.automationState().then(setSt);
    } catch (e) {
      setToast(errText(e));
    }
  };
  const edit = (a: AutomationRec | null) => {
    if (!meta) return;
    if (a) {
      setDraft({ id: a.id, name: a.name, enabled: a.enabled, every_hours: a.every_hours, steps: a.next_steps ?? a.steps, terminal: a.terminal });
      setOpen(a.id);
    } else {
      setDraft({ name: "", enabled: true, every_hours: 24, steps: [{ ...meta.defaults.campaign }, { ...meta.defaults.purge }], terminal: "AUTO" });
      setOpen("new");
    }
  };
  const close = () => {
    setOpen(null);
    setDraft(null);
  };
  const r = st.running;
  const runRec = r ? meta?.automations.find((a) => a.id === r.automation_id) : undefined;
  const list = meta?.automations ?? [];
  const editor = (id: string) => open === id && draft && meta ? (
    <Editor draft={draft} setDraft={setDraft} meta={meta} onClose={close}
      onSaved={(note) => { setToast(note); close(); void load(); }}
      onDelete={id === "new" ? null : () => { const a = list.find((x) => x.id === id); if (a) setDel(a); }} />
  ) : null;

  return (
    <div className="au">
      <div className="au-main">
        {(r || st.pending) && (
          <section className="dk-panel au-now" aria-label="Now">
            <div className="dk-ph"><span className="dk-pt">{r ? (r.phase === "INTERRUPTED" ? "Interrupted" : "Running") : "Next up"}</span>
              <span className="dk-ps">{r?.name ?? st.pending?.name}</span><span className="sp" />
              {r && <button type="button" className="rb rb-sm" title="Stop this run. A campaign this automation started stops too. Your own campaigns keep running."
                onClick={() => void act(() => api.automationCancel(r.automation_id), "STOPPING · THE CURRENT STEP ENDS FIRST")}>■ STOP</button>}</div>
            <div className="dk-pb au-now-b">
              {r && runRec && <Rings steps={runRec.steps} at={r.step_idx} progress={r.progress} />}
              {r && <div className="au-now-kv"><span>STEP {r.step_idx + 1} / {r.steps_total}</span><b>{r.phase}</b></div>}
              <div className="dk-note">{r?.phase === "INTERRUPTED" ? `ARQVIO restarted, carries on in ${Math.floor((r.resume_in ?? 0) / 60)}:${String((r.resume_in ?? 0) % 60).padStart(2, "0")}`
                : st.pending ? st.pending.why : st.last_result && !st.last_result.ok ? `LAST RUN: ${st.last_result.detail}` : ""}</div>
            </div>
          </section>
        )}
        <section className="dk-panel au-list" aria-label="Automations">
          <div className="dk-ph"><span className="dk-pt">Automations</span><span className="dk-ps">{list.length} automation{list.length === 1 ? "" : "s"}</span>
            <span className="sp" />{toast && <span className="au-toast">{toast}</span>}</div>
          <div className="dk-pb">
            {err && <div className="dk-note bad">{err}</div>}
            {editor("new")}
            {!list.length && open !== "new" && <div className="dk-empty">NO AUTOMATIONS YET – A SCHEDULE THAT RUNS ARQVIO FOR YOU</div>}
            {list.map((a) => {
              const [s1, s2, cls] = statusOf(a, st);
              return (
                <div key={a.id}>
                  <div className={`au-row${open === a.id ? " open" : ""}`}>
                    <button type="button" className="gb-tog" aria-pressed={a.enabled} aria-label={`${a.enabled ? "Switch off" : "Switch on"} ${a.name}`}
                      onClick={() => void act(() => api.automationToggle(a.id, !a.enabled))}>{a.enabled ? "ON" : "OFF"}</button>
                    <div className="au-name-c"><b>{a.name.toUpperCase()}</b>
                      <small>{a.every_text}{a.alongside ? " · RUNS ALONGSIDE" : ""}{a.next_steps ? " · NEW STEPS NEXT RUN" : ""}</small></div>
                    <Rings steps={a.steps} at={r?.automation_id === a.id ? r.step_idx : null} progress={r?.progress} />
                    <span className="au-hist">{a.history.slice(0, 14).reverse().map((h, i) => <i key={i} className={h.ok ? "ok" : h.paused ? "paused" : "failed"} title={`${fmt.time(h.ts)} · ${h.detail}`} />)}</span>
                    <div className={`au-st ${cls}`}><b>{s1}</b><small>{s2}</small></div>
                    <button type="button" className="rb rb-sm" disabled={!a.enabled || r?.automation_id === a.id} onClick={() => void act(() => api.automationRun(a.id))} title="Run now">▶</button>
                    <button type="button" className="rb rb-sm" onClick={() => (open === a.id ? close() : edit(a))} title="Settings">⚙</button>
                  </div>
                  {editor(a.id)}
                </div>
              );
            })}
            {open !== "new" && <button type="button" className="rb au-new" disabled={!meta} onClick={() => edit(null)}>+ NEW AUTOMATION</button>}
          </div>
        </section>
      </div>
      <div className="au-side">
        <Activity rows={meta?.log ?? []} />
        <LaunchQueue canQueue={canQueue} queueWhy={queueWhy} onQueueCurrent={onQueueCurrent} />
      </div>
      {del && (
        <ConfirmDialog title={`Delete automation ${del.name}?`} confirm="DELETE AUTOMATION"
          body="Its schedule and history go with it. Strategies and portfolios it created are not touched."
          onCancel={() => setDel(null)}
          onConfirm={() => { const a = del; setDel(null); close(); void act(() => api.automationDelete(a.id), "AUTOMATION DELETED"); }} />
      )}
    </div>
  );
}
