// Deploy (which MetaTrader 5 account, what lot, which guardrails) and Settings (theme, motion, data folder).

import { type ReactNode, useState } from "react";
import { createPortal } from "react-dom";
import { api, type StrategyRow, type TerminalsOverview } from "../api/ws";
import { errText, familyName, fmt, shortId } from "./util";

const GUARDS: { key: string; label: string; hint: string; step: number; min: number; max: number }[] = [
  { key: "lots", label: "Lot per trade", hint: "before the incubation scale (starts at 25%)", step: 0.01, min: 0.01, max: 100 },
  { key: "daily_loss_pct", label: "Daily loss limit %", hint: "0 = off; the bot flattens and waits for tomorrow", step: 0.5, min: 0, max: 100 },
  { key: "max_account_positions", label: "Max ARQVIO positions", hint: "on the whole account, all bots", step: 1, min: 1, max: 100 },
  { key: "friday_close_hour", label: "Friday close hour", hint: "server time; -1 = off", step: 1, min: -1, max: 23 },
  { key: "news_block_minutes", label: "News pause (minutes)", hint: "around high-impact events; 0 = off", step: 5, min: 0, max: 720 },
  { key: "profit_target_pct", label: "Profit target %", hint: "0 = off; stops after the target", step: 0.5, min: 0, max: 1000 },
];

/** Dialogs render into document.body: a panel's clip-path would clip even a fixed-position child. */
export function Layer({ children }: { children: ReactNode }) {
  return createPortal(children, document.body);
}

export function DeployDialog({ row, terms, defaultTerm, defaults, onClose, onDone }: {
  row: StrategyRow;
  terms: TerminalsOverview | null;
  defaultTerm: string | null;
  defaults: Record<string, number | boolean> | null;
  onClose: () => void;
  onDone: (msg: string) => void;
}) {
  const running = (terms?.terminals ?? []).filter((t) => t.running && t.account.state === "OK");
  const [termId, setTermId] = useState<string>(running.find((t) => t.id === defaultTerm)?.id ?? running[0]?.id ?? "");
  const [g, setG] = useState<Record<string, number>>(() => {
    const out: Record<string, number> = {};
    for (const f of GUARDS) out[f.key] = Number(defaults?.[f.key] ?? (f.key === "lots" ? 0.01 : 0));
    return out;
  });
  const [attach, setAttach] = useState(true);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const term = running.find((t) => t.id === termId) ?? null;
  const acc = term?.account;
  const bad = GUARDS.find((f) => !Number.isFinite(g[f.key]) || g[f.key] < f.min || g[f.key] > f.max);

  const go = async () => {
    if (!term) return;
    setBusy(true);
    setErr(null);
    try {
      const r = await api.go({ vault_id: row.id, terminal: term.id, guardrails: g, attach });
      onDone(attach ? `${r.bot} is starting in MetaTrader 5 (${r.symbol} ${row.timeframe}).`
        : `${r.bot} installed: drag it from Navigator › Expert Advisors › ARQVIO onto a ${r.symbol} ${row.timeframe} chart.`);
    } catch (e) {
      setErr(errText(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Layer>
      <div className="scrim" role="dialog" aria-modal="true" aria-label="Deploy to MetaTrader 5" onMouseDown={(e) => e.target === e.currentTarget && !busy && onClose()}>
        <section className="panel dialog">
          <div className="panel-h">Deploy to MetaTrader 5 <small>{shortId(row.id)} · {row.symbol} {row.timeframe} · {familyName(row.family)}</small></div>
          <div className="panel-b" style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            <div className="topo-row" aria-label="Deploy path">
              <div className="topo-box"><span className="label">Strategy</span><b>{shortId(row.id)} · grade {row.grade}</b><small>{familyName(row.family)}</small></div>
              <span className="topo-arrow" aria-hidden="true" />
              <div className="topo-box"><span className="label">Expert Advisor</span><b>compiled by MetaEditor</b><small>0 errors, 0 warnings required</small></div>
              <span className="topo-arrow" aria-hidden="true" />
              <div className="topo-box"><span className="label">Terminal</span><b title={term?.path}>{term ? term.install_dir : "—"}</b><small>{acc?.company ?? "choose below"}</small></div>
              <span className="topo-arrow" aria-hidden="true" />
              <div className="topo-box"><span className="label">Account</span><b>{acc?.server ?? "—"}</b><small>{acc ? `${acc.mode} · ${fmt.money(acc.balance, acc.currency)}` : ""}</small></div>
              <span className="topo-arrow" aria-hidden="true" />
              <div className="topo-box"><span className="label">Chart</span><b>{row.symbol} {row.timeframe}</b><small>broker name mapped</small></div>
            </div>

            <div className="field">
              <span className="label">Account (a running MetaTrader 5)</span>
              {running.length ? (
                <select className="select" value={termId} onChange={(e) => setTermId(e.target.value)} disabled={busy} aria-label="Terminal to deploy to">
                  {running.map((t) => (
                    <option key={t.id} value={t.id}>{t.account.company} · {t.account.server} · {t.account.mode} · {t.account.login} · {fmt.money(t.account.balance, t.account.currency)} ({t.install_dir})</option>
                  ))}
                </select>
              ) : (
                <div className="note">No MetaTrader 5 is running and logged in. Open the terminal you want the bot in, log in, then re-check
                  in the campaign panel.</div>
              )}
            </div>

            <div className="cols">
              {GUARDS.map((f) => (
                <label key={f.key} className="field">
                  <span className="label">{f.label}</span>
                  <input className="input" type="number" step={f.step} min={f.min} max={f.max} value={g[f.key]} disabled={busy}
                    onChange={(e) => setG({ ...g, [f.key]: Number(e.target.value) })} />
                  <span className="hint" style={{ fontSize: 11 }}>{f.hint}</span>
                </label>
              ))}
            </div>

            <label className="row" style={{ gap: 8 }}>
              <input type="checkbox" checked={attach} onChange={(e) => setAttach(e.target.checked)} disabled={busy} />
              <span>Open it on its chart now <span className="dim">(MetaTrader 5 restarts with algo trading on; it keeps running when ARQVIO is closed)</span></span>
            </label>
            <div className="hint">The EA trades the account it is attached to. Back-test it in MetaTrader 5's Strategy Tester first if you want:
              it runs there on any chart.</div>
            {bad && <div className="err">{bad.label}: between {bad.min} and {bad.max}</div>}
            {err && <div className="err">{err}</div>}
          </div>
          <div className="dialog-f">
            <button type="button" className="btn" onClick={onClose} disabled={busy}>Cancel</button>
            <button type="button" className="btn btn-primary" onClick={() => void go()} disabled={busy || !term || !!bad}>
              {busy ? "Deploying…" : attach ? "Deploy and start" : "Install EA only"}
            </button>
          </div>
        </section>
      </div>
    </Layer>
  );
}

export interface Prefs {
  theme: "dark" | "light";
  fx: "FULL" | "LITE" | "OFF";
  density?: "compact" | "comfort";  // research's COMPACT / COMFORTABLE (tokens.css data-density)
  hints?: "show" | "hide";          // research's "?" hints SHOW / HIDE: the grey explanation lines
}

export function SettingsForm({ prefs, setPrefs, home }: { prefs: Prefs; setPrefs: (p: Prefs) => void; home: string | null }) {
  return (
    <>
      <div className="field">
        <span className="label">Theme</span>
        <div className="seg" role="radiogroup" aria-label="Theme">
          {(["dark", "light"] as const).map((t) => (
            <button key={t} type="button" aria-pressed={prefs.theme === t} onClick={() => setPrefs({ ...prefs, theme: t })}>{t}</button>
          ))}
        </div>
      </div>
      <div className="field">
        <span className="label">Motion</span>
        <div className="seg" role="radiogroup" aria-label="Motion">
          {(["FULL", "LITE", "OFF"] as const).map((t) => (
            <button key={t} type="button" aria-pressed={prefs.fx === t} onClick={() => setPrefs({ ...prefs, fx: t })}>{t.toLowerCase()}</button>
          ))}
        </div>
        <span className="hint">Lite or off for remote desktop and slow GPUs.</span>
      </div>
      <div className="field">
        <span className="label">Density</span>
        <div className="seg" role="radiogroup" aria-label="Density">
          {(["compact", "comfort"] as const).map((t) => (
            <button key={t} type="button" aria-pressed={(prefs.density ?? "compact") === t} onClick={() => setPrefs({ ...prefs, density: t })}>{t}</button>
          ))}
        </div>
        <span className="hint">Comfort: taller rows and more padding.</span>
      </div>
      <div className="field">
        <span className="label">Explanations</span>
        <div className="seg" role="radiogroup" aria-label="Explanations">
          {(["show", "hide"] as const).map((t) => (
            <button key={t} type="button" aria-pressed={(prefs.hints ?? "show") === t} onClick={() => setPrefs({ ...prefs, hints: t })}>{t}</button>
          ))}
        </div>
      </div>
      <div className="field">
        <span className="label">Data folder</span>
        <span className="mono" style={{ fontSize: 12 }}>{home ?? "—"}</span>
        <span className="hint">Market data, campaigns, the vault and the log live here. Delete it to start empty.</span>
      </div>
      <div className="note">ARQVIO sends nothing to the internet: it talks only to its own engine on this computer and to your
        MetaTrader 5. Orders are placed only by the Expert Advisors you deploy.</div>
    </>
  );
}

export function SettingsDialog({ prefs, setPrefs, home, onClose }: {
  prefs: Prefs; setPrefs: (p: Prefs) => void; home: string | null; onClose: () => void;
}) {
  return (
    <Layer>
      <div className="scrim" role="dialog" aria-modal="true" aria-label="Settings" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
        <section className="panel dialog" style={{ width: 520 }}>
          <div className="panel-h">Settings</div>
          <div className="panel-b" style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            <SettingsForm prefs={prefs} setPrefs={setPrefs} home={home} />
          </div>
          <div className="dialog-f"><button type="button" className="btn btn-primary" onClick={onClose}>Done</button></div>
        </section>
      </div>
    </Layer>
  );
}

export function ConfirmDialog({ title, body, confirm, onConfirm, onCancel }: {
  title: string; body: string; confirm: string; onConfirm: () => void; onCancel: () => void;
}) {
  return (
    <Layer>
      <div className="scrim" role="alertdialog" aria-modal="true" aria-label={title}>
        <section className="panel dialog" style={{ width: 460 }}>
          <div className="panel-h">{title}</div>
          <div className="panel-b"><p style={{ margin: 0 }}>{body}</p></div>
          <div className="dialog-f">
            <button type="button" className="btn" onClick={onCancel}>Cancel</button>
            <button type="button" className="btn btn-primary" onClick={onConfirm}>{confirm}</button>
          </div>
        </section>
      </div>
    </Layer>
  );
}
