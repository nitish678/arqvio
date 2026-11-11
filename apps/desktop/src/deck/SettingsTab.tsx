// SETTINGS in four parts, one at a time: CAMPAIGN (what the next launch breeds), BREEDING (the genome: every gene and
// what the breeder may draw), SYSTEM (this computer and the engine's rules), SUPPORT (diagnostics and the report).
// The run status and the engine log stay on the right. The last part opened is remembered.
// idea: research §7.3 / BP06 §7 (settings tabs CAMPAIGN · BREEDING · SYSTEM · SUPPORT); content is ARQVIO's.

import { type ReactNode, useEffect, useState } from "react";
import { api, type AppSettings, type GenePoolDoc, type LaunchOverview } from "../api/ws";
import { type Prefs, SettingsForm } from "../ws/Dialogs";
import { LogTerminal } from "../ws/LeftRail";
import { errText, fmt, usePoll } from "../ws/util";
import { GeneBrowser } from "./GeneBrowser";
import { AutoPurgePanel, BugReportPanel, EchoPanel, InterfacePanel, PurgePanel, QueuePanel, RunStatus, SettingsVersion,
  TrainingPanel, V1Panel } from "./SettingsFields";
import { launchNow } from "./launchInfo";
import { Rules, Support, V1Rules } from "./Settings";

type Part = "CAMPAIGN" | "BREEDING" | "SYSTEM" | "SUPPORT";
const PARTS: { id: Part; hint: string }[] = [
  { id: "CAMPAIGN", hint: "terminal, markets, timeframes, history, engine, depth, power" },
  { id: "BREEDING", hint: "the genome: families, gates and every gene switch" },
  { id: "SYSTEM", hint: "this computer's display, and the rules the engine judges by" },
  { id: "SUPPORT", hint: "diagnostics, recent problems, the support report" },
];
const KEY = "arqvio.settings.part";

function readPart(): Part {
  try {
    const v = localStorage.getItem(KEY) as Part | null;
    return v && PARTS.some((p) => p.id === v) ? v : "CAMPAIGN";
  } catch {
    return "CAMPAIGN";
  }
}

function Problems() {
  const d = usePoll(() => api.diagnostics(), 60_000);
  const rows = d.data?.recent_problems ?? [];
  return (
    <section className="dk-panel" aria-label="Recent problems">
      <div className="dk-ph"><span className="dk-pt">Recent problems</span><span className="dk-ps">{rows.length ? `${rows.length} warnings / errors` : "none recently"}</span></div>
      <div className="dk-pb scroll" style={{ maxHeight: 420 }}>
        {d.load.state === "error" && <div className="dk-note bad">{errText(d.load.error)}</div>}
        <table className="dk-t">
          <tbody>
            {rows.map((r, i) => (
              <tr key={i} className="ro"><td className="l dim">{fmt.time(r.utc)}</td><td className="l acc">{r.src}</td>
                <td className="l" style={{ whiteSpace: "normal" }}>{r.msg}</td></tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

export function SettingsTab({ campaign, ov, prefs, setPrefs, pool, poolError, onPatchPool, depth, engine }: {
  campaign: ReactNode;
  depth: string;
  engine: string;
  ov: LaunchOverview | null;
  prefs: Prefs;
  setPrefs: (p: Prefs) => void;
  pool: GenePoolDoc | null;
  poolError: string | null;
  onPatchPool: (body: Record<string, unknown>) => Promise<void>;
}) {
  const [part, setPart] = useState<Part>(readPart);
  const pick = (p: Part) => {
    setPart(p);
    try {
      localStorage.setItem(KEY, p);
    } catch {
      /* a viewer convenience only */
    }
  };
  const eng = usePoll(() => api.engineSettings(), 60_000);
  const now = launchNow(ov);
  const [app, setApp] = useState<{ s: AppSettings; splits: number[] } | null>(null);
  const [appErr, setAppErr] = useState<string | null>(null);
  const [saved, setSaved] = useState(0);
  const loadApp = () => api.settings().then((r) => { setApp({ s: r.settings, splits: r.splits }); setAppErr(null); })
    .catch((e) => setAppErr(errText(e)));
  useEffect(() => void loadApp(), []);  // eslint-disable-line react-hooks/exhaustive-deps
  const patch = async (body: Record<string, unknown>) => {
    try {
      const r = await api.patchSettings(body);
      setApp((a) => (a ? { ...a, s: r.settings } : a));
      setAppErr(null);
    } catch (e) {
      setAppErr(errText(e));
    } finally {
      setSaved((n) => n + 1);  // every attempt: the fields show what the engine holds (SettingsFields Version)
    }
  };
  return (
    <SettingsVersion.Provider value={saved}>
    <div className="dk-settings">
      <div className="dk-settings-main">
        <div className="st-tabs" role="tablist" aria-label="Settings parts">
          {PARTS.map((p) => (
            <button key={p.id} type="button" role="tab" className="st-tab" aria-selected={part === p.id} title={p.hint}
              onClick={() => pick(p.id)}>{p.id}</button>
          ))}
          <span className="sp" />
          <span className="st-saved">SETTINGS SAVE AS YOU CHANGE THEM ✓</span>
        </div>
        {appErr && <div className="dk-note bad">{appErr}{!app && <> · <button type="button" className="lnk" onClick={() => void loadApp()}>RETRY</button></>}</div>}
        {part === "CAMPAIGN" && <>
          {campaign}
          {app && <div className="st-grid three">
            <TrainingPanel s={app.s} splits={app.splits} patch={patch} />
            <V1Panel s={app.s} patch={patch} />
            <EchoPanel depth={depth} engine={engine} version={saved} />
          </div>}
        </>}
        {part === "BREEDING" && <GeneBrowser pool={pool} busy={now.busy} onPatch={onPatchPool} error={poolError} />}
        {part === "SYSTEM" && (
          <div className="st-grid">
            <section className="dk-panel" aria-label="This computer">
              <div className="dk-ph"><span className="dk-pt">This computer</span></div>
              <div className="dk-pb" style={{ display: "flex", flexDirection: "column", gap: 14 }}>
                <SettingsForm prefs={prefs} setPrefs={setPrefs} home={ov?.home ?? null} />
              </div>
            </section>
            <InterfacePanel />
            {app && <PurgePanel s={app.s} patch={patch} />}
            {app && <AutoPurgePanel s={app.s} patch={patch} />}
            <section className="dk-panel" aria-label="Engine rules">
              <div className="dk-ph"><span className="dk-pt">Engine rules</span><span className="dk-ps">read from the engine, not editable</span></div>
              <div className="dk-pb">
                {eng.data ? <><div className="label" style={{ margin: "0 0 6px" }}>Engine V2 · CERTIFIED needs every gate</div><Rules s={eng.data} />
                  <div className="label" style={{ margin: "14px 0 6px" }}>Engine V1 · UNCERTIFIED until Certify (V2)</div><V1Rules s={eng.data} /></>
                  : eng.load.state === "error" ? <span className="warn">{errText(eng.load.error)}</span> : <span className="dim">reading the engine…</span>}
              </div>
            </section>
          </div>
        )}
        {part === "SUPPORT" && <div className="st-grid"><Support /><BugReportPanel /><Problems /></div>}
      </div>
      <div className="dk-settings-side">
        <RunStatus ov={ov} />
        <QueuePanel />
        <LogTerminal title="Engine log" compact />
      </div>
    </div>
    </SettingsVersion.Provider>
  );
}
