// DECK's header, the same on every tab: the mark and build, the run controls (QUEUE, REPEAT, LAUNCH or STOP, SKIP,
// PAUSE) with one line under them saying what they would do or are doing, the task corner, and the tabs.
// idea: research §7.3 / BP06 §2 (run controls on every tab, a one-line task state); layout and copy are ARQVIO's.

import type { Terminal } from "../api/ws";
import { BrandMark } from "../ui/BrandMark";
import type { LaunchNow } from "./launchInfo";

export type DeckTab = "COMMAND" | "CAMPAIGN" | "SETTINGS" | "VAULT" | "PORTFOLIOS" | "LIVE" | "AUTOMATION" | "MAP" | "INSIGHTS" | "GUIDE";

export const DECK_TABS: { id: DeckTab; hint: string }[] = [
  { id: "COMMAND", hint: "the whole engine on one screen: targets, the neural core, vault, genomes and the feed" },
  { id: "CAMPAIGN", hint: "the running campaign: progress, generations, the feed and what it added" },
  { id: "SETTINGS", hint: "campaign set-up, breeding genes, display and the engine's rules" },
  { id: "VAULT", hint: "every strategy that survived, and the selected one" },
  { id: "PORTFOLIOS", hint: "the deployed bots judged together: correlation, diversity, combined equity" },
  { id: "LIVE", hint: "the bots in MetaTrader 5, from their heartbeats" },
  { id: "AUTOMATION", hint: "the launch queue: campaigns that run one after another" },
  { id: "MAP", hint: "every strategy by Sharpe and drawdown, coloured by type" },
  { id: "INSIGHTS", hint: "what the vault holds, by market, type and grade" },
  { id: "GUIDE", hint: "ARQVIO in eight steps" },
];

const Ico = {
  play: <svg viewBox="0 0 12 12" aria-hidden="true"><path d="M3 1.5v9l7.5-4.5z" fill="currentColor" /></svg>,
  stop: <svg viewBox="0 0 12 12" aria-hidden="true"><rect x="2.5" y="2.5" width="7" height="7" fill="currentColor" /></svg>,
  skip: <svg viewBox="0 0 12 12" aria-hidden="true"><path d="M1.5 2v8l5-4zM6.5 2v8l5-4z" fill="currentColor" /></svg>,
  pause: <svg viewBox="0 0 12 12" aria-hidden="true"><path d="M3 2h2v8H3zM7 2h2v8H7z" fill="currentColor" /></svg>,
  repeat: <svg viewBox="0 0 12 12" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="1.3"><path d="M2 5.5a4 4 0 0 1 7-2.6M10 6.5a4 4 0 0 1-7 2.6M9 1v2.2H6.8M3 11V8.8h2.2" /></svg>,
};

export function DeckHeader({ now, canLaunch, launchWhy, idleLine, acting, repeat, onRepeat, onQueue, onLaunch, onStop,
  onSkip, onPause, onResume, tab, onTab, term, termReady, termNote, version, onSwitchLayout }: {
  now: LaunchNow;
  canLaunch: boolean;
  launchWhy: string | null;
  idleLine: string;
  acting: boolean;
  repeat: boolean;
  onRepeat: () => void;
  onQueue: () => void;
  onLaunch: () => void;
  onStop: () => void;
  onSkip: () => void;
  onPause: () => void;
  onResume: () => void;
  tab: DeckTab;
  onTab: (t: DeckTab) => void;
  term: Terminal | null;
  termReady: boolean;
  termNote?: string;  // set when the picked terminal is closed and another one is read instead
  version: string | null;
  onSwitchLayout?: () => void;
}) {
  const { busy, paused, cur, st } = now;
  const v1 = cur?.engine === "v1";
  const line = busy ? `Running: ${st?.id ?? "campaign"} · ${now.what}` : launchWhy ? `Launch needs: ${launchWhy}` : idleLine;
  const task = busy ? (paused ? "CAMPAIGN PAUSED" : "CAMPAIGN RUNNING") : "NO TASKS RUNNING";
  return (
    <header className="dk-head">
      <div className="dk-top">
        <div className="dk-brand">
          <BrandMark />
          <div>
            <div className="dk-name">ARQVIO</div>
            <div className="dk-build">BUILD {version ?? "…"}</div>
          </div>
        </div>
        <div className="dk-run">
          <div className="dk-run-btns">
            <button type="button" className="rb rb-q" onClick={onQueue} disabled={!!launchWhy || acting}
              title={launchWhy ?? "Add this set-up to the queue: it starts when the engine is free"}>+ QUEUE</button>
            <button type="button" className="rb rb-q" aria-pressed={repeat} onClick={onRepeat}
              title="REPEAT: when this launch finishes, the same set-up starts again (until you turn it off)">{Ico.repeat} REPEAT</button>
            {busy ? (
              <button type="button" className="rb rb-main" onClick={onStop} disabled={acting}
                title="End this launch: the running campaign stops at its checkpoint, the rest is skipped">{Ico.stop} STOP</button>
            ) : (
              <button type="button" className="rb rb-main" onClick={onLaunch} disabled={!canLaunch || acting}
                title={launchWhy ?? "Read the data, then breed and test strategies on every chosen market"}>{Ico.play} LAUNCH</button>
            )}
            <button type="button" className="rb" onClick={onSkip} disabled={!busy || !v1 || acting}
              title={v1 ? "Finish this generation, then go straight to the Final War and the vault gates" : "SKIP works on engine V1 campaigns"}>
              {Ico.skip} SKIP</button>
            {paused ? (
              <button type="button" className="rb" onClick={onResume} disabled={acting} title="Carry on from the checkpoint">{Ico.play} RESUME</button>
            ) : (
              <button type="button" className="rb" onClick={onPause} disabled={!busy || acting}
                title="Hold the running campaign at its checkpoint; resume later">{Ico.pause} PAUSE</button>
            )}
          </div>
          <div className={`dk-run-line${launchWhy && !busy ? " warn" : ""}`} title={line}>{line}</div>
        </div>
        <div className="dk-right">
          <div className="dk-mt5" title={termNote ?? term?.path ?? "No terminal"}>
            <span className={`led ${termReady ? "ok" : "bad"}`} />
            {termReady ? `${term?.account.server} · ${term?.account.mode}${termNote ? " · AUTO" : ""}` : term ? "MT5 closed" : "MT5 not found"}
          </div>
          <div className={`dk-task${busy ? " on" : ""}`} role="status">&gt; {task}</div>
          {onSwitchLayout && (
            <button type="button" className="rb rb-sm" onClick={onSwitchLayout} title="Switch between DECK and CLASSIC">CLASSIC</button>
          )}
        </div>
      </div>
      <nav className="dk-tabs" role="tablist" aria-label="Sections">
        {DECK_TABS.map((t) => (
          <button key={t.id} type="button" role="tab" className="dk-tab" aria-selected={tab === t.id} title={t.hint}
            onClick={() => onTab(t.id)}>{t.id}</button>
        ))}
      </nav>
    </header>
  );
}
