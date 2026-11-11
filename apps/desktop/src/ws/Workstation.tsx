// ARQVIO's one screen: the pipeline across the top, controls and the log on the left, the campaign and the vault in
// the middle, the selected strategy on the right. Everything shown is read from the engine; actions are one button
// each and their result lands in the log.

import { useCallback, useEffect, useMemo, useState } from "react";
import { api, type Bot, type GenePoolDoc, type LaunchRequest, type SymbolsOverview } from "../api/ws";
import { BrandMark } from "../ui/BrandMark";
import { CampaignPanel, type CampaignForm } from "./CampaignPanel";
import { ConfirmDialog, DeployDialog, type Prefs, SettingsDialog } from "./Dialogs";
import { EnvBanner, envState } from "./EnvBanner";
import { Inspector } from "./Inspector";
import { Pipeline, pipelineNodes } from "./Pipeline";
import { errText, fmt, readPrefs, usePoll, writePrefs } from "./util";
import { AutomationPage } from "../deck/AutomationPage";
import { Guide } from "../deck/Guide";
import { Insights } from "../deck/Insights";
import { LivePerf } from "../deck/LivePerf";
import { PortfoliosPage } from "../deck/PortfoliosPage";
import { MapPage } from "../deck/MapPage";
import { BotsTable, VaultPanel } from "./VaultPanel";
import { CampaignTab } from "../deck/CampaignTab";
import { Command } from "../deck/Command";
import { DeckHeader, type DeckTab } from "../deck/Header";
import { SettingsTab } from "../deck/SettingsTab";
import { VaultTab } from "../deck/VaultTab";
import { launchNow, useGenerations } from "../deck/launchInfo";
import { LeftRail } from "./LeftRail";

export type Layout = "classic" | "deck";
interface Saved extends Prefs {
  form: CampaignForm;
  termId: string | null;
  seeded: boolean;
  /** REPEAT in the header: the launch carries the "repeat" after-step, so the same set-up starts again */
  repeat?: boolean;
}

const FIRST: Saved = {
  theme: "dark",
  fx: "FULL",
  form: { symbols: [], timeframes: ["H1"], horizon: "research", years: 6, depth: "STANDARD", power: "high" },
  termId: null,
  seeded: false,
};
const STARTER = ["EURUSD", "GBPUSD", "XAUUSD"];

export function Workstation({ layout = "classic", onSwitchLayout }: { layout?: Layout; onSwitchLayout?: () => void } = {}) {
  const [tab, setTab] = useState<DeckTab>("COMMAND");
  const [saved, setSaved] = useState<Saved>(() => readPrefs(FIRST));
  const save = useCallback((p: Partial<Saved>) => setSaved((s) => {
    const next = { ...s, ...p };
    writePrefs(next);
    return next;
  }), []);
  useEffect(() => {
    document.documentElement.dataset.theme = saved.theme;
    document.documentElement.dataset.fx = saved.fx;
    document.documentElement.dataset.density = saved.density ?? "compact";
    document.documentElement.dataset.hints = saved.hints ?? "show";
  }, [saved.theme, saved.fx, saved.density, saved.hints]);

  const [engineUp, setEngineUp] = useState<boolean | null>(null);
  const [version, setVersion] = useState<string | null>(null);
  useEffect(() => {
    if (engineUp) api.health().then((h) => setVersion(h.version)).catch(() => undefined);
  }, [engineUp]);
  useEffect(() => {
    let alive = true;
    const check = () => api.ping().then(() => alive && setEngineUp(true)).catch(() => alive && setEngineUp(false));
    check();
    const t = setInterval(check, 4000);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, []);

  const terms = usePoll(() => api.terminals(), 20_000, [engineUp]);
  const termList = terms.data?.terminals ?? [];
  const pickedId = (saved.termId && termList.some((t) => t.id === saved.termId)) ? saved.termId : terms.data?.chosen ?? termList.find((t) => t.running)?.id ?? null;
  const picked = termList.find((t) => t.id === pickedId) ?? null;
  // The picked terminal is closed but exactly one other runs logged in: read from that one until the picked one opens
  // again. Not saved, and never the deploy default (deploy stays on the picked terminal).
  const ready = termList.filter((t) => t.account.state === "OK");
  const fallback = picked && picked.account.state !== "OK" && ready.length === 1 ? ready[0] : null;
  const termId = fallback?.id ?? pickedId;
  const term = fallback ?? picked;
  const termNote = fallback && picked
    ? `${picked.remembered_server ?? picked.path} is closed: reading from ${fallback.path} until it opens again` : undefined;
  const termReady = !!term && term.account.state === "OK";

  const [symbols, setSymbols] = useState<SymbolsOverview | null>(null);
  const [symbolsError, setSymbolsError] = useState<string | null>(null);
  useEffect(() => {
    setSymbols(null);
    setSymbolsError(null);
    if (!termId || !termReady) return undefined;
    let alive = true;
    api.symbols(termId).then((s) => {
      if (!alive) return;
      setSymbols(s);
      if (!saved.seeded) {  // first run: three main markets on H1 are picked, so the first LAUNCH is one press
        const pick = s.main_pairs.filter((m) => STARTER.includes(m.pair) && m.symbol).map((m) => m.symbol as string);
        save({ seeded: true, form: { ...saved.form, symbols: pick } });
      }
    }).catch((e) => alive && setSymbolsError(errText(e)));
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [termId, termReady]);

  const [busyHint, setBusyHint] = useState(false);
  const ov = usePoll(() => api.launchState(), busyHint ? 2000 : 6000, [engineUp]);
  const busy = !!ov.data?.busy;
  useEffect(() => setBusyHint(busy), [busy]);
  const vault = usePoll(() => api.vault(), busy ? 5000 : 15_000, [engineUp]);
  const dep = usePoll(() => api.deploy(), 10_000, [engineUp]);
  const [pool, setPool] = useState<GenePoolDoc | null>(null);
  const [poolError, setPoolError] = useState<string | null>(null);
  useEffect(() => {
    if (!engineUp) return;
    api.genePool().then(setPool).catch((e) => setPoolError(errText(e)));
  }, [engineUp]);
  const patchPool = async (body: Record<string, unknown>) => {
    try {
      setPool(await api.patchGenePool(body));
      setPoolError(null);
    } catch (e) {
      setPoolError(errText(e));
    }
  };

  const vaulted = ov.data?.launch?.vaulted ?? 0;
  useEffect(() => {
    void vault.reload();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [vaulted]);

  const [selected, setSelected] = useState<string | null>(null);
  const rows = vault.data?.strategies ?? [];
  const selRow = rows.find((r) => r.id === selected) ?? null;
  useEffect(() => {
    if (selected && vault.data && !rows.some((r) => r.id === selected)) setSelected(null);
  }, [rows, selected, vault.data]);

  const [acting, setActing] = useState(false);
  const [notice, setNotice] = useState<{ text: string; bad?: boolean } | null>(null);
  const act = async (fn: () => Promise<unknown>, ok?: string) => {
    setActing(true);
    try {
      await fn();
      if (ok) setNotice({ text: ok });
    } catch (e) {
      setNotice({ text: errText(e), bad: true });
    } finally {
      setActing(false);
      void ov.reload();
    }
  };

  const form = saved.form;
  const launchWhy = !engineUp ? "the engine is not running"
    : !term ? "no MetaTrader 5 found"
      : !termReady ? "open MetaTrader 5 and log in"
        : !form.symbols.length ? "pick at least one market"
          : !form.timeframes.length ? "pick at least one timeframe" : null;
  const launch = () => act(async () => {
    if (!termId) return;
    await api.chooseTerminal(termId);
    await api.launch({ terminal: termId, symbols: form.symbols, timeframes: form.timeframes, years: form.years,
      horizon: form.horizon ?? "research", depth: form.depth, power: form.power, search: form.engine ?? "standard", ...(saved.repeat ? { after: ["repeat"] as LaunchRequest["after"] } : {}) });
  }, "Launched: data first, then the safety check, then breeding. Follow it in the log.");
  const queueCurrent = async (after?: LaunchRequest["after"]) => {
    if (!termId) return;
    await api.enqueue({ terminal: termId, symbols: form.symbols, timeframes: form.timeframes, years: form.years,
      horizon: form.horizon ?? "research", depth: form.depth, power: form.power, search: form.engine ?? "standard", after });
  };
  const gens = useGenerations(ov.data ?? null);

  const [dlg, setDlg] = useState<null | "deploy" | "settings" | "delete">(null);
  const [removeBot, setRemoveBot] = useState<Bot | null>(null);

  const nodes = useMemo(() => pipelineNodes(term, ov.data?.launch ?? null, busy, vault.data ? rows.length : null,
    dep.data?.bots ?? null), [term, ov.data, busy, vault.data, rows.length, dep.data]);

  if (engineUp === false && !ov.data) {
    return (
      <div className="offline">
        <section className="panel">
          <div className="panel-h">ARQVIO</div>
          <div className="empty">
            <BrandMark />
            <h3>Starting the engine…</h3>
            <p>The ARQVIO engine runs on this computer (127.0.0.1:8741). If this stays, close ARQVIO and open it again;
              another program may be holding that port.</p>
          </div>
        </section>
      </div>
    );
  }

  const deck = layout === "deck";
  const show = (part: "center-campaign" | "center-vault" | "right") => !deck
    || (tab === "VAULT" && (part === "center-vault" || part === "right"));
  const open = (id: string) => { setSelected(id); setTab("VAULT"); };
  const campaignPanel = <CampaignPanel terms={terms.data} termId={termId} onTerm={(id) => save({ termId: id })}
    onRefreshTerms={() => void terms.reload()} symbols={symbols} symbolsError={symbolsError} form={form}
    setForm={(f) => save({ form: f })} ov={ov.data} busy={busy} pool={pool} poolError={poolError} onPatchPool={patchPool} />;
  const vaultPanel = <VaultPanel vault={vault.data} vaultError={vault.load.state === "error" ? vault.load.error : null}
    launch={ov.data?.launch ?? null} dep={dep.data} selected={selected} onSelect={setSelected}
    onDeploy={() => selRow && setDlg("deploy")} onDelete={() => selRow && setDlg("delete")}
    onStage={(b, s) => void act(async () => { await api.setStage(b.bot, s); await dep.reload(); },
      `${b.symbol} ${b.timeframe}: lot scale changed; press Deploy again to recompile it into MetaTrader 5`)}
    onRemoveBot={(b) => setRemoveBot(b)} onChanged={() => void vault.reload()}
    onStopAll={(stop) => void act(async () => { await api.stopAll(stop); await dep.reload(); },
      stop ? "STOP written: every ARQVIO bot closes its positions and stands by." : "Bots may trade again.")} />;
  const inspector = <Inspector id={selected} onDeploy={() => selRow && setDlg("deploy")} onDelete={() => selRow && setDlg("delete")}
    onChanged={() => void vault.reload()} />;
  const statusBar = (
    <footer className="ws-status">
      <span>engine <b className={engineUp ? "pos" : "neg"}>{engineUp ? "running" : engineUp === false ? "not answering" : "…"}</b></span>
      <span>vault <b>{rows.length}</b></span>
      <span>bots <b>{(dep.data?.bots ?? []).filter((b) => b.status !== "REMOVED").length}</b></span>
      {ov.data && <span>cores <b>{ov.data.logical_cores}</b></span>}
      {notice && <span className={notice.bad ? "neg" : ""} style={{ overflow: "hidden", textOverflow: "ellipsis" }} title={notice.text}>{notice.text}</span>}
      <span style={{ marginLeft: "auto" }}>{ov.data?.launch ? `last launch ${fmt.time(ov.data.launch.utc)} · ${ov.data.launch.state.toLowerCase()}` : "no launch yet"}</span>
    </footer>
  );
  const dialogs = (
    <>
      {dlg === "deploy" && selRow && (
        <DeployDialog row={selRow} terms={terms.data} defaultTerm={pickedId}
          defaults={(dep.data?.guardrail_defaults as Record<string, number | boolean>) ?? null}
          onClose={() => setDlg(null)} onDone={(msg) => { setDlg(null); setNotice({ text: msg }); void dep.reload(); }} />
      )}
      {dlg === "settings" && <SettingsDialog prefs={saved} setPrefs={(p) => save(p)} home={ov.data?.home ?? null} onClose={() => setDlg(null)} />}
      {dlg === "delete" && selRow && (
        <ConfirmDialog title="Delete this strategy?" confirm="Delete"
          body={`${selRow.symbol} ${selRow.timeframe} · ${selRow.family} (grade ${selRow.grade}) moves to the recycle bin (restorable for 30 days). Bots already deployed keep running.`}
          onCancel={() => setDlg(null)}
          onConfirm={() => { setDlg(null); void act(async () => { await api.removeStrategy(selRow.id); setSelected(null); await vault.reload(); }, "Moved to the recycle bin."); }} />
      )}
      {removeBot && (
        <ConfirmDialog title="Remove this bot?" confirm="Remove"
          body={`${removeBot.symbol} ${removeBot.timeframe} is removed from ARQVIO's list. Its open trades are kept and run to their own stop or target: ARQVIO never closes a position itself. If it is on a chart in MetaTrader 5, remove it there too (right-click the chart › Expert list).`}
          onCancel={() => setRemoveBot(null)}
          onConfirm={() => { const b = removeBot; setRemoveBot(null); void act(async () => { await api.removeBot(b.bot); await dep.reload(); }, "Bot removed."); }} />
      )}
    </>
  );

  if (deck) {
    const now = launchNow(ov.data ?? null);
    const runs = form.symbols.length * form.timeframes.length;
    const idleLine = ov.data && runs
      ? `${runs} run${runs === 1 ? "" : "s"} · engine ${(form.engine ?? "standard") === "v1" ? "V1" : "V2"} · up to ${fmt.int(ov.data.depths[form.depth].budget)} strategies each · ${ov.data.power[form.power]} of ${ov.data.logical_cores} cores`
      : "ready";
    return (
      <div className={`dk dk-tab-${tab.toLowerCase()}`}>
        <DeckHeader now={now} canLaunch={!launchWhy && !busy} launchWhy={launchWhy} idleLine={idleLine} acting={acting}
          repeat={!!saved.repeat} onRepeat={() => save({ repeat: !saved.repeat })}
          onQueue={() => void act(() => queueCurrent(saved.repeat ? ["repeat"] : undefined), "Added to the queue: it starts when the engine is free (AUTOMATION).")}
          onLaunch={() => void launch()}
          onStop={() => void act(() => api.stop(), "Stopping…")}
          onSkip={() => void act(() => api.skip(), "Skip: this generation finishes, then the Final War.")}
          onPause={() => void act(() => api.pause(), "Pausing at the next checkpoint…")}
          onResume={() => void act(() => api.resume(), "Resumed.")}
          tab={tab} onTab={setTab} term={term} termReady={termReady} termNote={termNote} version={version} onSwitchLayout={onSwitchLayout} />
        <EnvBanner state={envState(engineUp, termList, term)} />
        <main className="dk-main">
          {tab === "COMMAND" && <Command ov={ov.data ?? null} rows={rows} dep={dep.data ?? null} symbols={symbols} form={form}
            setForm={(f) => save({ form: f })} points={gens.points} onOpen={open} onTab={(t) => setTab(t)} />}
          {tab === "CAMPAIGN" && <CampaignTab ov={ov.data ?? null} rows={rows} points={gens.points} onOpen={open} />}
          {tab === "SETTINGS" && <SettingsTab ov={ov.data ?? null} prefs={saved} setPrefs={(p) => save(p)} pool={pool}
            depth={form.depth} engine={form.engine ?? "standard"}
            poolError={poolError} onPatchPool={patchPool}
            campaign={<CampaignPanel terms={terms.data} termId={termId} onTerm={(id) => save({ termId: id })}
              onRefreshTerms={() => void terms.reload()} symbols={symbols} symbolsError={symbolsError} form={form}
              setForm={(f) => save({ form: f })} ov={ov.data} busy={busy} pool={pool} poolError={poolError} onPatchPool={patchPool} hidePool />} />}
          {tab === "VAULT" && <div className="dk-vault"><div className="dk-vault-t"><VaultTab vault={vault.data ?? null}
            vaultError={vault.load.state === "error" ? vault.load.error : null} selected={selected} onSelect={setSelected}
            onDeploy={(id) => { setSelected(id); setDlg("deploy"); }} onChanged={() => void vault.reload()} /></div>
            <aside className="dk-vault-i">{inspector}</aside></div>}
          {tab === "INSIGHTS" && <Insights rows={rows} onOpen={open} />}
          {tab === "GUIDE" && <Guide onGo={(t) => setTab(t as DeckTab)} />}
          {tab === "AUTOMATION" && <AutomationPage canQueue={!launchWhy} queueWhy={launchWhy} onQueueCurrent={(after) => queueCurrent(after)} />}
          {tab === "MAP" && <MapPage rows={rows} onOpen={open} />}
          {tab === "PORTFOLIOS" && <PortfoliosPage rows={rows} dep={dep.data ?? null} onOpen={setSelected}
            onDeploy={(id) => { setSelected(id); setDlg("deploy"); }} onChanged={() => void vault.reload()} inspector={inspector} />}
          {tab === "LIVE" && (
            <div className="dk-live">
              <section className="panel grow" aria-label="Live bots">
                <div className="panel-h">Live bots <small>each row is the bot's own heartbeat; the chart dashboard in MetaTrader 5 shows the same status and its reason</small>
                  {dep.data && (
                    <button type="button" className="btn btn-sm" style={{ marginLeft: "auto" }}
                      onClick={() => void act(async () => { await api.stopAll(!dep.data?.stop_all); await dep.reload(); },
                        dep.data?.stop_all ? "Bots may trade again." : "STOP written: every ARQVIO bot closes its positions and stands by.")}>
                      {dep.data.stop_all ? "Resume all bots" : "Stop all bots"}
                    </button>
                  )}
                </div>
                {dep.data ? <BotsTable dep={dep.data}
                  onStage={(b, s) => void act(async () => { await api.setStage(b.bot, s); await dep.reload(); },
                    `${b.symbol} ${b.timeframe}: lot scale changed; press Deploy again to recompile it into MetaTrader 5`)}
                  onRemove={(b) => setRemoveBot(b)} /> : <div className="empty"><p>Reading the bots…</p></div>}
              </section>
              {(dep.data?.bots ?? []).some((b) => b.status !== "REMOVED") && <LivePerf />}
            </div>
          )}
        </main>
        {statusBar}
        {dialogs}
      </div>
    );
  }

  return (
    <div className="ws">
      <header className="ws-top">
        <div className="brand"><BrandMark />ARQVIO</div>
        <Pipeline nodes={nodes} />
        <div className="mt5-pill" title={term?.path ?? "No terminal"}>
          <span className={`led ${termReady ? "ok" : "bad"}`} />
          {termReady ? `${term?.account.server} · ${term?.account.mode}` : term ? "MT5 closed" : "MT5 not found"}
        </div>
        {onSwitchLayout && (
          <button type="button" className="btn btn-sm" onClick={onSwitchLayout} title="Switch between DECK and CLASSIC">DECK</button>
        )}
        <button type="button" className="icon-btn" onClick={() => setDlg("settings")} aria-label="Settings" title="Settings">
          <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4"><circle cx="8" cy="8" r="2.2" />
            <path d="M8 1.5v2M8 12.5v2M1.5 8h2M12.5 8h2M3.4 3.4l1.4 1.4M11.2 11.2l1.4 1.4M3.4 12.6l1.4-1.4M11.2 4.8l1.4-1.4" /></svg>
        </button>
      </header>

      <EnvBanner state={envState(engineUp, termList, term)} />

      <LeftRail ov={ov.data} canLaunch={!launchWhy && !busy} launchWhy={launchWhy} acting={acting} onLaunch={() => void launch()}
        onPause={() => void act(() => api.pause(), "Pausing at the next checkpoint…")}
        onResume={() => void act(() => api.resume(), "Resumed.")}
        onStop={() => void act(() => api.stop(), "Stopping…")}
        onSkip={() => void act(() => api.skip(), "Skip: this generation finishes, then the Final War.")} />

      <main className="ws-center">
        {show("center-campaign") && campaignPanel}
        {show("center-vault") && vaultPanel}
      </main>

      {show("right") && <aside className="ws-right" aria-label="Selected strategy">{inspector}</aside>}

      {statusBar}

      {dialogs}
    </div>
  );
}
