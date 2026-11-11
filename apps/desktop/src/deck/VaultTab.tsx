// DECK VAULT: every strategy, with the research toolbar and right-click menu. Scope ALL / LATEST CAMPAIGN (+ the
// campaign picker and DELETE CAMPAIGN), BROKER / PORTFOLIO / SEARCH / SYMBOL / TF filters and FILTERS ▾ criteria (all
// remembered), COLUMNS ▾, PURGE ▾ (criteria, preview first), DEDUPE, the recycle bin; multi-select (click, Ctrl/Shift
// click, Ctrl+A) and the menu COPY ID · COPY DNA · GET JSON · ADD TO PORTFOLIO › · DEPLOY · PARITY TEST · REASSESS ·
// DELETE, for one strategy or N SELECTED.
// idea: research VAULT (RF 69568-71660, PN 67/79/163/255/367/451); layout and code are ARQVIO's.

import { type MouseEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, type NamedPortfolio, type StrategyRow, type VaultOverview } from "../api/ws";
import { ConfirmDialog, Layer } from "../ws/Dialogs";
import { errText, FAMILY_SHORT, fmt, shortId } from "../ws/util";
import { VaultTools } from "../ws/VaultTools";

type Col = { id: string; label: string; get: (r: StrategyRow) => number | string | null; show: (r: StrategyRow) => string;
  cls?: (r: StrategyRow) => string; title?: string };

const pct = (v: number | null | undefined) => (v == null ? null : v * 100);
const ageDays = (r: StrategyRow) => Math.max(0, Math.floor((Date.now() - Date.parse(r.added_utc)) / 86_400_000));
const signCls = (v: number | null | undefined) => (v == null ? "" : v >= 0 ? "pos" : "neg");

const COLS: Col[] = [
  { id: "id", label: "ID", get: (r) => r.id, show: (r) => `#${shortId(r.id)}` },
  { id: "broker", label: "BROKER", get: (r) => r.broker ?? "", show: (r) => r.broker ?? "—" },
  { id: "asset", label: "ASSET", get: (r) => r.symbol, show: (r) => r.symbol },
  { id: "tf", label: "TF", get: (r) => r.timeframe, show: (r) => r.timeframe },
  { id: "trades", label: "TRADES", get: (r) => r.n_trades, show: (r) => fmt.int(r.n_trades) },
  { id: "return", label: "RETURN", get: (r) => r.total_return, show: (r) => fmt.pct(r.total_return), cls: (r) => signCls(r.total_return) },
  { id: "cagr", label: "CAGR", get: (r) => r.cagr ?? null, show: (r) => (r.cagr == null ? "—" : `${fmt.pct(r.cagr)}/yr`), title: "yearly growth over the strategy's window" },
  { id: "dd", label: "MAX DD", get: (r) => r.max_dd, show: (r) => fmt.pct(r.max_dd == null ? null : -Math.abs(r.max_dd)), cls: () => "neg" },
  { id: "type", label: "TYPE", get: (r) => r.family, show: (r) => FAMILY_SHORT[r.family] ?? r.family },
  { id: "score", label: "SCORE", get: (r) => r.score, show: (r) => `S ${r.score.toFixed(0)}`, cls: () => "acc" },
  { id: "grade", label: "GRADE", get: (r) => r.grade, show: (r) => r.grade },
  { id: "sharpe", label: "SHARPE", get: (r) => r.sharpe, show: (r) => fmt.num(r.sharpe) },
  { id: "win", label: "WIN %", get: (r) => r.win_rate ?? null, show: (r) => (r.win_rate == null ? "—" : fmt.pct(r.win_rate)), title: "from the inspector, once the strategy was opened" },
  { id: "oos", label: "OOS RET", get: (r) => r.held_out?.return ?? null, show: (r) => fmt.pct(r.held_out?.return), cls: (r) => signCls(r.held_out?.return), title: "the held-out 30% breeding never saw" },
  { id: "parity", label: "PARITY", get: (r) => r.parity_grade ?? "", show: (r) => r.parity_grade ?? "—" },
  { id: "robust", label: "ROBUST", get: (r) => r.robust ?? "", show: (r) => r.robust ?? "—" },
  { id: "portfolio", label: "PORTFOLIO", get: (r) => (r.portfolios ?? []).join(", "), show: (r) => (r.portfolios ?? []).join(", ") || "—" },
  { id: "window", label: "WINDOW", get: (r) => r.window_years ?? null, show: (r) => (r.window_years == null ? "—" : `${r.window_years.toFixed(1)}y`) },
  { id: "first", label: "FIRST BAR", get: (r) => r.first_bar ?? "", show: (r) => (r.first_bar ?? "—").slice(0, 10) },
  { id: "age", label: "AGE", get: (r) => ageDays(r), show: (r) => `${ageDays(r)}d` },
  { id: "tier", label: "TIER", get: (r) => r.tier, show: (r) => `${r.engine === "v1" ? "V1 · " : ""}${r.tier}` },
];
const DEFAULT_COLS = ["id", "broker", "asset", "tf", "trades", "return", "cagr", "dd", "type", "score", "grade", "oos", "parity", "window"];

const CRIT: { id: string; label: string; test: (r: StrategyRow, v: number) => boolean }[] = [
  { id: "min_score", label: "MIN SCORE", test: (r, v) => r.score >= v },
  { id: "min_return", label: "MIN RETURN %", test: (r, v) => (pct(r.total_return) ?? -Infinity) >= v },
  { id: "max_dd", label: "MAX DD %", test: (r, v) => (pct(r.max_dd) ?? Infinity) <= v },
  { id: "min_win", label: "MIN WIN RATE %", test: (r, v) => r.win_rate == null || r.win_rate * 100 >= v },
  { id: "min_trades", label: "MIN TRADES", test: (r, v) => r.n_trades >= v },
  { id: "max_age", label: "MAX AGE (DAYS)", test: (r, v) => ageDays(r) <= v },
  { id: "min_oos", label: "MIN OOS RETURN %", test: (r, v) => (pct(r.held_out?.return) ?? -Infinity) >= v },
  { id: "min_cagr", label: "MIN CAGR %", test: (r, v) => (pct(r.cagr) ?? -Infinity) >= v },
];
const PURGE_CRIT = [["min_score", "MIN SCORE"], ["min_trades", "MIN TRADES"], ["max_dd", "MAX DD %"], ["min_return", "MIN RETURN %"],
  ["min_oos_return", "MIN OOS RETURN %"], ["max_age_days", "MAX AGE (DAYS)"]] as const;

interface View { scope: "all" | "latest"; campaign: string; broker: string; portfolio: string; q: string; symbol: string; tf: string;
  crit: Record<string, number | null>; cols: string[]; sort: { key: string; desc: boolean } }
const KEY = "arqvio.vault.view";
const FIRST: View = { scope: "all", campaign: "", broker: "", portfolio: "", q: "", symbol: "", tf: "", crit: {}, cols: DEFAULT_COLS,
  sort: { key: "score", desc: true } };

function readView(): View {
  try {
    return { ...FIRST, ...JSON.parse(localStorage.getItem(KEY) ?? "{}") };
  } catch {
    return FIRST;
  }
}

function download(name: string, doc: unknown) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(doc, null, 1)], { type: "application/json" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

export function VaultTab({ vault, vaultError, selected, onSelect, onDeploy, onChanged }: {
  vault: VaultOverview | null;
  vaultError: string | null;
  selected: string | null;
  onSelect: (id: string | null) => void;
  onDeploy: (id: string) => void;
  onChanged: () => void;
}) {
  const [view, setViewState] = useState<View>(readView);
  const setView = (p: Partial<View>) => setViewState((v) => {
    const n = { ...v, ...p };
    try {
      localStorage.setItem(KEY, JSON.stringify(n));
    } catch {
      /* per-viewer convenience only */
    }
    return n;
  });
  const [sel, setSel] = useState<Set<string>>(new Set());
  const anchor = useRef<string | null>(null);
  const [menu, setMenu] = useState<{ x: number; y: number; ids: string[]; sub?: boolean } | null>(null);
  const [pop, setPop] = useState<null | "filters" | "columns" | "purge">(null);
  const [note, setNote] = useState<{ text: string; bad?: boolean } | null>(null);
  const [confirm, setConfirm] = useState<null | { title: string; body: string; go: string; run: () => Promise<void> }>(null);
  const [pfs, setPfs] = useState<NamedPortfolio[]>([]);
  const [newPf, setNewPf] = useState("");
  const [purgeCrit, setPurgeCrit] = useState<Record<string, number | null>>({});
  const [purgeCount, setPurgeCount] = useState<number | null>(null);
  const loadPfs = useCallback(() => api.portfolios().then((r) => setPfs(r.portfolios)).catch(() => undefined), []);
  useEffect(() => void loadPfs(), [loadPfs]);

  const rows = useMemo(() => vault?.strategies ?? [], [vault]);
  const campaigns = useMemo(() => [...new Set(rows.map((r) => r.campaign_id))].sort().reverse(), [rows]);
  const latest = campaigns[0] ?? "";
  const brokers = useMemo(() => [...new Set(rows.map((r) => r.broker).filter(Boolean) as string[])].sort(), [rows]);
  const symbols = useMemo(() => [...new Set(rows.map((r) => r.symbol))].sort(), [rows]);
  const tfs = useMemo(() => [...new Set(rows.map((r) => r.timeframe))].sort(), [rows]);
  const cols = COLS.filter((c) => view.cols.includes(c.id));

  // each active filter and how many rows it alone hides (research: "filter X hides N")
  const filters: { label: string; ok: (r: StrategyRow) => boolean; clear: () => void }[] = [];
  if (view.scope === "latest") filters.push({ label: "LATEST CAMPAIGN", ok: (r) => r.campaign_id === (view.campaign || latest), clear: () => setView({ scope: "all" }) });
  if (view.broker) filters.push({ label: `BROKER ${view.broker}`, ok: (r) => r.broker === view.broker, clear: () => setView({ broker: "" }) });
  if (view.portfolio === "__any") filters.push({ label: "IN ANY PORTFOLIO", ok: (r) => !!r.portfolios?.length, clear: () => setView({ portfolio: "" }) });
  else if (view.portfolio) filters.push({ label: `PORTFOLIO ${view.portfolio}`, ok: (r) => !!r.portfolios?.includes(view.portfolio), clear: () => setView({ portfolio: "" }) });
  if (view.q) {
    const q = view.q.toLowerCase();
    filters.push({ label: `SEARCH "${view.q}"`, ok: (r) => [r.id, r.symbol, r.campaign_id, r.family].some((x) => x.toLowerCase().includes(q)), clear: () => setView({ q: "" }) });
  }
  if (view.symbol) filters.push({ label: view.symbol, ok: (r) => r.symbol === view.symbol, clear: () => setView({ symbol: "" }) });
  if (view.tf) filters.push({ label: view.tf, ok: (r) => r.timeframe === view.tf, clear: () => setView({ tf: "" }) });
  for (const c of CRIT) {
    const v = view.crit[c.id];
    if (v != null) filters.push({ label: `${c.label} ${v}`, ok: (r) => c.test(r, v), clear: () => setView({ crit: { ...view.crit, [c.id]: null } }) });
  }
  const shown = useMemo(() => {
    const col = COLS.find((c) => c.id === view.sort.key) ?? COLS[9];
    const out = rows.filter((r) => !r.copy_of && filters.every((f) => f.ok(r)));
    return [...out].sort((a, b) => {
      const x = col.get(a);
      const y = col.get(b);
      const c = x == null ? 1 : y == null ? -1 : x < y ? -1 : x > y ? 1 : 0;
      return view.sort.desc ? -c : c;
    });
  }, [rows, view, latest]);  // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {  // a selection never holds rows that left the vault
    setSel((s) => new Set([...s].filter((i) => rows.some((r) => r.id === i))));
  }, [rows]);

  const click = (e: MouseEvent, r: StrategyRow) => {
    if (e.shiftKey && anchor.current) {
      const ids = shown.map((x) => x.id);
      const [a, b] = [ids.indexOf(anchor.current), ids.indexOf(r.id)].sort((p, q) => p - q);
      setSel(new Set([...sel, ...ids.slice(a, b + 1)]));
    } else if (e.ctrlKey || e.metaKey) {
      const n = new Set(sel);
      if (n.has(r.id)) n.delete(r.id);
      else n.add(r.id);
      setSel(n);
      anchor.current = r.id;
    } else {
      setSel(new Set([r.id]));
      anchor.current = r.id;
      onSelect(r.id);
    }
  };
  const context = (e: MouseEvent, r: StrategyRow) => {
    e.preventDefault();
    const ids = sel.has(r.id) && sel.size > 1 ? [...sel] : [r.id];
    if (!sel.has(r.id)) {
      setSel(new Set([r.id]));
      onSelect(r.id);
    }
    setMenu({ x: Math.min(e.clientX, window.innerWidth - 260), y: Math.min(e.clientY, window.innerHeight - 380), ids });
  };
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && ["INPUT", "SELECT", "TEXTAREA"].includes(t.tagName)) return;
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "a") {
        e.preventDefault();
        setSel(new Set(shown.map((r) => r.id)));
      } else if (e.key === "Escape") {
        setMenu(null);
        setPop(null);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [shown]);

  const run = async (fn: () => Promise<string>) => {
    setMenu(null);
    try {
      setNote({ text: await fn() });
      onChanged();
      void loadPfs();
    } catch (e) {
      setNote({ text: errText(e), bad: true });
    }
  };
  const label = (n: number, one: string) => (n > 1 ? `${one} · ${n} SELECTED` : one);
  const byId = (id: string) => rows.find((r) => r.id === id);

  const menuItems = menu ? [
    { k: "COPY ID", go: () => run(async () => { await navigator.clipboard?.writeText(menu.ids.join("\n")); return `Copied ${menu.ids.length} id(s).`; }) },
    { k: "COPY DNA", go: () => run(async () => {
      const dnas = await Promise.all(menu.ids.map((i) => api.strategy(i).then((d) => ({ id: i, dna: d.dna }))));
      await navigator.clipboard?.writeText(JSON.stringify(dnas.length === 1 ? dnas[0].dna : dnas, null, 1));
      return `Copied the DNA of ${dnas.length} strateg${dnas.length === 1 ? "y" : "ies"}.`;
    }) },
    { k: "GET JSON · FULL PASSPORT", go: () => run(async () => {
      const docs = await Promise.all(menu.ids.map((i) => api.passport(i)));
      download(menu.ids.length === 1 ? `arqvio_${shortId(menu.ids[0])}.json` : `arqvio_${menu.ids.length}_passports.json`, docs.length === 1 ? docs[0] : docs);
      return "Passport saved to your Downloads folder.";
    }) },
    { k: "ADD TO PORTFOLIO ›", sub: true },
    { k: "DEPLOY", off: menu.ids.length !== 1, go: () => { setMenu(null); onDeploy(menu.ids[0]); } },
    { k: label(menu.ids.length, "PARITY TEST"), go: () => run(async () => {
      const r = await api.vaultBulk("parity", menu.ids);
      const why = Object.values(r.refused ?? {})[0];
      return `${r.queued?.length ?? 0} parity test(s) queued${why ? `; ${Object.keys(r.refused ?? {}).length} cannot run: ${why}` : ""}.`;
    }) },
    { k: label(menu.ids.length, "REASSESS"), go: () => run(async () => {
      const r = await api.vaultBulk("reassess", menu.ids);
      return `${r.queued?.length ?? 0} robustness re-run(s) queued; they run one at a time.`;
    }) },
    { k: label(menu.ids.length, "DELETE"), danger: true, go: () => { const ids = menu.ids; setMenu(null); setConfirm({
      title: ids.length > 1 ? `Delete ${ids.length} strategies?` : "Delete this strategy?", go: "DELETE",
      body: "They move to the recycle bin and can be restored for 30 days. Bots already deployed keep running.",
      run: async () => { const r = await api.vaultBulk("delete", ids); setSel(new Set()); onSelect(null); setNote({ text: `${r.done?.length ?? 0} moved to the recycle bin.` }); onChanged(); },
    }); } },
  ] as { k: string; go?: () => void; sub?: boolean; off?: boolean; danger?: boolean }[] : [];

  const addTo = (p: NamedPortfolio | null) => menu && run(async () => {
    const target = p ?? (await api.portfolioCreate(newPf)).portfolio;
    const r = await api.portfolioAdd(target.id, menu.ids);
    setNewPf("");
    return `Added to portfolio '${r.portfolio.name}' (now ${r.portfolio.members.length}).`;
  });

  return (
    <section className="dk-panel vt2" aria-label="Strategy vault">
      <div className="vt2-bar">
        <span className="chips">
          <button type="button" className="chip" aria-pressed={view.scope === "all"} onClick={() => setView({ scope: "all" })}>ALL STRATEGIES</button>
          <button type="button" className="chip" aria-pressed={view.scope === "latest"} onClick={() => setView({ scope: "latest" })}>LATEST CAMPAIGN</button>
        </span>
        {view.scope === "latest" && <>
          <select className="select" value={view.campaign || latest} onChange={(e) => setView({ campaign: e.target.value })} aria-label="Campaign">
            {campaigns.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
          <button type="button" className="rb rb-sm" onClick={() => { const cid = view.campaign || latest; setConfirm({ title: "Delete this campaign?", go: "DELETE CAMPAIGN",
            body: `Every strategy of ${cid} moves to the recycle bin (restorable for 30 days).`,
            run: async () => { const r = await api.deleteCampaign(cid); setNote({ text: `${r.removed.length} moved to the recycle bin.` }); setView({ scope: "all", campaign: "" }); onChanged(); } }); }}>DELETE CAMPAIGN</button>
        </>}
        <select className="select" value={view.broker} onChange={(e) => setView({ broker: e.target.value })} aria-label="Broker">
          <option value="">ALL BROKERS</option>{brokers.map((b) => <option key={b} value={b}>{b}</option>)}</select>
        <select className="select" value={view.portfolio} onChange={(e) => setView({ portfolio: e.target.value })} aria-label="Portfolio">
          <option value="">ALL</option><option value="__any">IN ANY PORTFOLIO</option>{pfs.map((p) => <option key={p.id} value={p.name}>{p.name}</option>)}</select>
        <input className="as-text vt2-q" placeholder="id, symbol, campaign, type" value={view.q} onChange={(e) => setView({ q: e.target.value })} aria-label="Search" />
        <select className="select" value={view.symbol} onChange={(e) => setView({ symbol: e.target.value })} aria-label="Symbol">
          <option value="">SYMBOL: ALL</option>{symbols.map((s) => <option key={s} value={s}>{s}</option>)}</select>
        <select className="select" value={view.tf} onChange={(e) => setView({ tf: e.target.value })} aria-label="Timeframe">
          <option value="">TF: ALL</option>{tfs.map((s) => <option key={s} value={s}>{s}</option>)}</select>
        <button type="button" className="rb rb-sm" aria-pressed={pop === "filters"} onClick={() => setPop(pop === "filters" ? null : "filters")}>FILTERS ▾</button>
        <button type="button" className="rb rb-sm" aria-pressed={pop === "columns"} onClick={() => setPop(pop === "columns" ? null : "columns")}>COLUMNS ▾</button>
        <button type="button" className="rb rb-sm" aria-pressed={pop === "purge"} onClick={() => setPop(pop === "purge" ? null : "purge")}>PURGE ▾</button>
        <button type="button" className="rb rb-sm" title="Strategies that made the same trades as a better-ranked one go to the recycle bin; the best stays"
          onClick={() => void api.dedupe(false).then((p) => {
            if (!p.count) {
              setNote({ text: "No copies in the vault." });
              return;
            }
            setConfirm({ title: `DEDUPE: ${p.count} copies?`, go: "DEDUPE",
              body: "Each one made exactly the same trades as a better-ranked strategy, which stays. They move to the recycle bin (30 days).",
              run: async () => { const r = await api.dedupe(true); setNote({ text: `${r.count} copies moved to the recycle bin.` }); onChanged(); } });
          })}>DEDUPE</button>
        <VaultTools onChanged={onChanged} />
        <button type="button" className="rb rb-sm" onClick={() => void api.restoreAll().then((r) => { setNote({ text: `${r.restored.length} restored from the recycle bin.` }); onChanged(); })}>RESTORE ALL</button>
        <span className="sp" />
        <span className="vt2-count"><b>{shown.length}</b> / {rows.filter((r) => !r.copy_of).length} STRATEGIES{sel.size > 1 ? ` · ${sel.size} SELECTED` : ""}</span>
      </div>

      {pop === "filters" && (
        <div className="vt2-pop">
          {CRIT.map((c) => (
            <label key={c.id} className="af"><span>{c.label}</span>
              <input type="number" value={view.crit[c.id] ?? ""} placeholder="—"
                onChange={(e) => setView({ crit: { ...view.crit, [c.id]: e.target.value === "" ? null : Number(e.target.value) } })} /></label>
          ))}
          <button type="button" className="rb rb-sm" onClick={() => setView({ crit: {} })}>CLEAR ALL</button>
        </div>
      )}
      {pop === "columns" && (
        <div className="vt2-pop cols">
          {COLS.map((c) => (
            <label key={c.id} className="vt2-col"><input type="checkbox" checked={view.cols.includes(c.id)}
              onChange={() => setView({ cols: view.cols.includes(c.id) ? view.cols.filter((x) => x !== c.id) : COLS.filter((x) => x.id === c.id || view.cols.includes(x.id)).map((x) => x.id) })} />{c.label}</label>
          ))}
          <button type="button" className="rb rb-sm" onClick={() => setView({ cols: DEFAULT_COLS })}>DEFAULT</button>
        </div>
      )}
      {pop === "purge" && (
        <div className="vt2-pop">
          {PURGE_CRIT.map(([k, l]) => (
            <label key={k} className="af"><span>{l}</span>
              <input type="number" value={purgeCrit[k] ?? ""} placeholder="—"
                onChange={(e) => { setPurgeCount(null); setPurgeCrit({ ...purgeCrit, [k]: e.target.value === "" ? null : Number(e.target.value) }); }} /></label>
          ))}
          <button type="button" className="rb rb-sm" onClick={() => void api.purgeCriteria(purgeCrit, false).then((r) => setPurgeCount(r.count))}>PREVIEW</button>
          {purgeCount != null && <button type="button" className="rb rb-sm rb-main" disabled={!purgeCount}
            onClick={() => setConfirm({ title: `Purge ${purgeCount} strategies?`, go: "PURGE", body: "They move to the recycle bin for 30 days. Portfolio members and live bots are never touched.",
              run: async () => { const r = await api.purgeCriteria(purgeCrit, true); setPurgeCount(null); setNote({ text: `${r.count} moved to the recycle bin.` }); onChanged(); } })}>
            PURGE {purgeCount}</button>}
          <small className="dim">All blank = built-in criteria (30 trades, 35 % drawdown, no loss).</small>
        </div>
      )}

      {filters.length > 0 && (
        <div className="vt2-active">{filters.map((f) => {
          const hides = rows.filter((r) => !r.copy_of && !f.ok(r)).length;
          return <button key={f.label} type="button" className="chip" onClick={f.clear} title="clear this filter">{f.label} · hides {hides} ✕</button>;
        })}</div>
      )}
      {note && <div className={`dk-note ${note.bad ? "bad" : ""}`} style={{ padding: "0 14px" }}>{note.text}</div>}
      {vaultError && <div className="dk-note bad" style={{ padding: "0 14px" }}>{vaultError}</div>}

      <div className="vt2-scroll">
        <table className="dk-t vt2-t">
          <thead><tr>
            <th className="l"><input type="checkbox" aria-label="Select all shown" checked={shown.length > 0 && shown.every((r) => sel.has(r.id))}
              onChange={(e) => setSel(e.target.checked ? new Set(shown.map((r) => r.id)) : new Set())} /></th>
            {cols.map((c) => (
              <th key={c.id} className={c.id === "id" || c.id === "broker" || c.id === "type" || c.id === "portfolio" ? "l" : ""} title={c.title}
                aria-sort={view.sort.key === c.id ? (view.sort.desc ? "descending" : "ascending") : undefined}
                onClick={() => setView({ sort: { key: c.id, desc: view.sort.key === c.id ? !view.sort.desc : true } })}>
                {c.label}{view.sort.key === c.id ? (view.sort.desc ? " ▾" : " ▴") : ""}</th>
            ))}
          </tr></thead>
          <tbody>
            {shown.map((r) => (
              <tr key={r.id} aria-selected={sel.has(r.id) || selected === r.id} onClick={(e) => click(e, r)} onContextMenu={(e) => context(e, r)}>
                <td className="l"><input type="checkbox" checked={sel.has(r.id)} aria-label={`Select ${shortId(r.id)}`}
                  onClick={(e) => e.stopPropagation()} onChange={() => { const n = new Set(sel); if (n.has(r.id)) n.delete(r.id); else n.add(r.id); setSel(n); }} /></td>
                {cols.map((c) => <td key={c.id} className={`${c.id === "id" || c.id === "broker" || c.id === "type" || c.id === "portfolio" ? "l " : ""}${c.cls?.(r) ?? ""}`}>{c.show(r)}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
        {!rows.length && <div className="dk-empty">No strategies yet</div>}
        {rows.length > 0 && !shown.length && <div className="dk-empty">Every strategy is hidden by a filter above</div>}
      </div>

      {menu && (
        <Layer>
          <div className="vt2-menu-scrim" onClick={() => setMenu(null)} onContextMenu={(e) => { e.preventDefault(); setMenu(null); }} />
          <div className="vt2-menu" style={{ left: menu.x, top: menu.y }} role="menu">
            <div className="vt2-menu-h">{menu.ids.length > 1 ? `${menu.ids.length} SELECTED` : `#${shortId(menu.ids[0])} · ${byId(menu.ids[0])?.symbol ?? ""}`}</div>
            {menuItems.map((m) => m.sub ? (
              <div key={m.k} className="vt2-sub">
                <span>{m.k}</span>
                <div className="vt2-subm">
                  {pfs.map((p) => <button key={p.id} type="button" role="menuitem" onClick={() => addTo(p)}>{p.name} <small>{p.members.length}</small></button>)}
                  <div className="vt2-new"><input className="as-text" placeholder="NEW PORTFOLIO" value={newPf} onChange={(e) => setNewPf(e.target.value)}
                    onClick={(e) => e.stopPropagation()} />
                    <button type="button" disabled={!newPf.trim()} onClick={() => addTo(null)}>+ ADD</button></div>
                </div>
              </div>
            ) : (
              <button key={m.k} type="button" role="menuitem" className={m.danger ? "danger" : ""} disabled={m.off} onClick={m.go}>{m.k}</button>
            ))}
          </div>
        </Layer>
      )}
      {confirm && <ConfirmDialog title={confirm.title} confirm={confirm.go} body={confirm.body} onCancel={() => setConfirm(null)}
        onConfirm={() => { const c = confirm; setConfirm(null); void c.run().catch((e) => setNote({ text: errText(e), bad: true })); }} />}
    </section>
  );
}
