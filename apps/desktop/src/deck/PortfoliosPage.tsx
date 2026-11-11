// PORTFOLIOS: the named portfolios as cards (members, AVG RET, AVG DD, live, grade) plus two built-in views (the
// deployed bots, the vault's best 8); the chosen one in three parts -- STRATEGIES (its members, select and act),
// HEALTH (correlation, diversity, combined equity, TRUE SIM) and PERFORMANCE (its live bots against their backtest).
// idea: research PORTFOLIOS (RF 437-439, BP06 27, PN 121/367/461/471, GINC 8); layout and code are ARQVIO's.

import { useCallback, useEffect, useMemo, useState } from "react";
import { api, type DeployOverview, type PortfolioCard, type StrategyRow } from "../api/ws";
import { ConfirmDialog } from "../ws/Dialogs";
import { errText, FAMILY_SHORT, fmt, shortId } from "../ws/util";
import { LivePerf } from "./LivePerf";
import { Portfolio } from "./Portfolio";

type Part = "STRATEGIES" | "HEALTH" | "PERFORMANCE";
const BUILTIN = [
  { id: "deployed", name: "DEPLOYED BOTS", hint: "the strategies of the bots in MetaTrader 5" },
  { id: "top", name: "VAULT TOP 8", hint: "the vault's 8 best distinct strategies" },
];
const KEY = "arqvio.portfolios.view";

function read(): { pick: string; part: Part } {
  try {
    return { pick: "deployed", part: "STRATEGIES", ...JSON.parse(localStorage.getItem(KEY) ?? "{}") };
  } catch {
    return { pick: "deployed", part: "STRATEGIES" };
  }
}

export function PortfoliosPage({ rows, dep, onOpen, onDeploy, onChanged, inspector }: {
  rows: StrategyRow[];
  dep: DeployOverview | null;
  onOpen: (id: string) => void;
  onDeploy: (id: string) => void;
  onChanged: () => void;
  inspector: React.ReactNode;
}) {
  const [view, setViewState] = useState(read);
  const setView = (p: Partial<{ pick: string; part: Part }>) => setViewState((v) => {
    const n = { ...v, ...p };
    try {
      localStorage.setItem(KEY, JSON.stringify(n));
    } catch {
      /* per-viewer convenience only */
    }
    return n;
  });
  const [cards, setCards] = useState<PortfolioCard[]>([]);
  const [members, setMembers] = useState<Record<string, string[]>>({});
  const [err, setErr] = useState<string | null>(null);
  const [note, setNote] = useState<string | null>(null);
  const [newName, setNewName] = useState("");
  const [rename, setRename] = useState<{ id: string; name: string } | null>(null);
  const [del, setDel] = useState<PortfolioCard | null>(null);
  const [sel, setSel] = useState<Set<string>>(new Set());

  const load = useCallback(async () => {
    try {
      const [c, p] = await Promise.all([api.portfolioCards(), api.portfolios()]);
      setCards(c.cards);
      setMembers(Object.fromEntries(p.portfolios.map((x) => [x.id, x.members.map((m) => m.id)])));
      setErr(null);
    } catch (e) {
      setErr(errText(e));
    }
  }, []);
  useEffect(() => void load(), [load, rows.length]);

  const liveIds = useMemo(() => new Set((dep?.bots ?? []).filter((b) => b.status !== "REMOVED").map((b) => b.source.vault_id)), [dep]);
  const byId = useMemo(() => new Map(rows.map((r) => [r.id, r])), [rows]);
  const named = view.pick.startsWith("pf");
  const ids: string[] = useMemo(() => {
    if (named) return members[view.pick] ?? [];
    if (view.pick === "deployed") return [...liveIds];
    return [...rows].filter((r) => !r.copy_of).sort((a, b) => b.score - a.score).slice(0, 8).map((r) => r.id);
  }, [named, members, view.pick, liveIds, rows]);
  const memberRows = ids.map((i) => byId.get(i)).filter(Boolean) as StrategyRow[];
  const missing = ids.length - memberRows.length;
  const source = named ? `pf:${view.pick}` : view.pick;
  const bots = useMemo(() => new Set((dep?.bots ?? []).filter((b) => ids.includes(b.source.vault_id)).map((b) => b.bot)), [dep, ids]);
  const title = named ? cards.find((c) => c.id === view.pick)?.name ?? "" : BUILTIN.find((b) => b.id === view.pick)?.name ?? "";

  const act = async (fn: () => Promise<string>) => {
    try {
      setNote(await fn());
      void load();
      onChanged();
    } catch (e) {
      setNote(errText(e));
    }
  };
  const picked = [...sel].filter((i) => ids.includes(i));

  return (
    <div className="pfp">
      <aside className="dk-panel pfp-list" aria-label="Portfolios">
        <div className="dk-ph"><span className="dk-pt">Portfolios</span><span className="dk-ps">{cards.length} saved</span></div>
        <div className="dk-pb scroll">
          {err && <div className="dk-note bad">{err}</div>}
          {cards.map((c) => (
            <button key={c.id} type="button" className={`pfp-card${view.pick === c.id ? " on" : ""}`} onClick={() => { setView({ pick: c.id }); setSel(new Set()); }}
              onDoubleClick={() => setRename({ id: c.id, name: c.name })} title="double-click to rename">
              <div className="pfp-top">
                {rename?.id === c.id ? (
                  <input className="as-text" autoFocus value={rename.name} onClick={(e) => e.stopPropagation()}
                    onChange={(e) => setRename({ id: c.id, name: e.target.value })}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") void act(async () => { await api.portfolioRename(c.id, rename.name); setRename(null); return "Renamed."; });
                      if (e.key === "Escape") setRename(null);
                    }} />
                ) : <b>{c.name}</b>}
                <span className={`pfp-grade g-${c.grade ?? "x"}`} title={c.grade ? "diversity grade (the worst axis)" : "open HEALTH to grade it"}>{c.grade ?? "—"}</span>
              </div>
              <div className="pfp-kv"><span>{c.members} BOT{c.members === 1 ? "" : "S"}</span>
                <span>AVG RET <em className={(c.avg_return ?? 0) >= 0 ? "pos" : "neg"}>{fmt.pct(c.avg_return)}</em></span>
                <span>AVG DD <em className="neg">{fmt.pct(c.avg_dd == null ? null : -Math.abs(c.avg_dd))}</em></span></div>
              <div className="pfp-kv dim"><span>{c.live} LIVE</span>{c.members > c.in_vault && <span className="warn">{c.members - c.in_vault} left the vault</span>}
                <span>{c.brokers.join(", ")}</span></div>
            </button>
          ))}
          {BUILTIN.map((b) => (
            <button key={b.id} type="button" className={`pfp-card builtin${view.pick === b.id ? " on" : ""}`} title={b.hint}
              onClick={() => { setView({ pick: b.id }); setSel(new Set()); }}>
              <div className="pfp-top"><b>{b.name}</b></div><div className="pfp-kv dim"><span>{b.hint}</span></div>
            </button>
          ))}
          <div className="pfp-new">
            <input className="as-text" placeholder="NEW PORTFOLIO NAME" value={newName} onChange={(e) => setNewName(e.target.value)} />
            <button type="button" className="rb rb-sm" disabled={!newName.trim()}
              onClick={() => void act(async () => { const p = await api.portfolioCreate(newName); setNewName(""); setView({ pick: p.portfolio.id }); return `Portfolio '${p.portfolio.name}' created: add strategies from the VAULT (right-click › ADD TO PORTFOLIO).`; })}>
              + NEW PORTFOLIO</button>
          </div>
        </div>
      </aside>

      <section className="dk-panel pfp-main" aria-label={title}>
        <div className="dk-ph">
          <span className="dk-pt">{title}</span><span className="dk-ps">{memberRows.length} member{memberRows.length === 1 ? "" : "s"}{missing ? ` · ${missing} left the vault` : ""}</span>
          <span className="sp" />
          {(["STRATEGIES", "HEALTH", "PERFORMANCE"] as Part[]).map((p) => (
            <button key={p} type="button" className="st-tab" aria-selected={view.part === p} onClick={() => setView({ part: p })}>{p}</button>
          ))}
          {named && <button type="button" className="rb rb-sm" onClick={() => setDel(cards.find((c) => c.id === view.pick) ?? null)}>DELETE</button>}
        </div>
        {note && <div className="dk-note" style={{ padding: "0 14px" }}>{note}</div>}
        {view.part === "STRATEGIES" && (
          <div className="dk-pb pfp-body">
            <div className="pfp-acts">
              <span className="k">SCOPE</span><span className="dim">{picked.length ? `${picked.length} selected` : "select members to act on them"}</span>
              <span className="sp" />
              <button type="button" className="rb rb-sm" disabled={!picked.length}
                onClick={() => void act(async () => { const r = await api.vaultBulk("parity", picked); const why = Object.values(r.refused ?? {})[0]; return `${r.queued?.length ?? 0} parity test(s) queued${why ? `; ${Object.keys(r.refused ?? {}).length} cannot run: ${why}` : ""}.`; })}>PARITY TEST</button>
              <button type="button" className="rb rb-sm" disabled={!picked.length}
                onClick={() => void act(async () => { const r = await api.vaultBulk("reassess", picked); return `${r.queued?.length ?? 0} robustness re-run(s) queued.`; })}>REASSESS</button>
              <button type="button" className="rb rb-sm" disabled={picked.length !== 1} title="one strategy at a time" onClick={() => onDeploy(picked[0])}>DEPLOY</button>
              {named && <button type="button" className="rb rb-sm" disabled={!picked.length}
                onClick={() => void act(async () => { await api.portfolioRemove(view.pick, picked); setSel(new Set()); return `${picked.length} removed from this portfolio (they stay in the vault).`; })}>REMOVE FROM PORTFOLIO</button>}
            </div>
            {memberRows.length ? (
              <table className="dk-t vt2-t">
                <thead><tr><th className="l"><input type="checkbox" aria-label="Select all members" checked={memberRows.every((r) => sel.has(r.id))}
                  onChange={(e) => setSel(e.target.checked ? new Set(memberRows.map((r) => r.id)) : new Set())} /></th>
                  <th className="l">ID</th><th>ASSET</th><th>TF</th><th className="l">BROKER</th><th>RETURN</th><th>MAX DD</th><th>TRADES</th>
                  <th>SCORE</th><th className="l">TYPE</th><th>OOS RET</th><th>PARITY</th><th>WINDOW</th><th className="l">STATE</th></tr></thead>
                <tbody>
                  {memberRows.map((r) => (
                    <tr key={r.id} aria-selected={sel.has(r.id)} onClick={() => onOpen(r.id)}>
                      <td className="l"><input type="checkbox" checked={sel.has(r.id)} onClick={(e) => e.stopPropagation()}
                        onChange={() => { const n = new Set(sel); if (n.has(r.id)) n.delete(r.id); else n.add(r.id); setSel(n); }} aria-label={`Select ${shortId(r.id)}`} /></td>
                      <td className="l">#{shortId(r.id)}</td><td>{r.symbol}</td><td>{r.timeframe}</td><td className="l">{r.broker ?? "—"}</td>
                      <td className={(r.total_return ?? 0) >= 0 ? "pos" : "neg"}>{fmt.pct(r.total_return)}</td>
                      <td className="neg">{fmt.pct(r.max_dd == null ? null : -Math.abs(r.max_dd))}</td><td>{fmt.int(r.n_trades)}</td>
                      <td className="acc">S {r.score.toFixed(0)}</td><td className="l">{FAMILY_SHORT[r.family] ?? r.family}</td>
                      <td className={(r.held_out?.return ?? 0) >= 0 ? "pos" : "neg"}>{fmt.pct(r.held_out?.return)}</td>
                      <td>{r.parity_grade ?? "—"}</td><td>{r.window_years == null ? "—" : `${r.window_years.toFixed(1)}y`}</td>
                      <td className="l">{liveIds.has(r.id) ? <span className="tag ok">LIVE</span> : <span className="dim">not deployed</span>}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <div className="dk-empty">{named ? "EMPTY · ADD STRATEGIES FROM THE VAULT (RIGHT-CLICK › ADD TO PORTFOLIO) OR LET AN AUTOMATION FILL IT"
                : view.pick === "deployed" ? "NO BOTS DEPLOYED YET" : "THE VAULT IS EMPTY"}</div>
            )}
          </div>
        )}
        {view.part === "HEALTH" && <div className="pfp-body"><Portfolio key={source} fixed={source} onOpen={(id) => { onOpen(id); void load(); }} /></div>}
        {view.part === "PERFORMANCE" && <div className="dk-pb pfp-body"><LivePerf only={bots} /></div>}
      </section>

      <aside className="pfp-insp">{inspector}</aside>
      {del && <ConfirmDialog title={`Delete portfolio ${del.name}?`} confirm="DELETE PORTFOLIO"
        body="Only the portfolio goes. Its strategies stay in the vault and its deployed bots keep running."
        onCancel={() => setDel(null)}
        onConfirm={() => { const d = del; setDel(null); void act(async () => { await api.portfolioDelete(d.id); setView({ pick: "deployed" }); return "Portfolio deleted."; }); }} />}
    </div>
  );
}
