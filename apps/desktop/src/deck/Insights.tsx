// INSIGHTS (DECK): what the vault holds, by market and timeframe, by strategy type and grade, and the best of it.
// Every number is counted from the vault rows the engine returned; a tile with no strategies says so.

import { useMemo } from "react";
import type { StrategyRow } from "../api/ws";
import { FAMILY_NAMES, fmt } from "../ws/util";

function median(xs: number[]): number | null {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

export function Insights({ rows, onOpen }: { rows: StrategyRow[]; onOpen: (id: string) => void }) {
  const lead = useMemo(() => rows.filter((r) => !r.copy_of), [rows]);
  const stats = useMemo(() => {
    const rets = lead.map((r) => r.total_return).filter((v): v is number => v != null);
    const dds = lead.map((r) => r.max_dd).filter((v): v is number => v != null);
    const tfs = [...new Set(lead.map((r) => r.timeframe))].sort();
    const syms = [...new Set(lead.map((r) => r.symbol))].sort();
    const cell = (s: string, t: string) => lead.filter((r) => r.symbol === s && r.timeframe === t);
    const fams = [...new Set(lead.map((r) => r.family))].sort();
    const grades = ["A", "B", "C", "D", "F"].map((g) => ({ g, n: lead.filter((r) => r.grade === g).length }));
    const heldOut = lead.filter((r) => r.held_out?.state === "DONE");
    return {
      total: lead.length,
      certified: lead.filter((r) => r.tier === "CERTIFIED").length,
      medRet: median(rets),
      medDd: median(dds),
      profitable: rets.length ? rets.filter((v) => v > 0).length / rets.length : null,
      heldOutPos: heldOut.length ? heldOut.filter((r) => (r.held_out?.return ?? 0) > 0).length / heldOut.length : null,
      heldOutN: heldOut.length,
      tfs,
      syms,
      cell,
      fams: fams.map((f) => ({ f, n: lead.filter((r) => r.family === f).length,
        med: median(lead.filter((r) => r.family === f).map((r) => r.sharpe ?? 0)) })),
      grades,
      top: [...lead].sort((a, b) => b.score - a.score).slice(0, 8),
    };
  }, [lead]);

  if (!stats.total) {
    return (
      <section className="panel grow"><div className="panel-h">Insights</div>
        <div className="empty"><h3>No strategies yet</h3><p>Insights are counted from the vault. Run a campaign from COMMAND first.</p></div>
      </section>
    );
  }
  const maxFam = Math.max(1, ...stats.fams.map((x) => x.n));
  const maxGrade = Math.max(1, ...stats.grades.map((x) => x.n));
  return (
    <section className="panel grow insights" aria-label="Insights">
      <div className="panel-h">Insights <small>counted from the vault · copies counted once</small></div>
      <div className="panel-b ins-grid">
        <div className="ins-tiles">
          <div className="ins-tile"><span className="label">Strategies</span><b>{fmt.int(stats.total)}</b><small>{stats.certified} certified</small></div>
          <div className="ins-tile"><span className="label">Median return</span><b className={stats.medRet != null && stats.medRet < 0 ? "neg" : "pos"}>{fmt.pct(stats.medRet)}</b><small>whole window, costs in</small></div>
          <div className="ins-tile"><span className="label">Median max DD</span><b className="neg">{fmt.pct(stats.medDd)}</b><small>largest fall from a peak</small></div>
          <div className="ins-tile"><span className="label">Profitable</span><b>{fmt.pct(stats.profitable)}</b><small>return &gt; 0</small></div>
          <div className="ins-tile"><span className="label">Held-out positive</span><b>{stats.heldOutPos != null ? fmt.pct(stats.heldOutPos) : "—"}</b><small>{stats.heldOutN ? `of ${stats.heldOutN} tested on unseen bars` : "no held-out tests yet (v4 launches)"}</small></div>
        </div>

        <div className="ins-box">
          <div className="label">Market × timeframe <span className="dim">strategies · best score</span></div>
          <table className="vt ins-heat">
            <thead><tr><th className="l">Market</th>{stats.tfs.map((t) => <th key={t}>{t}</th>)}</tr></thead>
            <tbody>
              {stats.syms.map((s) => (
                <tr key={s}><td className="l">{s}</td>
                  {stats.tfs.map((t) => {
                    const c = stats.cell(s, t);
                    const best = c.length ? Math.max(...c.map((r) => r.score)) : null;
                    return <td key={t} className={best == null ? "dim" : best >= 65 ? "pos" : best < 50 ? "neg" : ""}
                      style={{ background: c.length ? `rgba(255,59,77,${Math.min(0.35, 0.05 + c.length / 40)})` : undefined }}>
                      {c.length ? <>{c.length} <span className="dim">· {best?.toFixed(0)}</span></> : "·"}</td>;
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="ins-box">
          <div className="label">By type <span className="dim">count · median Sharpe</span></div>
          {stats.fams.map((x) => (
            <div key={x.f} className="ins-bar"><span>{FAMILY_NAMES[x.f] ?? x.f}</span>
              <span className="m-bar"><span className="pos-bg" style={{ width: `${(x.n / maxFam) * 100}%` }} /></span>
              <b>{x.n}</b><span className="dim">{fmt.num(x.med)}</span></div>
          ))}
          <div className="label" style={{ marginTop: 12 }}>By grade</div>
          {stats.grades.map((x) => (
            <div key={x.g} className="ins-bar"><span className={`grade ${x.g}`}>{x.g}</span>
              <span className="m-bar"><span className="pos-bg" style={{ width: `${(x.n / maxGrade) * 100}%` }} /></span>
              <b>{x.n}</b><span /></div>
          ))}
        </div>

        <div className="ins-box">
          <div className="label">Top strategies <span className="dim">by score · click to inspect</span></div>
          <table className="vt">
            <tbody>
              {stats.top.map((r) => (
                <tr key={r.id} onClick={() => onOpen(r.id)} style={{ cursor: "pointer" }}>
                  <td className="l"><span className={`grade ${r.grade}`}>{r.grade}</span></td>
                  <td className="l">{r.symbol} {r.timeframe}</td>
                  <td className="l dim">{FAMILY_NAMES[r.family] ?? r.family}</td>
                  <td>{fmt.num(r.score, 1)}</td>
                  <td className={r.total_return != null && r.total_return < 0 ? "neg" : "pos"}>{fmt.pct(r.total_return)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}
