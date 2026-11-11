// Vault housekeeping: purge by a named rule (preview first, then move to the recycle bin) and the recycle bin itself
// (deleted strategies stay restorable for the engine's RECYCLE_DAYS).

import { useState } from "react";
import { api } from "../api/ws";
import { Layer } from "./Dialogs";
import { errText, fmt, shortId } from "./util";

const RULES: { rule: string; label: string }[] = [
  { rule: "grade_f", label: "Grade F" },
  { rule: "held_out_failed", label: "Failed held-out" },
  { rule: "copies", label: "Copies" },
];

export function VaultTools({ onChanged }: { onChanged: () => void }) {
  const [open, setOpen] = useState<null | "purge" | "bin">(null);
  const [rule, setRule] = useState(RULES[0].rule);
  const [preview, setPreview] = useState<{ what: string; ids: string[] } | null>(null);
  const [bin, setBin] = useState<Awaited<ReturnType<typeof api.recycle>> | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const look = async (r: string) => {
    setRule(r);
    setErr(null);
    try {
      const p = await api.purge(r, false);
      setPreview({ what: p.what, ids: p.ids });
    } catch (e) {
      setErr(errText(e));
    }
  };
  // GET JSON (research vault export): the whole vault as one file, every strategy with its address.
  const exportJson = async () => {
    setErr(null);
    try {
      const doc = await api.vaultExport();
      const url = URL.createObjectURL(new Blob([JSON.stringify(doc, null, 1)], { type: "application/json" }));
      const a = document.createElement("a");
      a.href = url;
      a.download = `arqvio-vault-${doc.exported_utc.slice(0, 10)}.json`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      setErr(errText(e));
    }
  };
  const openBin = async () => {
    setOpen("bin");
    setErr(null);
    try {
      setBin(await api.recycle());
    } catch (e) {
      setErr(errText(e));
    }
  };
  const run = async (fn: () => Promise<unknown>) => {
    setBusy(true);
    try {
      await fn();
      onChanged();
    } catch (e) {
      setErr(errText(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <button type="button" className="btn btn-sm" onClick={() => { setOpen("purge"); void look(rule); }}>Purge…</button>
      <button type="button" className="btn btn-sm" onClick={() => void openBin()}>Recycle bin</button>
      <button type="button" className="btn btn-sm" onClick={() => void exportJson()}
        title="Save the whole vault as one JSON file: every strategy, its numbers and where they come from">Export JSON</button>
      {open && (
        <Layer>
          <div className="scrim" role="dialog" aria-modal="true" aria-label={open === "purge" ? "Purge" : "Recycle bin"}
            onMouseDown={(e) => e.target === e.currentTarget && !busy && setOpen(null)}>
            <section className="panel dialog" style={{ maxWidth: 620 }}>
              {open === "purge" ? (
                <>
                  <div className="panel-h">Purge <small>removed strategies go to the recycle bin</small></div>
                  <div className="panel-b" style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                    <div className="tabs" role="tablist">
                      {RULES.map((r) => <button key={r.rule} type="button" role="tab" className="tab" aria-selected={rule === r.rule} onClick={() => void look(r.rule)}>{r.label}</button>)}
                    </div>
                    {preview && <div>{preview.ids.length ? `${preview.ids.length} strateg${preview.ids.length === 1 ? "y" : "ies"} match: ${preview.what}.` : `Nothing matches: ${preview.what}.`}</div>}
                    {preview && preview.ids.length > 0 && <div className="hint">{preview.ids.slice(0, 12).map(shortId).join(", ")}{preview.ids.length > 12 ? " …" : ""}</div>}
                    {err && <div className="err">{err}</div>}
                    <div className="row" style={{ justifyContent: "flex-end", gap: 8 }}>
                      <button type="button" className="btn" onClick={() => setOpen(null)} disabled={busy}>Close</button>
                      <button type="button" className="btn btn-primary" disabled={busy || !preview?.ids.length}
                        onClick={() => void run(async () => { await api.purge(rule, true); await look(rule); })}>Move {preview?.ids.length ?? 0} to the recycle bin</button>
                    </div>
                  </div>
                </>
              ) : (
                <>
                  <div className="panel-h">Recycle bin <small>{bin ? `restorable for ${bin.days} days` : ""}</small></div>
                  <div className="panel-b" style={{ maxHeight: 420, overflowY: "auto" }}>
                    {err && <div className="err">{err}</div>}
                    {bin && !bin.strategies.length && <div className="hint">Empty.</div>}
                    {bin?.strategies.map((r) => (
                      <div key={r.id} className="row" style={{ justifyContent: "space-between", gap: 8, padding: "4px 0" }}>
                        <span>{r.symbol} {r.timeframe} · {shortId(r.id)} <span className="dim">· {fmt.time(r.deleted_utc)} · {r.deleted_reason}</span></span>
                        <button type="button" className="btn btn-sm" disabled={busy} onClick={() => void run(async () => { await api.restore(r.id); setBin(await api.recycle()); })}>Restore</button>
                      </div>
                    ))}
                  </div>
                  <div className="dialog-f"><button type="button" className="btn" onClick={() => setOpen(null)}>Close</button></div>
                </>
              )}
            </section>
          </div>
        </Layer>
      )}
    </>
  );
}
