// AUTOMATION (DECK): the launch queue. Launches wait their turn in a file under the data folder, so the queue
// outlives the window and the engine; when a launch finishes the next one starts (a stopped launch halts the queue).

import { useCallback, useEffect, useState } from "react";
import { type AfterStep, api, type QueueItem } from "../api/ws";
import { errText, fmt } from "../ws/util";

const STEPS: { step: AfterStep; label: string; hint: string }[] = [
  { step: "purge_copies", label: "Purge copies", hint: "strategies that made exactly the same trades as a better one" },
  { step: "purge_held_out_failed", label: "Purge failed held-out", hint: "lost money or under 15 trades on the held-back bars" },
  { step: "purge_grade_f", label: "Purge grade F", hint: "V2 strategies graded F" },
  { step: "repeat", label: "Repeat", hint: "queue the same launch again when it finishes" },
];

export function Automation({ canQueue, queueWhy, onQueueCurrent }: {
  canQueue: boolean;
  queueWhy: string | null;
  onQueueCurrent: (after: AfterStep[]) => Promise<void>;
}) {
  const [after, setAfter] = useState<AfterStep[]>([]);
  const [items, setItems] = useState<QueueItem[]>([]);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [note, setNote] = useState<string | null>(null);
  const load = useCallback(async () => {
    try {
      const q = await api.queue();
      setItems(q.items);
      setBusy(q.busy);
      setErr(null);
    } catch (e) {
      setErr(errText(e));
    }
  }, []);
  useEffect(() => {
    void load();
    const t = setInterval(() => void load(), 8000);
    return () => clearInterval(t);
  }, [load]);

  return (
    <section className="panel grow" aria-label="Launch queue">
      <div className="panel-h">Launch queue <small>{busy ? "a launch is running; the next starts when it finishes" : "nothing running"}</small>
        <span className="sp" />
        <button type="button" className="btn btn-sm" disabled={!canQueue} title={queueWhy ?? "Add the campaign set up in COMMAND to the queue"}
          onClick={() => void onQueueCurrent(after).then(load).catch((e) => setErr(errText(e)))}>Queue the COMMAND campaign</button>
        <button type="button" className="btn btn-sm btn-primary" disabled={busy || !items.length}
          onClick={() => void api.queueStart().then((r) => { setNote(r.started ? `Started ${r.started}.` : r.detail ?? "Nothing started."); void load(); }).catch((e) => setErr(errText(e)))}>Run the queue</button>
      </div>
      <div className="panel-b">
        <div className="row" style={{ gap: 14, flexWrap: "wrap", marginBottom: 10 }} aria-label="After each queued launch">
          <span className="label">After it finishes</span>
          {STEPS.map((s) => (
            <label key={s.step} title={s.hint} className="row" style={{ gap: 6 }}>
              <input type="checkbox" checked={after.includes(s.step)}
                onChange={() => setAfter(after.includes(s.step) ? after.filter((x) => x !== s.step) : [...after, s.step])} />
              {s.label}
            </label>
          ))}
          <span className="hint">purges go to the recycle bin (30 days); a stopped launch runs no step</span>
        </div>
        {err && <div className="err">{err}</div>}
        {note && <div className="hint">{note}</div>}
        {!items.length ? (
          <div className="empty"><h3>The queue is empty</h3><p>Set up a campaign in COMMAND, then press "Queue the COMMAND campaign" here.
            Queued launches run one after another, even with the window closed (keep the engine running when you close ARQVIO).</p></div>
        ) : (
          <table className="vt">
            <thead><tr><th className="l">#</th><th className="l">Markets</th><th className="l">Timeframes</th><th>Years</th><th className="l">Depth</th><th className="l">Power</th><th className="l">Queued</th><th /></tr></thead>
            <tbody>
              {items.map((q, i) => (
                <tr key={q.id}>
                  <td className="l">{i + 1}</td>
                  <td className="l">{q.request.symbols.join(", ")}</td>
                  <td className="l">{q.request.timeframes.join(", ")}</td>
                  <td>{q.request.years}</td>
                  <td className="l">{q.request.depth}</td>
                  <td className="l">{q.request.power}{q.request.after?.length ? ` · then ${q.request.after.join(", ").replace(/_/g, " ")}` : ""}</td>
                  <td className="l dim">{fmt.time(q.utc)}</td>
                  <td><button type="button" className="btn btn-sm" onClick={() => void api.dequeue(q.id).then(load).catch((e) => setErr(errText(e)))}>Remove</button></td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </section>
  );
}
