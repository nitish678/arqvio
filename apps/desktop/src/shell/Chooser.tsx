// The interface chooser (ARQVIO_V4.md 12.2): shown on the first run, when Ctrl+Shift is held while ARQVIO starts, or
// when "Always use this" is off. Keyboard: Up/Down or 1/2 to pick, Enter to open. The choice lives in this browser's
// storage only; both interfaces read the same engine.

import { useEffect, useState } from "react";
import { BrandMark } from "../ui/BrandMark";
import type { Layout } from "../ws/Workstation";

const KEY = "arqvio.ui";

export interface UiChoice {
  layout: Layout;
  always: boolean;
}

export function readChoice(): UiChoice | null {
  try {
    const v = JSON.parse(localStorage.getItem(KEY) ?? "null") as UiChoice | null;
    return v && (v.layout === "deck" || v.layout === "classic") ? { layout: v.layout, always: !!v.always } : null;
  } catch {
    return null;
  }
}

export function writeChoice(c: UiChoice): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(c));
  } catch {
    /* private window: the choice simply is not remembered */
  }
}

const OPTIONS: { layout: Layout; title: string; tag: string; hint: string }[] = [
  { layout: "deck", title: "DECK", tag: "recommended", hint: "ARQVIO 4: sections in tabs — Command, Vault, Live" },
  { layout: "classic", title: "CLASSIC", tag: "v3 one screen", hint: "everything on one screen, as in ARQVIO 3" },
];

export function Chooser({ initial, onChoose }: { initial: UiChoice | null; onChoose: (c: UiChoice) => void }) {
  const [pick, setPick] = useState<number>(initial?.layout === "classic" ? 1 : 0);
  const [always, setAlways] = useState<boolean>(initial?.always ?? false);
  const open = () => onChoose({ layout: OPTIONS[pick].layout, always });

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "ArrowDown" || e.key === "2") setPick(1);
      else if (e.key === "ArrowUp" || e.key === "1") setPick(0);
      else if (e.key === "Enter") onChoose({ layout: OPTIONS[pick].layout, always });
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [pick, always, onChoose]);

  return (
    <div className="chooser">
      <section className="panel chooser-card" role="dialog" aria-label="Choose your interface">
        <div className="chooser-head">
          <BrandMark />
          <b style={{ letterSpacing: "0.24em" }}>ARQVIO</b>
          <span className="dim" style={{ marginLeft: "auto", fontSize: 11 }}>↑ ↓ · 1 2 · ENTER</span>
        </div>
        <div style={{ marginBottom: 12, fontWeight: 600 }}>Choose your interface</div>
        <div role="radiogroup" aria-label="Interface">
          {OPTIONS.map((o, i) => (
            <button key={o.layout} type="button" role="radio" aria-checked={pick === i} className="chooser-opt"
              onClick={() => setPick(i)} onDoubleClick={() => onChoose({ layout: o.layout, always })}>
              <span className="n">0{i + 1}</span>
              <span><b>{o.title} <small>· {o.tag}</small></b><small>{o.hint}</small></span>
            </button>
          ))}
        </div>
        <div className="row" style={{ justifyContent: "space-between", marginTop: 6 }}>
          <label className="row" style={{ gap: 8 }}>
            <input type="checkbox" checked={always} onChange={(e) => setAlways(e.target.checked)} />
            <span>Always use this <span className="dim">(hold Ctrl+Shift while ARQVIO starts to choose again)</span></span>
          </label>
          <button type="button" className="btn btn-primary" onClick={open}>Open</button>
        </div>
      </section>
    </div>
  );
}
