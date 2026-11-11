// First-run welcome (idea: research doc 06 §5 -- interface choice first, then a 3-card welcome, then the first-campaign
// call to action; ARQVIO's own words). Shown once, after the risk consent and the interface choice; Skip ends it.

import { useEffect, useRef, useState } from "react";
import { BrandMark } from "../ui/BrandMark";

const KEY = "arqvio.welcome";

export function welcomeSeen(): boolean {
  try {
    return localStorage.getItem(KEY) === "1";
  } catch {
    return true;  // no storage: never nag on every start
  }
}

export function markWelcomeSeen(): void {
  try {
    localStorage.setItem(KEY, "1");
  } catch {
    /* private window */
  }
}

const CARDS: { title: string; body: string }[] = [
  { title: "A strategy factory", body: "ARQVIO breeds trading strategies on your broker's own prices, tests them hard and keeps only the few that survive. It does not trade by itself: you decide what goes live." },
  { title: "Two engines", body: "V1 breeds by tribes: two tribes, twenty generations, a Final War. V2 adds strict validation and is the only one that certifies. Breed with V1, certify with V2." },
  { title: "From vault to chart", body: "Survivors land in the VAULT with every test and its number. Deploy compiles one into a MetaTrader 5 Expert Advisor that starts small, and LIVE shows what it does and why." },
];

export function Welcome({ onDone }: { onDone: (startCampaign: boolean) => void }) {
  const [i, setI] = useState(0);
  const next = useRef<HTMLButtonElement>(null);
  useEffect(() => next.current?.focus(), [i]);
  const last = i === CARDS.length - 1;
  const c = CARDS[i];
  return (
    <div className="boot" role="dialog" aria-modal="true" aria-label="Welcome to ARQVIO">
      <div className="boot-card panel" style={{ maxWidth: 560 }}>
        <div className="boot-mark"><BrandMark /></div>
        <div className="boot-title">ARQVIO <span className="dim">// {String(i + 1).padStart(2, "0")} / {String(CARDS.length).padStart(2, "0")}</span></div>
        <h2 style={{ margin: "12px 0 6px", fontSize: 18 }}>{c.title}</h2>
        <p style={{ margin: 0, lineHeight: 1.55 }}>{c.body}</p>
        <div className="row" style={{ gap: 8, justifyContent: "space-between", marginTop: 16 }}>
          <button type="button" className="btn" onClick={() => onDone(false)}>Skip</button>
          <span className="row" style={{ gap: 8 }}>
            {i > 0 && <button type="button" className="btn" onClick={() => setI(i - 1)}>Back</button>}
            <button ref={next} type="button" className="btn btn-primary" onClick={() => (last ? onDone(true) : setI(i + 1))}>
              {last ? "Pick my first market" : "Next"}
            </button>
          </span>
        </div>
      </div>
    </div>
  );
}
