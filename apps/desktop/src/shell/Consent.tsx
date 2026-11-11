// The first-run risk consent (idea: research APP-5 "DISCLAIMER: first-run full-screen risk-consent gate -- the deck
// must not be usable before consent"; ARQVIO's own words, no copy). Shown once per version of the text; the answer
// lives in this browser's storage. Quit closes the app (in the desktop shell) without touching the engine's work.

import { useEffect, useRef } from "react";
import { BrandMark } from "../ui/BrandMark";

const KEY = "arqvio.consent";
export const CONSENT_VERSION = 1;

export function hasConsent(): boolean {
  try {
    return Number(localStorage.getItem(KEY)) >= CONSENT_VERSION;
  } catch {
    return false;
  }
}

export function writeConsent(): void {
  try {
    localStorage.setItem(KEY, String(CONSENT_VERSION));
  } catch {
    /* private window: asked again next time */
  }
}

const POINTS = [
  "ARQVIO breeds and tests trading strategies on past prices. A strategy that did well on past data can lose money "
    + "from the first trade it takes on a live account.",
  "Every number ARQVIO shows is a backtest or a measurement of one; none is a promise. ARQVIO guarantees its process, "
    + "never a profit.",
  "Nothing in ARQVIO is financial advice. You decide what runs on your account, with what risk, and you carry the result.",
  "Trade a new strategy on a demo account first. Only risk money you can afford to lose.",
];

export function Consent({ onAccept, onQuit }: { onAccept: () => void; onQuit: (() => void) | null }) {
  const accept = useRef<HTMLButtonElement>(null);
  useEffect(() => accept.current?.focus(), []);
  return (
    <div className="boot" role="dialog" aria-modal="true" aria-label="Before you start: trading risk">
      <div className="boot-card panel" style={{ maxWidth: 620 }}>
        <div className="boot-mark"><BrandMark /></div>
        <div className="boot-title">ARQVIO <span className="dim">// BEFORE YOU START</span></div>
        <ol className="consent-list">
          {POINTS.map((p) => <li key={p}>{p}</li>)}
        </ol>
        <div className="row" style={{ gap: 8, justifyContent: "flex-end", marginTop: 14 }}>
          {onQuit && <button type="button" className="btn" onClick={onQuit}>Quit ARQVIO</button>}
          <button ref={accept} type="button" className="btn btn-primary" onClick={onAccept}>
            I understand the risk — continue
          </button>
        </div>
      </div>
    </div>
  );
}
