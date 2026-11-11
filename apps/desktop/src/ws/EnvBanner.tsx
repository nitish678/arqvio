// The environment banner (idea: research doc 06 §2 -- one amber line on every page, the most important problem only,
// priority srvdown > metaquotes > notconnected > nomt5 > nobroker; "activation guidance, not an upsell"). ARQVIO has
// no server of its own on the internet, so "offline" is not a state here; "srvdown" is its own engine not answering.
// Dismiss hides one state for this session; a different state shows again.

import { useState } from "react";
import type { Terminal } from "../api/ws";

export type EnvState = "srvdown" | "metaquotes" | "notconnected" | "nomt5" | "nobroker";

export function envState(engineUp: boolean | null, terms: Terminal[], term: Terminal | null): EnvState | null {
  if (engineUp === false) return "srvdown";
  if (engineUp === null) return null;
  if (term && term.account.state === "OK" && /metaquotes/i.test(`${term.account.server ?? ""} ${term.account.company ?? ""}`)) {
    return "metaquotes";
  }
  if (term && term.account.state === "OK" && term.account.connected === false) return "notconnected";
  if (!terms.length) return "nomt5";
  if (!term || term.account.state !== "OK") return "nobroker";
  return null;
}

const TEXT: Record<EnvState, string> = {
  srvdown: "The ARQVIO engine is not answering. It restarts with the app; if this stays, Settings › Support › Copy support report.",
  metaquotes: "This MetaTrader 5 is on a MetaQuotes demo server: its prices are synthetic. Log in to your broker's demo or real account for real spreads.",
  notconnected: "MetaTrader 5 is open but not connected to its broker: check the connection icon at the bottom right of MetaTrader 5.",
  nomt5: "No MetaTrader 5 found on this computer. Install it from your broker, open it once and log in.",
  nobroker: "Open MetaTrader 5 and log in to your broker account: ARQVIO reads bars, spreads and symbols from it.",
};

export function EnvBanner({ state }: { state: EnvState | null }) {
  const [hidden, setHidden] = useState<EnvState | null>(null);
  if (!state || hidden === state) return null;
  return (
    <div className="env-banner" role="status">
      <span>{TEXT[state]}</span>
      <button type="button" className="btn btn-sm" aria-label="Dismiss for this session" onClick={() => setHidden(state)}>✕</button>
    </div>
  );
}
