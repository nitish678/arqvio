// ARQVIO desktop: boot screen, the first-run risk consent, the interface chooser, then DECK or CLASSIC; the close
// dialog when a campaign runs (ARQVIO_V4.md 12.2). Every value is the engine's.
import { useContext, useState } from "react";
import { Chooser, readChoice, type UiChoice, writeChoice } from "./shell/Chooser";
import { Consent, hasConsent, writeConsent } from "./shell/Consent";
import { markWelcomeSeen, Welcome, welcomeSeen } from "./shell/Welcome";
import { CtrlShiftAtStart, Shell } from "./shell/Shell";
import { Workstation } from "./ws/Workstation";

function Interface() {
  const ctrlShift = useContext(CtrlShiftAtStart);
  const [choice, setChoice] = useState<UiChoice | null>(() => readChoice());
  const [asking, setAsking] = useState<boolean>(() => ctrlShift || !readChoice()?.always);
  const [consent, setConsent] = useState<boolean>(() => hasConsent());
  const [welcomed, setWelcomed] = useState<boolean>(() => welcomeSeen());
  if (!consent) {
    const tauri = typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
    const quit = tauri ? () => void import("@tauri-apps/api/core").then(({ invoke }) => invoke("app_exit", { stop: false })) : null;
    return <Consent onAccept={() => { writeConsent(); setConsent(true); }} onQuit={quit} />;
  }
  if (asking || !choice) {
    return <Chooser initial={choice} onChoose={(c) => { writeChoice(c); setChoice(c); setAsking(false); }} />;
  }
  if (!welcomed) {
    // the last card's call to action lands on COMMAND, the first tab: pick a market, press LAUNCH
    return <Welcome onDone={() => { markWelcomeSeen(); setWelcomed(true); }} />;
  }
  const other = choice.layout === "deck" ? "classic" : "deck";
  return <Workstation key={choice.layout} layout={choice.layout}
    onSwitchLayout={() => { const c = { ...choice, layout: other } as UiChoice; writeChoice(c); setChoice(c); }} />;
}

export default function App() {
  return (
    <Shell>
      <Interface />
    </Shell>
  );
}
