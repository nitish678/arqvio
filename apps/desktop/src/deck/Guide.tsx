// GUIDE (DECK): ARQVIO in eight steps, each with the button that does it. ARQVIO's own words.

const STEPS: { title: string; body: string; tab?: string }[] = [
  { title: "Open MetaTrader 5 and log in", body: "ARQVIO reads bars, spreads and symbol specs from a running MetaTrader 5. Any broker, demo or real: the EA trades the account it is attached to." },
  { title: "Pick markets, timeframes and the engine", body: "In COMMAND choose the terminal, the markets (or Browse for any symbol) and the timeframes. Engine: V1 breeds by tribes (16 seeds, 3,000 proving-ground strategies, two tribes for 20 generations, a Final War); V2 adds strict validation and is the only one that certifies. Depth and power: how hard to search.", tab: "COMMAND" },
  { title: "Press LAUNCH", body: "Data first (the last 30% of each window is held back), then a safety check, then breeding on the first 70%. Every step lands in the one log; you can pause or stop at the next checkpoint.", tab: "COMMAND" },
  { title: "Read the vault", body: "Survivors land in VAULT, best first. V2: grade is the score; CERTIFIED means every strict test passed: deflated Sharpe, the random-entry sham, the second market, robustness, profit not from one month, and the held-out bars. V1: the research score (55 and up GOOD) and UNCERTIFIED until you press Certify (V2) in its inspector.", tab: "VAULT" },
  { title: "Inspect one strategy", body: "Click a row: trade statistics, the equity curve, monthly profit with the months the spread gate blocked, and every test with its number. Run Check robustness for the last tests.", tab: "VAULT" },
  { title: "Deploy to MetaTrader 5", body: "Deploy compiles the EA (MetaEditor 0 errors, 0 warnings), installs it in the terminal you choose and opens it on its chart. It starts at 25% of your lot (incubation).", tab: "VAULT" },
  { title: "Watch it live", body: "LIVE lists every bot from its own heartbeat. On the chart in MetaTrader 5 the ARQVIO panel shows the same status and its reason: LIVE, WAITING FOR SIGNAL, BLOCKED (and why), STANDBY.", tab: "LIVE" },
  { title: "Scale up or stop", body: "Raise the lot 25% → 50% → 100% as the bot earns your trust; Stop all bots flattens every ARQVIO position and stands the bots by. Closing ARQVIO never stops MetaTrader 5.", tab: "LIVE" },
];

export function Guide({ onGo }: { onGo: (tab: string) => void }) {
  return (
    <section className="panel grow" aria-label="Guide">
      <div className="panel-h">Guide <small>ARQVIO in eight steps</small></div>
      <div className="panel-b guide">
        {STEPS.map((s, i) => (
          <div key={s.title} className="guide-step">
            <span className="n">{String(i + 1).padStart(2, "0")}</span>
            <div><b>{s.title}</b><p>{s.body}</p></div>
            {s.tab ? <button type="button" className="btn btn-sm" onClick={() => onGo(s.tab as string)}>{s.tab}</button> : <span />}
          </div>
        ))}
      </div>
    </section>
  );
}
