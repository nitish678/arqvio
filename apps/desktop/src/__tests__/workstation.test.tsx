// The workstation against a fake engine: first run, launch, the one log, the vault, the inspector, deploy.
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Workstation } from "../ws/Workstation";
import { DETAIL, fakeEngine, INSPECT, LAUNCH_IDLE, ROW, TERMINAL } from "./fakeEngine";

beforeEach(() => {
  localStorage.clear();
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

describe("first run", () => {
  it("says the engine is starting while it does not answer, and shows no numbers", async () => {
    render(<Workstation />);
    expect(await screen.findByText("Starting the engine…")).toBeTruthy();
    expect(document.querySelector(".vt")).toBeNull();
  });

  it("starts empty: no strategies, three main markets on H1 picked, LAUNCH ready", async () => {
    const eng = fakeEngine();
    render(<Workstation />);
    expect(await screen.findByText("No strategies yet")).toBeTruthy();
    await waitFor(() => expect(document.querySelectorAll(".asset.on").length).toBe(3));
    const on = [...document.querySelectorAll(".asset.on")].map((a) => a.textContent);
    expect(on.join(" ")).toContain("EURUSD");
    expect(on.join(" ")).toContain("XAUUSD");
    expect(screen.getByRole("button", { name: "H1" }).getAttribute("aria-pressed")).toBe("true");
    const off = [...document.querySelectorAll(".asset.off")].map((a) => a.textContent);
    expect(off).toEqual(["US30"]);  // not offered by this terminal: shown, but cannot be picked
    expect(eng.calls.some((c) => c.path === "/api/v2/app/terminals/t1/symbols")).toBe(true);
  });

  it("LAUNCH sends the chosen terminal, markets, timeframes, history, depth, power and engine", async () => {
    const eng = fakeEngine();
    render(<Workstation />);
    await waitFor(() => expect(document.querySelectorAll(".asset.on").length).toBe(3));
    fireEvent.click(screen.getByRole("button", { name: "Quick" }));
    fireEvent.click(screen.getByRole("button", { name: "M15" }));
    fireEvent.click(screen.getByRole("button", { name: "3y" }));
    const launch = screen.getByRole("button", { name: "Launch" });
    await waitFor(() => expect((launch as HTMLButtonElement).disabled).toBe(false));
    fireEvent.click(launch);
    await waitFor(() => expect(eng.calls.some((c) => c.method === "POST" && c.path === "/api/v2/app/launch")).toBe(true));
    const choose = eng.calls.findIndex((c) => c.path === "/api/v2/app/terminals/choose");
    const post = eng.calls.findIndex((c) => c.method === "POST" && c.path === "/api/v2/app/launch");
    expect(choose).toBeGreaterThanOrEqual(0);
    expect(choose).toBeLessThan(post);
    expect(eng.calls[post].body).toEqual({ terminal: "t1", symbols: ["EURUSDm", "GBPUSDm", "XAUUSDm"],
      timeframes: ["H1", "M15"], horizon: "research", years: 3, depth: "QUICK", power: "high", search: "standard" });
  });

  it("a closed terminal blocks LAUNCH and says what to do", async () => {
    fakeEngine({
      "GET /api/v2/app/terminals": () => ({ terminals: [{ ...TERMINAL, running: false, account: { state: "CLOSED", detail: "not running" } }],
        chosen: "t1", read_utc: "x" }),
    });
    render(<Workstation />);
    expect(await screen.findByText(/Open MetaTrader 5 \(C:\\mt5\\2\\terminal64\.exe\) and log in/)).toBeTruthy();
    expect((screen.getByRole("button", { name: "Launch" }) as HTMLButtonElement).disabled).toBe(true);
    expect(document.querySelector('[data-node="mt5"]')?.getAttribute("data-state")).toBe("bad");
  });
});

describe("the one log", () => {
  it("appends each engine line once, even when an answer is slower than the poll", async () => {
    let n = 0;
    fakeEngine({
      "GET /api/v2/app/log": async (_b, url) => {
        const after = Number(url.searchParams.get("after"));
        n += 1;
        await new Promise((r) => setTimeout(r, n === 1 ? 2000 : 10));  // the first answer outlives a poll interval
        return after < 2 ? { lines: [
          { seq: 1, utc: "2026-10-23T06:00:00Z", level: "OK", src: "engine", msg: "LAUNCH: EURUSDm H1" },
          { seq: 2, utc: "2026-10-23T06:00:01Z", level: "INFO", src: "data", msg: "Reading EURUSDm H1" },
        ].filter((l) => l.seq > after), last: 2 } : { lines: [], last: after };
      },
    });
    render(<Workstation />);
    await screen.findByText(/LAUNCH: EURUSDm H1/, undefined, { timeout: 6000 });
    await new Promise((r) => setTimeout(r, 3200));
    expect(screen.getAllByText(/LAUNCH: EURUSDm H1/)).toHaveLength(1);
    expect(screen.getAllByText(/Reading EURUSDm H1/)).toHaveLength(1);
  });
});

describe("vault, inspector, deploy", () => {
  const withRow = () => fakeEngine({
    "GET /api/v2/app/library": () => ({ strategies: [ROW], counts: { CANDIDATE: 1 }, robustness_running: null }),
    [`GET /api/v2/app/library/strategies/${encodeURIComponent(ROW.id)}`]: () => DETAIL,
    [`GET /api/v2/app/library/strategies/${encodeURIComponent(ROW.id)}/inspect`]: () => INSPECT,
    "POST /api/v2/app/deploy/go": () => ({ bot: "arqvio_rsi_x", symbol: "EURUSDm", terminal: TERMINAL.path }),
  });

  it("a vault row opens the inspector with the engine's numbers", async () => {
    withRow();
    render(<Workstation />);
    const cell = await screen.findByText("ABCDEF12");
    fireEvent.click(cell);
    const insp = await screen.findByLabelText("Inspector");
    await waitFor(() => expect(within(insp).getByText("52.9%")).toBeTruthy());  // win rate from the re-run
    expect(within(insp).getByText("1.82")).toBeTruthy();  // profit factor
    expect(within(insp).getByText("8 / 26")).toBeTruthy();  // long / short
    expect(screen.getByText("matches breeding")).toBeTruthy();
  });

  it("a failed re-run shows why, never a number", async () => {
    fakeEngine({
      "GET /api/v2/app/library": () => ({ strategies: [ROW], counts: {}, robustness_running: null }),
      [`GET /api/v2/app/library/strategies/${encodeURIComponent(ROW.id)}`]: () => DETAIL,
      [`GET /api/v2/app/library/strategies/${encodeURIComponent(ROW.id)}/inspect`]: () => { throw new Error("REFUSED: the strategy could not be re-run: no data"); },
    });
    render(<Workstation />);
    fireEvent.click(await screen.findByText("ABCDEF12"));
    expect(await screen.findByText(/Trade statistics: the strategy could not be re-run/)).toBeTruthy();
    const insp = screen.getByLabelText("Inspector");
    expect(within(insp).queryByText("52.9%")).toBeNull();
  });

  it("Deploy asks which account and sends the strategy, the terminal and the guardrails", async () => {
    const eng = withRow();
    render(<Workstation />);
    fireEvent.click(await screen.findByText("ABCDEF12"));
    const deploys = await screen.findAllByRole("button", { name: "Deploy" });
    fireEvent.click(deploys[0]);
    const dlg = await screen.findByRole("dialog", { name: "Deploy to MetaTrader 5" });
    expect(within(dlg).getByLabelText("Terminal to deploy to")).toBeTruthy();
    const lots = within(dlg).getAllByRole("spinbutton")[0];
    fireEvent.change(lots, { target: { value: "0.05" } });
    fireEvent.click(within(dlg).getByRole("button", { name: "Deploy and start" }));
    await waitFor(() => expect(eng.calls.some((c) => c.path === "/api/v2/app/deploy/go")).toBe(true));
    const go = eng.calls.find((c) => c.path === "/api/v2/app/deploy/go");
    expect(go?.body).toMatchObject({ vault_id: ROW.id, terminal: "t1", attach: true, guardrails: { lots: 0.05, daily_loss_pct: 3 } });
  });
});

describe("a running launch", () => {
  it("lights the pipeline and the ring from the engine's progress", async () => {
    fakeEngine({
      "GET /api/v2/app/launch": () => ({ ...LAUNCH_IDLE, busy: true, launch: {
        id: "l1", utc: "2026-10-23T06:00:00+00:00", state: "RUNNING", vaulted: 0,
        request: { terminal: "t1", symbols: ["EURUSDm"], timeframes: ["H1"], years: 3, depth: "QUICK", power: "high", terminal_path: "x" },
        args: { pop: 50, gen: 5, budget: 400, final_n: 20 },
        tasks: [
          { kind: "data", role: "market", symbol: "EURUSDm", timeframe: "H1", cell: "eurusdm_h1", state: "DONE", detail: "18,669 bars" },
          { kind: "breed", symbol: "EURUSDm", timeframe: "H1", cell: "eurusdm_h1", state: "RUNNING", detail: null,
            started_utc: "2026-10-23T06:01:00+00:00",
            progress: { generation: 1, generations: 5, trials: 100, budget: 400, best_sharpe: 1.35, alive: 30, utc: "2026-10-23T06:02:00+00:00" } },
        ] } }),
    });
    render(<Workstation />);
    await waitFor(() => expect(document.querySelector('[data-node="breed"]')?.getAttribute("data-state")).toBe("active"));
    expect(document.querySelector('[data-node="data"]')?.getAttribute("data-state")).toBe("ok");
    expect(screen.getByText("100 / 400")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Pause" })).toBeTruthy();
    expect(screen.getByText("~3 min")).toBeTruthy();  // 60 s for 100 tested, 300 to go: an estimate
    await act(async () => undefined);
  });
});
