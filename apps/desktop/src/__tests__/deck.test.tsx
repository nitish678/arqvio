// DECK: real sections in tabs, unbuilt sections shown but never clickable; the chooser by keyboard.
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Chooser } from "../shell/Chooser";
import { Workstation } from "../ws/Workstation";
import { fakeEngine } from "./fakeEngine";

beforeEach(() => localStorage.clear());
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("DECK", () => {
  it("opens on COMMAND with the launch controls; VAULT and LIVE switch sections; later tabs are disabled", async () => {
    fakeEngine();
    render(<Workstation layout="deck" />);
    expect(await screen.findByRole("tab", { name: "COMMAND" })).toBeTruthy();
    expect(screen.getByRole("tab", { name: "COMMAND" }).getAttribute("aria-selected")).toBe("true");
    fireEvent.click(screen.getByRole("tab", { name: "VAULT" }));
    expect(await screen.findByText("No strategies yet")).toBeTruthy();
    fireEvent.click(screen.getByRole("tab", { name: "LIVE" }));
    expect(await screen.findByLabelText("Live bots")).toBeTruthy();
    fireEvent.click(screen.getByRole("tab", { name: "INSIGHTS" }));
    expect(await screen.findByText(/Insights are counted from the vault/)).toBeTruthy();
    fireEvent.click(screen.getByRole("tab", { name: "AUTOMATION" }));
    expect(await screen.findByLabelText("Launch queue")).toBeTruthy();
    fireEvent.click(screen.getByRole("tab", { name: "MAP" }));
    expect(await screen.findByText(/The map draws the vault/)).toBeTruthy();
    fireEvent.click(screen.getByRole("tab", { name: "GUIDE" }));
    expect(await screen.findByText("ARQVIO in eight steps")).toBeTruthy();
  });
});

describe("interface chooser", () => {
  it("picks with the keyboard and remembers 'Always use this'", () => {
    const onChoose = vi.fn();
    render(<Chooser initial={null} onChoose={onChoose} />);
    expect(screen.getByRole("radio", { name: /DECK/ }).getAttribute("aria-checked")).toBe("true");
    fireEvent.keyDown(window, { key: "2" });
    fireEvent.click(screen.getByRole("checkbox"));
    fireEvent.keyDown(window, { key: "Enter" });
    expect(onChoose).toHaveBeenCalledWith({ layout: "classic", always: true });
  });
});

describe("CAMPAIGN generation fan and trade tape", () => {
  it("draws the survivors' legend and lists backtest trades with their exit reason", async () => {
    const { GenerationFan } = await import("../deck/GenerationFan");
    const { TradeTape } = await import("../deck/TradeTape");
    const view = {
      task: { symbol: "XAUUSDm", timeframe: "H1", state: "RUNNING", engine: "v1" as const, campaign_id: "c1" },
      live: { generation: 4, generations: 20, phase: "TRIBES", survivors: 2, utc: "2026-11-01T10:00:00+00:00",
        bots: [{ id: "AB12CD", tribe: "TRIBE A", ret: 61.5, dd: 4.2, fit: 12.9, curve: [0, 5, 61.5] },
          { id: "EF34GH", tribe: "TRIBE B", ret: -3.1, dd: 8, fit: 4.1, curve: [0, -1, -3.1] }],
        tape: [{ bot: "AB12CD", dir: 1, bar: 900, pct: 0.98, reason: "TP" }, { bot: "EF34GH", dir: -1, bar: 880, pct: -1.02, reason: "SL" }] },
    };
    render(<><GenerationFan view={view} /><TradeTape view={view} running /></>);
    expect(screen.getAllByText("#AB12CD").length).toBe(2);
    expect(screen.getByText("+61.5%")).toBeTruthy();
    expect(screen.getByText("-1.02%")).toBeTruthy();
    expect(screen.getByText("SL")).toBeTruthy();
    expect(screen.getByText(/GEN 4 · 2 backtest trades/)).toBeTruthy();
  });

  it("says what fills the tape when there is no generation yet", async () => {
    const { TradeTape } = await import("../deck/TradeTape");
    render(<TradeTape view={{ task: null, live: null }} running={false} />);
    expect(screen.getByText(/TRADES APPEAR AFTER THE FIRST GENERATION/)).toBeTruthy();
  });
});

describe("AUTOMATION", () => {
  const AUTO = { id: "au1", name: "Metals night", enabled: true, every_hours: 24, every_text: "EVERY 24H", terminal: "AUTO",
    steps: [{ kind: "campaign" }, { kind: "portfolio" }, { kind: "purge" }], history: [{ ts: "2026-11-01T10:00:00+00:00", ok: true, detail: "finished all 3 steps" }],
    next_at: "2099-01-01T00:00:00+00:00", run_now: false, last_run_utc: null, alongside: false };
  const LIST = { automations: [AUTO], log: [{ ts: "2026-11-01T10:00:00+00:00", id: "au1", name: "Metals night", status: "ok", detail: "step 2: added 3 member(s) to portfolio 'Auto Metals' (now 3 of 5)" }],
    groups: [{ id: "fx_majors", label: "FX MAJORS", count: 7 }, { id: "energy", label: "ENERGY", count: 0 }], timeframes: ["H1", "H4"],
    kinds: ["campaign", "portfolio", "parity", "purge", "replace", "reassess"], built: ["campaign", "portfolio", "parity", "purge"],
    planned_why: { replace: "swaps live bots", reassess: "needs a refresh" },
    defaults: { campaign: { kind: "campaign", preset_groups: ["fx_majors"], extra_symbols: [], timeframes: ["H1"], overrides: {} },
      portfolio: { kind: "portfolio", name: "" }, parity: { kind: "parity", pass_mark: "B", tests_per_run: 10 }, purge: { kind: "purge", criteria: {} },
      replace: { kind: "replace" }, reassess: { kind: "reassess" } },
    launch_defaults: { years: 3, depth: "QUICK", power: "medium", search: "standard" }, terminals: [], portfolios: [] };

  it("shows the run in progress, the list and the activity, and opens the editor with research's save messages", async () => {
    const { AutomationPage } = await import("../deck/AutomationPage");
    const eng = fakeEngine({
      "GET /api/v2/app/automation": () => LIST,
      "GET /api/v2/app/automation/state": () => ({ running: { automation_id: "au1", name: "Metals night", step_idx: 1, steps_total: 3, phase: "PORTFOLIO", progress: null } }),
      "GET /api/v2/app/launch/queue": () => ({ items: [], busy: false }),
      "POST /api/v2/app/automation/save": () => { throw Object.assign(new Error("x"), { status: 422, detail: "NAME THIS AUTOMATION; STEP 1: PICK AT LEAST ONE TIMEFRAME" }); },
    });
    render(<AutomationPage canQueue queueWhy={null} onQueueCurrent={async () => undefined} />);
    expect(await screen.findByText("METALS NIGHT")).toBeTruthy();
    expect(await screen.findByText("STEP 2 / 3")).toBeTruthy();
    expect(screen.getByText(/added 3 member\(s\) to portfolio 'Auto Metals'/)).toBeTruthy();
    fireEvent.click(screen.getByText("+ NEW AUTOMATION"));
    expect(screen.getByPlaceholderText("NAME IT")).toBeTruthy();
    expect(screen.getByText("+ REPLACE").hasAttribute("disabled")).toBe(true);
    expect(screen.getByText("ENERGY 0").hasAttribute("disabled")).toBe(true);
    fireEvent.click(screen.getByText("SAVE"));
    expect(eng.calls.some((c) => c.path === "/api/v2/app/automation/save")).toBe(true);
  });
});

describe("VAULT (DECK)", () => {
  it("selects several with Ctrl+click, opens the right-click menu for N SELECTED, and names what each filter hides", async () => {
    localStorage.removeItem("arqvio.vault.view");
    const { VaultTab } = await import("../deck/VaultTab");
    const { ROW } = await import("./fakeEngine");
    fakeEngine({ "GET /api/v2/app/portfolios": () => ({ portfolios: [{ id: "pf1", name: "Team A", created_utc: "", members: [] }] }) });
    const rows = [
      { ...ROW, id: "aaaaaaaa11111111@eurusdm_h1", score: 70, broker: "Exness-MT5Trial6", copies: 1, copy_of: null },
      { ...ROW, id: "bbbbbbbb22222222@eurusdm_h1", score: 60, broker: "AiwaCapital-Server", copies: 1, copy_of: null },
      { ...ROW, id: "cccccccc33333333@eurusdm_h1", score: 50, broker: "AiwaCapital-Server", copies: 1, copy_of: null },
    ];
    render(<VaultTab vault={{ strategies: rows as never, counts: {}, robustness_running: null }} vaultError={null} selected={null}
      onSelect={() => undefined} onDeploy={() => undefined} onChanged={() => undefined} />);
    expect(screen.getByText(/3 STRATEGIES/)).toBeTruthy();
    fireEvent.click(screen.getByText("#AAAAAAAA"));
    fireEvent.click(screen.getByText("#BBBBBBBB"), { ctrlKey: true });
    expect(screen.getByText(/2 SELECTED/)).toBeTruthy();
    fireEvent.contextMenu(screen.getByText("#BBBBBBBB"));
    expect(screen.getByText("DELETE · 2 SELECTED")).toBeTruthy();
    expect(screen.getByText("DEPLOY").hasAttribute("disabled")).toBe(true);  // one strategy at a time
    expect(await within(screen.getByRole("menu")).findByText("Team A")).toBeTruthy();
    fireEvent.keyDown(window, { key: "Escape" });
    fireEvent.change(screen.getByLabelText("Broker"), { target: { value: "AiwaCapital-Server" } });
    expect(screen.getByText("BROKER AiwaCapital-Server · hides 1 ✕")).toBeTruthy();
  });
});

describe("MAP", () => {
  it("groups the vault into archetype clusters, filters by symbol, and opens a cluster's callout", async () => {
    localStorage.removeItem("arqvio.map.view");
    const { MapPage } = await import("../deck/MapPage");
    const { ROW } = await import("./fakeEngine");
    const rows = [
      { ...ROW, id: "a1@x", family: "MA_CROSS", symbol: "EURUSDm", score: 70, total_return: 0.4, copy_of: null },
      { ...ROW, id: "a2@x", family: "MA_CROSS", symbol: "XAUUSDm", score: 50, total_return: 0.1, copy_of: null },
      { ...ROW, id: "b1@x", family: "MACD_CROSS", symbol: "EURUSDm", score: 60, total_return: 0.2, copy_of: null },
      { ...ROW, id: "c1@x", family: "MACD_CROSS", symbol: "EURUSDm", score: 60, total_return: 0.2, copy_of: "b1@x" },
    ];
    render(<MapPage rows={rows as never} onOpen={() => undefined} />);
    expect(document.querySelector(".mapp-count")?.textContent).toBe("3 STRATEGIES");  // the copy is drawn once
    fireEvent.click(screen.getByText("TREND FOLLOWER"));
    expect(screen.getByRole("dialog", { name: "TREND FOLLOWER cluster" }).textContent).toContain("2 · 67% of the view");
    fireEvent.change(screen.getByLabelText("Symbol"), { target: { value: "XAUUSDm" } });
    expect(document.querySelector(".mapp-count")?.textContent).toBe("1 STRATEGIES");
    fireEvent.click(screen.getByText("SCATTER"));
    expect(screen.getByLabelText("Strategy map")).toBeTruthy();
  });
});
