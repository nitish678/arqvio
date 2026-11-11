// The boot screen: each line is an engine check, a failed check is red with its reason, the app opens after the checks.
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Shell } from "../shell/Shell";
import { fakeEngine } from "./fakeEngine";

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

const CHECKS = [
  { check: "engine", ok: true, detail: "ARQVIO engine 3.0.1, up 1.2 s", ms: 0 },
  { check: "data folder", ok: true, detail: "C:\\Users\\me\\ARQVIO", ms: 2 },
  { check: "MetaTrader 5", ok: false, detail: "no MetaTrader 5 terminal found (data and deploy need one)", ms: 30 },
  { check: "vault", ok: true, detail: "40 strategies, 0 certified", ms: 9 },
  { check: "gene registry", ok: true, detail: "90 genes, 11 ready to breed", ms: 1 },
  { check: "campaign", ok: true, detail: "none running", ms: 0 },
];

describe("boot screen", () => {
  it("waits for the engine and says so, without numbers", async () => {
    render(<Shell><div>APP</div></Shell>);
    expect(await screen.findByText(/waiting for the engine/)).toBeTruthy();
    expect(screen.queryByText("APP")).toBeNull();
  });

  it("shows every check the engine ran, a failed one in red, then opens the app", async () => {
    fakeEngine({ "GET /api/v2/app/boot": () => ({ version: "3.0.1", checks: CHECKS, ok: true }) });
    render(<Shell><div>APP</div></Shell>);
    expect(await screen.findByText("90 genes, 11 ready to breed")).toBeTruthy();
    const bad = screen.getByText(/no MetaTrader 5 terminal found/).closest("li");
    expect(bad?.className).toBe("bad");
    // the lines print one after another; all six arrive, five passed
    await waitFor(() => expect(document.querySelectorAll(".boot-lines li.ok").length).toBe(5));
    expect(await screen.findByText(/ready with 1 warning/)).toBeTruthy();
    await waitFor(() => expect(screen.getByText("APP")).toBeTruthy(), { timeout: 3000 });
  });
});

describe("first-run risk consent", () => {
  it("blocks until accepted, remembers the answer, and offers no Quit outside the desktop shell", async () => {
    const { Consent, hasConsent, writeConsent } = await import("../shell/Consent");
    localStorage.removeItem("arqvio.consent");
    expect(hasConsent()).toBe(false);
    const accepted = vi.fn();
    render(<Consent onAccept={() => { writeConsent(); accepted(); }} onQuit={null} />);
    expect(screen.getByText(/never a profit/)).toBeTruthy();
    expect(screen.queryByText("Quit ARQVIO")).toBeNull();
    screen.getByText(/I understand the risk/).click();
    expect(accepted).toHaveBeenCalledOnce();
    expect(hasConsent()).toBe(true);
  });
});

describe("environment banner", () => {
  it("names the most important problem only, in research priority order", async () => {
    const { envState } = await import("../ws/EnvBanner");
    const ok = { id: "t", path: "", install_dir: "", data_dir: null, portable: false, running: true, remembered_server: null,
      account: { state: "OK" as const, server: "Broker-Demo", company: "Broker", connected: true } };
    expect(envState(false, [ok], ok)).toBe("srvdown");
    expect(envState(true, [], null)).toBe("nomt5");
    expect(envState(true, [ok], { ...ok, account: { ...ok.account, state: "CLOSED" as const } })).toBe("nobroker");
    expect(envState(true, [ok], { ...ok, account: { ...ok.account, server: "MetaQuotes-Demo" } })).toBe("metaquotes");
    expect(envState(true, [ok], { ...ok, account: { ...ok.account, connected: false } })).toBe("notconnected");
    expect(envState(true, [ok], ok)).toBeNull();
  });
});

describe("first-run welcome", () => {
  it("walks three cards, ends on the first-campaign call to action, and is shown once", async () => {
    const { Welcome, welcomeSeen, markWelcomeSeen } = await import("../shell/Welcome");
    localStorage.removeItem("arqvio.welcome");
    expect(welcomeSeen()).toBe(false);
    const done = vi.fn();
    render(<Welcome onDone={(start) => { markWelcomeSeen(); done(start); }} />);
    screen.getByText("Next").click();
    expect(await screen.findByText("Two engines")).toBeTruthy();
    screen.getByText("Next").click();
    (await screen.findByText("Pick my first market")).click();
    expect(done).toHaveBeenCalledWith(true);
    expect(welcomeSeen()).toBe(true);
  });
});
