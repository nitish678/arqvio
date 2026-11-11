import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import type { LaunchOverview, LaunchState } from "../api/ws";
import { CampaignTab } from "../deck/CampaignTab";
import { fakeEngine, LAUNCH_IDLE } from "./fakeEngine";

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

it.each(["BLOCKED_DATA", "PARTIAL_DATA", "FAILED"] as const)("shows %s and its actual cause", (state) => {
  fakeEngine();
  const launch: LaunchState = {
    id: "fixture", utc: "2026-11-10T00:00:00Z", state, detail: "GBPUSD H1: insufficient history",
    request: { terminal: "fake", terminal_path: "unused", symbols: ["GBPUSD"], timeframes: ["H1"],
      years: 3, depth: "QUICK", power: "eco" },
    args: { pop: 50, gen: 5, budget: 400, final_n: 20 }, tasks: [], vaulted: 0,
    finished_utc: "2026-11-10T00:01:00Z",
  };
  const ov: LaunchOverview = { ...LAUNCH_IDLE, launch };
  render(<CampaignTab ov={ov} rows={[]} points={[]} onOpen={() => undefined} />);
  expect(screen.getByRole("status").textContent).toBe(`${state} · GBPUSD H1: insufficient history`);
  expect(screen.getByRole("status").classList.contains("bad")).toBe(true);
});
