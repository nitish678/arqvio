import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";
import { afterEach, expect, it, vi } from "vitest";
import { CampaignPanel, type CampaignForm } from "../ws/CampaignPanel";
import { fakeEngine } from "./fakeEngine";

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

const initial: CampaignForm = { symbols: ["XAUUSDm"], timeframes: ["H1"], horizon: "research", years: 3, depth: "QUICK", power: "medium" };

function Harness() {
  const [form, setForm] = useState<CampaignForm>(initial);
  return <CampaignPanel terms={null} termId={null} onTerm={() => undefined} onRefreshTerms={() => undefined}
    symbols={null} symbolsError={null} form={form} setForm={setForm} ov={null} busy={false}
    pool={null} poolError={null} onPatchPool={async () => undefined} />;
}

it("shows horizon choices and explains why scalping is unavailable", () => {
  fakeEngine();
  render(<Harness />);
  expect(screen.getByRole("radiogroup", { name: "Strategy horizon" })).toBeTruthy();
  fireEvent.click(screen.getByRole("button", { name: "Scalping" }));
  expect(screen.getByText(/Scalping is unavailable:/)).toBeTruthy();
});
