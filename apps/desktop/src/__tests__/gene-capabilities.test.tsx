import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import type { GenePoolDoc } from "../api/ws";
import { GeneBrowser } from "../deck/GeneBrowser";
import { fakeEngine } from "./fakeEngine";

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

it.each([true, false])("shows declared support without inventing parity; matching hash=%s", async (matching) => {
  const gene = { name: "bias.sma", group: "bias", kind: "switch", origin: "research", status: "READY", params: [],
    dna_field: "bias_switches", note: "Fixed SMA rule" };
  const point = (declared: boolean) => ({ declared, evidence_status: "UNVERIFIED", proof_runs: [] });
  fakeEngine({
    "GET /api/v1/gene-pool/registry": () => ({ version: 1, hash: "current", genes: [gene],
      counts: { total: 1, by_status: { READY: 1 }, by_group: { bias: 1 }, engines: {} } }),
    "GET /api/v1/gene-pool/capabilities": () => ({ schema: "arqvio.gene_capabilities", version: 1,
      registry_hash: matching ? "current" : "stale", scope: "declarations only",
      genes: [{ name: "bias.sma", legacy_status: "READY", breedable_declared: true,
        support: { oracle: point(true), kernel: point(true), live_ea: point(true), tester_ea: point(false) }, note: "" }] }),
  });
  const pool: GenePoolDoc = { families: {}, genes: {}, hash: "pool", gates: { bias_switches: true } };
  render(<GeneBrowser pool={pool} busy={false} onPatch={async () => undefined} error={null} />);
  const text = await screen.findByLabelText("Support for bias.sma");
  if (matching) {
    await screen.findByText(/Declared: oracle, kernel, live ea · Evidence: UNVERIFIED/);
    expect(text.textContent).not.toContain("tester ea");
  } else {
    expect(text.textContent).toContain("Support evidence unavailable");
  }
  expect(screen.getByText(/READY is a legacy implementation declaration/)).toBeTruthy();
});
