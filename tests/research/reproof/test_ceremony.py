"""The U3 worker_status ceremony (DR-2026-10-08-u3-worker §5): both cells PROVEN with guardrail provenance, the
proof path appended to evidence, fills_in and every other field as U1 / U2 left them, and the file still verifies."""
from __future__ import annotations

from engine.research.registry.cells import list_cells
from engine.research.reproof.checks import STATUS_GRAMMAR

# U2's sealed values for every field the U3 ceremony must NOT touch (configs/cells_registry.json at 9b0f1a9), as data.
U2_UNTOUCHED = {
    "usdcadm_h1": {"dataset_hash16": "aed3bf314539ec5a", "cross_cell": "xauusdm_h1", "annualisation_ppy": 6048,
                   "entered_by_stage": "U1", "entered_at": "2026-10-07T13:01:49.159937+00:00",
                   "power_check_status": "COMPUTED", "evidence_before": [
                       "artifacts/reports/s4_stage2_report.md", "artifacts/reports/s4_stage3_report.md",
                       "artifacts/reports/s6_report.md", "docs/decisions/DR-2026-10-07-u1-cell-registry.md"]},
    "xauusdm_h1": {"dataset_hash16": "447b843f87438a9a", "cross_cell": "usdcadm_h1", "annualisation_ppy": 6048,
                   "entered_by_stage": "U1", "entered_at": "2026-10-07T13:01:49.159937+00:00",
                   "power_check_status": "COMPUTED", "evidence_before": [
                       "artifacts/reports/xau_h1_report.md", "artifacts/reports/xau_m5/BLOCKED.md",
                       "docs/decisions/DR-2026-10-07-u1-cell-registry.md"]},
}


def test_both_cells_are_proven_with_their_guardrail_provenance() -> None:
    entries = {e["cell_id"]: e for e in list_cells()}
    # U4 appended gbpjpym_h1 (DR-2026-10-09-u4-wizard §3: the v2 package supersedes U1-SL3 for U4); U3's two stand
    assert sorted(entries) == ["gbpjpym_h1", "usdcadm_h1", "xauusdm_h1"]
    for cell in U2_UNTOUCHED:
        e = entries[cell]
        ws = e["worker_status"]
        assert ws["status"].startswith("PROVEN@") and STATUS_GRAMMAR.fullmatch(ws["status"])
        assert ws["fills_in"] == "U3"
        assert "battery==S6(704/1096/2967/1994); era: S6-bank" in ws["note"]
        assert "20/20 candidates" in ws["note"] and f"artifacts/reports/u3/worker/{cell}.proof.json" in ws["note"]
    assert "tier: own cell" in entries["usdcadm_h1"]["worker_status"]["note"]
    assert "tier: global tier R4(a)" in entries["xauusdm_h1"]["worker_status"]["note"]


def test_the_ceremony_appended_the_proof_and_moved_nothing_else() -> None:
    entries = {e["cell_id"]: e for e in list_cells()}
    for cell, want in U2_UNTOUCHED.items():
        e = entries[cell]
        assert e["evidence"] == [*want["evidence_before"], f"artifacts/reports/u3/worker/{cell}.proof.json"]
        assert e["power_check"]["status"] == want["power_check_status"]
        for k in ("dataset_hash16", "cross_cell", "annualisation_ppy", "entered_by_stage", "entered_at"):
            assert e[k] == want[k], (cell, k)
