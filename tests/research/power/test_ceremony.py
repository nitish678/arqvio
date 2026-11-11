"""The U2 registry ceremony (DR-2026-10-08-u2 §6): both cells' power_check COMPUTED with their refs, nothing else
moved from the U1 migration, and the sealed file still verifies."""
from __future__ import annotations

from engine.research.power.status import power_status
from engine.research.registry.cells import PinnedCellsView, list_cells

# U1's migrated values for every field the ceremony must NOT touch (the U1 registry, configs/cells_registry.json at
# 4a85481), as data.
U1_UNTOUCHED = {
    "usdcadm_h1": {"dataset_hash16": "aed3bf314539ec5a", "cross_cell": "xauusdm_h1", "annualisation_ppy": 6048,
                   "entered_by_stage": "U1", "entered_at": "2026-10-07T13:01:49.159937+00:00",
                   "dataset_window": {"start": "2014-01-14T00:00:00+00:00", "end": "2026-09-01T00:00:00+00:00",
                                      "bars": 59957}},
    "xauusdm_h1": {"dataset_hash16": "447b843f87438a9a", "cross_cell": "usdcadm_h1", "annualisation_ppy": 6048,
                   "entered_by_stage": "U1", "entered_at": "2026-10-07T13:01:49.159937+00:00",
                   "dataset_window": {"start": "2014-01-14T00:00:00+00:00", "end": "2026-09-01T00:00:00+00:00",
                                      "bars": 57011}},
}


def test_both_cells_are_computed_with_their_refs() -> None:
    usd, xau = power_status("usdcadm_h1"), power_status("xauusdm_h1")
    assert usd["status"] == xau["status"] == "COMPUTED"
    assert usd["ref"].startswith("artifacts/reports/s4_stage3_report.md:276 (VERDICT smallest_powered_strength 3.0")
    assert "re-derived byte-identically by U2" in usd["ref"]
    assert xau["ref"].startswith("artifacts/reports/u2/xau_power_run.log:")
    assert "smallest_powered_strength 2.0" in xau["ref"] and "controlled false" in xau["ref"]
    assert "discharges XAU-H1's deferral" in xau["ref"]


def test_the_ceremony_moved_nothing_but_power_check() -> None:
    entries = {e["cell_id"]: e for e in list_cells()}
    # U4 appended gbpjpym_h1 (DR-2026-10-09-u4-wizard §3: the v2 package supersedes U1-SL3 for U4); U2's two stand
    assert sorted(entries) == ["gbpjpym_h1", "usdcadm_h1", "xauusdm_h1"]
    for cell, want in U1_UNTOUCHED.items():
        assert {k: entries[cell][k] for k in want} == want
    assert dict(PinnedCellsView()) == {
        "usdcadm_h1": ("USDCADm", "H1", "aed3bf314539ec5a"),
        "xauusdm_h1": ("XAUUSDm", "H1", "447b843f87438a9a"),
        "gbpjpym_h1": ("GBPJPYm", "H1", "5979a019aa6b3e0c"),
    }
