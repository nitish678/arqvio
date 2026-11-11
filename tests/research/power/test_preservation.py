"""P2, the preservation gate as a test: S4 Stage 3's recorded USDCADm power numbers, encoded as DATA, against the U2
post-surgery re-derivation archived at artifacts/reports/u2/preservation/power_out (DR-2026-10-08-u2 §4).

The archive is an untracked run product, so the test skips where it is absent (a staged worktree, another machine).
"""
from __future__ import annotations

import hashlib
import json

import pytest

from engine.core.paths import PROJECT_ROOT

# artifacts/reports/s4_stage3_report.md:222-276 and power_run.log:28, verbatim values.
S4_ROWS_SHA256 = "e8a192e5cc77fd7611cd296764a4a283c78155b4755f924083c9216db3428705"
S4_DETECTED_AT_4760 = {0.0: 0.0, 0.5: 0.0, 1.0: 0.0, 1.5: 0.0, 2.0: 0.225, 3.0: 0.97, 5.0: 1.0}  # both regimes
S4_VERDICT = {
    "controlled": True,
    "fpr": {"matched@N=1000": 0.0, "matched@N=20": 0.005, "matched@N=200": 0.0, "matched@N=4760": 0.0,
            "v2@N=1000": 0.0, "v2@N=20": 0.005, "v2@N=200": 0.0, "v2@N=4760": 0.0},
    "powered_at_1_5": False,
    "rate_1_5": {"matched": 0.0, "v2": 0.0},
    "smallest_powered_strength": 3.0,
}
S4_FILLS = {"v2": {"requested": 89342, "filled": 54454}, "matched": {"requested": 89342, "filled": 88486}}
S4_ADMITTED_SHARE = 0.6107  # as printed, 4 places

ARCHIVE = PROJECT_ROOT / "artifacts" / "reports" / "u2" / "preservation" / "power_out"


@pytest.fixture
def u2_run() -> dict:
    if not (ARCHIVE / "power_table.json").is_file():
        pytest.skip("the U2 USDCADm re-derivation archive is not in this tree")
    return json.loads((ARCHIVE / "power_table.json").read_text(encoding="utf-8"))


def test_the_rows_are_byte_identical_to_s4s(u2_run) -> None:
    assert hashlib.sha256((ARCHIVE / "power_rows.json").read_bytes()).hexdigest() == S4_ROWS_SHA256
    assert u2_run["rows_sha256"] == u2_run["rows_rederived_sha256"] == S4_ROWS_SHA256


def test_the_primary_curve_verdict_and_shams_are_s4s(u2_run) -> None:
    got = {(t["strength"], t["regime"]): t["detected"] for t in u2_run["table"] if t["N"] == 4760}
    assert got == {(s, g): rate for s, rate in S4_DETECTED_AT_4760.items() for g in ("v2", "matched")}
    assert u2_run["verdict"] == S4_VERDICT
    assert u2_run["shams"]["fills"] == S4_FILLS
    assert round(u2_run["shams"]["admitted_share"], 4) == S4_ADMITTED_SHARE


def test_the_run_resolved_the_retired_literals(u2_run) -> None:
    assert u2_run["context"] == {"campaign_id": "ga-usdcadm_h1-s7-cbcb2ae061e70bef", "cell_id": "usdcadm_h1",
                                 "dataset_hash16": "aed3bf314539ec5a", "ppy": 6048, "primary_n": 4760,
                                 "symbol": "USDCADm", "timeframe": "H1"}
