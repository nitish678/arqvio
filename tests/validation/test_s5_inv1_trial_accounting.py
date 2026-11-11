"""S5 invariant I1 - trial accounting separates search from validation.

Search trials inflate the multiple-testing budget. Validation runs do NOT.
The DSR bar must move on a SEARCH_TRIAL row and stay frozen on a
VALIDATION_RUN row, even when 1,000 validation runs are appended.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from engine.validation.cell_dsr import (
    CellDSRStatus,
    cell_dsr_from_registry,
)
from engine.validation.minbtl import minbtl, search_trial_count
from engine.validation.op_types import OpType
from engine.validation.trial_registry import TrialRegistry


@pytest.fixture
def reg(tmp_path: Path) -> TrialRegistry:
    return TrialRegistry(tmp_path / "trials.sqlite")


def _seed(reg: TrialRegistry, campaign_id: str = "c1") -> None:
    reg.open_campaign(
        campaign_id, symbol="USDCADm", timeframe="H2",
        dataset_hash="d", git_sha="g", config_hash="c", seed=7,
    )


def test_validation_run_does_not_move_dsr_bar(reg: TrialRegistry) -> None:
    """Adding 1,000 VALIDATION_RUN rows must not change the DSR probability
    to full float precision."""
    _seed(reg)
    rng = np.random.default_rng(11)
    returns = rng.normal(0.0002, 0.01, 500)
    # Five real search trials, one of them strong.
    for i, sr in enumerate((0.1, 0.2, 0.3, 0.4, 0.5), start=1):
        reg.record(
            "c1", f"search-{i}", "EVALUATED", sharpe=sr,
            symbol="USDCADm", timeframe="H2",
            op_type=OpType.SEARCH_TRIAL.value,
            dataset_hash="d", source="BUNDLED", as_of="2026-09-01T00:00:00Z",
        )
    before = cell_dsr_from_registry(reg, "c1", returns, "USDCADm", "H2")
    # 1,000 validation runs with astronomical Sharpes.
    for i in range(1000):
        reg.record(
            "c1", f"val-{i}", "EVALUATED", sharpe=10.0,
            symbol="EURUSDm", timeframe="D1",
            op_type=OpType.VALIDATION_RUN.value,
            dataset_hash="d2", source="CACHED", as_of="2026-09-10T00:00:00Z",
        )
    after = cell_dsr_from_registry(reg, "c1", returns, "USDCADm", "H2")
    # Same cell, same inputs, same n_trials in DSR accounting.
    assert before.n_trials == after.n_trials == 5
    assert before.dsr is not None
    assert after.dsr is not None
    assert math.isclose(float(before.dsr), float(after.dsr), rel_tol=1e-15, abs_tol=1e-15), (
        f"DSR moved on a validation run: {before.dsr} -> {after.dsr}"
    )
    # Registry audit: validation runs visible but excluded from stats.
    s = reg.stats("c1")
    assert s.validation_runs == 1000
    assert s.n_trials == 5


def test_unknown_legacy_rows_are_excluded_from_dsr(reg: TrialRegistry) -> None:
    """Pre-S5 rows with op_type='UNKNOWN' must not inflate the trial count."""
    _seed(reg, "c2")
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0, 0.01, 500)
    # Simulate legacy rows by writing them as UNKNOWN.
    with reg._conn() as c:
        for i in range(200):
            c.execute(
                "INSERT INTO trials (campaign_id, dna_hash, symbol, timeframe, op_type, "
                "sharpe, outcome) VALUES (?,?,?,?,?,?,?)",
                ("c2", f"legacy-{i}", "USDCADm", "H2",
                 OpType.UNKNOWN.value, 5.0, "EVALUATED"),
            )
    cell = cell_dsr_from_registry(reg, "c2", returns, "USDCADm", "H2")
    assert cell.n_trials == 0  # the 200 UNKNOWN rows do not count
    # NO_TRIALS, not REJECTED: there was nothing to correct for, which is a
    # different fact from a correction that failed.
    assert cell.status is CellDSRStatus.NO_TRIALS
    assert reg.stats("c2").unknown_rows == 200


def test_minbtl_counts_only_search_trials(reg: TrialRegistry) -> None:
    _seed(reg, "c3")
    for i in range(10):
        reg.record(
            "c3", f"s-{i}", "EVALUATED", sharpe=1.0,
            op_type=OpType.SEARCH_TRIAL.value,
            dataset_hash="d", source="BUNDLED", as_of="2026-09-01T00:00:00Z",
        )
    for i in range(50):
        reg.record(
            "c3", f"v-{i}", "EVALUATED", sharpe=1.0,
            op_type=OpType.VALIDATION_RUN.value,
            dataset_hash="d", source="BUNDLED", as_of="2026-09-01T00:00:00Z",
        )
    assert search_trial_count(reg, "c3") == 10
    out = minbtl(target_sharpe=1.0, n_trials=search_trial_count(reg, "c3"))
    assert out.n_trials == 10
    # The 50 validation runs are recorded but do NOT inflate the budget.
    assert out.min_years < 50.0


def test_invalid_op_type_rejected(reg: TrialRegistry) -> None:
    _seed(reg)
    with pytest.raises(ValueError, match="op_type"):
        reg.record("c1", "x", "EVALUATED", sharpe=1.0, op_type="NONSENSE")
