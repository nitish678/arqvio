"""S6-A regression tests: operation scope and per-cell DSR."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from engine.validation.cell_dsr import (
    CellDSRStatus,
    cell_dsr_from_registry,
    compute_cell_dsr,
    compute_cross_cell_dsr,
)
from engine.validation.trial_registry import TrialRegistry


def _open(reg: TrialRegistry) -> None:
    reg.open_campaign(
        "c1",
        symbol="USDCADm",
        timeframe="H2",
        dataset_hash="d",
        git_sha="g",
        config_hash="c",
        seed=7,
    )


def test_validation_run_does_not_move_dsr_bar(tmp_path: Path) -> None:
    reg = TrialRegistry(tmp_path / "trials.sqlite")
    _open(reg)
    returns = np.random.default_rng(11).normal(0.0002, 0.01, 500)
    for i, sharpe in enumerate((0.1, 0.2, 0.3), start=1):
        reg.record(
            "c1", f"search-{i}", "EVALUATED", sharpe=sharpe,
            symbol="USDCADm", timeframe="H2", op_type="SEARCH_TRIAL",
        )
    before = cell_dsr_from_registry(reg, "c1", returns, "USDCADm", "H2")
    reg.record(
        "c1", "validation-1", "EVALUATED", sharpe=4.0,
        symbol="EURUSDm", timeframe="D1", op_type="VALIDATION_RUN",
    )
    after = cell_dsr_from_registry(reg, "c1", returns, "USDCADm", "H2")
    assert before.dsr == after.dsr
    assert before.n_trials == after.n_trials == 3
    assert reg.stats("c1").validation_runs == 1


def test_dsr_is_computed_per_cell_not_from_pooled_trials() -> None:
    rng = np.random.default_rng(12)
    short = compute_cell_dsr(
        np.zeros(60), n_trials=100, trial_sharpes=np.linspace(-.2, .2, 100),
        symbol="A", timeframe="D1",
    )
    long = compute_cell_dsr(
        rng.normal(0, .01, 2400), n_trials=100,
        trial_sharpes=np.linspace(-.02, .02, 100), symbol="B", timeframe="M5",
    )
    pooled = compute_cross_cell_dsr((short, long))
    assert short.status is CellDSRStatus.UNRELIABLE
    assert long.status is CellDSRStatus.READY
    assert pooled.status is CellDSRStatus.REJECTED
    assert pooled.dsr_across_cells != long.dsr


def test_sixty_observations_are_unreliable_even_with_high_sharpe() -> None:
    result = compute_cell_dsr(
        np.full(60, 0.01), n_trials=1, trial_sharpes=[10.0],
        symbol="USDCADm", timeframe="D1", min_obs_for_dsr=100,
    )
    assert result.status is CellDSRStatus.UNRELIABLE
    assert result.dsr is None


def test_cross_cell_result_requires_both_corrections() -> None:
    rng = np.random.default_rng(13)
    a = compute_cell_dsr(
        rng.normal(.001, .01, 500), n_trials=2,
        trial_sharpes=[.1, .2], symbol="A", timeframe="D1",
    )
    b = compute_cell_dsr(
        rng.normal(.001, .01, 500), n_trials=2,
        trial_sharpes=[.1, .2], symbol="B", timeframe="D1",
    )
    result = compute_cross_cell_dsr((a, b))
    assert result.n_cells == 2
    assert result.dsr_within_cell is not None
    assert result.dsr_across_cells is not None
