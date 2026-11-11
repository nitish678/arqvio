"""S5 invariant I3 - minimum observation gate on DSR.

A cell with n_obs < 100 returns UNRELIABLE, not a number. Such a strategy
cannot leave WEAK grade regardless of Sharpe.
"""
from __future__ import annotations

import numpy as np

from engine.validation.cell_dsr import (
    UNRELIABLE,
    CellDSRStatus,
    compute_cell_dsr,
    compute_cross_cell_dsr,
)
from engine.validation.minbtl import MIN_OBS_FOR_DSR


def test_dsr_unreliable_below_min_obs() -> None:
    """A 60-observation cell returns UNRELIABLE - regardless of Sharpe."""
    returns = np.full(60, 0.01)  # perfectly positive
    result = compute_cell_dsr(
        returns, n_trials=1, trial_sharpes=[10.0],
        symbol="USDCADm", timeframe="D1", min_obs_for_dsr=MIN_OBS_FOR_DSR,
    )
    assert result.status is CellDSRStatus.UNRELIABLE
    assert result.dsr is None
    assert result.display() == f"UNRELIABLE (n_obs={60} < {MIN_OBS_FOR_DSR})"
    # The literal UNRELIABLE sentinel must surface in the display string,
    # so downstream reports and the UI can recognise it.
    assert UNRELIABLE in result.display()


def test_99_observations_still_unreliable() -> None:
    result = compute_cell_dsr(
        np.zeros(99), n_trials=1, trial_sharpes=[1.0],
        symbol="X", timeframe="D1", min_obs_for_dsr=100,
    )
    assert result.status is CellDSRStatus.UNRELIABLE


def test_100_observations_are_reliable() -> None:
    rng = np.random.default_rng(0)
    result = compute_cell_dsr(
        rng.normal(0.0, 0.01, 100), n_trials=1, trial_sharpes=[0.5],
        symbol="X", timeframe="D1", min_obs_for_dsr=100,
    )
    assert result.status is CellDSRStatus.READY
    assert result.dsr is not None


def test_unreliable_cell_cannot_clear_weak_grade() -> None:
    """If even one cell is UNRELIABLE, the cross-cell result is REJECTED
    and dsr_within_cell / dsr_across_cells are None - the strategy cannot
    leave WEAK grade."""
    rng = np.random.default_rng(1)
    reliable = compute_cell_dsr(
        rng.normal(0.001, 0.01, 500), n_trials=50, trial_sharpes=[0.1, 0.2, 0.3],
        symbol="A", timeframe="D1",
    )
    weak = compute_cell_dsr(
        np.zeros(50), n_trials=50, trial_sharpes=[0.1, 0.2, 0.3],
        symbol="B", timeframe="D1",
    )
    assert reliable.status is CellDSRStatus.READY
    assert weak.status is CellDSRStatus.UNRELIABLE
    cross = compute_cross_cell_dsr((reliable, weak))
    assert cross.status is CellDSRStatus.REJECTED
    assert cross.dsr_across_cells is None
    assert cross.n_reliable_cells == 1
