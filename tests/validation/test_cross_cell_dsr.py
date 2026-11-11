"""dsr_across is a second selection step, not a copy of dsr_within.

The campaign reported dsr_across equal to dsr_within in every cell. That is
what a degenerate single-cell case looks like and it is also what a copy
looks like. It was a copy: run_campaign did

    cross = compute_cross_cell_dsr((...)) if False else None
    dsr_across = dsr_all.dsr_within

with the real call dead behind ``if False``.

Choosing the best cell is itself a multiple-testing problem, so the correct
answer is the same Bailey deflation with the trial population swapped:
n_trials = number of cells, var_sr_trials = variance of the per-cell winner
Sharpes, sr_hat = the best of them. One cell is DEGENERATE, never a copy.
"""
from __future__ import annotations

import numpy as np

from engine.validation.cell_dsr import (
    AcrossCellDSR,
    CellDSRResult,
    CellDSRStatus,
    compute_cross_cell_dsr,
    deflated_sharpe_across_cells,
)

PPY = float(np.sqrt(252.0))


def _cell(
    dsr: float,
    per_bar: float,
    *,
    n_obs: int = 3915,
    status: CellDSRStatus = CellDSRStatus.READY,
    symbol: str = "USDCADm",
) -> CellDSRResult:
    """A cell whose winner has per-bar Sharpe ``per_bar``."""
    return CellDSRResult(
        symbol=symbol,
        timeframe="D1",
        status=status,
        dsr=dsr,
        n_obs=n_obs,
        n_trials=5000,
        var_sr_trials=0.17,
        winner_sharpe=per_bar * PPY,   # annualised, as the registry stores it
        min_obs_threshold=100,
        sr_hat_per_bar=per_bar,
        expected_max_sr=0.13,
    )


def test_dsr_across_uses_cross_cell_variance() -> None:
    """Identical per-cell DSR, different winner Sharpes -> a NEW number.

    Every cell has dsr=0.80. A copy of any of them returns 0.80. A second
    deflation cannot, because it reads the winner Sharpes, which differ, and
    the variance ACROSS cells, which a single cell does not have.
    """
    cells = [_cell(0.80, 0.130), _cell(0.80, 0.135), _cell(0.80, 0.140)]
    out = deflated_sharpe_across_cells(cells)

    assert out.status is CellDSRStatus.READY
    assert out.n_cells == 3
    assert out.dsr is not None
    for c in cells:
        assert out.dsr != c.dsr, "across-cell DSR is a copy of a cell's own"

    # It is driven by the cross-cell variance, not the within-cell one.
    expected_var = float(np.var([0.130, 0.135, 0.140], ddof=1))
    assert out.var_sr_trials == expected_var
    assert out.var_sr_trials != cells[0].var_sr_trials
    assert out.sr_hat == 0.140  # the best cell winner


def test_more_dispersed_cell_sharpes_raise_the_across_bar() -> None:
    """Same best cell, wider spread beneath it -> a higher bar to clear.

    Holding sr_hat fixed at 0.140 isolates dispersion: only the variance of
    the cell winners changes, so only the bar can move.
    """
    tight = deflated_sharpe_across_cells(
        [_cell(0.80, 0.130), _cell(0.80, 0.135), _cell(0.80, 0.140)]
    )
    wide = deflated_sharpe_across_cells(
        [_cell(0.80, 0.110), _cell(0.80, 0.125), _cell(0.80, 0.140)]
    )
    assert tight.sr_hat == wide.sr_hat == 0.140
    assert wide.var_sr_trials > tight.var_sr_trials
    assert wide.expected_max_sr > tight.expected_max_sr, (
        f"dispersed cells gave bar {wide.expected_max_sr:.8f}, not above "
        f"the tight {tight.expected_max_sr:.8f}"
    )
    assert wide.dsr < tight.dsr, (
        f"a higher bar must lower the DSR: wide {wide.dsr!r} vs tight "
        f"{tight.dsr!r}"
    )


def test_a_high_bar_actually_moves_the_dsr_off_saturation() -> None:
    """With enough dispersion the bar reaches the winner and DSR falls.

    The three-similar-cells case saturates near 1.0, which is the honest
    answer there: three agreeing cells carry almost no selection penalty.
    This pins that the statistic is live, not pinned at 1.
    """
    # Cell winners centred near zero: the best of them barely exceeds the
    # bar their own dispersion creates, which is exactly the case where the
    # across-cell correction decides something.
    spread = deflated_sharpe_across_cells(
        [_cell(0.80, s) for s in (-0.10, -0.05, 0.0, 0.05, 0.10)]
    )
    assert spread.dsr is not None
    assert 0.0 < spread.dsr < 1.0, spread


def test_a_single_cell_is_degenerate_not_a_copy() -> None:
    """One cell must be LABELLED, never silently equal to dsr_within."""
    out = deflated_sharpe_across_cells([_cell(0.62, 0.0986)])
    assert out.status is CellDSRStatus.DEGENERATE
    assert out.dsr is None, "a single cell must not report a number at all"
    assert out.n_cells == 1
    assert "not a selection" in out.note

    # And through the public entry point the campaign uses.
    cross = compute_cross_cell_dsr([_cell(0.62, 0.0986)])
    assert cross.status is CellDSRStatus.DEGENERATE
    assert cross.dsr_across_cells is None
    assert cross.dsr_within_cell == 0.62
    assert cross.dsr_across_cells != cross.dsr_within_cell


def test_missing_per_bar_sharpe_is_unreliable_not_guessed() -> None:
    """Without a per-bar scale the deflation refuses rather than assuming.

    winner_sharpe is annualised; scaling it needs periods-per-year, which a
    CellDSRResult does not carry. Guessing would silently inflate z by
    sqrt(ppy) and saturate every DSR at 1.0.
    """
    bare = CellDSRResult(
        symbol="USDCADm", timeframe="D1", status=CellDSRStatus.READY,
        dsr=0.8, n_obs=3915, n_trials=5000, var_sr_trials=0.17,
        winner_sharpe=2.2, min_obs_threshold=100,
    )
    out = deflated_sharpe_across_cells([bare, bare])
    assert out.status is CellDSRStatus.UNRELIABLE
    assert out.dsr is None
    assert "per-bar" in out.note


def test_one_unreliable_cell_refuses_the_whole_thing() -> None:
    out = compute_cross_cell_dsr(
        [
            _cell(0.80, 0.130),
            _cell(0.80, 0.135),
            _cell(0.10, 0.02, status=CellDSRStatus.UNRELIABLE),
        ]
    )
    assert out.status is CellDSRStatus.REJECTED
    assert out.dsr_across_cells is None
    assert "UNRELIABLE" in out.note


def test_within_is_the_worst_cell_not_the_best() -> None:
    """A strategy is only as good as its weakest cell."""
    out = compute_cross_cell_dsr(
        [_cell(0.95, 0.140), _cell(0.55, 0.135), _cell(0.80, 0.130)]
    )
    assert out.dsr_within_cell == 0.55


def test_result_type_is_explicit() -> None:
    out = deflated_sharpe_across_cells([_cell(0.8, 0.13), _cell(0.8, 0.14)])
    assert isinstance(out, AcrossCellDSR)
    assert out.expected_max_sr is not None
    assert out.var_sr_trials is not None
