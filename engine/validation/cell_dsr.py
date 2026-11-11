"""Per-cell and cross-cell Deflated Sharpe accounting.

A cell is exactly one (symbol, timeframe) pair. Trial Sharpes are never pooled
across cells because their estimator variance depends on sample size.

Two invariants live here:

1. Per-cell DSR: deflated_sharpe is computed using that cell's own
   ``n_trials`` and ``var_sr_trials``. Pooling the trial population across
   cells is forbidden - it gives low-trade-count cells a free ride and
   falsely punishes high-trade-count cells.

2. Cross-cell DSR: selecting the *best* cell is itself a second selection
   step. We treat the per-cell winners as a new trial population of size
   ``n_cells`` and deflate again. Both numbers are reported; a strategy
   clears WEAK grade only if it passes both.

3. Min-observation gate: when ``n_obs < min_obs_for_dsr`` (default 100),
   DSR is reported as the string ``"UNRELIABLE"`` rather than a number.
   A UNRELIABLE cell cannot clear WEAK grade regardless of its Sharpe.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats

from engine.validation.dsr import _expected_max_sharpe, dsr
from engine.validation.minbtl import MIN_OBS_FOR_DSR

if TYPE_CHECKING:
    from engine.validation.trial_registry import TrialRegistry


# Bars per year by timeframe. Trial Sharpes are stored ANNUALIZED in the
# registry (real_backtest_factory annualizes by sqrt(ppy)); DSR's sr_hat and
# expected-max are on the PER-BAR scale. The trial-Sharpe variance must be
# de-annualized (divide by ppy) before it can scale the per-bar luck bar, or
# the two scales mismatch and DSR collapses to 0 (or 1). One scale only.
# Every row is the law 252 x 24 x 60 / tf_minutes (DR-2026-10-07-u1 §4).
PERIODS_PER_YEAR: dict[str, int] = {
    "D1": 252, "H4": 1512, "H2": 3024, "H1": 6048, "M30": 12096, "M15": 24192, "M5": 72576, "M1": 362880,
}
DEFAULT_PERIODS_PER_YEAR = 252



# Sentinel string returned by DSR when the sample is too small to trust the
# Gaussian approximation behind PSR/DSR. Returning a string - never a number -
# is the whole point: downstream code cannot accidentally treat UNRELIABLE as
# a probability.
UNRELIABLE = "UNRELIABLE"


class CellDSRStatus(str, Enum):
    READY = "READY"
    UNRELIABLE = "UNRELIABLE"
    # "No trials" and "rejected" are different facts. NO_TRIALS means the
    # cell had nothing to correct for - no SEARCH_TRIAL rows at all, so a
    # multiple-testing correction is undefined. REJECTED means a correction
    # was computed and the result did not clear its bar. Collapsing the two
    # loses the distinction in any stored record.
    NO_TRIALS = "NO_TRIALS"
    # Cross-cell only: one cell is not a selection problem, so the second
    # deflation has nothing to deflate. Reported rather than silently
    # echoing the within-cell number back.
    DEGENERATE = "DEGENERATE"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class CellDSRResult:
    symbol: str
    timeframe: str
    status: CellDSRStatus
    dsr: float | None              # a real number in [0, 1] OR None
    n_obs: int
    n_trials: int
    var_sr_trials: float
    winner_sharpe: float | None
    min_obs_threshold: int
    note: str = ""
    # A DSR cannot be read without these two. sr_hat_per_bar is THIS
    # candidate's own per-bar Sharpe; expected_max_sr is the Bailey luck bar
    # it has to clear, on the same per-bar scale. The bar rises with
    # n_trials, so the same candidate passes at n=200 and fails at n=5000.
    sr_hat_per_bar: float | None = None
    expected_max_sr: float | None = None

    def display(self) -> str:
        if self.status is CellDSRStatus.UNRELIABLE:
            return f"UNRELIABLE (n_obs={self.n_obs} < {self.min_obs_threshold})"
        if self.status is CellDSRStatus.NO_TRIALS:
            return "NO_TRIALS (no search trials to correct for)"
        if self.status is CellDSRStatus.REJECTED:
            return "REJECTED"
        # READY with no number would be a contradiction; say so rather than
        # printing a status this result does not carry.
        return f"{self.dsr:.4f}" if self.dsr is not None else "READY (dsr=None)"


@dataclass(frozen=True)
class CrossCellDSRResult:
    status: CellDSRStatus
    dsr_within_cell: float | None
    dsr_across_cells: float | None
    n_cells: int
    n_reliable_cells: int
    note: str = ""

    def display(self) -> str:
        within = f"{self.dsr_within_cell:.4f}" if self.dsr_within_cell is not None else "n/a"
        across = f"{self.dsr_across_cells:.4f}" if self.dsr_across_cells is not None else "n/a"
        return (
            f"within={within} across={across} "
            f"({self.n_reliable_cells}/{self.n_cells} reliable)"
        )


def _is_reliable(returns: np.ndarray, min_obs: int) -> tuple[bool, int]:
    n = int(np.asarray(returns, dtype=np.float64).ravel().size)
    return n >= min_obs, n


def compute_cell_dsr(
    returns: np.ndarray,
    *,
    n_trials: int,
    trial_sharpes: Iterable[float],
    symbol: str,
    timeframe: str,
    min_obs_for_dsr: int = MIN_OBS_FOR_DSR,
) -> CellDSRResult:
    """Compute DSR for one cell, or mark it unreliable below the sample gate."""
    r = np.asarray(returns, dtype=np.float64).ravel()
    sharpes = np.asarray(list(trial_sharpes), dtype=np.float64)
    finite = sharpes[np.isfinite(sharpes)]
    variance = float(np.var(finite, ddof=1)) if finite.size >= 2 else float("nan")
    # De-annualize the trial-Sharpe variance to the per-bar scale so it matches
    # dsr()'s per-bar sr_hat. Trial Sharpes are stored annualized (x sqrt(ppy)),
    # so their variance carries a factor ppy that must be divided out.
    ppy = PERIODS_PER_YEAR.get(timeframe, DEFAULT_PERIODS_PER_YEAR)
    var_sr_trials_perbar = (
        variance / float(ppy) if finite.size >= 2 and np.isfinite(variance) else None
    )
    winner = float(np.max(finite)) if finite.size else None
    reliable, n_obs = _is_reliable(r, min_obs_for_dsr)
    if not reliable:
        return CellDSRResult(
            symbol=symbol, timeframe=timeframe,
            status=CellDSRStatus.UNRELIABLE, dsr=None,
            n_obs=n_obs, n_trials=int(n_trials),
            var_sr_trials=variance, winner_sharpe=winner,
            min_obs_threshold=int(min_obs_for_dsr),
            note=f"n_obs={n_obs} < {min_obs_for_dsr}; Gaussian PSR/DSR approximation unreliable",
        )
    if n_trials < 1:
        return CellDSRResult(
            symbol=symbol, timeframe=timeframe,
            status=CellDSRStatus.NO_TRIALS, dsr=None,
            n_obs=n_obs, n_trials=0,
            var_sr_trials=variance, winner_sharpe=winner,
            min_obs_threshold=int(min_obs_for_dsr),
            note="n_trials < 1",
        )
    result = dsr(r, n_trials=n_trials, var_sr_trials=var_sr_trials_perbar)
    return CellDSRResult(
        symbol=symbol, timeframe=timeframe,
        status=CellDSRStatus.READY, dsr=float(result.dsr),
        n_obs=n_obs, n_trials=int(n_trials),
        var_sr_trials=variance, winner_sharpe=winner,
        min_obs_threshold=int(min_obs_for_dsr),
        sr_hat_per_bar=float(result.sr_hat),
        expected_max_sr=float(result.expected_max_sr),
    )


def cell_dsr_from_registry(
    registry: TrialRegistry,
    campaign_id: str,
    returns: np.ndarray,
    symbol: str,
    timeframe: str,
    *,
    min_obs_for_dsr: int = MIN_OBS_FOR_DSR,
) -> CellDSRResult:
    """Read only SEARCH_TRIAL rows for one registry cell."""
    stats_for_cell = registry.stats(campaign_id, symbol=symbol, timeframe=timeframe)
    return compute_cell_dsr(
        returns,
        n_trials=stats_for_cell.n_trials,
        trial_sharpes=registry.search_sharpes(campaign_id, symbol, timeframe),
        symbol=symbol,
        timeframe=timeframe,
        min_obs_for_dsr=min_obs_for_dsr,
    )


@dataclass(frozen=True)
class AcrossCellDSR:
    """The second deflation, over the per-cell winners."""

    status: CellDSRStatus
    dsr: float | None
    sr_hat: float | None            # best cell winner Sharpe
    expected_max_sr: float | None   # the bar it had to clear
    n_cells: int
    var_sr_trials: float | None     # variance ACROSS cells, not within one
    note: str = ""


def deflated_sharpe_across_cells(
    cell_winners: Sequence[CellDSRResult],
) -> AcrossCellDSR:
    """Deflate the best cell winner against the population of cell winners.

    Choosing the best cell is itself a multiple-testing problem: with enough
    cells, one of them looks good by luck. This is the same Bailey
    deflation used within a cell, with the trial population swapped:

        n_trials      = number of cells
        var_sr_trials = variance of the per-cell winner Sharpes
        sr_hat        = the best of them

    A single cell is DEGENERATE, not a pass: there is no selection to
    correct, so the answer is "not applicable", never a copy of the
    within-cell number.

    Moment caveat, stated rather than buried: a cell winner is a scalar
    Sharpe with no return series attached here, so the skew and excess
    kurtosis that ``dsr()`` reads off the returns are unavailable. This
    computes the Gaussian case (skew 0, excess kurtosis 0), which is what
    Bailey's expected-maximum formula assumes anyway. The within-cell DSR
    keeps using the real moments of the real series.
    """
    rows = tuple(cell_winners)
    # PER-BAR Sharpes, not annualised ones. The z below scales by
    # sqrt(n_obs - 1), which is the per-bar convention; feeding it annualised
    # Sharpes inflates z by sqrt(periods_per_year) and saturates every DSR at
    # 1.0. This is the same scale trap the within-cell var_sr_trials hit.
    # sr_hat_per_bar is required rather than derived: without it the periods
    # per year are unknown here, and guessing would be fabrication.
    sr = [
        float(r.sr_hat_per_bar)
        for r in rows
        if r.sr_hat_per_bar is not None and np.isfinite(r.sr_hat_per_bar)
    ]
    if not sr:
        return AcrossCellDSR(
            status=CellDSRStatus.UNRELIABLE, dsr=None, sr_hat=None,
            expected_max_sr=None, n_cells=len(rows), var_sr_trials=None,
            note=(
                "no cell carries a finite sr_hat_per_bar; the across-cell "
                "deflation needs per-bar Sharpes and will not guess a scale"
            ),
        )
    if len(sr) < 2:
        return AcrossCellDSR(
            status=CellDSRStatus.DEGENERATE, dsr=None, sr_hat=float(sr[0]),
            expected_max_sr=None, n_cells=len(rows), var_sr_trials=None,
            note=(
                "one cell: choosing the best of one is not a selection, so "
                "there is no second deflation to apply"
            ),
        )

    n_obs = int(rows[0].n_obs)
    var_across = float(np.var(np.asarray(sr, dtype=np.float64), ddof=1))
    sr_hat = float(np.max(sr))
    expected_max = _expected_max_sharpe(
        n_obs=n_obs,
        n_trials=len(sr),
        skew=0.0,
        excess_kurt=0.0,
        var_sr_trials=var_across if var_across > 0.0 else None,
    )
    if n_obs < 2:
        return AcrossCellDSR(
            status=CellDSRStatus.UNRELIABLE, dsr=None, sr_hat=sr_hat,
            expected_max_sr=float(expected_max), n_cells=len(rows),
            var_sr_trials=var_across, note=f"n_obs={n_obs} < 2",
        )
    z = (sr_hat - expected_max) * math.sqrt(n_obs - 1)
    return AcrossCellDSR(
        status=CellDSRStatus.READY,
        dsr=float(stats.norm.cdf(z)),
        sr_hat=sr_hat,
        expected_max_sr=float(expected_max),
        n_cells=len(rows),
        var_sr_trials=var_across,
    )


def compute_cross_cell_dsr(
    cells: Iterable[CellDSRResult],
) -> CrossCellDSRResult:
    """Apply a second selection correction to the per-cell winners.

    All input cells must be READY for the cross-cell number to be meaningful.
    A single UNRELIABLE cell blocks the strategy from clearing WEAK grade
    regardless of the other cells' results.
    """
    rows = tuple(cells)
    reliable = tuple(r for r in rows if r.status is CellDSRStatus.READY)
    unreliable = len(rows) - len(reliable)
    if unreliable > 0 or not reliable:
        return CrossCellDSRResult(
            status=CellDSRStatus.REJECTED,
            dsr_within_cell=None,
            dsr_across_cells=None,
            n_cells=len(rows),
            n_reliable_cells=len(reliable),
            note=(
                f"{unreliable} cell(s) UNRELIABLE; cross-cell DSR refused"
                if unreliable
                else "no cells available"
            ),
        )
    within_values = [float(r.dsr) for r in reliable if r.dsr is not None]
    if not within_values:
        return CrossCellDSRResult(
            status=CellDSRStatus.REJECTED,
            dsr_within_cell=None, dsr_across_cells=None,
            n_cells=len(rows), n_reliable_cells=len(reliable),
            note="no cell DSR values",
        )
    within = min(within_values)
    # The second deflation, over the per-cell winners. A single cell comes
    # back DEGENERATE with dsr=None rather than echoing `within`.
    acr = deflated_sharpe_across_cells(reliable)
    across = acr.dsr
    if acr.status is CellDSRStatus.DEGENERATE:
        status = CellDSRStatus.DEGENERATE
    else:
        status = (
            CellDSRStatus.READY
            if (across is not None and within > 0.5 and across > 0.5)
            else CellDSRStatus.REJECTED
        )
    return CrossCellDSRResult(
        status=status,
        dsr_within_cell=within,
        dsr_across_cells=across,
        note=acr.note,
        n_cells=len(rows),
        n_reliable_cells=len(reliable),
    )


__all__ = [
    "CellDSRResult",
    "CellDSRStatus",
    "CrossCellDSRResult",
    "UNRELIABLE",
    "cell_dsr_from_registry",
    "compute_cell_dsr",
    "compute_cross_cell_dsr",
]
