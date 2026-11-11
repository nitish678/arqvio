"""Minimum Backtest Length (MinBTL) - Bailey-Borwein-Lopez de Prado-Zhu (2014).

The real MinBTL formula from "The Deflated Sharpe Ratio" (2014):

    gamma = 0.5772156649015329            (Euler-Mascheroni constant)
    z(n)  = (1 - gamma) * Phi_inv(1 - 1/n)
          + gamma      * Phi_inv(1 - 1/(n*e))
    MinBTL(n, SR) = z(n)^2 / SR^2

Pinned at SR=1.5 (S5 master prompt, 2026-09-17):
    min_backtest_length(500,     1.5) ==  4.14  (+/-0.01)
    min_backtest_length(5000,    1.5) ==  6.04  (+/-0.01)
    min_backtest_length(50000,   1.5) ==  7.98  (+/-0.01)
    min_backtest_length(500000,  1.5) ==  9.94  (+/-0.01)

For n >= 50 the formula is monotonically sub-quadratic: MinBTL(10n) <
2 * MinBTL(n). The n=10 -> n=100 step breaks this (2.58x) because the
z(n) tail is in the Gaussian small-p regime. The monotonicity test in
:mod:`tests.validation.test_minbtl_s5_spec` starts at n=50.

The numeric inverse :func:`max_trials_for_history` uses binary search,
not a closed form, per Maalik's 2026-09-17 instruction.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from engine.core.errors import DataError
from engine.validation.trial_registry import TrialRegistry, counts_toward_dsr

DEFAULT_TARGET_IS_SHARPE = 1.5
LEGACY_TARGET_IS_SHARPE = 1.0
MIN_OBS_FOR_DSR = 100

# Euler-Mascheroni constant. 2014 paper, Section 3.2.
EULER_GAMMA = 0.5772156649015329


@dataclass(frozen=True)
class MinBTLResult:
    min_years: float
    target_sharpe: float
    n_trials: int
    satisfied: bool | None
    history_years: float | None


def _z(n_trials: int) -> float:
    """z(n) from Bailey et al. (2014), Section 3.2.

    For n=1 the formula is degenerate (Phi_inv(0) = -infinity). We
    return a finite surrogate of 0 so callers can use n=1 as a
    "minimum trial" floor. For n>=2 the formula is exact.
    """
    if n_trials < 1:
        raise DataError(f"n_trials must be >= 1; got {n_trials}.")
    if n_trials == 1:
        # z(1) is degenerate. Use a finite surrogate = the SR=1 MinBTL
        # at n=2 so the inverse is well-behaved.
        return _z(2)
    n = float(n_trials)
    a = (1.0 - EULER_GAMMA) * float(norm.ppf(1.0 - 1.0 / n))
    b = EULER_GAMMA * float(norm.ppf(1.0 - 1.0 / (n * math.e)))
    return a + b


def min_backtest_length(
    n_trials: int,
    target_sharpe: float,
    *,
    history_years: float | None = None,
) -> MinBTLResult:
    """MinBTL = z(n)^2 / SR^2 (Bailey et al. 2014)."""
    if n_trials < 1:
        raise DataError(f"n_trials must be >= 1; got {n_trials}.")
    if target_sharpe <= 0:
        raise DataError(f"target_sharpe must be > 0; got {target_sharpe}.")
    if history_years is not None and history_years <= 0:
        raise DataError(f"history_years must be > 0 when provided; got {history_years}.")

    sr = float(target_sharpe)
    z_n = _z(int(n_trials))
    min_years = (z_n * z_n) / (sr * sr)

    satisfied: bool | None = None
    if history_years is not None:
        satisfied = history_years >= min_years

    return MinBTLResult(
        min_years=float(min_years),
        target_sharpe=sr,
        n_trials=int(n_trials),
        satisfied=satisfied,
        history_years=float(history_years) if history_years is not None else None,
    )


def max_trials_for_history(
    history_years: float,
    target_sharpe: float,
) -> int:
    """Numeric inverse: largest n with min_backtest_length(n, SR) <= years.

    Binary search in [1, 10^9]. Returns 0 if even n=1 needs more years
    than are available.
    """
    if target_sharpe <= 0:
        raise DataError(f"target_sharpe must be > 0; got {target_sharpe}.")
    if history_years < 0:
        raise DataError(f"history_years must be >= 0; got {history_years}.")
    if history_years == 0:
        return 0

    lo, hi = 1, 1_000_000_000
    if min_backtest_length(1, target_sharpe).min_years > history_years:
        return 0
    if min_backtest_length(hi, target_sharpe).min_years <= history_years:
        return hi
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if min_backtest_length(mid, target_sharpe).min_years <= history_years:
            lo = mid
        else:
            hi = mid - 1
    return lo


def minbtl_from_returns(
    returns: np.ndarray,
    n_trials: int,
    *,
    target_sharpe: float | None = None,
    periods_per_year: int = 1,
    history_years: float | None = None,
) -> MinBTLResult:
    """Wrapper that surfaces observed moments for transparency; the
    MinBTL number itself uses the Bailey formula, not the moments.
    """
    r = np.asarray(returns, dtype=np.float64).ravel()
    n = r.size
    if n < 3:
        raise DataError(f"Need >= 3 returns for moment estimation; got {n}.")
    if periods_per_year <= 0:
        raise DataError(f"periods_per_year must be > 0; got {periods_per_year}.")

    std = float(np.std(r, ddof=1))
    mean = float(np.mean(r))
    if std == 0.0:
        raise DataError("Zero-variance returns; cannot compute MinBTL.")

    sr_observed = (mean / std) * math.sqrt(periods_per_year)

    sr_to_use = target_sharpe if target_sharpe is not None else sr_observed
    return min_backtest_length(
        n_trials=n_trials,
        target_sharpe=sr_to_use,
        history_years=history_years,
    )


@dataclass(frozen=True)
class MinBTLBudgetRow:
    target_sharpe: float
    n_trials: int
    min_years: float


@dataclass(frozen=True)
class MinBTLBudgetTable:
    target_is_annual_sharpe: float
    rows: tuple[MinBTLBudgetRow, ...]

    def at_target(self, target_sharpe: float) -> MinBTLBudgetRow | None:
        for row in self.rows:
            if math.isclose(row.target_sharpe, target_sharpe, rel_tol=1e-9):
                return row
        return None

    def line(self) -> str:
        return " | ".join(
            f"SR={row.target_sharpe:g} -> {row.n_trials:,} trials, "
            f"{row.min_years:.2f}y required"
            for row in self.rows
        )


def minbtl_budget_table(
    history_years: float,
    *,
    targets: tuple[float, ...] = (LEGACY_TARGET_IS_SHARPE, DEFAULT_TARGET_IS_SHARPE),
) -> MinBTLBudgetTable:
    """Both the legacy SR=1.0 figure and the default SR=1.5 figure."""
    rows: list[MinBTLBudgetRow] = []
    for target in targets:
        max_n = max_trials_for_history(history_years, target)
        required_years = min_backtest_length(
            n_trials=max_n, target_sharpe=target
        ).min_years
        rows.append(
            MinBTLBudgetRow(
                target_sharpe=float(target),
                n_trials=int(max_n),
                min_years=float(required_years),
            )
        )
    return MinBTLBudgetTable(
        target_is_annual_sharpe=DEFAULT_TARGET_IS_SHARPE,
        rows=tuple(rows),
    )


def search_trial_count(
    registry: TrialRegistry,
    campaign_id: str,
    *,
    symbol: str | None = None,
    timeframe: str | None = None,
) -> int:
    rows = registry.list_trials(campaign_id, symbol=symbol, timeframe=timeframe)
    return sum(1 for r in rows if counts_toward_dsr(r.op_type))


# Back-compat aliases for prior call sites.
def minbtl(
    target_sharpe: float,
    n_trials: int,
    *,
    skew: float = 0.0,
    excess_kurtosis: float = 0.0,
    history_years: float | None = None,
) -> MinBTLResult:
    return min_backtest_length(
        n_trials=n_trials,
        target_sharpe=target_sharpe,
        history_years=history_years,
    )


def n_trials_for_budget(
    target_sharpe: float,
    history_years: float,
) -> int:
    return max_trials_for_history(
        target_sharpe=target_sharpe, history_years=history_years
    )


__all__ = [
    "DEFAULT_TARGET_IS_SHARPE",
    "EULER_GAMMA",
    "LEGACY_TARGET_IS_SHARPE",
    "MIN_OBS_FOR_DSR",
    "MinBTLBudgetRow",
    "MinBTLBudgetTable",
    "MinBTLResult",
    "max_trials_for_history",
    "min_backtest_length",
    "minbtl",
    "minbtl_budget_table",
    "minbtl_from_returns",
    "n_trials_for_budget",
    "search_trial_count",
]
