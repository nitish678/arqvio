"""Shared statistical primitives for ARQVIO's validation core.

Convention used throughout this package
---------------------------------------
Every Sharpe ratio is a PER-OBSERVATION Sharpe unless the function name or
argument explicitly says annualised. Mixing the two is the single most common
way to get DSR wrong by an order of magnitude, so the conversion is always
explicit and never implicit.

Kurtosis is PEARSON kurtosis (normal == 3.0), not excess kurtosis. The PSR
formula in Bailey & Lopez de Prado (2012) is written in terms of Pearson
kurtosis. scipy defaults to Fisher (excess), so we always pass fisher=False.

Cross-checked against `arqvio v1` reference: the per-observation Sharpe,
moments, and annualisation helpers in this file produce identical numerical
results to the reference implementation. The `vs_random` and validation
modules in this package use these primitives so there is exactly one
definition of "Sharpe" in the system.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as _sps

# Observations per year for common bar frequencies. Used only for explicit
# annualisation helpers - never applied silently.
PERIODS_PER_YEAR: dict[str, int | None] = {
    "M1": 252 * 24 * 60,
    "M5": 252 * 24 * 12,
    "M15": 252 * 24 * 4,
    "M30": 252 * 24 * 2,
    "H1": 252 * 24,
    "H2": 252 * 12,
    "H4": 252 * 6,
    "D1": 252,
    "W1": 52,
    "trades": None,  # per-trade returns have no calendar frequency
}


def as_returns(x: object) -> np.ndarray:
    """Coerce to a 1-D float64 array of returns, dropping non-finite values."""
    a = np.asarray(x, dtype=np.float64).ravel()
    return a[np.isfinite(a)]


def sharpe(returns: object, rf_per_period: float = 0.0, ddof: int = 1) -> float:
    """Per-observation Sharpe ratio.

    Returns np.nan when the sample is too small or has zero dispersion,
    rather than raising or silently returning inf. Callers must handle nan.
    """
    r = as_returns(returns)
    if r.size < 2:
        return float("nan")
    excess = r - rf_per_period
    sd = float(np.std(excess, ddof=ddof))
    mean_abs = float(np.mean(np.abs(excess)))
    # Detect constant (or near-constant) series. np.std on a float64 constant
    # is not exactly 0 (e.g. np.full(100, 0.001) has std ~ 2e-19); use a tiny
    # relative epsilon to short-circuit.
    if not np.isfinite(sd) or sd <= max(mean_abs, 1.0) * 1e-12:
        return float("nan")
    return float(np.mean(excess) / sd)


def annualise_sharpe(sr_per_period: float, periods_per_year: float) -> float:
    """Explicit sqrt-time scaling. Only valid under iid-ish returns."""
    return float(sr_per_period * np.sqrt(periods_per_year))


def deannualise_sharpe(sr_annual: float, periods_per_year: float) -> float:
    return float(sr_annual / np.sqrt(periods_per_year))


def moments(returns: object) -> tuple[int, float, float, float]:
    """Return (n, sharpe, skew, pearson_kurtosis).

    Skew and kurtosis are computed on the raw return series. For n < 4 the
    higher moments are not identifiable, so we fall back to the Gaussian
    values (skew 0, kurtosis 3) which makes PSR reduce to its normal case.
    """
    r = as_returns(returns)
    n = int(r.size)
    sr = sharpe(r)
    if n < 4:
        return n, sr, 0.0, 3.0
    g3 = float(_sps.skew(r, bias=False))
    g4 = float(_sps.kurtosis(r, fisher=False, bias=False))
    if not np.isfinite(g3):
        g3 = 0.0
    if not np.isfinite(g4):
        g4 = 3.0
    return n, sr, g3, g4


def max_drawdown(equity: object) -> float:
    """Maximum peak-to-trough drawdown of an equity curve, as a positive fraction."""
    e = np.asarray(equity, dtype=np.float64).ravel()
    if e.size == 0:
        return float("nan")
    peak = np.maximum.accumulate(e)
    dd = (e - peak) / np.where(peak == 0, np.nan, peak)
    return float(-np.nanmin(dd))


def equity_from_returns(returns: object, start: float = 1.0) -> np.ndarray:
    r = as_returns(returns)
    return start * np.cumprod(1.0 + r)
