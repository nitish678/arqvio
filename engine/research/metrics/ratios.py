"""Annualised risk-adjusted ratios.

WHAT: Sharpe, Sortino, Calmar - all on per-bar returns, all annualised
      by a caller-supplied ``periods_per_year`` (S4's orchestrator
      derives this from the bar frequency or accepts an override).

WHY: S1's DSR consumes per-observation Sharpe; S4's ``ResearchMetrics``
     surfaces the ANNUALISED form because that is what humans read.
     Two distinct consumers, two scales - no silent conversion.

HOW: Reuse :func:`engine.validation.stats.sharpe` (per-observation) and
     ``sqrt-time`` scale for annualisation. Sortino uses downside
     deviation against 0. Calmar is CAGR / |max DD|, both defined
     elsewhere in this module.

TRADE-OFF: When ``sharpe`` returns ``nan`` (zero variance, too few obs)
           the annualised value is also ``None`` - not 0.0. Downstream
           dashboards must show "N/A", not "0.0", when a ratio is
           genuinely undefined.
"""
from __future__ import annotations

import math

import numpy as np

from engine.validation.stats import sharpe as _per_obs_sharpe


def annualised_sharpe(returns: np.ndarray, periods_per_year: int) -> float | None:
    """Per-observation Sharpe scaled by sqrt(periods_per_year).

    Returns ``None`` if S1's ``sharpe`` returns ``nan`` (insufficient
    data or zero dispersion). ``periods_per_year`` must be > 0.
    """
    if periods_per_year <= 0:
        raise ValueError(f"periods_per_year must be > 0; got {periods_per_year}")
    sr = _per_obs_sharpe(returns)
    if not math.isfinite(sr):
        return None
    return float(sr * math.sqrt(periods_per_year))


def annualised_sortino(returns: np.ndarray, periods_per_year: int) -> float | None:
    """Annualised Sortino: mean excess return / downside deviation.

    Downside deviation uses returns < 0 only, with the same ddof=1
    convention as :func:`sharpe`. Returns ``None`` when the downside
    deviation is zero (no losing bars) or the input is too short.
    """
    r = np.asarray(returns, dtype=np.float64).ravel()
    r = r[np.isfinite(r)]
    if r.size < 2 or periods_per_year <= 0:
        return None
    downside = r[r < 0.0]
    if downside.size < 2:
        return None
    dd = float(np.sqrt(np.mean(downside ** 2)))
    if dd <= max(abs(float(np.mean(r))), 1.0) * 1e-12:
        return None
    return float((np.mean(r) / dd) * math.sqrt(periods_per_year))


def calmar(cagr_value: float | None, max_drawdown: float) -> float | None:
    """Calmar = CAGR / |max_drawdown|.

    Returns ``None`` if either input is ``None`` or if ``max_drawdown``
    is zero. ``max_drawdown`` is expected to be a negative number
    (e.g. ``-0.18``).
    """
    if cagr_value is None or not math.isfinite(cagr_value):
        return None
    if max_drawdown is None or not math.isfinite(max_drawdown) or max_drawdown == 0.0:
        return None
    return float(cagr_value / abs(max_drawdown))
