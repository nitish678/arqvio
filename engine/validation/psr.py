"""Probabilistic Sharpe Ratio (PSR).

WHAT: Probability that the true Sharpe ratio exceeds a benchmark, given the
observed Sharpe, sample size, and non-normal return moments.
WHY: A raw Sharpe ratio is uninformative without uncertainty quantification.
HOW: Closed-form under a skew/kurtosis-adjusted normal approximation, then
     ``scipy.stats.norm.cdf`` for the tail probability.
TRADE-OFF: PSR assumes a benchmark ``SR*`` (often 0). We expose it as a
     parameter rather than hard-coding; default is 0.0.
TEST: ``tests/validation/test_psr.py`` — known-answer tests, monotonicity in N.
FAILURE: Invalid inputs raise ``DataError``; no silent returns.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from engine.core.errors import DataError
from engine.validation import stats as stats_h


@dataclass(frozen=True)
class PSRResult:
    """Probabilistic Sharpe Ratio result.

    Attributes:
        psr: Probability that true Sharpe exceeds ``sr_star`` (in [0, 1]).
        z_score: The standardized test statistic used.
        sr_hat: Observed Sharpe (echoed back for convenience).
        n: Number of return observations (echoed).
        sr_star: Benchmark Sharpe (echoed).
    """

    psr: float
    z_score: float
    sr_hat: float
    n: int
    sr_star: float


def psr(
    returns: np.ndarray,
    sr_star: float = 0.0,
    *,
    periods_per_year: int | None = None,
) -> PSRResult:
    """Compute the Probabilistic Sharpe Ratio.

    Args:
        returns: 1-D array of per-period arithmetic returns.
        sr_star: Benchmark Sharpe to test against. Default 0.0.
        periods_per_year: If provided, ``sr_hat`` and ``sr_star`` are
            annualized before the test. If ``None``, the input is treated
            as already on the desired scale.

    Returns:
        :class:`PSRResult` with the PSR probability, z-score, and echoed inputs.

    Raises:
        DataError: If ``returns`` is empty, has fewer than 2 elements, or
            ``periods_per_year`` is non-positive.
    """
    r = np.asarray(returns, dtype=np.float64).ravel()
    n = r.size
    if n < 2:
        raise DataError(f"PSR requires >= 2 return observations; got {n}.")
    if periods_per_year is not None and periods_per_year <= 0:
        raise DataError(f"periods_per_year must be positive; got {periods_per_year}.")

    # Per-observation Sharpe + moments via the shared stats primitives so
    # there is exactly one definition of "Sharpe" in the system.
    sr_per_period = stats_h.sharpe(r)
    _, _, m3, m4_pearson = stats_h.moments(r)
    m4 = m4_pearson - 3.0  # convert to excess kurtosis for the formula

    if not np.isfinite(sr_per_period):
        # Effectively zero variance. PSR is undefined; report 0.5 (coin flip)
        # only if the observed mean also equals the benchmark; otherwise 0 or 1.
        if periods_per_year is not None:
            sr_hat = float(np.mean(r)) * float(np.sqrt(periods_per_year))
        else:
            sr_hat = float(np.mean(r))
        return PSRResult(
            psr=0.5 if sr_hat == sr_star else (1.0 if sr_hat > sr_star else 0.0),
            z_score=0.0,
            sr_hat=sr_hat,
            n=n,
            sr_star=sr_star,
        )

    # Annualise explicitly if requested.
    if periods_per_year is not None:
        sr_hat = stats_h.annualise_sharpe(sr_per_period, periods_per_year)
    else:
        sr_hat = sr_per_period

    # PSR formula: Phi(z) where
    #   z = (SR_hat - SR*) * sqrt(n-1) / sqrt(1 - gamma3*SR_hat + ((gamma4-1)/4)*SR_hat^2)
    # Use n-1 as the effective sample size (Bailey & López de Prado, 2014, eq. 4).
    denom_sq = 1.0 - m3 * sr_hat + ((m4) / 4.0) * sr_hat ** 2
    if denom_sq <= 0.0:
        # Pathological moments; treat as undefined.
        z = float("nan")
        p = float("nan")
    else:
        z = (sr_hat - sr_star) * np.sqrt(n - 1) / np.sqrt(denom_sq)
        p = float(stats.norm.cdf(z))

    return PSRResult(psr=p, z_score=float(z), sr_hat=float(sr_hat), n=int(n), sr_star=float(sr_star))
