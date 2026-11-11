"""Deflated Sharpe Ratio (DSR).

WHAT: Adjusts the observed Sharpe for (a) the number of trials that produced
      it, (b) the non-normality of returns, and (c) the sample size.
      Returns the probability that the observed Sharpe would arise from an
      unskilled strategy that produced the *best* of N trials.
WHY: A high Sharpe selected from N=1 is informative. The same Sharpe selected
      from N=100,000 is almost certainly an overfit.
HOW: Uses PSR machinery (skew/kurtosis-adjusted), plus the expected-maximum
      Sharpe under the null (E[max SR] for N i.i.d. unskilled trials).
      The DSR is then ``Phi( (SR_hat - E[max SR]) * sqrt(n-1) / sqrt(...) )``.
TRADE-OFF: DSR is one number. It is informative, not dispositive. A low DSR
      rejects the strategy; a high DSR raises it for further checks (PBO,
      OOS) but does not certify it.
TEST: ``tests/validation/test_dsr.py`` — synthetic noise DSR<=0, increasing
      trials reduces DSR.
FAILURE: Invalid inputs raise ``DataError``.

References: Bailey & López de Prado (2014), eq. (5)-(7).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

from engine.core.errors import DataError
from engine.validation import stats as stats_h
from engine.validation.psr import PSRResult, psr


@dataclass(frozen=True)
class DSRResult:
    """Deflated Sharpe Ratio result.

    Attributes:
        dsr: Probability the observed SR exceeds the expected best SR under
            the null of N unskilled trials (in [0, 1]).
        psr_components: The PSR computation at the same returns/benchmark.
        sr_hat: Observed (annualized if applicable) Sharpe.
        n: Number of return observations.
        n_trials: Number of trials in the campaign.
        sr_star: Benchmark Sharpe.
        expected_max_sr: E[max SR] under the null (informational).
    """

    dsr: float
    psr_components: PSRResult
    sr_hat: float
    n: int
    n_trials: int
    sr_star: float
    expected_max_sr: float


def _expected_max_sharpe(
    n_obs: int,
    n_trials: int,
    skew: float,
    excess_kurt: float,
    *,
    var_sr_trials: float | None = None,
) -> float:
    """Expected maximum Sharpe under the null of N iid unskilled trials.

    Formula (Bailey & López de Prado, 2014, eq. (6)):

        E[max SR_i] = sqrt(V[SR]) * ((1 - gamma) Phi^{-1}(1 - 1/N)
                                     + gamma * Phi^{-1}(1 - 1/(N*e)))

    The scale ``sqrt(V[SR])`` is the standard deviation of the trial-Sharpe
    estimator. Two sources are possible:

    - ``var_sr_trials`` supplied: the OBSERVED cross-trial variance of the
      actual candidate Sharpes in the cell. This is the correct scale for a
      real search — a tightly clustered population and a wildly dispersed one
      have different luck bars. This is what defect-fix A2 wires through.
    - ``var_sr_trials`` None: fall back to the theoretical estimator variance
      ``1 / (n_obs - 1)`` (valid only when all trials share one return series
      of length ``n_obs`` and the benchmark SR is 0).
    """
    if n_trials <= 1:
        return 0.0
    if n_obs < 2:
        return 0.0

    if var_sr_trials is not None and math.isfinite(var_sr_trials) and var_sr_trials > 0.0:
        sqrt_var = math.sqrt(var_sr_trials)
    else:
        # Theoretical estimator variance at benchmark SR* = 0.
        var_sr_star = 1.0
        sqrt_var = math.sqrt(max(var_sr_star, 1e-12) / max(n_obs - 1, 1))
    z_n = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z_ne = stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    euler_gamma = 0.5772156649015329
    em1 = (1.0 - euler_gamma) * z_n + euler_gamma * z_ne
    return float(em1 * sqrt_var)


def dsr(
    returns: np.ndarray,
    n_trials: int,
    sr_star: float = 0.0,
    *,
    periods_per_year: int | None = None,
    var_sr_trials: float | None = None,
) -> DSRResult:
    """Compute the Deflated Sharpe Ratio.

    Args:
        returns: 1-D array of per-period arithmetic returns.
        n_trials: Number of trials that produced this strategy (>= 1).
        sr_star: Benchmark Sharpe to test against. Default 0.0.
        periods_per_year: If provided, ``sr_hat`` is annualized before
            evaluation. If ``None``, returns are taken as already scaled.
        var_sr_trials: Observed variance of the trial-Sharpe population in
            this cell. When supplied, the expected-max-Sharpe luck bar is
            scaled by ``sqrt(var_sr_trials)`` instead of the theoretical
            ``1/(n_obs-1)``. A dispersed search then earns a higher bar than
            a clustered one, as it must.

    Returns:
        :class:`DSRResult`.

    Raises:
        DataError: If inputs are invalid.
    """
    if n_trials < 1:
        raise DataError(f"n_trials must be >= 1; got {n_trials}.")
    if var_sr_trials is not None and var_sr_trials < 0.0:
        raise DataError(f"var_sr_trials must be >= 0; got {var_sr_trials}.")


    # First compute PSR (this also validates the returns array).
    p = psr(returns, sr_star=sr_star, periods_per_year=periods_per_year)

    r = np.asarray(returns, dtype=np.float64).ravel()
    n = r.size
    scale = float(np.sqrt(periods_per_year)) if periods_per_year is not None else 1.0
    raw_std = float(np.std(r, ddof=1))
    if raw_std <= max(abs(float(np.mean(r))) * scale, 1.0) * 1e-12:
        # Degenerate; DSR is undefined.
        return DSRResult(
            dsr=0.5 if p.sr_hat == sr_star else (1.0 if p.sr_hat > sr_star else 0.0),
            psr_components=p,
            sr_hat=p.sr_hat,
            n=n,
            n_trials=n_trials,
            sr_star=sr_star,
            expected_max_sr=0.0,
        )

    sr_hat = p.sr_hat
    # Moments on the per-period scale (Pearson kurtosis, Gaussian=3).
    _, _, m3, m4_pearson = stats_h.moments(r)
    m4 = m4_pearson - 3.0  # convert to excess kurtosis for the formula

    expected_max = _expected_max_sharpe(
        n, n_trials, skew=m3, excess_kurt=m4, var_sr_trials=var_sr_trials
    )

    # Apply the same denominator used by PSR, then z = (sr_hat - expected_max) / denom.
    denom_sq = 1.0 - m3 * sr_hat + ((m4) / 4.0) * sr_hat ** 2
    if denom_sq <= 0.0:
        z = float("nan")
        prob = float("nan")
    else:
        z = (sr_hat - expected_max) * np.sqrt(n - 1) / np.sqrt(denom_sq)
        prob = float(stats.norm.cdf(z))

    return DSRResult(
        dsr=prob,
        psr_components=p,
        sr_hat=float(sr_hat),
        n=int(n),
        n_trials=int(n_trials),
        sr_star=float(sr_star),
        expected_max_sr=float(expected_max),
    )
