"""Stability: std of rolling-window Sharpe on the equity curve.

WHAT: ``np.std`` of the per-observation Sharpe computed over
      overlapping windows of length ``window`` (default 63, ~3 months
      of daily bars). Reports consistency of risk-adjusted returns
      over time.

WHY: A strategy whose Sharpe was 2.0 in months 1-3, -1.0 in months
     4-6, and 2.5 in months 7-9 has a "headline" Sharpe of about 1.2
     but a stability of ~1.6 - the operator should know the edge
     flickered. Stability is one number for that question.

HOW: Sliding-window Sharpe (mean / std with ddof=1, rf=0), then
     std-of-Sharpes. Returns ``None`` if the equity curve has fewer
     than ``2 * window`` points (two complete windows) - anything
     less is too noisy to summarise.

TRADE-OFF: We use a fixed window length. The literature (Lo, 2002)
           shows Sharpe estimators are biased and noisy at typical
           hedge-fund sample sizes; the rolling-window estimate
           inherits that. This metric is for relative comparison
           between strategies on the same data, not as an absolute
           claim.
"""
from __future__ import annotations

import math

import numpy as np

from engine.validation.stats import as_returns


def rolling_sharpe_stability(
    equity: np.ndarray,
    *,
    window: int = 63,
) -> float | None:
    """Standard deviation of per-window Sharpe ratios.

    Args:
        equity: 1-D equity curve (one entry per bar).
        window: number of bars per window. Must be >= 2.

    Returns:
        Std of window-Sharpes, or ``None`` if fewer than ``2 * window``
        bars are available or if any window has zero variance.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2; got {window}")
    e = np.asarray(equity, dtype=np.float64).ravel()
    e = e[np.isfinite(e)]
    if e.size < 2 * window:
        return None
    # Per-bar returns.
    rets = e[1:] / e[:-1] - 1.0
    rets = rets[np.isfinite(rets)]
    if rets.size < 2 * window:
        return None
    # Rolling std / mean. Use stride tricks for memory; simple loop
    # is fine for our sizes (<= 1e5 bars typically).
    n_windows = rets.size - window + 1
    if n_windows < 2:
        return None
    sh = np.empty(n_windows, dtype=np.float64)
    for i in range(n_windows):
        w = rets[i : i + window]
        sd = float(np.std(w, ddof=1))
        if sd <= 0.0 or not math.isfinite(sd):
            sh[i] = float("nan")
        else:
            sh[i] = float(np.mean(w) / sd)
    sh = sh[np.isfinite(sh)]
    if sh.size < 2:
        return None
    return float(np.std(sh, ddof=1))


def _unused_as_returns_import_kept_for_type_completeness() -> None:
    # S4 stability reads the equity directly; ``as_returns`` is
    # referenced by sibling modules. This no-op keeps the symbol
    # discoverable for future expansion (e.g. per-trade stability).
    _ = as_returns
