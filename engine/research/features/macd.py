"""MACD line and signal line features (structural look-ahead safety).

Each function reads a window ending at bar t (inclusive) and never reaches
into the future. The registry enforces the lookback guard at the indexing layer.
"""
from __future__ import annotations

import numpy as np


def ema_series(close: np.ndarray, period: int) -> np.ndarray:
    """EMA with SMA seed (standard alpha = 2/(period+1)). NaN during warmup.

    The seed is the SMA of the first ``period`` FINITE values in the input.
    This allows calling ema_series on a series that has a NaN warmup (e.g.
    an EMA-of-EMA, or EMA of a MACD line that starts as NaN). The output is
    NaN for all indices before the seed index, and the seed itself is at the
    index of the last bar included in the seed window.
    """
    close = np.asarray(close, dtype=np.float64)
    n = close.size
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0 or period < 1:
        return out
    if period > n:
        return out
    alpha = 2.0 / (period + 1)
    finite_idx = np.where(np.isfinite(close))[0]
    if finite_idx.size < period:
        return out
    seed_start = int(finite_idx[0])
    seed_end = int(finite_idx[period - 1])
    seed_vals = close[seed_start : seed_end + 1]
    out[seed_end] = float(np.mean(seed_vals))
    for i in range(seed_end + 1, n):
        if not np.isfinite(close[i]):
            continue
        prev = out[i - 1]
        if not np.isfinite(prev):
            prev = out[seed_end]
        out[i] = alpha * close[i] + (1.0 - alpha) * prev
    return out


def macd_line_series(
    close: np.ndarray,
    *, fast_period: int = 12, slow_period: int = 26,
) -> np.ndarray:
    """MACD line = EMA(fast) - EMA(slow) for every bar."""
    close = np.asarray(close, dtype=np.float64)
    fast = ema_series(close, fast_period)
    slow = ema_series(close, slow_period)
    return fast - slow


def macd_signal_series(
    close: np.ndarray,
    *, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (macd_line, signal_line) for every bar."""
    macd_line = macd_line_series(close, fast_period=fast_period, slow_period=slow_period)
    signal_line = ema_series(macd_line, signal_period)
    return macd_line, signal_line


__all__ = ["macd_line_series", "macd_signal_series"]
