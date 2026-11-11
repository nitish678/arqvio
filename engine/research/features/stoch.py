"""Stochastic oscillator features (structural look-ahead safety).

Each function reads a window ending at bar t (inclusive) and never reaches
into the future. The registry enforces the lookback guard at the indexing layer;
these functions operate on the already-sliced window.

STOCH_K / STOCH_D are the two line values the StochCross strategy needs.
"""
from __future__ import annotations

import numpy as np


def _raw_stoch_into(
    high: np.ndarray, low: np.ndarray, close: np.ndarray,
    period: int, start: int, stop: int, out: np.ndarray,
) -> None:
    """out[i], i in [start, stop): 100*(C - LL)/(HH - LL) over [i-period+1, i].
    Left untouched (NaN) before period-1 and where HH == LL."""
    for i in range(max(start, period - 1), stop):
        lo = i - period + 1
        ll = float(np.min(low[lo : i + 1]))
        hh = float(np.max(high[lo : i + 1]))
        denom = hh - ll
        if denom > 0:
            out[i] = (close[i] - ll) / denom * 100.0


def _full_window_mean_into(
    src: np.ndarray, size: int, start: int, stop: int, out: np.ndarray,
) -> None:
    """out[i], i in [start, stop): mean(src[i-size+1 : i+1]) when none is NaN."""
    for i in range(max(start, size - 1), stop):
        window = src[i - size + 1 : i + 1]
        valid = window[~np.isnan(window)]
        if valid.size == size:
            out[i] = float(np.mean(valid))


def stoch_k_series(
    high: np.ndarray, low: np.ndarray, close: np.ndarray,
    *, period: int = 14, smoothing: int = 3,
) -> np.ndarray:
    """Smoothed %K for every bar. NaN during warmup."""
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    n = high.size
    raw_k = np.full(n, np.nan, dtype=np.float64)
    if n == 0 or period > n or smoothing > n:
        return raw_k

    raw_stoch = np.full(n, np.nan, dtype=np.float64)
    _raw_stoch_into(high, low, close, period, 0, n, raw_stoch)
    _full_window_mean_into(raw_stoch, smoothing, 0, n, raw_k)
    return raw_k


def stoch_d_series(
    high: np.ndarray, low: np.ndarray, close: np.ndarray,
    *, period: int = 14, d_period: int = 3, smoothing: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (raw_K, slow_D) for every bar."""
    raw_k = stoch_k_series(high, low, close, period=period, smoothing=smoothing)
    n = raw_k.size
    slow_d = np.full(n, np.nan, dtype=np.float64)
    _full_window_mean_into(raw_k, d_period, 0, n, slow_d)
    return raw_k, slow_d


class StochPrefixCache:
    """stoch_d_series over a window that grows, extended at the tail only.

    Every per-bar value is causal (index t reads bars 0..t alone), so the
    series of window [0, n) is the series of [0, m) plus indices m..n-1
    computed by the same helpers -- bit for bit what stoch_d_series(window)
    returns, at O(new bars) per call instead of O(window). A window that
    is not an extension of the cached one (shorter, or any cached bar
    changed) starts over from bar 0.
    """

    def __init__(self, *, period: int, d_period: int, smoothing: int) -> None:
        self.params = (period, d_period, smoothing)
        self._n = 0
        empty = np.zeros(0, dtype=np.float64)
        self._high = self._low = self._close = empty
        self._raw = self._k = self._d = empty

    def _reserve(self, n: int, keep: int) -> None:
        if n <= self._high.size:
            return
        cap = max(n, 2 * self._high.size)

        def grown(a: np.ndarray) -> np.ndarray:
            out = np.full(cap, np.nan, dtype=np.float64)
            out[:keep] = a[:keep]
            return out

        self._high, self._low, self._close = (grown(a) for a in (self._high, self._low, self._close))
        self._raw, self._k, self._d = (grown(a) for a in (self._raw, self._k, self._d))

    def series(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """(raw_K, slow_D) of the window, as views of length ``high.size``."""
        high = np.asarray(high, dtype=np.float64)
        low = np.asarray(low, dtype=np.float64)
        close = np.asarray(close, dtype=np.float64)
        n = high.size
        m = self._n
        if n < m or not (
            np.array_equal(high[:m], self._high[:m], equal_nan=True)
            and np.array_equal(low[:m], self._low[:m], equal_nan=True)
            and np.array_equal(close[:m], self._close[:m], equal_nan=True)
        ):
            m = 0
        self._reserve(n, m)
        self._high[m:n] = high[m:n]
        self._low[m:n] = low[m:n]
        self._close[m:n] = close[m:n]
        self._raw[m:n] = np.nan
        self._k[m:n] = np.nan
        self._d[m:n] = np.nan
        period, d_period, smoothing = self.params
        _raw_stoch_into(self._high, self._low, self._close, period, m, n, self._raw)
        _full_window_mean_into(self._raw, smoothing, m, n, self._k)
        _full_window_mean_into(self._k, d_period, m, n, self._d)
        self._n = n
        return self._k[:n], self._d[:n]


__all__ = ["StochPrefixCache", "stoch_d_series", "stoch_k_series"]
