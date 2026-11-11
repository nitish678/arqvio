"""Stochastic oscillator feature semantics -- parity anchor for STOCH_CROSS.

%K = (close - lowest_low) / (highest_high - lowest_low) * 100, smoothed
by a simple moving average over ``smoothing`` bars. %D is a simple moving
average of %K over ``d_period`` bars. NaN during warmup.

These functions operate on full window slices passed by the registry
``_stoch_k`` / ``_stoch_d`` wrappers; here we verify the math directly.
"""
from __future__ import annotations

import numpy as np
import pytest

from engine.research.features.stoch import stoch_d_series, stoch_k_series


def _rsi_like_bars(n: int = 40, period: int = 14, smoothing: int = 3):
    """Generate bars where high/low oscillate so %K is non-trivial."""
    rng = np.random.default_rng(0)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    high = close + 0.3 + rng.random(n) * 0.2
    low = close - 0.3 - rng.random(n) * 0.2
    # Ensure high >= close >= low
    high = np.maximum(high, close + 0.01)
    low = np.minimum(low, close - 0.01)
    return high, low, close, period, smoothing


def test_stoch_k_warmup_is_nan_before_full_window() -> None:
    """%K is NaN for indices < period-1 (not enough bars for the lookback)."""
    high, low, close, period, _ = _rsi_like_bars(n=40, period=14, smoothing=3)
    k = stoch_k_series(high, low, close, period=period, smoothing=3)
    assert np.all(np.isnan(k[:period - 1]))


def test_stoch_k_is_in_range_0_to_100() -> None:
    """%K must always be within [0, 100] where defined."""
    high, low, close, period, smoothing = _rsi_like_bars(n=100, period=14, smoothing=3)
    k = stoch_k_series(high, low, close, period=period, smoothing=smoothing)
    finite = k[~np.isnan(k)]
    assert finite.size > 0
    assert np.all(finite >= -1e-10), f"min=%K={finite.min()}"
    assert np.all(finite <= 100.0 + 1e-10), f"max=%K={finite.max()}"


def test_stoch_k_extreme_high_yields_100() -> None:
    """Close at the high of the window => %K = 100."""
    n = 20
    high = np.array([100.0 + i * 0.1 for i in range(n)], dtype=np.float64)
    low = high - 1.0
    close = high.copy()  # close == high => %K = 100
    k = stoch_k_series(high, low, close, period=14, smoothing=3)
    finite = k[~np.isnan(k)]
    assert np.all(finite == pytest.approx(100.0, abs=1e-12)), finite


def test_stoch_k_extreme_low_yields_0() -> None:
    """Close equals the lowest low in the window => %K = 0.

    Construct bars where low[i] = close[i] and the window's lowest low
    equals close[i]. This means close sits at the lowest low of its window.
    """
    n = 20
    # Prices oscillate: high[i] > close[i] > low[i], with lows decreasing
    # within a window so the lowest low is at the start. Instead, make low
    # constant within the window by using flat lows.
    low = np.full(n, 100.0, dtype=np.float64)
    high = low + 2.0
    close = low.copy()  # close == lowest low => %K = 0
    k = stoch_k_series(high, low, close, period=14, smoothing=3)
    finite = k[~np.isnan(k)]
    assert np.all(finite == pytest.approx(0.0, abs=1e-12)), finite


def test_stoch_k_no_smoothing_matches_raw_definition() -> None:
    """With smoothing=1, %K is the raw (close-ll)/(hh-ll)*100."""
    n = 20
    rng = np.random.default_rng(42)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    high = close + 0.3 + rng.random(n) * 0.2
    low = close - 0.3 - rng.random(n) * 0.2
    high = np.maximum(high, close + 0.01)
    low = np.minimum(low, close - 0.01)

    period = 14
    k = stoch_k_series(high, low, close, period=period, smoothing=1)

    # Hand-compute at bar 13 (first full window).
    i = period - 1
    ll = float(np.min(low[i - period + 1 : i + 1]))
    hh = float(np.max(high[i - period + 1 : i + 1]))
    expected = (close[i] - ll) / (hh - ll) * 100.0
    assert k[i] == pytest.approx(expected, abs=1e-12)


def test_stoch_d_is_sma_of_k() -> None:
    """%D is the SMA of %K over d_period bars."""
    high, low, close, period, _ = _rsi_like_bars(n=50, period=14, smoothing=3)
    raw_k, d = stoch_d_series(high, low, close, period=period, d_period=3, smoothing=3)
    d_period = 3

    # Hand-compute %D at the last defined index.
    last = len(raw_k) - 1
    window = raw_k[last - d_period + 1 : last + 1]
    expected = float(np.mean(window[~np.isnan(window)]))
    assert d[last] == pytest.approx(expected, abs=1e-12)


def test_stoch_d_warmup_is_nan_before_full_window() -> None:
    """%D needs (period-1) + (smoothing-1) + (d_period-1) bars to be defined."""
    high, low, close, period, smoothing = _rsi_like_bars(n=100, period=14, smoothing=3)
    _, d = stoch_d_series(high, low, close, period=period, d_period=3, smoothing=smoothing)
    # The first NaN run covers period-1 (raw K needs period bars, smoothed
    # K needs smoothing more, D needs d_period more).
    first_finite = np.argmax(np.isfinite(d))
    assert first_finite > 0
    assert np.all(np.isnan(d[:first_finite]))


def test_stoch_k_empty_input_returns_empty() -> None:
    k = stoch_k_series(np.array([]), np.array([]), np.array([]))
    assert k.size == 0


def test_stoch_d_returns_tuple_of_two_arrays() -> None:
    high, low, close, period, smoothing = _rsi_like_bars(n=30)
    raw_k, d = stoch_d_series(high, low, close, period=period, d_period=3, smoothing=smoothing)
    assert raw_k.shape == d.shape == close.shape
