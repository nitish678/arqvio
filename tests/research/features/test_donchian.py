"""Donchian channel feature semantics -- parity anchor for DONCHIAN_BREAKOUT.

Upper channel = max(high) over ``period`` bars ending at bar t.
Lower channel = min(low) over ``period`` bars ending at bar t.

NaN for the first ``period-1`` bars (window not yet full).
"""
from __future__ import annotations

import numpy as np
import pytest

from engine.research.features.donchian import (
    donchian_high_series,
    donchian_low_series,
)

# The registry in registry.py uses lookback=DEFAULT_DONCHIAN_PERIOD (20).
# The feature module itself doesn't export it, so define it here.
DEFAULT_DONCHIAN_PERIOD = 20


def test_donchian_high_warmup_is_nan_for_first_period_minus_one() -> None:
    """Upper channel is NaN for indices 0..period-2; bar period-1 is the first."""
    n = 25
    high = np.array([100.0 + i * 0.1 for i in range(n)], dtype=np.float64)
    upper = donchian_high_series(high, period=DEFAULT_DONCHIAN_PERIOD)
    assert np.all(np.isnan(upper[:DEFAULT_DONCHIAN_PERIOD - 1]))
    assert np.isfinite(upper[DEFAULT_DONCHIAN_PERIOD - 1])


def test_donchian_low_warmup_is_nan_for_first_period_minus_one() -> None:
    n = 25
    low = np.array([100.0 - i * 0.1 for i in range(n)], dtype=np.float64)
    lower = donchian_low_series(low, period=DEFAULT_DONCHIAN_PERIOD)
    assert np.all(np.isnan(lower[:DEFAULT_DONCHIAN_PERIOD - 1]))
    assert np.isfinite(lower[DEFAULT_DONCHIAN_PERIOD - 1])


def test_donchian_high_is_running_max_of_window() -> None:
    """Upper channel at bar t = max(high[t-period+1 : t+1])."""
    n = 30
    rng = np.random.default_rng(0)
    high = 100.0 + rng.random(n) * 10.0

    period = DEFAULT_DONCHIAN_PERIOD
    upper = donchian_high_series(high, period=period)

    for t in range(period - 1, n):
        window = high[t - period + 1 : t + 1]
        expected = float(np.max(window))
        assert upper[t] == pytest.approx(expected, abs=1e-12), f"t={t}"


def test_donchian_low_is_running_min_of_window() -> None:
    """Lower channel at bar t = min(low[t-period+1 : t+1])."""
    n = 30
    rng = np.random.default_rng(1)
    low = 90.0 - rng.random(n) * 10.0

    period = DEFAULT_DONCHIAN_PERIOD
    lower = donchian_low_series(low, period=period)

    for t in range(period - 1, n):
        window = low[t - period + 1 : t + 1]
        expected = float(np.min(window))
        assert lower[t] == pytest.approx(expected, abs=1e-12), f"t={t}"


def test_donchian_high_at_peak_equals_peak_value() -> None:
    """If bar t is the highest in its window, upper = high[t]."""
    n = 25
    high = np.array([100.0 + i * 0.1 for i in range(n)], dtype=np.float64)
    period = DEFAULT_DONCHIAN_PERIOD
    upper = donchian_high_series(high, period=period)
    for t in range(period - 1, n):
        assert upper[t] == pytest.approx(high[t], abs=1e-12), f"t={t}"


def test_donchian_low_at_trough_equals_trough_value() -> None:
    """If bar t is the lowest in its window, lower = low[t]."""
    n = 25
    low = np.array([100.0 - i * 0.1 for i in range(n)], dtype=np.float64)
    period = DEFAULT_DONCHIAN_PERIOD
    lower = donchian_low_series(low, period=period)
    for t in range(period - 1, n):
        assert lower[t] == pytest.approx(low[t], abs=1e-12), f"t={t}"


def test_donchian_upper_always_geq_lower_on_same_series() -> None:
    """For any high/low pair, upper >= lower at every defined bar."""
    n = 30
    rng = np.random.default_rng(2)
    high = 100.0 + rng.random(n) * 10.0
    low = high - rng.random(n) * 5.0  # low < high

    period = DEFAULT_DONCHIAN_PERIOD
    upper = donchian_high_series(high, period=period)
    lower = donchian_low_series(low, period=period)

    mask = np.isfinite(upper) & np.isfinite(lower)
    assert mask.sum() > 0
    assert np.all(upper[mask] >= lower[mask]), (
        "upper channel < lower channel at some bar"
    )


def test_donchian_custom_period() -> None:
    """A custom period changes the warmup length and window size."""
    period = 10
    n = 20
    rng = np.random.default_rng(3)
    high = 100.0 + rng.random(n) * 10.0
    upper = donchian_high_series(high, period=period)
    assert np.all(np.isnan(upper[:period - 1]))
    assert np.isfinite(upper[period - 1])

    # Verify the window is exactly `period` bars wide.
    expected = float(np.max(high[0 : period]))  # high[0..period-1]
    assert upper[period - 1] == pytest.approx(expected, abs=1e-12)


def test_donchian_empty_input_returns_empty() -> None:
    assert donchian_high_series(np.array([]), period=14).size == 0
    assert donchian_low_series(np.array([]), period=14).size == 0


def test_donchian_period_too_large_returns_all_nan() -> None:
    """If the series is shorter than the period, all values are NaN."""
    short = np.array([100.0, 101.0, 102.0])
    upper = donchian_high_series(short, period=DEFAULT_DONCHIAN_PERIOD)
    assert upper.size == 3
    assert np.all(np.isnan(upper))
