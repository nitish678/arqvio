"""MACD feature semantics -- parity anchor for MACD_CROSS.

MACD line = EMA(fast) - EMA(slow), where EMA uses the standard
``alpha = 2 / (period + 1)`` recursion with a simple-MA seed for the
first value. The signal line is EMA(signal_period) of the MACD line.

NaN during warmup (no value until the slow EMA has its seed).
"""
from __future__ import annotations

import numpy as np
import pytest

from engine.research.features.macd import (
    ema_series,
    macd_line_series,
    macd_signal_series,
)

# The registry in registry.py uses these defaults; the macd module doesn't
# export them, so define them here to match.
DEFAULT_FAST_PERIOD = 12
DEFAULT_SLOW_PERIOD = 26
DEFAULT_SIGNAL_PERIOD = 9


def test_ema_seed_is_simple_moving_average() -> None:
    """The first EMA value (at index period-1) is the SMA of the first period closes."""
    close = np.array([100.0, 101.0, 99.0, 102.0, 103.0, 104.0, 98.0, 105.0,
                      106.0, 107.0], dtype=np.float64)
    period = 5
    ema = ema_series(close, period)
    assert np.isnan(ema[:period - 1]).all()
    expected_seed = float(np.mean(close[:period]))
    assert ema[period - 1] == pytest.approx(expected_seed, abs=1e-12)


def test_ema_recursive_update_matches_definition() -> None:
    """EMA[i] = alpha * close[i] + (1-alpha) * EMA[i-1] for i >= period."""
    close = np.array([100.0, 101.0, 99.0, 102.0, 103.0, 104.0, 98.0, 105.0,
                      106.0, 107.0, 108.0, 109.0], dtype=np.float64)
    period = 5
    alpha = 2.0 / (period + 1)
    ema = ema_series(close, period)

    # Step through manually from the seed.
    manual = np.full(close.size, np.nan)
    manual[period - 1] = float(np.mean(close[:period]))
    for i in range(period, close.size):
        manual[i] = alpha * close[i] + (1.0 - alpha) * manual[i - 1]

    finite = np.isfinite(ema)
    np.testing.assert_allclose(ema[finite], manual[finite], rtol=1e-14, atol=0.0)


def test_macd_line_equals_ema_fast_minus_ema_slow() -> None:
    """MACD line = EMA(fast, fast_period) - EMA(slow, slow_period)."""
    rng = np.random.default_rng(0)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, 60))

    fast_ema = ema_series(close, DEFAULT_FAST_PERIOD)
    slow_ema = ema_series(close, DEFAULT_SLOW_PERIOD)
    macd = macd_line_series(close)

    expected = fast_ema - slow_ema
    finite = np.isfinite(macd)
    np.testing.assert_allclose(macd[finite], expected[finite], rtol=1e-14, atol=0.0)


def test_macd_line_warmup_respects_slow_period() -> None:
    """MACD is NaN until the slow EMA seeds (>= slow_period bars)."""
    rng = np.random.default_rng(1)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, 60))
    macd = macd_line_series(close)
    assert np.isnan(macd[:DEFAULT_SLOW_PERIOD - 1]).all()
    assert np.isfinite(macd[DEFAULT_SLOW_PERIOD - 1])


def test_macd_signal_is_ema_of_macd_line() -> None:
    """Signal line = EMA(macd_line, signal_period) on the finite portion.

    The signal EMA seeds at the index of the 9th finite MACD value
    (slow_period-1 + signal_period-1 = 25 + 8 = 33 with defaults). We verify
    the seed is the SMA of those 9 MACD values, then check the recursive
    update matches the standard EMA formula.
    """
    rng = np.random.default_rng(2)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, 100))
    macd_line, signal = macd_signal_series(close)

    alpha = 2.0 / (DEFAULT_SIGNAL_PERIOD + 1)
    finite_idx = np.where(np.isfinite(macd_line))[0]
    seed_idx = int(finite_idx[DEFAULT_SIGNAL_PERIOD - 1])

    # Seed: SMA of the first signal_period finite MACD values.
    seed_window = macd_line[finite_idx[0] : seed_idx + 1]
    seed = float(np.mean(seed_window))
    assert signal[seed_idx] == pytest.approx(seed, abs=1e-12)

    # Step forward: signal[i] = alpha * macd_line[i] + (1-alpha) * signal[i-1]
    for i in range(seed_idx + 1, 100):
        if not np.isfinite(signal[i]):
            continue
        expected = alpha * macd_line[i] + (1.0 - alpha) * signal[i - 1]
        assert signal[i] == pytest.approx(expected, abs=1e-12), f"i={i}"


def test_macd_signal_warmup() -> None:
    """Signal line needs slow_period-1 + signal_period-1 bars to be defined.

    With 100 bars and default periods (12/26/9), this is sufficient.
    """
    rng = np.random.default_rng(3)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, 100))
    _, signal = macd_signal_series(close)

    # Signal seeds at (slow_period-1) + (signal_period-1) = 25 + 8 = 33.
    min_warmup = DEFAULT_SLOW_PERIOD - 1 + DEFAULT_SIGNAL_PERIOD - 1
    assert np.isnan(signal[:min_warmup]).all(), "signal should be NaN during warmup"
    assert np.isfinite(signal[seed_idx := min_warmup])
    assert np.isfinite(signal[-1])


def test_macd_line_crosses_signal_at_extremes() -> None:
    """On a steep uptrend, MACD line should be above signal line eventually."""
    rng = np.random.default_rng(4)
    # Strong uptrend.
    close = 100.0 + np.arange(100) * 0.5 + rng.normal(0, 0.1, 100)
    macd_line, signal = macd_signal_series(close)
    assert macd_line[-1] > signal[-1]


def test_macd_zero_variance_input_gives_zero_macd() -> None:
    """Flat prices => EMA fast == EMA slow => MACD = 0."""
    close = np.full(60, 100.0, dtype=np.float64)
    macd = macd_line_series(close)
    # After both EMAs seed, the difference is zero.
    finite = macd[~np.isnan(macd)]
    np.testing.assert_allclose(finite, 0.0, atol=1e-12)


def test_macd_custom_periods() -> None:
    """Custom fast/slow/signal periods are respected."""
    rng = np.random.default_rng(5)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, 60))
    fast, slow, sig_p = 5, 10, 4
    macd = macd_line_series(close, fast_period=fast, slow_period=slow)
    expected = ema_series(close, fast) - ema_series(close, slow)
    finite = np.isfinite(expected)
    np.testing.assert_allclose(macd[finite], expected[finite], rtol=1e-14, atol=0.0)

    _, signal = macd_signal_series(
        close, fast_period=fast, slow_period=slow, signal_period=sig_p
    )
    expected_sig = ema_series(macd, sig_p)
    finite_sig = np.isfinite(expected_sig)
    np.testing.assert_allclose(
        signal[finite_sig], expected_sig[finite_sig], rtol=1e-14, atol=0.0
    )


def test_ema_empty_returns_empty() -> None:
    assert ema_series(np.array([]), 14).size == 0


def test_macd_line_empty_returns_empty() -> None:
    assert macd_line_series(np.array([])).size == 0
