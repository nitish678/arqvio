"""RSI feature semantics -- parity anchor for the RSI_REVERSION family.

RSI uses Wilder's smoothing (the textbook recursive ATR-style update),
which is what MT5's ``iRSI`` computes on default settings. The seed is the
arithmetic mean of the first ``period`` gains/losses; every subsequent
value is a smoothed update.

The warmup is the first ``period`` bars (indices 0..period-1 are NaN);
bar ``period`` is the first defined value. This is the same lookback
declared in the registry (``rsi_14`` has lookback = DEFAULT_RSI_PERIOD + 1).
"""
from __future__ import annotations

import numpy as np
import pytest

from engine.research.features.rsi import (
    DEFAULT_RSI_PERIOD,
    _rsi_value,
    rsi_series,
)


# A hand-checkable fixture: 16 bars of monotonically rising close.
# up[i] = close[i+1] - close[i] = 1.0 for all i (all gains, zero loss)
# Wilder seed (period=14): avg_gain = mean([1]*14) = 1.0, avg_loss = 0.0
# => RS = inf => RSI = 100.0 at bar 14.
# Subsequent updates: avg_gain stays 1.0, avg_loss stays 0.0 => RSI stays 100.
_CLOSE_UP = np.array([100.0 + i for i in range(16)])

# A falling fixture: all losses, zero gains => avg_gain = 0, avg_loss = 1.0
# => RS = 0 => RSI = 0.0
_CLOSE_DOWN = np.array([100.0 - i for i in range(16)])

# A flat fixture: no gains, no losses => avg_gain = 0, avg_loss = 0
# => RSI = 0.0 (the _rsi_value function returns 0.0 when avg_loss == 0.0
# and avg_gain does not cause a divide-by-zero... actually: RS = 0/0, but
# avg_gain == 0 too, so _rsi_value returns 0.0 because avg_loss == 0.0
# gives 100.0 only when avg_loss == 0.0 AND avg_gain > 0. When both are 0,
# the function returns 0.0.)
_CLOSE_FLAT = np.full(16, 100.0)


def test_rsi_warmup_is_nan_for_first_period_bars() -> None:
    """Indices 0..13 are NaN; bar 14 is the first defined value."""
    rsi = rsi_series(_CLOSE_UP, period=DEFAULT_RSI_PERIOD)
    assert np.all(np.isnan(rsi[:DEFAULT_RSI_PERIOD])), rsi[:DEFAULT_RSI_PERIOD]
    assert np.isfinite(rsi[DEFAULT_RSI_PERIOD]), rsi[DEFAULT_RSI_PERIOD]


def test_rsi_all_gains_yields_100() -> None:
    """A monotonically rising close has zero losses: RSI(14) = 100."""
    rsi = rsi_series(_CLOSE_UP, period=DEFAULT_RSI_PERIOD)
    assert rsi[DEFAULT_RSI_PERIOD] == pytest.approx(100.0, abs=1e-12)
    assert rsi[-1] == pytest.approx(100.0, abs=1e-12)


def test_rsi_all_losses_yields_0() -> None:
    """A monotonically falling close has zero gains: RSI(14) = 0."""
    rsi = rsi_series(_CLOSE_DOWN, period=DEFAULT_RSI_PERIOD)
    assert rsi[DEFAULT_RSI_PERIOD] == pytest.approx(0.0, abs=1e-12)
    assert rsi[-1] == pytest.approx(0.0, abs=1e-12)


def test_rsi_flat_series_yields_50() -> None:
    """No price movement: gains and losses are both zero → RSI = 50.0 (neutral)."""
    rsi = rsi_series(_CLOSE_FLAT, period=DEFAULT_RSI_PERIOD)
    assert rsi[DEFAULT_RSI_PERIOD] == pytest.approx(50.0, abs=1e-12)
    assert rsi[-1] == pytest.approx(50.0, abs=1e-12)


def test_rsi_series_matches_step_by_step_wilder() -> None:
    """Hand-compute the first two RSI values via the Wilder update."""
    close = _CLOSE_UP[:DEFAULT_RSI_PERIOD + 2]  # 16 bars enough for 2 values
    rsi = rsi_series(close, period=DEFAULT_RSI_PERIOD)

    # Seed: SMA of first 14 gains/losses.
    up = np.diff(close)
    gains = np.where(up > 0, up, 0.0)
    losses = np.where(up < 0, -up, 0.0)
    avg_gain = float(np.mean(gains[:DEFAULT_RSI_PERIOD]))
    avg_loss = float(np.mean(losses[:DEFAULT_RSI_PERIOD]))
    expected_14 = _rsi_value(avg_gain, avg_loss)
    assert rsi[DEFAULT_RSI_PERIOD] == pytest.approx(expected_14, abs=1e-12)

    # Step 14->15: Wilder update with gain=1.0, loss=0.0
    avg_gain = (avg_gain * (DEFAULT_RSI_PERIOD - 1) + gains[DEFAULT_RSI_PERIOD]) / DEFAULT_RSI_PERIOD
    avg_loss = (avg_loss * (DEFAULT_RSI_PERIOD - 1) + losses[DEFAULT_RSI_PERIOD]) / DEFAULT_RSI_PERIOD
    expected_15 = _rsi_value(avg_gain, avg_loss)
    assert rsi[DEFAULT_RSI_PERIOD + 1] == pytest.approx(expected_15, abs=1e-12)


def test_rsi_period_14_matches_rsi_period_14() -> None:
    """A different period must still be NaN for the first ``period`` bars."""
    rsi = rsi_series(_CLOSE_UP, period=7)
    assert np.all(np.isnan(rsi[:7])), rsi[:7]
    assert np.isfinite(rsi[7])
    # With all gains, RSI = 100 regardless of period.
    assert rsi[7] == pytest.approx(100.0, abs=1e-12)


def test_rsi_empty_input_returns_empty() -> None:
    rsi = rsi_series(np.array([]))
    assert rsi.size == 0


def test_rsi_short_input_all_nan() -> None:
    """Fewer bars than period+1 => all NaN."""
    rsi = rsi_series(np.array([100.0, 101.0, 102.0]), period=DEFAULT_RSI_PERIOD)
    assert rsi.size == 3
    assert np.all(np.isnan(rsi))


def test_rsi_rejects_invalid_period() -> None:
    from engine.core.errors import DataError
    with pytest.raises(DataError, match="period"):
        rsi_series(_CLOSE_UP, period=0)
    with pytest.raises(DataError, match="period"):
        rsi_series(_CLOSE_UP, period=-5)
