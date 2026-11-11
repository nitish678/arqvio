"""C0: engine-level hour-bucket alignment fixture.

Proves the no-trade gate hour, the SpreadMatrix hour, and the bar hour
all agree — no drift post-C0. Uses the real SpreadMatrix.no_trade_hours
(computed from a zero spread at hour 21), feeds synthetic H1 bars
spanning hours 20/21/22 on a UTC basis, and asserts the engine blocks
fills in hour 21 and only hour 21.

This is the engine-level half of the C0 alignment contract (the preflight
half is tests/research/test_campaign_preflight_basis.py).
"""
from __future__ import annotations

import numpy as np
import pytest

from engine.data.spread_calibration import SpreadMatrix
from engine.research.backtest.engine import (
    BacktestConfig,
    BacktestEngine,
    OrderIntent,
)
from engine.research.data.mt5_source import BarData, Timeframe
from engine.research.search.real_backtest import spec_for

SYMBOL = "EURUSDm"
HOUR_S = 3600


def _hourly_bars(n: int, start_hour: int = 18, price: float = 100.0) -> BarData:
    """One H1 bar per hour from `start_hour` UTC on 1970-01-02, flat price."""
    base = 86400 + start_hour * HOUR_S
    times = np.array([base + i * HOUR_S for i in range(n)], dtype=np.int64)
    close = np.full(n, price, dtype=np.float64)
    return BarData(
        time=times,
        open=close.copy(),
        high=close + 1.0,
        low=close - 1.0,
        close=close.copy(),
        tick_volume=np.ones(n),
        spread=np.zeros(n, dtype=np.int32),
        real_volume=np.zeros(n),
        symbol=SYMBOL,
        timeframe=Timeframe.H1,
    )


def _hour_of(bar_time: int) -> int:
    return (int(bar_time) // HOUR_S) % 24


def _engine_from_matrix(matrix: SpreadMatrix):
    """BacktestEngine wired from a real SpreadMatrix — not a hand-coded tuple."""
    config = BacktestConfig(
        strategy_id="c0-hour-alignment",
        symbol=SYMBOL,
        timeframe="H1",
        initial_equity=10_000.0,
        charge_swap=False,
        charge_commission=False,
        slippage_mean_points=0.0,
        slippage_std_points=0.0,
        no_trade_hours=tuple(matrix.no_trade_hours),
        hourly_spread_points=tuple(matrix.mean_buckets),
    )
    return BacktestEngine(spec=spec_for(SYMBOL), config=config, rng_seed=0)


class _EnterOnBar:
    """Emits one buy intent on a chosen bar; fills on the next bar's open."""

    strategy_id = "enter-on-bar"

    def __init__(self, signal_bar: int) -> None:
        self.signal_bar = signal_bar

    def on_bar(self, bar_index, bars, state):
        if bar_index == self.signal_bar:
            return OrderIntent(direction="long", volume=0.1, bar_index=bar_index)
        return None


# A real spread matrix with hour 21 zero (no_trade), all other hours non-zero.
_MATRIX_H21 = SpreadMatrix(
    symbol=SYMBOL,
    buckets=tuple([1.2] * 21 + [0.0, 1.3, 1.1]),
    mean_buckets=tuple([1.0] * 21 + [0.0, 1.4, 1.2]),
    sample_count=1000,
    source="BUNDLED",
    calibrator_version=2,
)


def test_matrix_derives_no_trade_hours_21() -> None:
    """The fixture matrix must carry hour 21 in its derived no_trade set."""
    assert _MATRIX_H21.no_trade_hours == frozenset({21})


def test_engine_blocks_entry_fill_at_hour_21_only() -> None:
    """A signal whose fill lands in hour 21 is deferred; hour 20 and 22 fill."""
    # Bars start at 18:00 → index 0=18, 1=19, 2=20, 3=21, 4=22.
    # Signal on bar 2 (20:00) → fill would be bar 3 (21:00) → blocked.
    # Signal on bar 3 (21:00) → fill would be bar 4 (22:00) → fills normally.
    bars = _hourly_bars(5, start_hour=20)
    assert _hour_of(bars.time[0]) == 20
    assert _hour_of(bars.time[1]) == 21
    assert _hour_of(bars.time[2]) == 22
    assert _hour_of(bars.time[3]) == 23
    assert _hour_of(bars.time[4]) == 0

    # Re-derive the matrix so we also carry hour 0 as blocked (buckets[0] = 0.0).
    # Actually: keep hour 21 as the only zero hour and span 20/21/22 properly.
    # Use start_hour=19 so index 1=20, 2=21, 3=22.
    bars = _hourly_bars(4, start_hour=19)
    assert _hour_of(bars.time[1]) == 20
    assert _hour_of(bars.time[2]) == 21
    assert _hour_of(bars.time[3]) == 22

    engine = _engine_from_matrix(_MATRIX_H21)

    # Signal on hour-20 bar → fill lands at hour 21 → must be deferred.
    blocked_result = engine.run(bars, _EnterOnBar(signal_bar=1))
    assert blocked_result.n_trades == 1, "deferred, not dropped"
    filled_hour = _hour_of(blocked_result.trades[0].entry_fill.bar_time)
    assert filled_hour == 22, (
        f"fill landed at hour {filled_hour}, expected 22 (deferred past 21)"
    )
    assert blocked_result.blocked_zero_spread_events >= 1

    # Signal on hour-21 bar itself → fills at hour 22 (next tradeable bar).
    blocked2 = engine.run(bars, _EnterOnBar(signal_bar=2))
    assert blocked2.n_trades == 1
    filled_hour2 = _hour_of(blocked2.trades[0].entry_fill.bar_time)
    assert filled_hour2 == 22, (
        f"signal inside window (hour 21) should fill at 22, got {filled_hour2}"
    )


def test_no_window_fills_at_hour_21_normally() -> None:
    """Without the gate, hour-21 fills return to normal — proves the gate fired."""
    # Build a matrix with NO zero hours.
    no_block_matrix = SpreadMatrix(
        symbol=SYMBOL,
        buckets=tuple([1.0] * 24),
        mean_buckets=tuple([0.9] * 24),
        sample_count=1000,
        source="BUNDLED",
        calibrator_version=2,
    )
    assert no_block_matrix.no_trade_hours == frozenset()

    bars = _hourly_bars(4, start_hour=19)
    assert _hour_of(bars.time[2]) == 21

    engine = _engine_from_matrix(no_block_matrix)
    result = engine.run(bars, _EnterOnBar(signal_bar=1))

    assert result.n_trades == 1
    filled_hour = _hour_of(result.trades[0].entry_fill.bar_time)
    assert filled_hour == 21, (
        f"without a gate the fill should be at 21, got {filled_hour}"
    )
    assert result.blocked_zero_spread_events == 0


def test_matrix_hour_21_matches_bar_hour_no_drift() -> None:
    """The SpreadMatrix.zero hour (21), the bar's UTC hour (21), and the
    gate's blocked hour (21) must all be the same integer.

    This is the core C0 invariant: the hour the artifact says is
    untradeable is the same hour the bar timestamp encodes.
    """
    # The matrix says hour 21 is blocked.
    assert 21 in _MATRIX_H21.no_trade_hours

    # A bar whose timestamp lands on hour 21 UTC.
    bar_time_21 = 86400 + 21 * HOUR_S  # 1970-01-02 21:00:00 UTC
    assert _hour_of(bar_time_21) == 21

    # The engine's gate uses (bar_time // 3600) % 24, which must equal 21.
    # If the basis were wrong (e.g., broker time with no conversion), the bar
    # at "21:00 server" would map to a different hour and the gate would not
    # fire or would fire on the wrong bar.
    engine = _engine_from_matrix(_MATRIX_H21)
    no_trade = frozenset(int(h) for h in engine.config.no_trade_hours)
    assert 21 in no_trade
    # And the bar time that should be blocked:
    assert (bar_time_21 // 3600) % 24 in no_trade


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
