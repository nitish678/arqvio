"""B2: the behaviour metrics, against a hand-computed 6-trade series.

Every expected number below is written out so it can be checked
without running the code.

The series (net P&L per trade, in order):

    +100, +50, -30, -20, +80, -10

    wins   at 0, 1, 4          losses at 2, 3, 5
    win runs   : [2, 1]        loss runs : [2, 1]
    total      : +170
"""
from __future__ import annotations

import numpy as np
import pytest

from engine.research.backtest.order import Fill, Trade
from engine.research.metrics.behaviour import (
    DEFAULT_IS_FRACTION,
    PERSISTENCE_WINDOW_TRADES,
    avg_hold_bars,
    in_out_sample,
    linearity_r2,
    long_short_split,
    persistence,
    recovery_factor,
    streaks,
)

PNLS = np.array([100.0, 50.0, -30.0, -20.0, 80.0, -10.0])


def _fill(bar: int) -> Fill:
    return Fill(
        fill_id=f"f{bar}", order_id=str(bar), direction="long", volume=0.1,
        requested_price=1.1, filled_price=1.1, spread_cost=0.0,
        slippage_cost=0.0, commission_cost=0.0, bar_index=bar,
        bar_time=86400 + bar * 3600,
    )


def _trade(entry: int, exit_: int | None, direction: str = "long") -> Trade:
    return Trade(
        trade_id=f"t{entry}", strategy_id="s", symbol="USDCADm",
        direction=direction, entry_fill=_fill(entry),
        exit_fill=_fill(exit_) if exit_ is not None else None,
        entry_bar_index=entry, exit_bar_index=exit_, volume=0.1,
    )


def test_streaks_hand_computed() -> None:
    """win runs [2, 1] -> max 2, avg 1.5; loss runs [2, 1] -> same."""
    s = streaks(PNLS)
    assert s.max_wins == 2
    assert s.avg_wins == pytest.approx(1.5)
    assert s.max_losses == 2
    assert s.avg_losses == pytest.approx(1.5)


def test_a_zero_trade_breaks_a_streak_rather_than_extending_it() -> None:
    s = streaks(np.array([10.0, 0.0, 10.0]))
    assert s.max_wins == 1, "a zero-P&L trade extended a winning streak"


def test_streaks_with_no_losses() -> None:
    s = streaks(np.array([1.0, 2.0, 3.0]))
    assert s.max_wins == 3
    assert s.max_losses == 0
    assert s.avg_losses is None, "no losses means undefined, not 0.0"


def test_avg_hold_bars_hand_computed() -> None:
    """Spans 5, 3, 10 -> mean 6.0. The open trade is excluded."""
    trades = (_trade(0, 5), _trade(10, 13), _trade(20, 30), _trade(40, None))
    assert avg_hold_bars(trades) == pytest.approx(6.0)
    assert avg_hold_bars(()) is None


def test_long_short_split_hand_computed() -> None:
    trades = (
        _trade(0, 1, "long"), _trade(2, 3, "long"),
        _trade(4, 5, "long"), _trade(6, 7, "short"),
    )
    n_long, n_short, frac = long_short_split(trades)
    assert (n_long, n_short) == (3, 1)
    assert frac == pytest.approx(0.75)
    assert long_short_split(())[2] is None


def test_recovery_factor_hand_computed() -> None:
    """+170 return over a -0.20 drawdown -> 850.0."""
    assert recovery_factor(170.0, -0.20) == pytest.approx(850.0)
    # Sign of the drawdown must not matter.
    assert recovery_factor(170.0, 0.20) == pytest.approx(850.0)


def test_recovery_factor_without_a_drawdown_is_undefined() -> None:
    """Not 0.0: that would rank it below a strategy that recovered."""
    assert recovery_factor(170.0, 0.0) is None


def test_linearity_is_one_on_a_straight_line() -> None:
    equity = np.array([100.0, 110.0, 120.0, 130.0, 140.0])
    assert linearity_r2(equity) == pytest.approx(1.0, abs=1e-12)


def test_linearity_separates_a_steady_climb_from_one_lucky_jump() -> None:
    """Both end at the same place; only one earned it steadily."""
    steady = np.array([100.0, 105.0, 110.0, 115.0, 120.0, 125.0])
    jump = np.array([100.0, 100.0, 100.0, 125.0, 125.0, 125.0])
    assert linearity_r2(steady) > linearity_r2(jump)


def test_linearity_undefined_on_a_flat_curve() -> None:
    assert linearity_r2(np.full(10, 100.0)) is None
    assert linearity_r2(np.array([1.0, 2.0])) is None


def test_persistence_hand_computed() -> None:
    """Window 3 over the 6-trade series.

        [100, 50, -30] = +120  profitable
        [50, -30, -20] =    0  NOT (> 0 is strict)
        [-30, -20, 80] =  +30  profitable
        [-20, 80, -10] =  +50  profitable

    3 of 4 windows -> 0.75
    """
    assert persistence(PNLS, window=3) == pytest.approx(0.75)


def test_persistence_needs_a_full_window() -> None:
    assert persistence(PNLS, window=PERSISTENCE_WINDOW_TRADES) is None
    assert persistence(np.arange(PERSISTENCE_WINDOW_TRADES, dtype=float)) is not None


def test_in_out_sample_hand_computed() -> None:
    """6 trades at the 70% default -> cut at round(4.2) = 4.

        IS  = [100, 50, -30, -20] = +100, 4 trades
        OOS = [80, -10]           =  +70, 2 trades
    """
    out = in_out_sample(PNLS)
    assert out is not None
    assert out.is_fraction == pytest.approx(DEFAULT_IS_FRACTION)
    assert (out.n_trades_is, out.n_trades_oos) == (4, 2)
    assert out.return_is == pytest.approx(100.0)
    assert out.return_oos == pytest.approx(70.0)


def test_in_out_sample_always_leaves_both_sides_non_empty() -> None:
    """An extreme fraction must not produce an empty half."""
    for fraction in (0.0, 0.01, 0.99, 1.0):
        out = in_out_sample(PNLS, is_fraction=fraction)
        assert out is not None
        assert out.n_trades_is >= 1
        assert out.n_trades_oos >= 1
        assert out.n_trades_is + out.n_trades_oos == PNLS.size


def test_in_out_sample_undefined_below_two_trades() -> None:
    assert in_out_sample(np.array([1.0])) is None
