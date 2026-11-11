"""The strategy inspector's trade statistics (pure arithmetic over closed trades and the equity curve)."""
from __future__ import annotations

import numpy as np
import pytest

from engine.research.vault.inspect import CURVE_POINTS, summarise


def test_trade_statistics_are_read_from_the_closed_trades() -> None:
    pnl = [50.0, -20.0, 30.0, 30.0, -10.0, -10.0, -10.0]
    dirs = [1, -1, 1, 1, -1, -1, 1]
    holds = [2, 4, 6, 2, 2, 2, 2]
    eq = np.array([10_000.0, 10_050.0, 10_030.0, 10_060.0, 10_090.0, 10_080.0, 10_070.0, 10_060.0])
    times = np.arange(len(eq)) * 3600
    s = summarise(pnl, dirs, holds, eq, times, "H1", 10_000.0)
    assert s["trades"] == 7 and s["long_trades"] == 4 and s["short_trades"] == 3
    assert s["win_rate"] == pytest.approx(3 / 7)
    assert s["profit_factor"] == pytest.approx(110 / 50)
    assert s["avg_win"] == pytest.approx(110 / 3) and s["avg_loss"] == pytest.approx(-50 / 4)
    assert s["largest_win"] == 50.0 and s["largest_loss"] == -20.0
    assert (s["max_win_streak"], s["max_loss_streak"]) == (2, 3)
    assert s["avg_hold_hours"] == pytest.approx(20 / 7)  # H1: one bar is one hour
    assert s["total_return"] == pytest.approx(0.006) and s["net_profit"] == pytest.approx(60.0)
    assert s["max_drawdown"] == pytest.approx(30 / 10_090)


def test_no_trades_gives_no_ratios_rather_than_zeros() -> None:
    s = summarise([], [], [], np.array([10_000.0, 10_000.0]), np.array([0, 300]), "M5", 10_000.0)
    assert s["trades"] == 0 and s["win_rate"] is None and s["profit_factor"] is None and s["avg_hold_hours"] is None


def test_the_curve_is_thinned_to_a_fixed_number_of_points() -> None:
    eq = np.linspace(10_000, 11_000, 5000)
    s = summarise([1.0], [1], [1], eq, np.arange(5000) * 900, "M15", 10_000.0)
    assert len(s["curve"]) == CURVE_POINTS
    assert s["curve"][0] == [0, 10_000.0] and s["curve"][-1] == [4999 * 900, 11_000.0]
