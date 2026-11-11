"""Charging swap must not replace the Trade object.

``_charge_swap`` rebuilt the Trade to add the night's cost. The rebuilt
object replaced ``position.trade``, but ``_fill_intents`` had already
appended the ORIGINAL to the run's trades list. From that point the
engine held one object and the results held another: the close was
recorded on the engine's copy, and the copy that was reported stayed
open forever.

It only bit positions held across a day roll, which is why it survived
-- the synthetic fixtures ran with ``charge_swap=False``. On real
USDCADm D1 bars it hid 121 of 125 trades: n_round_trips read 4 instead
of 59, and every trade-distribution metric (win rate, expectancy,
profit factor, avg win/loss, cost impact) was computed over those 4.
"""
from __future__ import annotations

import numpy as np

from engine.research.backtest.engine import BacktestConfig, BacktestEngine
from engine.research.backtest.order import OrderIntent
from engine.research.data.mt5_source import BarData, Timeframe
from engine.research.search.real_backtest import spec_for

SYMBOL = "USDCADm"


class _FlipEveryNBars:
    """Opens alternating positions, so each new entry closes the last."""

    strategy_id = "flip"

    def __init__(self, every: int = 30) -> None:
        self.every = every

    def on_bar(self, bar_index, bars, state):
        if bar_index < self.every or bar_index % self.every:
            return None
        direction = "long" if (bar_index // self.every) % 2 else "short"
        return OrderIntent(direction=direction, volume=0.1, bar_index=bar_index)


def _hourly_bars(n: int) -> BarData:
    # Hourly bars over several days, so positions are held across
    # multiple daily rollovers and swap is charged repeatedly.
    price = np.full(n, 1.10, dtype=np.float64)
    return BarData(
        time=np.array([86400 + i * 3600 for i in range(n)], dtype=np.int64),
        open=price.copy(),
        high=price + 0.001,
        low=price - 0.001,
        close=price.copy(),
        tick_volume=np.ones(n),
        spread=np.zeros(n, dtype=np.int32),
        real_volume=np.zeros(n),
        symbol=SYMBOL,
        timeframe=Timeframe.H1,
    )


def _run(*, charge_swap: bool):
    config = BacktestConfig(
        strategy_id="swap-identity",
        symbol=SYMBOL,
        timeframe="H1",
        initial_equity=10_000.0,
        charge_swap=charge_swap,
        charge_commission=False,
        slippage_mean_points=0.0,
        slippage_std_points=0.0,
    )
    engine = BacktestEngine(spec=spec_for(SYMBOL), config=config, rng_seed=0)
    return engine.run(_hourly_bars(24 * 10), _FlipEveryNBars(every=30))


def test_swap_does_not_orphan_the_reported_trade() -> None:
    """Every position but the last one closes, swap on or off."""
    with_swap = _run(charge_swap=True)
    without_swap = _run(charge_swap=False)

    assert with_swap.n_trades == without_swap.n_trades
    assert with_swap.n_trades > 2, "fixture must open several positions"

    # Exactly one position is still open at the end of the data.
    assert without_swap.n_open == 1
    assert with_swap.n_open == 1, (
        f"{with_swap.n_open} of {with_swap.n_trades} trades read as still "
        "open with swap charged; the Trade object was replaced and the "
        "close landed on an orphan"
    )
    assert with_swap.n_round_trips == without_swap.n_round_trips


def test_swap_is_actually_charged_on_the_reported_trade() -> None:
    """The fix must not lose the cost it was there to add.

    Mutating in place is only correct if the charge still lands on the
    object the caller reads.
    """
    with_swap = _run(charge_swap=True)
    without_swap = _run(charge_swap=False)

    charged = [t.swap_charged for t in with_swap.trades]
    assert any(c != 0.0 for c in charged), (
        "no trade carries a swap charge; the fixture holds positions "
        "across daily rollovers so some must"
    )
    assert all(t.swap_charged == 0.0 for t in without_swap.trades)

    # And the charge reaches P&L rather than only the field.
    assert with_swap.final_equity != without_swap.final_equity
