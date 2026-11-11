"""Structural look-ahead blocking at the strategy boundary.

The feature registry already makes FEATURES safe: a BarWindow is sliced to
end at bar t. Strategies were not covered — Strategy.on_bar received the
whole BarData, so `bars.close[-1]` was the last bar of the backtest and
nothing refused it.

A planted look-ahead oracle at Sharpe 35 survived both DSR and PBO
(arXiv 2608.27734). Statistical correction does not catch leakage, so the
refusal has to be structural and at the data-access boundary.
"""
from __future__ import annotations

import numpy as np
import pytest

import engine.research.backtest.engine as engine_module
from engine.research.backtest.engine import BacktestConfig, BacktestEngine
from engine.research.backtest.strategies.ma_crossover import (
    MACrossConfig,
    MACrossStrategy,
)
from engine.research.data.mt5_source import BarData, Timeframe
from engine.research.features import LookaheadError
from engine.research.features.forward_block import (
    ForwardAccessBlocker,
    bars_up_to,
)
from engine.research.search.real_backtest import spec_for


def _bars(n: int = 300) -> BarData:
    rng = np.random.default_rng(0)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    return BarData(
        time=np.arange(n, dtype=np.int64) * 86400,
        open=close,
        high=close + 0.1,
        low=close - 0.1,
        close=close,
        tick_volume=np.ones(n),
        spread=np.zeros(n, dtype=np.int32),
        real_volume=np.zeros(n),
        symbol="USDCADm",
        timeframe=Timeframe.D1,
    )


# --- the blocker itself ----------------------------------------------------
def test_reading_bar_t_plus_one_raises() -> None:
    arr = np.arange(100.0)
    view = ForwardAccessBlocker(arr, t=50)
    assert view[50] == 50.0
    with pytest.raises(LookaheadError, match="past bar t=50"):
        view[51]


def test_negative_index_raises_because_it_reaches_the_end() -> None:
    """close[-1] is the last bar of the FULL series. It looks innocent."""
    arr = np.arange(100.0)
    view = ForwardAccessBlocker(arr, t=50)
    with pytest.raises(LookaheadError):
        view[-1]


def test_open_ended_and_overlong_slices_raise() -> None:
    arr = np.arange(100.0)
    view = ForwardAccessBlocker(arr, t=50)
    with pytest.raises(LookaheadError):
        view[:]          # the whole series
    with pytest.raises(LookaheadError):
        view[40:]        # runs to the end
    with pytest.raises(LookaheadError):
        view[:60]        # ten bars into the future
    with pytest.raises(LookaheadError):
        view[-10:51]     # negative start


def test_everything_up_to_t_is_allowed() -> None:
    arr = np.arange(100.0)
    view = ForwardAccessBlocker(arr, t=50)
    assert view[0] == 0.0
    assert np.array_equal(view[45:51], arr[45:51])
    assert np.array_equal(view[:51], arr[:51])
    assert len(view) == 51
    assert view.size == 51
    # np.mean must see only the visible window, not the whole array.
    assert float(np.mean(view)) == float(np.mean(arr[:51]))


def test_index_array_reaching_past_t_raises() -> None:
    arr = np.arange(100.0)
    view = ForwardAccessBlocker(arr, t=50)
    assert np.array_equal(view[np.array([1, 2, 50])], arr[[1, 2, 50]])
    with pytest.raises(LookaheadError):
        view[np.array([1, 51])]


def test_bars_view_bounds_every_array_and_the_length() -> None:
    bars = _bars(300)
    view = bars_up_to(bars, 100)
    assert view.n_bars == 101, "the length of the series is future information"
    with pytest.raises(LookaheadError):
        view.close[101]
    with pytest.raises(LookaheadError):
        view.high[-1]
    assert view.close[100] == bars.close[100]
    # Fields are proxied generically, so a field added later is covered too.
    with pytest.raises(LookaheadError):
        view.low[200]


# --- the engine actually uses it -------------------------------------------
class _LeakingStrategy:
    """Reads one bar into the future. The classic accidental oracle."""

    strategy_id = "leak"

    def on_bar(self, bar_index, bars, state):
        _ = bars.close[bar_index + 1]
        return None


class _EndPeekStrategy:
    """Reads the final bar of the series via a negative index."""

    strategy_id = "endpeek"

    def on_bar(self, bar_index, bars, state):
        _ = bars.close[-1]
        return None


def _run(strategy, bars):
    spec = spec_for("USDCADm")
    config = BacktestConfig(
        strategy_id=strategy.strategy_id,
        symbol="USDCADm",
        timeframe="D1",
        initial_equity=10_000.0,
    )
    return BacktestEngine(spec=spec, config=config, rng_seed=0).run(bars, strategy)


def test_engine_refuses_a_strategy_that_reads_the_next_bar() -> None:
    with pytest.raises(LookaheadError):
        _run(_LeakingStrategy(), _bars(300))


def test_engine_refuses_a_strategy_that_peeks_at_the_end() -> None:
    with pytest.raises(LookaheadError):
        _run(_EndPeekStrategy(), _bars(300))


def test_the_blocker_changes_no_honest_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """MACrossStrategy reads only [a:b] with b <= t+1, so it is unaffected.

    A guard that changed results would be silently rewriting history; this
    pins that it does not.
    """
    bars = _bars(300)
    strategy = MACrossStrategy(MACrossConfig(fast=5, slow=20, volume=0.1))

    blocked = _run(strategy, bars)
    monkeypatch.setattr(engine_module, "bars_up_to", lambda b, t: b)
    raw = _run(MACrossStrategy(MACrossConfig(fast=5, slow=20, volume=0.1)), bars)

    assert blocked.n_trades == raw.n_trades
    assert np.array_equal(
        np.asarray(blocked.equity_curve), np.asarray(raw.equity_curve)
    )
