"""F1: a zero-spread hour is untradeable, not cheap.

Several Exness symbols (US30m, US500m, XAUUSDm, XAGUSDm) calibrate to a
median spread of 0 in hour 21 UTC. A zero spread is a free trade -- enter
and exit at the same price at no cost -- so any search will find it and
build a strategy on it, and that strategy will not survive a real broker.

During a no-trade hour NOTHING fills: no entries, no pending-order
triggers, no SL/TP. Open positions carry through, and an exit owed from
inside the window is taken on the first bar after it at that bar's open.
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

SYMBOL = "USDCADm"
HOUR_S = 3600


def _hourly_bars(n: int, start_hour: int = 18, price: float = 100.0) -> BarData:
    """One bar per hour from `start_hour` UTC on 1970-01-02, flat price."""
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


class _EnterOnBar:
    """Emits one buy intent on a chosen bar. Fills on the NEXT bar's open."""

    strategy_id = "enter-on-bar"

    def __init__(self, signal_bar: int) -> None:
        self.signal_bar = signal_bar

    def on_bar(self, bar_index, bars, state):
        if bar_index == self.signal_bar:
            return OrderIntent(direction="long", volume=0.1, bar_index=bar_index)
        return None


def _engine(no_trade_hours: tuple[int, ...]):
    config = BacktestConfig(
        strategy_id="f1",
        symbol=SYMBOL,
        timeframe="H1",
        initial_equity=10_000.0,
        charge_swap=False,
        charge_commission=False,
        slippage_mean_points=0.0,
        slippage_std_points=0.0,
        no_trade_hours=no_trade_hours,
    )
    return BacktestEngine(spec=spec_for(SYMBOL), config=config, rng_seed=0)


def test_zero_spread_hour_blocks_trades() -> None:
    """A signal whose fill lands in hour 21 does not trade until hour 22."""
    # Bars start at 18:00, so index 2 is 20:00 and index 3 is 21:00.
    bars = _hourly_bars(8, start_hour=18)
    assert _hour_of(bars.time[3]) == 21

    # Signal on bar 2 (20:00) -> would fill on bar 3 (21:00), which is shut.
    strategy = _EnterOnBar(signal_bar=2)

    blocked = _engine(no_trade_hours=(21,)).run(bars, strategy)
    assert blocked.n_trades == 1, "the intent must survive the window, not vanish"
    entry_bar = blocked.trades[0].entry_fill.bar_time
    assert _hour_of(entry_bar) == 22, (
        f"entry filled in hour {_hour_of(entry_bar)}; the 21:00 window is shut "
        "and the fill belongs on the first tradeable bar"
    )
    assert blocked.blocked_zero_spread_events >= 1

    # Same bars, no window: the fill happens in hour 21 as normal.
    allowed = _engine(no_trade_hours=()).run(bars, _EnterOnBar(signal_bar=2))
    assert _hour_of(allowed.trades[0].entry_fill.bar_time) == 21
    assert allowed.blocked_zero_spread_events == 0


def test_no_entry_bar_falls_in_a_no_trade_hour() -> None:
    """Across a whole run, no filled entry may sit in a blocked hour."""
    bars = _hourly_bars(72, start_hour=0)

    class _EnterEveryBar:
        strategy_id = "enter-every-bar"

        def on_bar(self, bar_index, bars, state):
            if state.get("open"):
                return None
            state["open"] = True
            return OrderIntent(direction="long", volume=0.1, bar_index=bar_index)

    result = _engine(no_trade_hours=(21,)).run(bars, _EnterEveryBar())
    for trade in result.trades:
        assert _hour_of(trade.entry_fill.bar_time) != 21, (
            f"trade {trade.trade_id} entered inside the 21:00 window"
        )


def test_zero_spread_sl_tp_deferred() -> None:
    """An SL due inside the window fills after it, at that bar's open.

    The production engine's _check_sl_tp is a deliberate no-op -- SL/TP is
    deferred to S6 -- so the trigger is injected here. This pins the
    DEFERRAL, which is the part F1 owns: the exit must not execute at the
    stop price inside a window where nothing could have executed.
    """
    bars = _hourly_bars(8, start_hour=18, price=100.0)
    # Make the post-window bar a different price so "that bar's open" is
    # distinguishable from any stop level.
    bars.open[5] = 90.0
    assert _hour_of(bars.time[3]) == 21
    assert _hour_of(bars.time[4]) == 22

    engine = _engine(no_trade_hours=(21,))
    fired: list[int] = []
    real_check = engine._check_sl_tp

    def _sl_from_bar_3(position, bar_high, bar_low, bar_index):
        # The stop is "touched" on every bar from 3 onward.
        if bar_index >= 3:
            fired.append(bar_index)
            closed = real_check(position, bar_high, bar_low, bar_index=bar_index)
            if closed is not None:
                return closed
            # Stand in for a real SL close at the stop price.
            position.realised_pnl = -999.0
            position.trade.exit_bar_index = bar_index
            return position
        return None

    engine._check_sl_tp = _sl_from_bar_3  # type: ignore[method-assign]
    result = engine.run(bars, _EnterOnBar(signal_bar=0))

    assert fired, "the injected SL never ran; the test proves nothing"
    assert 3 in fired, "the SL should have been touched inside the window"
    trade = result.trades[0]
    assert trade.exit_bar_index is not None
    assert _hour_of(bars.time[trade.exit_bar_index]) != 21, (
        "the exit executed inside the window"
    )
    # Executed at the post-window bar's OPEN, not the stop price.
    assert trade.realized_pnl != -999.0
    assert result.blocked_zero_spread_events >= 1


def test_symbol_without_zero_hours_is_unchanged() -> None:
    bars = _hourly_bars(8, start_hour=18)
    with_cfg = _engine(no_trade_hours=()).run(bars, _EnterOnBar(signal_bar=2))
    assert with_cfg.blocked_zero_spread_events == 0
    assert with_cfg.n_trades == 1


def test_matrix_reports_and_persists_its_no_trade_hours() -> None:
    m = SpreadMatrix(
        symbol="US30m",
        buckets=tuple([12.0] * 21 + [0.0, 14.0, 13.0]),
        mean_buckets=tuple([12.0] * 21 + [0.0, 14.0, 13.0]),
        sample_count=1000,
        source="MT5",
    )
    assert m.no_trade_hours == frozenset({21})
    assert m.to_dict()["no_trade_hours"] == [21]
    # Recomputed on load, so an artifact written before this existed is
    # still handled correctly.
    assert SpreadMatrix.from_json(m.to_json()).no_trade_hours == frozenset({21})

    clean = SpreadMatrix(
        symbol="EURUSDm",
        buckets=tuple(float(9 + (h % 3)) for h in range(24)),
        mean_buckets=tuple(float(9 + (h % 3)) for h in range(24)),
        sample_count=1000,
        source="MT5",
    )
    assert clean.no_trade_hours == frozenset()


if __name__ == "__main__":
    pytest.main([__file__, "-q"])


# ---------------------------------------------------------------------------
# R1: the deferral policy, and the counter that reports it
# ---------------------------------------------------------------------------

def test_zero_spread_market2_intent_defers_rather_than_drops() -> None:
    """R1: a blocked MARKET2 intent is neither dropped nor discarded.

    The distinction matters for the equity curve. Dropping the signal
    deletes a trade the strategy asked for, which makes the window look
    cheaper than it is; deferring keeps the trade and charges it the
    reopened spread, which is what a live venue does across a freeze.
    """
    bars = _hourly_bars(8, start_hour=18)
    assert _hour_of(bars.time[3]) == 21

    result = _engine(no_trade_hours=(21,)).run(bars, _EnterOnBar(signal_bar=2))

    # Not dropped: the trade exists.
    assert result.n_trades == 1
    # Not filled in the window: it waited.
    assert _hour_of(result.trades[0].entry_fill.bar_time) == 22
    # And the wait is reported, not silent.
    assert result.deferred_zero_spread_fills >= 1, (
        "a deferred fill must be counted; blocked minus deferred is the "
        "work the window absorbed rather than merely delayed"
    )


def test_deferred_counter_is_zero_when_the_window_never_bites() -> None:
    """No window, no deferrals -- the counter cannot report phantom work."""
    bars = _hourly_bars(8, start_hour=18)
    result = _engine(no_trade_hours=()).run(bars, _EnterOnBar(signal_bar=2))
    assert result.n_trades == 1
    assert result.blocked_zero_spread_events == 0
    assert result.deferred_zero_spread_fills == 0


def test_only_market2_exists_so_expiry_cannot_be_tested_yet() -> None:
    """Tripwire for R1's pending-order clause.

    R1 specifies that a blocked LIMIT/LIMIT2/STOP keeps its own expiry, so
    an intent whose expiry passes before the window reopens is DROPPED
    rather than deferred. That path cannot be tested today: EXEC_VALUES is
    ("MARKET2",), OrderIntent has no expiry field, and engine.py states
    LIMIT/LIMIT2/STOP arrive in S6. Writing the test now would mean
    inventing the expiry semantics.

    This fails the moment a second exec mode is added, so R1's dropped-on-
    expiry rule and its counter get implemented with it rather than
    forgotten.
    """
    from engine.research.backtest.order import OrderIntent as _Intent
    from engine.research.search.dna import EXEC_VALUES

    assert EXEC_VALUES == ("MARKET2",), (
        f"EXEC_VALUES is now {EXEC_VALUES}. R1 requires that a blocked "
        "pending order keeps its own expiry and is DROPPED, not deferred, "
        "when that expiry passes before the no-trade window reopens. "
        "Implement that at the gate in engine.py, add a "
        "dropped_zero_spread_expired counter, and replace this test with "
        "test_zero_spread_pending_expiry."
    )
    assert not hasattr(_Intent, "expiry_bars"), (
        "OrderIntent grew an expiry field; see the message above."
    )


def test_final_bar_fill_path_obeys_the_window() -> None:
    """The end-of-data fill path is a SECOND fill path, and it was open.

    run() fills pending intents once inside the bar loop and again after
    it, for the final bar. Only the first was gated, so an intent pending
    into a final bar that fell in a no-trade hour filled at that bar's
    open -- inside the very window the gate exists to close. The A/B on
    real USDCADm H1 bars hid this: the last bar there is rarely hour 21.

    There is no bar after the last one, so a blocked intent here cannot be
    deferred. It never executes, and that is the correct outcome.
    """
    # 4 bars from 18:00 => the last bar IS 21:00, and the signal on bar 2
    # would otherwise fill there.
    bars = _hourly_bars(4, start_hour=18)
    assert _hour_of(bars.time[-1]) == 21

    result = _engine(no_trade_hours=(21,)).run(bars, _EnterOnBar(signal_bar=2))

    assert result.n_trades == 0, (
        "the final-bar path filled inside the no-trade window; every fill "
        "path must obey it, not just the one in the bar loop"
    )
    assert result.blocked_zero_spread_events >= 1
    assert result.deferred_zero_spread_fills == 0, (
        "nothing can be deferred past the end of the data"
    )

    # Without the window the same bars DO trade, so the assertion above is
    # about the gate and not about the fixture being unable to trade.
    allowed = _engine(no_trade_hours=()).run(bars, _EnterOnBar(signal_bar=2))
    assert allowed.n_trades == 1
    assert _hour_of(allowed.trades[0].entry_fill.bar_time) == 21


def test_deferred_never_exceeds_blocked() -> None:
    """You cannot defer more fills than you refused.

    The first implementation counted the pending LIST on the reopening
    bar, which also holds the intent emitted at the end of the blocked bar
    -- one that was never refused a fill. That reported 8251 deferrals
    against 4251 blocks on real bars. Counting by intent id fixed it; this
    pins the invariant.
    """
    bars = _hourly_bars(72, start_hour=0)

    class _EnterEveryBar:
        strategy_id = "enter-every-bar"

        def on_bar(self, bar_index, bars, state):
            if state.get("open"):
                return None
            state["open"] = True
            return OrderIntent(direction="long", volume=0.1, bar_index=bar_index)

    result = _engine(no_trade_hours=(21,)).run(bars, _EnterEveryBar())
    assert result.deferred_zero_spread_fills <= result.blocked_zero_spread_events, (
        f"{result.deferred_zero_spread_fills} deferrals against "
        f"{result.blocked_zero_spread_events} blocks"
    )
