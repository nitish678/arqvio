"""B4 phase-1 guardrails: daily loss limit and Friday close.

The verify sweep found NO in-backtest guardrails at all: daily_dd_limit
and friday_close were sampled into every DNA and read by nothing. Two
are implemented here. The rest (time-window, news filter,
max-agg-risk) stay for the pre-GA stage and are listed in the report.
"""
from __future__ import annotations

import numpy as np
import pytest

from engine.research.backtest.engine import BacktestConfig, BacktestEngine
from engine.research.backtest.order import EXIT_FRIDAY_CLOSE, OrderIntent
from engine.research.data.mt5_source import BarData, Timeframe
from engine.research.search.real_backtest import spec_for

SYMBOL = "USDCADm"
DAY = 86400


def _bars_from(day0: int, n: int, closes: list[float] | None = None) -> BarData:
    """Hourly bars starting at midnight of epoch-day ``day0``."""
    price = (
        np.array(closes, dtype=np.float64) if closes
        else np.full(n, 1.10000, dtype=np.float64)
    )
    return BarData(
        time=np.array([day0 * DAY + i * 3600 for i in range(n)], dtype=np.int64),
        open=price.copy(), high=price + 0.0010, low=price - 0.0010,
        close=price.copy(), tick_volume=np.ones(n),
        spread=np.zeros(n, dtype=np.int32), real_volume=np.zeros(n),
        symbol=SYMBOL, timeframe=Timeframe.H1,
    )


def _engine(**overrides) -> BacktestEngine:
    config = BacktestConfig(
        strategy_id="guardrails", symbol=SYMBOL, timeframe="H1",
        initial_equity=10_000.0, charge_swap=False, charge_commission=False,
        slippage_mean_points=0.0, slippage_std_points=0.0, **overrides,
    )
    return BacktestEngine(spec=spec_for(SYMBOL), config=config, rng_seed=0)


class _EnterEveryNBars:
    strategy_id = "churn"

    def __init__(self, every: int = 4) -> None:
        self.every = every

    def on_bar(self, bar_index, bars, state):
        if bar_index < self.every or bar_index % self.every:
            return None
        side = "long" if (bar_index // self.every) % 2 else "short"
        return OrderIntent(direction=side, volume=0.5, bar_index=bar_index)


class _EnterOnce:
    strategy_id = "once"

    def __init__(self, bar: int) -> None:
        self.bar = bar

    def on_bar(self, bar_index, bars, state):
        if bar_index != self.bar:
            return None
        return OrderIntent(direction="long", volume=0.1, bar_index=bar_index)


# ---------------------------------------------------------------------------
# Friday close
# ---------------------------------------------------------------------------

def test_friday_is_identified_correctly() -> None:
    """1970-01-01 was a Thursday, so epoch day 1 is a Friday."""
    from engine.research.backtest.engine import _is_friday

    assert _is_friday(1 * DAY)          # Fri 1970-01-02
    assert not _is_friday(2 * DAY)      # Sat
    assert not _is_friday(0 * DAY)      # Thu
    assert _is_friday(8 * DAY)          # the next Friday


def test_friday_close_flattens_an_open_position() -> None:
    bars = _bars_from(1, 24)            # a Friday, hours 0..23
    result = _engine(friday_close_hour=17).run(bars, _EnterOnce(bar=2))

    trade = result.trades[0]
    assert trade.exit_reason == EXIT_FRIDAY_CLOSE
    assert result.friday_flattens == 1
    closed_hour = (int(bars.time[trade.exit_bar_index]) // 3600) % 24
    assert closed_hour == 17, f"flattened at hour {closed_hour}, not 17"

    # Without the guardrail the same position runs to the end of data.
    plain = _engine().run(bars, _EnterOnce(bar=2))
    assert plain.trades[0].exit_reason != EXIT_FRIDAY_CLOSE
    assert plain.friday_flattens == 0


def test_friday_close_does_nothing_on_other_days() -> None:
    bars = _bars_from(4, 24)            # a Monday
    result = _engine(friday_close_hour=17).run(bars, _EnterOnce(bar=2))
    assert result.friday_flattens == 0
    assert result.trades[0].exit_reason != EXIT_FRIDAY_CLOSE


def test_friday_close_respects_no_trade_hours() -> None:
    """It must flatten at a TRADEABLE bar, not inside a shut window.

    With hour 17 shut, nothing can fill there; the flatten waits for
    hour 18 rather than pretending a fill happened in a window the
    engine refuses fills in.
    """
    bars = _bars_from(1, 24)
    result = _engine(friday_close_hour=17, no_trade_hours=(17,)).run(
        bars, _EnterOnce(bar=2)
    )
    trade = result.trades[0]
    assert result.friday_flattens == 1
    closed_hour = (int(bars.time[trade.exit_bar_index]) // 3600) % 24
    assert closed_hour == 18, (
        f"flattened at hour {closed_hour}; hour 17 is a no-trade hour"
    )


# ---------------------------------------------------------------------------
# Daily loss limit
# ---------------------------------------------------------------------------

def _losing_series(n: int) -> list[float]:
    """A price path that loses money on every alternating position."""
    return list(1.10000 - np.arange(n) * 0.00050)


def test_daily_loss_limit_blocks_new_entries() -> None:
    bars = _bars_from(1, 40, _losing_series(40))
    limited = _engine(daily_loss_limit_pct=0.5).run(bars, _EnterEveryNBars(4))
    unlimited = _engine().run(bars, _EnterEveryNBars(4))

    assert unlimited.blocked_daily_loss == 0
    assert limited.blocked_daily_loss > 0, (
        "the limit never fired on a series that only loses"
    )
    assert limited.n_trades < unlimited.n_trades, (
        "entries were blocked but the trade count did not fall"
    )


def test_daily_loss_limit_does_not_liquidate_an_open_position() -> None:
    """It stops you digging; it does not close what you already hold."""
    bars = _bars_from(1, 40, _losing_series(40))
    result = _engine(daily_loss_limit_pct=0.01).run(bars, _EnterOnce(bar=2))
    # One position opened before the limit could bite, and it is still
    # running at the end of data rather than force-closed.
    assert result.n_trades == 1
    assert result.trades[0].exit_reason == "EOD"


def test_daily_loss_limit_resets_at_the_day_boundary() -> None:
    """Yesterday's loss must not gate today's entries."""
    # Two days: heavy losses on day 1, flat on day 2.
    day1 = _losing_series(24)
    day2 = [day1[-1]] * 24
    bars = _bars_from(1, 48, day1 + day2)

    result = _engine(daily_loss_limit_pct=0.5).run(bars, _EnterEveryNBars(4))
    entry_days = {
        (int(bars.time[t.entry_bar_index]) // DAY)
        for t in result.trades
    }
    assert len(entry_days) == 2, (
        f"entries only on day(s) {entry_days}; the limit did not reset "
        "at the day boundary"
    )


def test_guardrails_are_off_by_default() -> None:
    """A run that did not ask for them must be unchanged."""
    bars = _bars_from(1, 40, _losing_series(40))
    result = _engine().run(bars, _EnterEveryNBars(4))
    assert result.blocked_daily_loss == 0
    assert result.friday_flattens == 0


def test_exit_reason_distribution_still_partitions_with_guardrails() -> None:
    bars = _bars_from(1, 40, _losing_series(40))
    result = _engine(daily_loss_limit_pct=0.5, friday_close_hour=20).run(
        bars, _EnterEveryNBars(4)
    )
    assert sum(result.exit_reasons.values()) == result.n_trades, (
        f"{result.exit_reasons} does not partition {result.n_trades} trades"
    )
    assert all(t.exit_reason is not None for t in result.trades)


@pytest.mark.parametrize("pct", [0.0, -1.0])
def test_a_non_positive_limit_never_fires(pct: float) -> None:
    """A zero or negative limit is not "block everything"."""
    bars = _bars_from(1, 40, _losing_series(40))
    result = _engine(daily_loss_limit_pct=pct).run(bars, _EnterEveryNBars(4))
    assert result.blocked_daily_loss == 0
