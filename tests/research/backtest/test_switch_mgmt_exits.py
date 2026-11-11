"""W4-2.5: management exit gates for SwitchStrategy.

Tests:
1. friday_close_profit fires on a Friday bar when in profit >= threshold.
2. friday_close_profit does NOT fire on a non-Friday bar.
3. friday_close_profit does NOT fire when not in profit.
4. eod_close fires when bar UTC hour >= eod_hour_utc (default 20).
5. eod_close does NOT fire when hour < eod_hour_utc.
6. PLANNED mgmt switch (partial_tp) raises SwitchGateNotImplemented at eval.
7. No mgmt switches -> no exit intent (flat passthrough, even with position open).

Parity contract (CLAUDE.md law 7): each wired gate returns a volume=0 close
intent -- the engine fills at the next bar's open, a price the strategy does
not decide. PLANNED gates raise rather than guessing.
"""
from __future__ import annotations

import datetime

import numpy as np
import pytest

from engine.research.backtest.order import OrderIntent
from engine.research.backtest.strategies.switch import (
    SwitchConfig,
    SwitchGateNotImplemented,
    SwitchStrategy,
)
from engine.research.data.mt5_source import BarData


# --- a minimal fake family strategy that opens a long on bar 0 -------------
class _LongOnlyFamily:
    strategy_id = "longonly"

    def __init__(self):
        self._signaled = False

    def on_bar(self, bar_index, bars, state):
        # open a long exactly once, on the first bar.
        if not self._signaled:
            self._signaled = True
            return OrderIntent(direction="long", volume=0.1, bar_index=bar_index)
        return None


def _bars_at(times):
    """Build a BarData from a list of (unix_time, close) pairs.

    OHLC derived so the position is in profit when close rises. high/low
    span a 0.02 window so atr_at returns a sane positive value."""
    n = len(times)
    t = np.array([x[0] for x in times], dtype=np.int64)
    close = np.array([x[1] for x in times], dtype=np.float64)
    high = close + 0.01
    low = close - 0.01
    open_ = close - 0.005
    return BarData(
        time=t, open=open_, high=high, low=low, close=close,
        tick_volume=np.ones(n, dtype=np.int64),
        spread=np.zeros(n, dtype=np.float64),
        real_volume=np.zeros(n, dtype=np.int64),
        symbol="TEST", timeframe="H1", source="test", time_basis="close",
    )


def _utc(year, month, day, hour):
    return int(datetime.datetime(year, month, day, hour, tzinfo=datetime.timezone.utc).timestamp())


# --- 1. friday_close_profit fires on Friday in profit ----------------------
def test_friday_close_profit_fires_on_friday_in_profit():
    # 2025-06-13 is a Friday (UTC). bar0=thu 18:00 (entry), bar1=fri 19:00
    # (Friday, in profit). sl_atr=2, atr ~ 0.02-window -> risk_distance small;
    # with close rising move exceeds threshold.
    times = [
        (_utc(2025, 6, 12, 18), 1.000),   # Thu, entry bar (long opened)
        (_utc(2025, 6, 13, 19), 1.050),   # Fri, in profit -> closes
    ]
    bars = _bars_at(times)
    cfg = SwitchConfig(
        inner_strategy_id="longonly",
        mgmt_switches=("friday_close_profit",),
        params={"fcp_trigger_pct": 50.0, "sl_atr": 2.0, "atr_period": 14},
    )
    sw = SwitchStrategy(_LongOnlyFamily(), cfg)
    state = {}
    # bar0: long opens; state tracked for mgmt on bar 1.
    sw.on_bar(0, bars, state)
    # bar 1: Friday, in profit -> mgmt exit fires (before inner runs).
    intent = sw.on_bar(1, bars, state)
    assert intent is not None
    assert intent.volume == 0.0  # close intent


# --- 2. friday_close_profit NOT on a non-Friday ----------------------------
def test_friday_close_profit_does_not_fire_on_nonfriday():
    # Thu -> Mon, both long position open. Friday gate must not fire.
    times = [
        (_utc(2025, 6, 12, 18), 1.000),   # Thu entry
        (_utc(2025, 6, 16, 18), 1.050),   # Mon, in profit but not Friday
    ]
    bars = _bars_at(times)
    cfg = SwitchConfig(
        inner_strategy_id="longonly",
        mgmt_switches=("friday_close_profit",),
        params={"fcp_trigger_pct": 50.0, "sl_atr": 2.0},
    )
    sw = SwitchStrategy(_LongOnlyFamily(), cfg)
    state = {}
    sw.on_bar(0, bars, state)
    intent = sw.on_bar(1, bars, state)
    # Not Friday -> no mgmt exit; inner returns None (already signaled).
    assert intent is None


# --- 3. friday_close_profit NOT when flat / not in profit --------------------
def test_friday_close_profit_does_not_fire_when_not_in_profit():
    times = [
        (_utc(2025, 6, 12, 18), 1.050),   # Thu: entry at 1.050
        (_utc(2025, 6, 13, 19), 1.040),   # Fri: slightly down -> not in profit
    ]
    bars = _bars_at(times)
    cfg = SwitchConfig(
        inner_strategy_id="longonly",
        mgmt_switches=("friday_close_profit",),
        params={"fcp_trigger_pct": 50.0, "sl_atr": 2.0},
    )
    sw = SwitchStrategy(_LongOnlyFamily(), cfg)
    state = {}
    sw.on_bar(0, bars, state)
    intent = sw.on_bar(1, bars, state)
    assert intent is None  # not in profit -> no close


# --- 4. eod_close fires when hour >= default 20 ----------------------------
def test_eod_close_fires_at_default_hour():
    # bar0 entry at hour 18, bar1 at hour 20 (>= default eod_hour_utc=20).
    times = [
        (_utc(2025, 6, 12, 18), 1.000),
        (_utc(2025, 6, 12, 20), 1.010),   # hour 20 -> EOD fires
    ]
    bars = _bars_at(times)
    cfg = SwitchConfig(
        inner_strategy_id="longonly",
        mgmt_switches=("eod_close",),
        params={"eod_hour_utc": 20},
    )
    sw = SwitchStrategy(_LongOnlyFamily(), cfg)
    state = {}
    sw.on_bar(0, bars, state)
    intent = sw.on_bar(1, bars, state)
    assert intent is not None
    assert intent.volume == 0.0


# --- 5. eod_close does NOT fire below threshold ----------------------------
def test_eod_close_does_not_fire_below_hour():
    times = [
        (_utc(2025, 6, 12, 18), 1.000),
        (_utc(2025, 6, 12, 19), 1.010),   # hour 19 < 20 -> no fire
    ]
    bars = _bars_at(times)
    cfg = SwitchConfig(
        inner_strategy_id="longonly",
        mgmt_switches=("eod_close",),
        params={"eod_hour_utc": 20},
    )
    sw = SwitchStrategy(_LongOnlyFamily(), cfg)
    state = {}
    sw.on_bar(0, bars, state)
    intent = sw.on_bar(1, bars, state)
    assert intent is None  # below EOD hour, inner silent


# --- 6. PLANNED mgmt switch raises at eval ----------------------------------
def test_planned_mgmt_switch_raises_at_eval():
    times = [
        (_utc(2025, 6, 12, 18), 1.000),
        (_utc(2025, 6, 12, 20), 1.010),
    ]
    bars = _bars_at(times)
    cfg = SwitchConfig(
        inner_strategy_id="longonly",
        mgmt_switches=("partial_tp",),
        params={},
    )
    sw = SwitchStrategy(_LongOnlyFamily(), cfg)
    state = {}
    sw.on_bar(0, bars, state)
    with pytest.raises(SwitchGateNotImplemented, match="partial_tp"):
        sw.on_bar(1, bars, state)


# --- 7. no mgmt switches -> no exit (passthrough) --------------------------
def test_no_mgmt_switches_no_exit():
    times = [
        (_utc(2025, 6, 12, 18), 1.000),
        (_utc(2025, 6, 12, 20), 1.010),
    ]
    bars = _bars_at(times)
    cfg = SwitchConfig(inner_strategy_id="longonly")  # no mgmt switches
    sw = SwitchStrategy(_LongOnlyFamily(), cfg)
    state = {}
    sw.on_bar(0, bars, state)
    intent = sw.on_bar(1, bars, state)
    # inner already signaled on bar 0; no mgmt exit on bar 1.
    assert intent is None
