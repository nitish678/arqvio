"""W4-2.2: SwitchStrategy gates + dna_to_strategy dispatch + kernel fallback.

Tests:
1. SwitchStrategy with no switches == passthrough to inner (bit-identical to bare family strategy).
2. SwitchStrategy with a PLANNED switch raises SwitchGateNotImplemented at construction-of-intent
   (so a PLANNED gene never silently affects a trial).
3. dna_to_strategy returns SwitchStrategy when any switch is ON.
4. expressible() returns False for a switched DNA (kernel falls back to slow oracle).
5. session gate + trend_ema (already DNA v3) still work through SwitchStrategy wrapping.
"""
from __future__ import annotations

import numpy as np
import pytest

from engine.research.backtest.order import OrderIntent
from engine.research.backtest.strategies.switch import (
    SwitchConfig,
    SwitchGateNotImplemented,
    SwitchStrategy,
)
from engine.research.data.mt5_source import BarData
from engine.research.search.dna import DNA


# --- a minimal fake family strategy -----------------------------------------
class _FakeFamily:
    def __init__(self):
        self.strategy_id = "fakefamily"
        self._calls = 0

    def on_bar(self, bar_index, bars, state):
        self._calls += 1
        # open long on even bars, short on odd bars (deterministic for the gate)
        direction = "long" if bar_index % 2 == 0 else "short"
        return OrderIntent(direction=direction, volume=0.1)


def _bars(n=20):
    t = np.arange(n, dtype=np.int64) * 60
    close = np.linspace(1.1, 1.2, n)
    high = close + 0.01
    low = close - 0.01
    open_ = close - 0.005
    return BarData(
        time=t, open=open_, high=high, low=low, close=close,
        tick_volume=np.zeros(n), spread=np.zeros(n), real_volume=np.zeros(n),
        symbol="TEST", timeframe="M1", source="test", time_basis="close",
    )


def _minimal_dna(**over):
    base = dict(
        bias="NONE", filter="NONE", signal="RSI_REVERSION", signal2="",
        exec_mode="MARKET2", risk="FIXED_ATR", mgmt="NONE",
        sl_atr=2.0, tp_atr=3.0, risk_pct=0.01,
        friday_close=False, news_filter=False,
        daily_dd_limit=0.05, max_agg_risk=0.05,
        max_feature_lookback=50, max_hold_time=10,
        window_start_hour_utc=0, window_end_hour_utc=23,
    )
    base.update(over)
    return DNA(**base)


# --- 1. no switches = passthrough -------------------------------------------
def test_no_switches_passthrough():
    inner = _FakeFamily()
    cfg = SwitchConfig(inner_strategy_id="fakefamily")
    sw = SwitchStrategy(inner, cfg)
    bars = _bars(5)
    for i in range(5):
        intent = sw.on_bar(i, bars, {})
        assert intent is not None
        assert intent.direction == inner.on_bar(i, bars, {}).direction
    assert sw.strategy_id.startswith("fakefamily")


def test_no_switches_drops_intent_when_gate_rejects_session():
    """Even with no switches, the session/trend gate wraps the inner strategy."""
    inner = _FakeFamily()
    # session off-hours for all bars -> all entries refused
    cfg = SwitchConfig(inner_strategy_id="fakefamily", params={"session": "ASIA"})
    sw = SwitchStrategy(inner, cfg)
    bars = _bars(5)
    for i in range(5):
        # bar times are 0..4 min -> hours 0,0,0,0,0 -> all in ASIA (0-8) -> allowed actually
        intent = sw.on_bar(i, bars, {})
        assert intent is not None  # ASIA 00-08, bar hour 0 -> allowed
    # now a session that never allows (NEWYORK 12-21, all bars hour 0)
    cfg2 = SwitchConfig(inner_strategy_id="fakefamily", params={"session": "NEWYORK"})
    sw2 = SwitchStrategy(inner, cfg2)
    for i in range(5):
        intent = sw2.on_bar(i, bars, {})
        assert intent is None  # refused


# --- 3. dna_to_strategy dispatch -------------------------------------------
def test_dna_to_strategy_returns_switch_when_switches_on():
    """v4 batch B: READY bias/filter switches are entry gates (GatedStrategy, kernel gate arrays); management and
    risk switches still need the SwitchStrategy."""
    from engine.research.backtest.strategies.gated import GatedStrategy
    from engine.research.search.real_backtest import dna_to_strategy
    cfg, strat = dna_to_strategy(_minimal_dna(bias_switches=("sma",)))
    assert isinstance(strat, GatedStrategy) and not isinstance(strat, SwitchStrategy)
    cfg, strat = dna_to_strategy(_minimal_dna(mgmt_switches=("eod_close",)))
    assert isinstance(strat, SwitchStrategy)


def test_dna_to_strategy_returns_plain_when_switches_off():
    from engine.research.search.real_backtest import dna_to_strategy
    dna = _minimal_dna()  # no switches
    cfg, strat = dna_to_strategy(dna)
    assert not isinstance(strat, SwitchStrategy)


# --- 4. kernel expressible fallback ----------------------------------------
def test_expressible_false_when_switches_on():
    from engine.research.backtest.fast_path import expressible, Expressibility
    cfg = object()  # config shape doesn't matter; switches trigger refusal
    exp = expressible(_minimal_dna(mgmt_switches=("partial_tp",)), cfg)
    assert exp.ok is False
    assert "switch" in exp.reason.lower()
    assert expressible(_minimal_dna(bias_switches=("sma",), filter_switches=("rsi",)), cfg).ok is True
    assert expressible(_minimal_dna(mgmt_switches=("eod_close", "friday_close_profit")), cfg).ok is True


# --- 2. planned switch raises at eval ---------------------------------------
def test_planned_bias_switch_raises_at_eval():
    inner = _FakeFamily()
    cfg = SwitchConfig(inner_strategy_id="fakefamily", bias_switches=("sma",))
    sw = SwitchStrategy(inner, cfg)
    bars = _bars(2)
    with pytest.raises(SwitchGateNotImplemented, match="sma"):
        sw.on_bar(0, bars, {})
