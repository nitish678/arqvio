"""v4 management genes friday_close_profit and eod_close (strategies/switch.py SwitchStrategy close rules; kernel.py
step 5 mg_exit): the kernel takes exactly the trades the slow oracle takes on pinned real bars, for every DNA family,
alone and with entry gates (SwitchStrategy wraps the GatedStrategy since 2026-11-01)."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from engine.research.search.dna import SIGNAL_VALUES
from engine.research.search.real_backtest import (
    CellBars,
    dna_to_strategy,
    fast_backtest_factory,
    load_cell_bars_from_snapshot,
    real_backtest_factory,
    slice_cell_bars,
)
from engine.research.search.sampler import random_dna

DATASET_ID = "aed3bf314539ec5a"
SYMBOL, TIMEFRAME = "USDCADm", "H1"
SLICE = 6_000


@pytest.fixture(scope="module")
def cell():
    return slice_cell_bars(load_cell_bars_from_snapshot(SYMBOL, TIMEFRAME, DATASET_ID), SLICE)


def _family(signal: str):
    rng = np.random.default_rng(11)
    while True:
        d = random_dna(rng)
        if d.signal == signal:
            return dataclasses.replace(d, risk_pct=0.01, sl_atr=2.0, tp_atr=3.0)


def _signals():
    return dataclasses.replace(random_dna(np.random.default_rng(7)), signal="SIGNALS", signal2="",
                               signal_switches=("rsi", "engulfing", "kalman"), risk_pct=0.01, sl_atr=2.0, tp_atr=3.0)


def _check(cell: CellBars, dna, *, must_differ: bool = True) -> None:
    fast, slow = fast_backtest_factory(cell), real_backtest_factory(cell)
    a, b = fast(dna), slow(dna)
    assert fast.fallback_state["fallbacks"] == 0, fast.fallback_state["reasons"]
    assert (a.n_trades, a.sharpe, a.total_return) == (b.n_trades, b.sharpe, b.total_return)
    if must_differ:
        plain = slow(dataclasses.replace(dna, mgmt_switches=()))
        assert (b.n_trades, b.total_return) != (plain.n_trades, plain.total_return), "the rule never fired"


@pytest.mark.parametrize("signal", [*SIGNAL_VALUES, "SIGNALS"])
@pytest.mark.parametrize("mgmt", [("eod_close",), ("friday_close_profit",)])
def test_kernel_matches_oracle_for_every_family(cell: CellBars, signal, mgmt):
    dna = _signals() if signal == "SIGNALS" else _family(signal)
    _check(cell, dataclasses.replace(dna, mgmt_switches=mgmt))


@pytest.mark.parametrize("extra", [
    {"bias_switches": ("sma",), "filter_switches": ("volatility",)},
    {"bias_switches": ("psar",), "filter_switches": ("doji",), "session": "LONDON"},
    {"trend_ema": 200, "mgmt_switches": ("breakeven", "atr_trail", "friday_close_profit", "eod_close")},
])
def test_with_entry_gates_and_the_stop_rules(cell: CellBars, extra):
    dna = dataclasses.replace(_signals(), mgmt_switches=("friday_close_profit", "eod_close"))
    _check(cell, dataclasses.replace(dna, **extra))


def test_a_gated_dna_with_a_close_rule_runs_its_gates_through_gated_strategy():
    from engine.research.backtest.strategies.gated import GatedStrategy
    from engine.research.backtest.strategies.switch import SwitchStrategy

    dna = dataclasses.replace(_signals(), bias_switches=("sma",), mgmt_switches=("eod_close",))
    _, strategy = dna_to_strategy(dna)
    assert isinstance(strategy, SwitchStrategy) and isinstance(strategy.inner, GatedStrategy)
    assert not strategy.config.bias_switches and not strategy.config.filter_switches
