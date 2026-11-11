"""SIGNALS family (v4 batch 2.6): the kernel takes exactly the trades the slow oracle takes, on pinned real bars,
for each research signal alone and for combinations (with a bias/filter switch and breakeven on top)."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from engine.research.backtest.signal_switches import READY_SIGNALS
from engine.research.search.real_backtest import (
    CellBars,
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


def _dna(switches, **extra):
    base = random_dna(np.random.default_rng(7))
    return dataclasses.replace(base, signal="SIGNALS", signal2="", signal_switches=tuple(switches), risk_pct=0.01,
                               sl_atr=2.0, tp_atr=3.0, **extra)


CASES = [(s,) for s in READY_SIGNALS] + [("engulfing", "pin_bar"), ("rsi", "williams", "bb"), ("squeeze", "fade"),
                                        ("fib", "orb"), ("vwap", "kalman", "atr_run")]


@pytest.mark.parametrize("switches", CASES, ids=lambda s: "+".join(s))
def test_kernel_matches_oracle(cell: CellBars, switches):
    dna = _dna(switches)
    fast, slow = fast_backtest_factory(cell), real_backtest_factory(cell)
    a, b = fast(dna), slow(dna)
    assert fast.fallback_state["fallbacks"] == 0, fast.fallback_state["reasons"]
    assert a.n_trades == b.n_trades
    assert a.sharpe == b.sharpe and a.total_return == b.total_return


def test_with_switch_gates_and_breakeven(cell: CellBars):
    dna = _dna(("rsi", "engulfing"), bias_switches=("sma",), filter_switches=("volatility",),
               mgmt_switches=("breakeven",))
    fast, slow = fast_backtest_factory(cell), real_backtest_factory(cell)
    a, b = fast(dna), slow(dna)
    assert fast.fallback_state["fallbacks"] == 0, fast.fallback_state["reasons"]
    assert (a.n_trades, a.sharpe, a.total_return) == (b.n_trades, b.sharpe, b.total_return)
    assert b.n_trades > 0


@pytest.mark.parametrize(("bias", "filters"), [(("trailing", "vwap"), ("doji",)), (("psar",), ("event", "sma")),
                                               (("market_struct",), ("receding",)), ((), ("regime",))])
def test_with_the_last_bias_and_filter_switches(cell: CellBars, bias, filters):
    dna = _dna(("kalman", "vwap", "engulfing"), bias_switches=bias, filter_switches=filters)
    fast, slow = fast_backtest_factory(cell), real_backtest_factory(cell)
    a, b = fast(dna), slow(dna)
    assert fast.fallback_state["fallbacks"] == 0, fast.fallback_state["reasons"]
    assert (a.n_trades, a.sharpe, a.total_return) == (b.n_trades, b.sharpe, b.total_return)
    assert b.n_trades > 0


def test_dna_rule_switches_only_on_the_signals_family():
    with pytest.raises(ValueError, match="exactly for the SIGNALS family"):
        dataclasses.replace(random_dna(np.random.default_rng(1)), signal_switches=("rsi",))
    with pytest.raises(ValueError, match="exactly for the SIGNALS family"):
        _dna(())


def test_with_the_stop_moving_management_switches(cell: CellBars):
    dna = _dna(("rsi", "engulfing", "kalman"), mgmt_switches=("breakeven", "sl_reduce", "sl_lock", "atr_trail"))
    fast, slow = fast_backtest_factory(cell), real_backtest_factory(cell)
    a, b = fast(dna), slow(dna)
    plain = slow(dataclasses.replace(dna, mgmt_switches=()))
    assert fast.fallback_state["fallbacks"] == 0, fast.fallback_state["reasons"]
    assert (a.n_trades, a.sharpe, a.total_return) == (b.n_trades, b.sharpe, b.total_return)
    assert b.total_return != plain.total_return  # the rules moved stops on real bars
