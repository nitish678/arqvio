"""v4 management genes sl_reduce, sl_lock, atr_trail (engine loop step 4b, engine.py _stop_rules; kernel mirror).

Each moves the stop only towards the trade, judged on a bar's close and acting from the next bar, in R = the
initial stop distance: sl_reduce to entry - 0.5 R once 0.5 R ahead, sl_lock to entry + 1 R once 2 R ahead, atr_trail
to close - 2 ATR(14) on every close once 1 R ahead (strategies/switch.py SL_REDUCE_AT_R, SL_LOCK_AT_R, ATR_TRAIL).
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from engine.research.backtest.engine import BacktestConfig, BacktestEngine
from engine.research.backtest.fast_path import expressible, record_to_dict, run_fast
from engine.research.backtest.kernel import TR_SIZE
from engine.research.backtest.prep import prep_cell
from engine.research.backtest.strategies.ma_crossover import MACrossConfig, MACrossStrategy
from engine.research.backtest.strategies.switch import ATR_TRAIL, SL_LOCK_AT_R, SL_REDUCE_AT_R, stop_rule_kwargs
from engine.research.search import sampler
from engine.research.search.real_backtest import spec_for
from tests.research.backtest.test_breakeven import SYMBOL, _bars
from tests.research.backtest.test_kernel_parity_steps import slow_trade_rows, trade_list_diffs
from tests.research.backtest.test_spread_regime_parity import _bars as walk


def _run(bars, side: str, **rules):
    from engine.research.backtest.strategies.sham import ShamConfig, ShamStrategy

    cfg = BacktestConfig(strategy_id="sr", symbol=SYMBOL, timeframe="H1", initial_equity=10_000.0, charge_swap=False,
                         charge_commission=False, slippage_mean_points=0.0, slippage_std_points=0.0, **rules)
    sham = ShamConfig(entry_bars=(19,), sides=(side,), volume=0.1, strategy_id="sr", sl_atr=1.0, tp_atr=10.0,
                      atr_period=14)
    (trade,) = BacktestEngine(spec=spec_for(SYMBOL), config=cfg, rng_seed=0).run(bars, ShamStrategy(sham)).trades
    return trade


# entry 1.10 (bar 20 open), stop 1R = 0.002 below; closes 1.1012 (0.6 R), 1.1045 (2.25 R), then back down
RUN_UP = [(1.10, 1.1013, 1.0999, 1.1012), (1.1012, 1.1046, 1.1010, 1.1045), (1.1045, 1.1047, 1.1040, 1.1042),
          (1.1042, 1.1043, 1.0975, 1.0985)] + [(1.0985, 1.0986, 1.0984, 1.0985)] * 3


def test_sl_reduce_halves_the_risk_and_sl_lock_locks_one_r() -> None:
    bars = _bars(RUN_UP)
    t = _run(bars, "long", sl_reduce_at_r=SL_REDUCE_AT_R)
    entry = float(t.entry_fill.filled_price)
    assert t.exit_reason == "SL" and float(t.exit_fill.filled_price) == round(entry - 0.001, 5)
    t = _run(bars, "long", sl_lock_at_r=SL_LOCK_AT_R)
    assert t.exit_reason == "SL" and float(t.exit_fill.filled_price) == round(entry + 0.002, 5)
    off = _run(bars, "long")
    assert off.exit_reason == "SL" and float(off.exit_fill.filled_price) == round(entry - 0.002, 5)


def test_atr_trail_follows_the_close_and_a_short_mirrors_a_long() -> None:
    t = _run(_bars(RUN_UP), "long", atr_trail=ATR_TRAIL)
    entry = float(t.entry_fill.filled_price)
    assert t.exit_reason == "SL" and float(t.exit_fill.filled_price) > entry  # trailed above the entry
    mirror = [(2.2 - o, 2.2 - lo, 2.2 - h, 2.2 - c) for o, h, lo, c in RUN_UP]
    s = _run(_bars(mirror), "short", atr_trail=ATR_TRAIL)
    gain_long = float(t.exit_fill.filled_price) - entry
    gain_short = float(s.entry_fill.filled_price) - float(s.exit_fill.filled_price)
    assert s.exit_reason == "SL" and round(gain_short, 5) == round(gain_long, 5) > 0.0


@pytest.mark.parametrize("mgmt", [("sl_reduce",), ("sl_lock",), ("atr_trail",), ("breakeven", "sl_lock", "atr_trail"),
                                  ("sl_reduce", "sl_lock", "atr_trail")])
@pytest.mark.parametrize("slip", [0.0, 0.3])
def test_the_kernel_moves_the_stop_exactly_when_the_oracle_does(mgmt, slip) -> None:
    bars, spec = walk(), spec_for(SYMBOL)
    common = dict(initial_equity=10_000.0, charge_swap=True, charge_commission=True, commission_per_lot_per_side=3.5,
                  slippage_mean_points=slip, slippage_std_points=slip / 2)
    rules = stop_rule_kwargs(mgmt)
    cfg = BacktestConfig(strategy_id="sr", symbol=SYMBOL, timeframe="H1", max_spread_points=None, **rules, **common)
    ma = MACrossConfig(fast=5, slow=21, volume=0.1, sl_atr=1.5, tp_atr=4.0, strategy_id="sr")
    slow_result = BacktestEngine(spec=spec, config=cfg, rng_seed=0).run(bars, MACrossStrategy(ma))
    plain = BacktestEngine(spec=spec, config=dataclasses.replace(cfg, **stop_rule_kwargs(())), rng_seed=0).run(
        bars, MACrossStrategy(ma))
    assert [t.sl_price for t in slow_result.trades] != [t.sl_price for t in plain.trades], (
        "the rule never fired on this fixture; it proves nothing")
    prepped = prep_cell(bars, spec)
    trades = np.zeros((prepped.n_bars, TR_SIZE), dtype=np.float64)
    record, _, status = run_fast(prepped, trades_out=trades, fast=5, slow=21, volume=0.1, sl_atr=1.5, tp_atr=4.0,
                                 max_spread_points=None, rng_seed=0, **rules, **common)
    assert status == 0
    got = record_to_dict(record)
    assert got["n_trades"] == slow_result.n_trades
    assert not trade_list_diffs([tuple(x) for x in trades[: got["n_trades"]]], slow_trade_rows(slow_result))
    assert got["final_equity"] == slow_result.final_equity


def test_the_stop_rules_run_on_the_kernel_and_the_close_rules_fall_back() -> None:
    dna = dataclasses.replace(sampler.random_dna(np.random.default_rng(3)),
                              mgmt_switches=("sl_reduce", "sl_lock", "atr_trail"))
    cfg = BacktestConfig(strategy_id="x", symbol=SYMBOL, timeframe="H1")
    assert expressible(dna, cfg).ok
    assert not expressible(dataclasses.replace(dna, mgmt_switches=("atr_trail", "partial_tp")), cfg).ok
