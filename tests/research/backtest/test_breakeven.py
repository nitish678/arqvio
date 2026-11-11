"""v4 management gene breakeven (engine loop): the stop moves to the entry fill once a bar closes 1R ahead.

The oracle owns it (BacktestConfig.breakeven_at_r); the kernel refuses management switches, so a breakeven DNA runs
on the oracle (fast_path.expressible) -- the fallback is the parity guarantee until the kernel learns it.
"""
from __future__ import annotations

import numpy as np
import pytest

from engine.research.backtest.engine import BacktestConfig, BacktestEngine
from engine.research.backtest.fast_path import expressible
from engine.research.backtest.strategies.sham import ShamConfig, ShamStrategy
from engine.research.backtest.strategies.switch import READY_MGMT
from engine.research.data.mt5_source import BarData, Timeframe
from engine.research.search import sampler
from engine.research.search.real_backtest import spec_for

SYMBOL = "USDCADm"


def _bars(path: list[tuple[float, float, float, float]], warm: int = 20) -> BarData:
    """``warm`` quiet bars at 1.10 (range 0.002: ATR 0.002), then ``path`` as (open, high, low, close)."""
    rows = [(1.10, 1.101, 1.099, 1.10)] * warm + path
    o, h, lo, c = (np.array(col, dtype=np.float64) for col in zip(*rows, strict=True))
    n = len(rows)
    return BarData(time=np.array([86400 * 4 + i * 3600 for i in range(n)], dtype=np.int64), open=o, high=h, low=lo,
                   close=c, tick_volume=np.ones(n), spread=np.full(n, 10, dtype=np.int32), real_volume=np.zeros(n),
                   symbol=SYMBOL, timeframe=Timeframe.H1)


def _run(bars: BarData, side: str, be: float | None):
    cfg = BacktestConfig(strategy_id="be", symbol=SYMBOL, timeframe="H1", initial_equity=10_000.0, charge_swap=False,
                         charge_commission=False, slippage_mean_points=0.0, slippage_std_points=0.0,
                         breakeven_at_r=be)
    sham = ShamConfig(entry_bars=(19,), sides=(side,), volume=0.1, strategy_id="be", sl_atr=1.0, tp_atr=10.0,
                      atr_period=14)
    return BacktestEngine(spec=spec_for(SYMBOL), config=cfg, rng_seed=0).run(bars, ShamStrategy(sham))


UP_THEN_BACK = [(1.10, 1.1012, 1.0998, 1.1010), (1.1010, 1.1032, 1.1008, 1.1030),  # closes 1.5R ahead
                (1.1030, 1.1036, 1.1020, 1.1025), (1.1025, 1.1026, 1.0995, 1.0997),  # back through the entry
                (1.0997, 1.0999, 1.0985, 1.0990)] + [(1.0990, 1.0991, 1.0989, 1.0990)] * 3


def test_a_trade_1r_ahead_at_a_close_is_stopped_at_its_entry_not_its_stop() -> None:
    bars = _bars(UP_THEN_BACK)
    off, on = _run(bars, "long", None), _run(bars, "long", 1.0)
    (t_off,), (t_on,) = off.trades, on.trades
    entry = float(t_on.entry_fill.filled_price)
    assert t_on.exit_reason == "SL" and float(t_on.exit_fill.filled_price) == round(entry, 5)
    assert t_on.exit_bar_index == 23  # the bar that came back through the entry
    assert (t_off.exit_reason, t_off.sl_price) == ("EOD", 1.098)  # off: the original stop, never touched


def test_a_close_short_of_1r_leaves_the_stop_alone_and_a_short_mirrors_a_long() -> None:
    timid = [(1.10, 1.1012, 1.0998, 1.1010), (1.1010, 1.1019, 1.1005, 1.1015),  # never closes 1R ahead
             (1.1015, 1.1016, 1.0995, 1.0997)] + [(1.0997, 1.0998, 1.0996, 1.0997)] * 3
    a, b = _run(_bars(timid), "long", 1.0), _run(_bars(timid), "long", None)
    assert [(t.exit_bar_index, t.realized_pnl) for t in a.trades] == [(t.exit_bar_index, t.realized_pnl) for t in b.trades]
    mirror = [(2.2 - o, 2.2 - lo, 2.2 - h, 2.2 - c) for o, h, lo, c in UP_THEN_BACK]
    (short,) = _run(_bars(mirror), "short", 1.0).trades
    assert short.exit_reason == "SL" and short.exit_bar_index == 23
    assert float(short.exit_fill.filled_price) == round(float(short.entry_fill.filled_price), 5)


def test_a_breakeven_dna_runs_on_the_kernel_and_other_management_falls_back() -> None:
    import dataclasses

    dna = dataclasses.replace(sampler.random_dna(np.random.default_rng(3)), mgmt_switches=("breakeven",))
    cfg = BacktestConfig(strategy_id="x", symbol=SYMBOL, timeframe="H1")
    assert expressible(dna, cfg).ok
    assert not expressible(dataclasses.replace(dna, mgmt_switches=("breakeven", "partial_tp")), cfg).ok


@pytest.mark.parametrize(("r", "slip", "sl_atr"), [(1.0, 0.0, 1.5), (1.0, 0.3, 1.5), (0.5, 0.3, 1.0), (1.5, 0.0, 1.0)])
def test_the_kernel_moves_the_stop_exactly_when_the_oracle_does(r, slip, sl_atr) -> None:
    """Bit-for-bit: trades, final equity and every cost total, with breakeven on, on a walk with many trades."""
    from engine.research.backtest.fast_path import record_to_dict, run_fast
    from engine.research.backtest.kernel import TR_SIZE
    from engine.research.backtest.prep import prep_cell
    from engine.research.backtest.strategies.ma_crossover import MACrossConfig, MACrossStrategy
    from tests.research.backtest.test_kernel_parity_steps import slow_trade_rows, trade_list_diffs
    from tests.research.backtest.test_spread_regime_parity import _bars as walk

    bars, spec = walk(), spec_for(SYMBOL)
    common = dict(initial_equity=10_000.0, charge_swap=True, charge_commission=True, commission_per_lot_per_side=3.5,
                  slippage_mean_points=slip, slippage_std_points=slip / 2)
    cfg = BacktestConfig(strategy_id="be", symbol=SYMBOL, timeframe="H1", max_spread_points=None, breakeven_at_r=r,
                         **common)
    ma = MACrossConfig(fast=5, slow=21, volume=0.1, sl_atr=sl_atr, tp_atr=4.0, strategy_id="be")
    slow_result = BacktestEngine(spec=spec, config=cfg, rng_seed=0).run(bars, MACrossStrategy(ma))
    without = BacktestEngine(spec=spec, config=dataclasses_replace(cfg, breakeven_at_r=None), rng_seed=0).run(
        bars, MACrossStrategy(ma))
    assert [t.sl_price for t in slow_result.trades] != [t.sl_price for t in without.trades], (
        "breakeven never fired on this fixture; it proves nothing")
    prepped = prep_cell(bars, spec)
    trades = np.zeros((prepped.n_bars, TR_SIZE), dtype=np.float64)
    record, _, status = run_fast(prepped, trades_out=trades, fast=5, slow=21, volume=0.1, sl_atr=sl_atr, tp_atr=4.0,
                                 max_spread_points=None, breakeven_at_r=r, rng_seed=0, **common)
    assert status == 0
    got = record_to_dict(record)
    assert got["n_trades"] == slow_result.n_trades
    assert not trade_list_diffs([tuple(x) for x in trades[: got["n_trades"]]], slow_trade_rows(slow_result))
    assert got["final_equity"] == slow_result.final_equity
    for k in ("cost_commission", "cost_spread", "cost_slippage", "cost_swap"):
        assert got[k] == getattr(slow_result, k)


def test_breakeven_with_entry_gates_runs_on_the_gated_family_strategy() -> None:
    """A breakeven DNA bred with bias / filter switches must not land on SwitchStrategy, whose bias and filter
    evaluators are PLANNED (it raised SwitchGateNotImplemented before)."""
    import dataclasses

    from engine.research.search.real_backtest import dna_to_strategy
    from tests.research.backtest.test_spread_regime_parity import _bars as walk

    base = sampler.random_dna(np.random.default_rng(5))
    for extra in ({}, {"bias_switches": ("sma",)}, {"filter_switches": ("rsi",)}):
        dna = dataclasses.replace(base, mgmt_switches=("breakeven",), **extra)
        _, strategy = dna_to_strategy(dna)
        assert type(strategy).__name__ != "SwitchStrategy"
        cfg = BacktestConfig(strategy_id="be", symbol=SYMBOL, timeframe="H1", breakeven_at_r=1.0)
        assert BacktestEngine(spec=spec_for(SYMBOL), config=cfg, rng_seed=0).run(walk(), strategy).n_trades > 0


def test_a_pool_with_management_on_breeds_breakeven_and_one_without_draws_as_before() -> None:
    from engine.research.search import gene_pool

    from engine.research.search.dna import SIGNAL_VALUES

    base = {"families": {f: {"enabled": True, "weight": 1.0} for f in SIGNAL_VALUES},
            "gates": {"bias_switches": True, "filter_switches": True}}
    old = gene_pool.canonicalise(base)
    new = gene_pool.canonicalise({**base, "gates": {**base["gates"], "mgmt_switches": True}})
    a = [sampler.random_dna(np.random.default_rng(9), gene_pool=old) for _ in range(1)]
    b = [sampler.random_dna(np.random.default_rng(9), gene_pool=new) for _ in range(1)]
    # the management draw comes last: every earlier gene is the same draw as before
    assert [dataclasses_replace(x, mgmt_switches=()) for x in b] == a
    drawn = {m for i in range(200)
             for m in sampler.random_dna(np.random.default_rng(i), gene_pool=new).mgmt_switches}
    assert drawn == set(READY_MGMT)
    assert all(not sampler.random_dna(np.random.default_rng(i), gene_pool=old).mgmt_switches for i in range(50))


def dataclasses_replace(dna, **kw):
    import dataclasses

    return dataclasses.replace(dna, **kw)
