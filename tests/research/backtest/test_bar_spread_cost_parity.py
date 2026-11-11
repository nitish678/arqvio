"""v4 1.2 cost: a fill charges the signal candle's own historical spread -- the kernel against the oracle.

The oracle charges ``fill_spread_points[i] * point`` on a fill at bar i (the hour's bucket where it is NaN); the
kernel reads the same price from ``prepped.fill_spread_price``. Same trades, same money, bit for bit, with and
without the hourly matrix behind the gaps.
"""
from __future__ import annotations

import numpy as np
import pytest

from engine.research.backtest.engine import BacktestConfig, BacktestEngine
from engine.research.backtest.fast_path import record_to_dict, run_fast
from engine.research.backtest.kernel import TR_SIZE
from engine.research.backtest.prep import prep_cell
from engine.research.backtest.strategies.ma_crossover import MACrossConfig, MACrossStrategy
from engine.research.costs.spread_regime import fill_spread_points
from engine.research.search.real_backtest import spec_for
from tests.research.backtest.test_kernel_parity_steps import slow_trade_rows, trade_list_diffs
from tests.research.backtest.test_spread_regime_parity import SYMBOL, _bars

HOURLY = tuple(float(h % 7 + 2) for h in range(24))


def _run(bars, fill: np.ndarray | None, hourly: tuple[float, ...] | None, *, slip: float = 0.0):
    spec = spec_for(SYMBOL)
    cfg = BacktestConfig(strategy_id="barcost", symbol=SYMBOL, timeframe="H1", initial_equity=10_000.0,
                         charge_swap=True, charge_commission=True, commission_per_lot_per_side=3.5,
                         slippage_mean_points=slip, slippage_std_points=slip / 2, max_spread_points=None,
                         hourly_spread_points=hourly, fill_spread_points=fill)
    ma = MACrossConfig(fast=5, slow=21, volume=0.1, sl_atr=1.5, tp_atr=3.0, strategy_id="barcost")
    slow_result = BacktestEngine(spec=spec, config=cfg, rng_seed=0).run(bars, MACrossStrategy(ma))
    prepped = prep_cell(bars, spec, hourly_spread_points=hourly, fill_spread_points=fill)
    trades = np.zeros((prepped.n_bars, TR_SIZE), dtype=np.float64)
    record, _, status = run_fast(prepped, trades_out=trades, fast=5, slow=21, volume=0.1, sl_atr=1.5, tp_atr=3.0,
                                 initial_equity=10_000.0, max_spread_points=None, charge_swap=True,
                                 charge_commission=True, commission_per_lot_per_side=3.5,
                                 slippage_mean_points=slip, slippage_std_points=slip / 2, rng_seed=0)
    assert status == 0
    return slow_result, record_to_dict(record), trades, prepped


@pytest.mark.parametrize(("hourly", "slip"), [(HOURLY, 0.0), (HOURLY, 0.3), (None, 0.0), (None, 0.3)])
def test_the_kernel_charges_each_fill_what_the_oracle_charges(hourly, slip) -> None:
    bars = _bars()
    fill = fill_spread_points(bars.spread)
    fill[np.arange(fill.size) % 9 == 0] = np.nan  # gaps: the hour's bucket (or the flat spread) there
    slow_result, got, trades, prepped = _run(bars, fill, hourly, slip=slip)
    assert slow_result.n_trades > 10, "fixture traded too little; it proves nothing"
    assert got["n_trades"] == slow_result.n_trades
    assert not trade_list_diffs([tuple(r) for r in trades[: got["n_trades"]]], slow_trade_rows(slow_result))
    assert got["final_equity"] == slow_result.final_equity
    for k in ("cost_commission", "cost_spread", "cost_slippage", "cost_swap"):
        assert got[k] == getattr(slow_result, k)
    assert prepped.spread_cost_source.startswith("bar_spread+")


def test_the_bar_spread_changes_the_cost_and_none_keeps_the_hourly_charge() -> None:
    """The expensive era costs more under its own spread than under the hourly buckets; without the per-bar
    vector the fill price is exactly the hour's bucket (the W0 behaviour, digest unchanged)."""
    bars = _bars()
    own, _, _, _ = _run(bars, fill_spread_points(bars.spread), HOURLY)
    hourly, _, _, prepped = _run(bars, None, HOURLY)
    assert own.n_trades == hourly.n_trades and own.cost_spread > hourly.cost_spread
    spec = spec_for(SYMBOL)
    assert np.array_equal(prepped.fill_spread_price,
                          np.array([HOURLY[h] * spec.point for h in prepped.bar_hour_index]))
    assert prepped.spread_cost_source == "mean_buckets"
    assert prepped.cost_vector_hash == prep_cell(bars, spec, hourly_spread_points=HOURLY).cost_vector_hash


def test_the_cost_mode_comes_from_settings_unless_a_re_run_names_its_own(monkeypatch) -> None:
    from types import SimpleNamespace

    from engine.config.settings import get_settings
    from engine.research.search.real_backtest import fill_spread_for, spread_cost_mode

    cell = SimpleNamespace(bars=SimpleNamespace(spread=np.array([4, 6, 8], dtype=np.int32)))
    get_settings.cache_clear()
    monkeypatch.setenv("ARQVIO_SPREAD_COST_MODE", "bar")
    try:
        assert fill_spread_for(cell, flat_cost=False)[1:].tolist() == [4.0, 6.0]
        assert fill_spread_for(cell, flat_cost=True) is None
        with spread_cost_mode("hourly"):
            assert fill_spread_for(cell, flat_cost=False) is None
        assert fill_spread_for(cell, flat_cost=False) is not None
        with pytest.raises(ValueError), spread_cost_mode("tick"):
            pass
    finally:
        get_settings.cache_clear()


def test_fill_spread_points_is_the_signal_candles_spread_and_zero_is_unknown() -> None:
    out = fill_spread_points(np.array([5, 0, 7, 9], dtype=np.int32))
    assert np.isnan(out[0]) and out[1] == 5.0 and np.isnan(out[2]) and out[3] == 7.0
