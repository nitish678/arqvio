"""v4 spread-regime gate: the kernel against the oracle, and the gate's own arithmetic.

The oracle refuses an entry when ``spread[i] > cap[i]``; the kernel gets ``spread - cap`` against 0.0. Both must
refuse the same entries and produce the same trades and money, bit for bit.
"""
from __future__ import annotations

import numpy as np
import pytest

from engine.research.backtest.engine import BacktestConfig, BacktestEngine
from engine.research.backtest.fast_path import record_to_dict, run_fast
from engine.research.backtest.kernel import TR_SIZE
from engine.research.backtest.prep import prep_cell
from engine.research.backtest.strategies.ma_crossover import MACrossConfig, MACrossStrategy
from engine.research.costs.spread_regime import blocked_by_month, regime_cap, window_bars
from engine.research.data.mt5_source import BarData, Timeframe
from engine.research.search.real_backtest import spec_for
from tests.research.backtest.test_kernel_parity_steps import slow_trade_rows, trade_list_diffs

SYMBOL = "USDCADm"


def _bars(n: int = 1500, seed: int = 11) -> BarData:
    """A cheap era, an expensive era (3x wider -- the XAGUSD shape), and a few blow-outs inside each."""
    rng = np.random.default_rng(seed)
    walk = np.cumsum(rng.normal(0.0, 0.0016, size=n)) + 1.10
    spread = np.where(np.arange(n) < n // 2, 12, 36).astype(np.int32) + rng.integers(0, 4, size=n).astype(np.int32)
    spread[rng.choice(n, size=40, replace=False)] *= 5
    return BarData(
        time=np.array([86400 * 3 + i * 3600 for i in range(n)], dtype=np.int64),
        open=walk.copy(), high=walk + 0.0012, low=walk - 0.0012, close=walk.copy(),
        tick_volume=np.ones(n), spread=spread, real_volume=np.zeros(n), symbol=SYMBOL, timeframe=Timeframe.H1)


def _run(bars: BarData, cap: np.ndarray, *, fast: int = 5, slow: int = 21, sl_atr: float = 1.5,
         tp_atr: float = 3.0, slip: float = 0.0):
    spec = spec_for(SYMBOL)
    cfg = BacktestConfig(strategy_id="regime", symbol=SYMBOL, timeframe="H1", initial_equity=10_000.0,
                         charge_swap=True, charge_commission=True, commission_per_lot_per_side=3.5,
                         slippage_mean_points=slip, slippage_std_points=slip / 2, max_spread_points=None,
                         spread_cap_by_bar=cap)
    ma = MACrossConfig(fast=fast, slow=slow, volume=0.1, sl_atr=sl_atr, tp_atr=tp_atr, strategy_id="regime")
    slow_result = BacktestEngine(spec=spec, config=cfg, rng_seed=0).run(bars, MACrossStrategy(ma))
    prepped = prep_cell(bars, spec, spread_cap_by_bar=cap)
    trades = np.zeros((prepped.n_bars, TR_SIZE), dtype=np.float64)
    record, _, status = run_fast(prepped, trades_out=trades, fast=fast, slow=slow, volume=0.1, sl_atr=sl_atr,
                                 tp_atr=tp_atr, initial_equity=10_000.0, max_spread_points=prepped.max_spread_points,
                                 charge_swap=True, charge_commission=True, commission_per_lot_per_side=3.5,
                                 slippage_mean_points=slip, slippage_std_points=slip / 2, rng_seed=0)
    assert status == 0
    return slow_result, record_to_dict(record), trades


@pytest.mark.parametrize(("window", "multiple", "slip"), [(48, 3.0, 0.0), (48, 1.5, 0.3), (240, 3.0, 0.3),
                                                          (24, 1.1, 0.0)])
def test_the_kernel_refuses_exactly_what_the_oracle_refuses(window, multiple, slip) -> None:
    bars = _bars()
    cap = regime_cap(bars.spread, window=window, multiple=multiple)
    slow_result, got, trades = _run(bars, cap, slip=slip)
    assert slow_result.blocked_max_spread > 0, "fixture refused nothing; it proves nothing"
    assert got["blocked_max_spread"] == slow_result.blocked_max_spread
    assert got["n_trades"] == slow_result.n_trades
    assert not trade_list_diffs([tuple(r) for r in trades[: got["n_trades"]]], slow_trade_rows(slow_result))
    assert got["final_equity"] == slow_result.final_equity
    for k in ("cost_commission", "cost_spread", "cost_slippage", "cost_swap"):
        assert got[k] == getattr(slow_result, k)


def test_the_expensive_era_trades_under_the_regime_gate_and_not_under_a_cheap_era_cap() -> None:
    """The XAGUSD failure in miniature: a fixed cap from the cheap era refuses the whole expensive era."""
    bars = _bars()
    half = len(bars.spread) // 2
    fixed = np.full(len(bars.spread), float(np.percentile(bars.spread[:half], 95)))
    regime = regime_cap(bars.spread, window=48, multiple=3.0)
    fixed_result, _, _ = _run(bars, fixed)
    regime_result, _, _ = _run(bars, regime)
    late = lambda r: sum(1 for t in r.trades if t.entry_fill.bar_index >= half)  # noqa: E731
    assert late(fixed_result) == 0
    assert late(regime_result) > 5
    assert regime_result.blocked_max_spread < fixed_result.blocked_max_spread


def test_the_cap_uses_only_past_bars_and_starts_after_half_a_window() -> None:
    s = np.array([10] * 30 + [1000] + [10] * 30, dtype=np.int32)
    cap = regime_cap(s, window=20, multiple=3.0)
    assert np.isnan(cap[:10]).all() and cap[10] == 30.0
    assert cap[30] == 30.0          # the blow-out bar is judged by the 20 bars before it
    assert cap[31] == 30.0          # and its own value barely moves the median after it
    assert s[30] > cap[30]


def test_a_zero_spread_stretch_is_floored() -> None:
    cap = regime_cap(np.zeros(50, dtype=np.int32), window=10, multiple=3.0)
    assert np.nanmin(cap) == 3.0


def test_window_bars_and_blocked_by_month() -> None:
    assert window_bars("M5") == 5760 and window_bars("H1") == 480 and window_bars("D1") == 20
    t = np.array([0, 86400 * 40, 86400 * 41], dtype=np.int64)
    assert blocked_by_month(t, np.array([5, 50, 50]), np.array([10.0, 10.0, np.nan])) == {"1970-02": 1}
