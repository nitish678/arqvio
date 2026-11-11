"""W1 parity fixture: MACD_CROSS family.

Since W1.F3 the kernel expresses MACD_CROSS (full oracle parity, trade lists
exact: test_family_kernel_parity). This fixture pins a real cell and asserts:

  - MACD_CROSS DNAs pass expressible() (the fast path gate).
  - fast_backtest_factory runs them on the kernel with zero fallbacks and
    agrees with the slow factory on spot-checked candidates.
  - The slow engine produces deterministic results -- two runs with the same
    DNA, spec, and rng_seed match trade-for-trade.

The slow engine IS the oracle here; the kernel mirrors it, never the other
way round.
"""
from __future__ import annotations

import numpy as np
import pytest

from engine.research.backtest.engine import BacktestConfig, BacktestEngine
from engine.research.backtest.fast_path import expressible
from engine.research.backtest.strategies.macd_cross import MACDCrossStrategy
from engine.research.data.dataset_snapshot import find_snapshot, load_snapshot
from engine.research.search.dna import (
    BIAS_VALUES,
    DNA,
    EXEC_VALUES,
    FILTER_VALUES,
    RISK_VALUES,
    SIGNAL2_VALUES,
)
from engine.research.search.real_backtest import (
    CellBars,
    dna_to_macd_config,
    dna_to_strategy,
    fast_backtest_factory,
    real_backtest_factory,
    spec_for,
)
from engine.research.search.sampler import (
    MACD_FAST_VALUES,
    MACD_SIGNAL_PERIOD_VALUES,
    _macd_slow_choices,
)

SYMBOL = "USDCADm"
DATASET_ID = "aed3bf314539ec5a"
N_CANDIDATES = 20
SLICE_BARS = 4_000

_CFG = {
    "charge_swap": True,
    "charge_commission": True,
    "commission_per_lot_per_side": 0.0,
    "slippage_mean_points": 0.3,
    "slippage_std_points": 0.2,
    "no_trade_hours": (),
    "max_spread_points": None,
    "hourly_spread_points": tuple(float(h % 20 + 1) for h in range(24)),
}


def _macd_cross_dna(
    rng: np.random.Generator,
) -> DNA:
    """Construct a DNA with signal='MACD_CROSS'."""
    return DNA(
        bias=str(rng.choice(BIAS_VALUES)),
        filter=str(rng.choice(FILTER_VALUES)),
        signal="MACD_CROSS",
        signal2=str(rng.choice(SIGNAL2_VALUES)),
        exec_mode=str(rng.choice(EXEC_VALUES)),
        risk=str(rng.choice(RISK_VALUES)),
        mgmt="NONE",
        sl_atr=float(rng.uniform(1.0, 4.0)),
        tp_atr=float(rng.uniform(1.0, 6.0)),
        risk_pct=float(rng.uniform(0.005, 0.02)),
        friday_close=bool(rng.random() < 0.5),
        news_filter=bool(rng.random() < 0.3),
        daily_dd_limit=float(rng.uniform(0.03, 0.10)),
        max_agg_risk=float(rng.uniform(0.02, 0.10)),
        max_feature_lookback=int(rng.integers(10, 200)),
        max_hold_time=int(rng.integers(20, 500)),
        window_start_hour_utc=int(rng.integers(0, 14)),
        window_end_hour_utc=int(rng.integers(15, 24)),
        macd_fast=(fast := int(rng.choice(MACD_FAST_VALUES))),
        # The sampler's legal pairs (slow - fast >= MACD_MIN_GAP, 01f2b55);
        # an independent slow draw builds configs MACDCrossConfig refuses.
        macd_slow=int(rng.choice(_macd_slow_choices(fast))),
        macd_signal_period=int(rng.choice(MACD_SIGNAL_PERIOD_VALUES)),
        uses_future_bar=False,
    )


def _make_cfg(idx: int) -> BacktestConfig:
    return BacktestConfig(
        strategy_id=f"macd-parity-{idx}",
        symbol=SYMBOL,
        timeframe="H1",
        initial_equity=10_000.0,
        **_CFG,
    )


@pytest.fixture(scope="module")
def cell_slice():
    if find_snapshot(DATASET_ID) is None:
        pytest.skip(f"snapshot {DATASET_ID} not taken on this machine")
    from engine.research.data.mt5_source import BarData

    bars = load_snapshot(DATASET_ID)
    n = min(SLICE_BARS, int(np.asarray(bars.close).size))
    sliced = BarData(
        time=np.asarray(bars.time)[-n:],
        open=np.asarray(bars.open)[-n:],
        high=np.asarray(bars.high)[-n:],
        low=np.asarray(bars.low)[-n:],
        close=np.asarray(bars.close)[-n:],
        tick_volume=np.asarray(bars.tick_volume)[-n:],
        spread=np.asarray(bars.spread)[-n:],
        real_volume=np.asarray(bars.real_volume)[-n:],
        symbol=bars.symbol,
        timeframe=bars.timeframe,
    )
    closes = np.asarray(sliced.close, dtype=np.float64)
    log_ret = np.diff(np.log(closes))
    return CellBars(
        symbol=SYMBOL,
        timeframe="H1",
        bars=sliced,
        n_bars=n,
        n_returns=int(log_ret.size),
        returns=log_ret,
        source="MT5_SNAPSHOT",
        dataset_hash=f"{DATASET_ID}-sliced{n}",
    )


def test_macd_cross_dnas_are_expressible() -> None:
    """W1.F3: every MACD_CROSS DNA passes the fast-path gate.

    Renamed from test_macd_cross_dnAs_are_not_expressible: the family has a kernel
    branch with 60-DNA oracle parity (test_family_kernel_parity), so the
    gate's contract flipped from "always refuse" to "always accept"."""
    rng = np.random.default_rng(20260924)
    for idx in range(N_CANDIDATES):
        dna = _macd_cross_dna(rng)
        verdict = expressible(dna, _make_cfg(idx))
        assert verdict.ok, f"candidate {idx} ({dna.dna_hash}) refused: {verdict.reason}"


def test_macd_cross_runs_on_kernel_through_fast_factory(cell_slice) -> None:
    """W1.F3: fast_backtest_factory runs every MACD_CROSS candidate on the
    kernel (zero fallbacks) and agrees with the slow factory on the ones it
    spot-checks. Replaces test_macd_cross_falls_back_through_fast_factory: the
    family no longer falls back."""
    fast_fn = fast_backtest_factory(cell_slice, rng_seed=0, flat_cost=False)
    slow_fn = real_backtest_factory(cell_slice, flat_cost=False)
    rng = np.random.default_rng(20260925)
    for idx in range(N_CANDIDATES):
        dna = _macd_cross_dna(rng)
        result = fast_fn(dna)
        assert result.outcome == "EVALUATED"
        if idx < 2:
            ref = slow_fn(dna)
            assert result.n_trades == ref.n_trades
            assert result.exit_reasons == ref.exit_reasons
            assert result.sharpe == pytest.approx(ref.sharpe, rel=1e-9, abs=1e-12)
    assert fast_fn.fallback_state["fallbacks"] == 0, fast_fn.fallback_state["reasons"]


def test_macd_cross_slow_engine_is_deterministic(cell_slice) -> None:
    """Two slow-engine runs with the same DNA + rng_seed must match."""
    spec = spec_for(SYMBOL)
    rng = np.random.default_rng(20260926)
    dna = _macd_cross_dna(rng)
    cfg = _make_cfg(0)
    macd_cfg = dna_to_macd_config(dna)
    strategy = MACDCrossStrategy(macd_cfg)

    run1 = BacktestEngine(spec=spec, config=cfg, rng_seed=0).run(
        cell_slice.bars, strategy
    )
    run2 = BacktestEngine(spec=spec, config=cfg, rng_seed=0).run(
        cell_slice.bars, strategy
    )

    assert run1.n_trades == run2.n_trades
    assert run1.final_equity == pytest.approx(run2.final_equity, rel=0, abs=1e-6)
    assert run1.exit_reasons == run2.exit_reasons


def test_dna_to_strategy_routes_macd_cross() -> None:
    """dna_to_strategy must return MACDCrossStrategy for signal='MACD_CROSS'."""
    rng = np.random.default_rng(20260927)
    dna = _macd_cross_dna(rng)
    cfg, strategy = dna_to_strategy(dna)
    assert isinstance(strategy, MACDCrossStrategy)
    assert cfg is not None
    assert strategy.strategy_id == cfg.strategy_id
