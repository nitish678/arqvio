"""B3 parity fixtures: RSI_REVERSION mean-reversion family.

RSI_REVERSION is the first non-MA-cross family. Since W1.F1 the kernel
expresses it (full oracle parity: test_family_kernel_parity). These
fixtures pin a real cell and assert:

  - RSI_REVERSION DNAs pass expressible() (the fast path gate).
  - fast_backtest_factory runs them on the kernel with zero fallbacks and
    agrees with the slow factory; a still-unexpressible RSI DNA falls back
    with a reason naming the refused gene.
  - The slow engine produces deterministic results -- two runs with the same
    DNA, spec, and rng_seed match trade-for-trade.
  - rsi_series uses Wilder EMA smoothing, not an SMA approximation.
  - RSIStrategy refuses oversold >= overbought configs.
  - effective_genes tracks dna_to_ma_config for MA_CROSS.

The slow engine IS the oracle here; the kernel mirrors it, never the
other way round.
"""
from __future__ import annotations

import dataclasses
import numpy as np
import pytest

from engine.core.errors import DataError
from engine.research.backtest.engine import BacktestConfig, BacktestEngine
from engine.research.backtest.fast_path import expressible
from engine.research.backtest.strategies.rsi import RSIStrategy
from engine.research.data.dataset_snapshot import find_snapshot, load_snapshot
from engine.research.search.dna import (
    BIAS_VALUES,
    DNA,
    EXEC_VALUES,
    FILTER_VALUES,
    RISK_VALUES,
    SIGNAL2_VALUES,
)
from engine.research.search.effective_hash import (
    EFFECTIVE_HASH_VERSION,
    effective_gene_hash,
    effective_genes,
)
from engine.research.search.real_backtest import (
    CellBars,
    dna_to_rsi_config,
    dna_to_strategy,
    dna_to_ma_config,
    fast_backtest_factory,
    real_backtest_factory,
    spec_for,
)

SYMBOL = "USDCADm"
DATASET_ID = "aed3bf314539ec5a"
N_CANDIDATES = 60
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


def _rsi_reversion_dna(
    rng: np.random.Generator,
    *,
    sl_atr: float | None = None,
    tp_atr: float | None = None,
    period: int | None = None,
) -> DNA:
    """Construct a DNA with signal='RSI_REVERSION'.

    B2: period is sampled from [5, 21] when not given explicitly.
    """
    return DNA(
        bias=str(rng.choice(BIAS_VALUES)),
        filter=str(rng.choice(FILTER_VALUES)),
        signal="RSI_REVERSION",
        signal2=str(rng.choice(SIGNAL2_VALUES)),
        exec_mode=str(rng.choice(EXEC_VALUES)),
        risk=str(rng.choice(RISK_VALUES)),
        mgmt="NONE",
        sl_atr=float(rng.uniform(1.0, 4.0)) if sl_atr is None else sl_atr,
        tp_atr=float(rng.uniform(1.0, 6.0)) if tp_atr is None else tp_atr,
        risk_pct=float(rng.uniform(0.005, 0.02)),
        friday_close=bool(rng.random() < 0.5),
        news_filter=bool(rng.random() < 0.3),
        daily_dd_limit=float(rng.uniform(0.03, 0.10)),
        max_agg_risk=float(rng.uniform(0.02, 0.10)),
        max_feature_lookback=int(rng.integers(10, 200)),
        max_hold_time=int(rng.integers(20, 500)),
        window_start_hour_utc=int(rng.integers(0, 14)),
        window_end_hour_utc=int(rng.integers(15, 24)),
        rsi_period=int(rng.choice(range(5, 22))) if period is None else period,
        rsi_oversold=float(rng.uniform(15.0, 35.0)),
        rsi_overbought=float(rng.uniform(65.0, 85.0)),
        uses_future_bar=False,
    )


def _make_cfg(idx: int) -> BacktestConfig:
    return BacktestConfig(
        strategy_id=f"rsi-parity-{idx}",
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


def test_rsi_reversion_dnas_are_expressible() -> None:
    """W1.F1: every RSI_REVERSION DNA passes the fast-path gate.

    Renamed from test_rsi_reversion_dnAs_are_not_expressible: the family has a kernel
    branch with 60-DNA oracle parity (test_family_kernel_parity), so the
    gate's contract flipped from "always refuse" to "always accept"."""
    rng = np.random.default_rng(20260924)
    for idx in range(N_CANDIDATES):
        dna = _rsi_reversion_dna(rng)
        verdict = expressible(dna, _make_cfg(idx))
        assert verdict.ok, f"candidate {idx} ({dna.dna_hash}) refused: {verdict.reason}"


def test_rsi_reversion_runs_on_kernel_through_fast_factory(cell_slice) -> None:
    """W1.F1: fast_backtest_factory runs every RSI_REVERSION candidate on the
    kernel (zero fallbacks) and agrees with the slow factory on the ones it
    spot-checks. Replaces test_rsi_reversion_falls_back_through_fast_factory: the
    family no longer falls back."""
    fast_fn = fast_backtest_factory(cell_slice, rng_seed=0, flat_cost=False)
    slow_fn = real_backtest_factory(cell_slice, flat_cost=False)
    rng = np.random.default_rng(20260925)
    for idx in range(N_CANDIDATES):
        dna = _rsi_reversion_dna(rng)
        result = fast_fn(dna)
        assert result.outcome == "EVALUATED"
        if idx < 2:
            ref = slow_fn(dna)
            assert result.n_trades == ref.n_trades
            assert result.exit_reasons == ref.exit_reasons
            assert result.sharpe == pytest.approx(ref.sharpe, rel=1e-9, abs=1e-12)
    assert fast_fn.fallback_state["fallbacks"] == 0, fast_fn.fallback_state["reasons"]


def test_rsi_reversion_slow_engine_is_deterministic(cell_slice) -> None:
    """Two slow-engine runs with the same DNA + rng_seed must match."""
    spec = spec_for(SYMBOL)
    rng = np.random.default_rng(20260926)
    dna = _rsi_reversion_dna(rng, sl_atr=2.0, tp_atr=3.0, period=14)
    cfg = _make_cfg(0)
    rsi_cfg = dna_to_rsi_config(dna)
    strategy = RSIStrategy(rsi_cfg)

    run1 = BacktestEngine(spec=spec, config=cfg, rng_seed=0).run(
        cell_slice.bars, strategy
    )
    run2 = BacktestEngine(spec=spec, config=cfg, rng_seed=0).run(
        cell_slice.bars, strategy
    )

    assert run1.n_trades == run2.n_trades
    assert run1.final_equity == pytest.approx(run2.final_equity, rel=0, abs=1e-6)
    assert run1.exit_reasons == run2.exit_reasons


def test_dna_to_strategy_routes_rsi_reversion() -> None:
    """dna_to_strategy must return RSIStrategy for signal='RSI_REVERSION'."""
    rng = np.random.default_rng(20260927)
    dna = _rsi_reversion_dna(rng)
    cfg, strategy = dna_to_strategy(dna)
    assert isinstance(strategy, RSIStrategy)
    assert cfg is not None
    assert strategy.strategy_id == cfg.strategy_id


def test_rsi_unexpressible_candidate_records_named_fallback_reason(cell_slice) -> None:
    """The fallback reason must name WHY, not just 'not fast'.

    Renamed from test_rsi_fired_fixture_records_fallback_reason: RSI_REVERSION
    is kernel-expressible since W1.F1, so a plain RSI DNA no longer falls
    back. The contract is kept on an RSI DNA the kernel still refuses (a
    management type nothing implements): exactly one fallback, and its
    reason names the gene that caused it."""
    bt_fn = fast_backtest_factory(cell_slice, rng_seed=0, flat_cost=False)
    dna = dataclasses.replace(
        _rsi_reversion_dna(np.random.default_rng(20260928)), mgmt="BREAKEVEN"
    )
    bt_fn(dna)
    assert bt_fn.fallback_state["fallbacks"] == 1
    reasons = bt_fn.fallback_state["reasons"]
    assert any("mgmt" in r for r in reasons), (
        f"fallback reason must name the refused gene; got {reasons!r}"
    )


def test_rsi_uses_wilder_ema_not_sma() -> None:
    """RSI must use Wilder EMA smoothing, not a simple SMA.

    Wilder: avg_gain[t] = (prev * (p-1) + gain) / p
    SMA:    avg_gain[t] = mean(gains[t-p+1..t])

    They differ after the seed bar. This anchors the baseline so a refactor
    that swaps Wilder for SMA is caught immediately -- the resulting RSI values
    would be systematically different, silently shifting every trade gate.
    """
    from engine.research.features.rsi import rsi_series, DEFAULT_RSI_PERIOD

    close = np.array([100.0, 101.0, 99.5, 102.0, 98.0, 103.0, 101.5, 104.0,
                      99.0, 105.0, 102.5, 106.0, 100.0, 107.0, 103.0, 108.0,
                      105.0, 109.0, 106.0, 110.0, 107.0, 104.0, 108.0], dtype=np.float64)
    period = 5
    rsi = rsi_series(close, period=period)

    # First value at index == period (seed is the SMA over bars 0..period-1).
    # Wilder takes that SMA seed and then applies the EMA update.
    gains = np.maximum(np.diff(close), 0.0)
    losses = np.maximum(-np.diff(close), 0.0)
    seed_gain = float(np.mean(gains[:period]))
    seed_loss = float(np.mean(losses[:period]))

    # At index `period`, Wilder uses the seed directly.
    rs = seed_gain / seed_loss if seed_loss > 0 else (float("inf") if seed_gain > 0 else 0.0)
    wilder_val_period = float(100.0 - 100.0 / (1.0 + rs)) if seed_loss > 0 else (
        100.0 if seed_gain > 0 else 0.0
    )
    assert np.isfinite(rsi[period]), "RSI at seed index must be finite"
    assert abs(rsi[period] - wilder_val_period) < 1e-10, (
        f"RSI at index {period}: Wilder seed = {wilder_val_period}, "
        f"but rsi_series gave {rsi[period]}"
    )

    # At index period+1, Wilder applies the EMA update:
    #   avg_gain = (seed_gain * (p-1) + gain) / p
    # An SMA-based RSI would use a fresh SMA window. They diverge.
    g1 = gains[period]  # gain for bar at index period (diff uses [period-1..period])
    l1 = losses[period]
    wilder_gain = (seed_gain * (period - 1) + g1) / period
    wilder_loss = (seed_loss * (period - 1) + l1) / period
    rs2 = wilder_gain / wilder_loss if wilder_loss > 0 else (
        float("inf") if wilder_gain > 0 else 0.0
    )
    wilder_val_next = float(100.0 - 100.0 / (1.0 + rs2)) if wilder_loss > 0 else (
        100.0 if wilder_gain > 0 else 0.0
    )
    assert abs(rsi[period + 1] - wilder_val_next) < 1e-10, (
        f"RSI at index {period + 1}: Wilder EMA = {wilder_val_next}, "
        f"but rsi_series gave {rsi[period + 1]}"
    )

    # Explicitly verify divergence from SMA: recompute the SMA-based RSI at
    # index period+1 and confirm it differs from the Wilder result.
    sma_gain = float(np.mean(gains[1 : period + 1]))
    sma_loss = float(np.mean(losses[1 : period + 1]))
    rs_sma = sma_gain / sma_loss if sma_loss > 0 else (
        float("inf") if sma_gain > 0 else 0.0
    )
    sma_val_next = float(100.0 - 100.0 / (1.0 + rs_sma)) if sma_loss > 0 else (
        100.0 if sma_gain > 0 else 0.0
    )
    if wilder_val_next != sma_val_next:
        assert rsi[period + 1] == pytest.approx(wilder_val_next, abs=1e-10), (
            f"rsi_series does not match Wilder EMA at index {period + 1}"
        )
    # If they happen to coincide on this path, at least the Wilder match holds.
    assert rsi[period + 1] == pytest.approx(wilder_val_next, abs=1e-10)


def test_rsi_rejects_oversold_ge_overbought() -> None:
    """RSIConfig with oversold >= overbought must be refused at construction.

    Mean reversion requires a band: oversold < overbought. If they cross or
    meet, the long and short conditions overlap or invert -- a config that
    cannot mean-revert is not a strategy, it is a noise generator. The
    RSIConfig __post_init__ is the guardrail; it fails fast so the bad
    config never reaches the engine.
    """
    from engine.research.backtest.strategies.rsi import RSIConfig

    # oversold == overbought: every RSI value is in one of two bands that
    # share a boundary -- a trade fires on every bar.
    with pytest.raises(DataError, match="must be <"):
        RSIConfig(oversold=50.0, overbought=50.0)

    # oversold > overbought: the "long" condition is always true, so the
    # strategy is permanently long and never short -- not reversion.
    with pytest.raises(DataError, match="must be <"):
        RSIConfig(oversold=70.0, overbought=30.0)

    # The default (30 < 70) is fine.
    cfg = RSIConfig()
    assert cfg.oversold < cfg.overbought


def test_rsi_warmup_nans_before_period() -> None:
    """RSI must be NaN for the first ``period`` bars (warmup).

    The Wilder seed needs a full window before it produces a value, and
    on_bar must treat NaN as "do not trade yet". A non-NaN warmup value
    would let the strategy trade on an unseeded indicator.
    """
    from engine.research.features.rsi import rsi_series

    close = np.linspace(100.0, 110.0, 50, dtype=np.float64)
    period = 14
    rsi = rsi_series(close, period=period)
    # Indices 0..period-1 are NaN (need `period` gains to seed).
    for i in range(period):
        assert np.isnan(rsi[i]), f"RSI at index {i} should be NaN (warmup)"
    # Index period onward: first real value.
    assert np.isfinite(rsi[period]), (
        f"RSI at index {period} should be finite (seeded)"
    )


def test_effective_hash_tracks_dna_to_ma_config_for_ma_cross() -> None:
    """effective_genes must match dna_to_ma_config for the MA_CROSS family.

    This is the B3 effective_hash fixture for the fast-path family. It pins
    the contract: the dedupe cache keyed on effective_gene_hash groups
    MA_CROSS DNAs that differ only in engine-invisible genes. If the two
    functions drift, two DNAs that produce different runs get the same hash,
    or two that produce identical runs get different hashes.
    """
    from engine.research.search.sampler import random_dna

    rng = np.random.default_rng(20260929)
    for _ in range(200):
        dna = random_dna(rng)
        dna = dataclasses.replace(dna, signal="MA_CROSS")
        cfg = dna_to_ma_config(dna)
        _, _sig, fast, slow, volume, sl_atr, tp_atr = effective_genes(dna)
        assert (fast, slow, volume, sl_atr, tp_atr) == (
            cfg.fast, cfg.slow, cfg.volume, cfg.sl_atr, cfg.tp_atr
        ), f"effective_genes / dna_to_ma_config drift for {dna.dna_hash}"


def test_effective_hash_tracks_dna_to_rsi_config_for_rsi_reversion() -> None:
    """effective_genes must match dna_to_rsi_config for the RSI_REVERSION family.

    C3-A': threshold genes (rsi_oversold, rsi_overbought, rsi_exit_midline)
    reach the engine via RSIStrategy and must appear in the projection. If they
    drift, two DNAs with different gates but the same period hash together --
    a dedupe collision that masks a real trial difference.
    """
    from engine.research.search.sampler import random_dna
    from engine.research.backtest.strategies.rsi import RSIConfig

    rng = np.random.default_rng(20260930)
    for _ in range(200):
        dna = random_dna(rng)
        dna = dataclasses.replace(dna, signal="RSI_REVERSION")
        cfg = dna_to_rsi_config(dna)
        _ver, _sig, period, oversold, overbought, exit_midline, volume, \
            sl_atr, tp_atr = effective_genes(dna)
        assert (period, oversold, overbought, exit_midline, volume, sl_atr,
                tp_atr) == (
            cfg.period, cfg.oversold, cfg.overbought, cfg.exit_midline,
            cfg.volume, cfg.sl_atr, cfg.tp_atr,
        ), f"effective_genes / dna_to_rsi_config drift for {dna.dna_hash}"


def test_rsi_threshold_genes_change_hash() -> None:
    """Varying rsi_oversold, rsi_overbought, rsi_exit_midline must each change
    the hash.

    These genes reach the engine via RSIStrategy: rsi_oversold/rsi_overbought
    gate entry (a long fires only when RSI < oversold), and rsi_exit_midline
    enables midline-cross exit for an open position. Changing any must produce
    a different effective_gene_hash so trials aren't deduplicated.
    """
    from engine.research.search.sampler import random_dna

    rng = np.random.default_rng(20261001)
    base = dataclasses.replace(
        random_dna(rng), signal="RSI_REVERSION",
        rsi_oversold=20.0, rsi_overbought=80.0,
        rsi_exit_midline=False,  # force False so d3's toggle is a real change
    )
    assert effective_genes(base)[0] == EFFECTIVE_HASH_VERSION

    # Only oversold changes
    d1 = dataclasses.replace(base, rsi_oversold=25.0)
    assert effective_gene_hash(d1) != effective_gene_hash(base), (
        "rsi_oversold change did not alter effective hash"
    )

    # Only overbought changes
    d2 = dataclasses.replace(base, rsi_overbought=85.0)
    assert effective_gene_hash(d2) != effective_gene_hash(base), (
        "rsi_overbought change did not alter effective hash"
    )

    # Only exit_midline changes (base is False, so True is a real toggle)
    d3 = dataclasses.replace(base, rsi_exit_midline=True)
    assert effective_gene_hash(d3) != effective_gene_hash(base), (
        "rsi_exit_midline change did not alter effective hash"
    )

    # All change together -> still different from base
    d4 = dataclasses.replace(base, rsi_oversold=25.0, rsi_overbought=85.0,
                             rsi_exit_midline=True)
    assert effective_gene_hash(d4) != effective_gene_hash(base), (
        "combined threshold change did not alter effective hash"
    )


def test_rsi_config_uses_threshold_genes() -> None:
    """dna_to_rsi_config must carry threshold genes into RSIConfig."""
    from engine.research.search.sampler import random_dna

    rng = np.random.default_rng(20261002)
    dna = dataclasses.replace(
        random_dna(rng),
        signal="RSI_REVERSION",
        rsi_period=14,
        rsi_oversold=20.0,
        rsi_overbought=80.0,
        rsi_exit_midline=True,
        sl_atr=2.0,
        tp_atr=3.0,
        risk_pct=0.01,  # volume = 0.1
    )
    cfg = dna_to_rsi_config(dna)
    assert cfg.oversold == 20.0
    assert cfg.overbought == 80.0
    assert cfg.exit_midline is True
    assert "os20" in cfg.strategy_id
    assert "ob80" in cfg.strategy_id
    assert "_midline" in cfg.strategy_id


def test_rsi_config_exit_midline_false_by_default() -> None:
    """dna_to_rsi_config must propagate exit_midline=False when the DNA has it off."""
    from engine.research.search.sampler import random_dna

    rng = np.random.default_rng(20261007)
    dna = dataclasses.replace(random_dna(rng), rsi_exit_midline=False)
    cfg = dna_to_rsi_config(dna)
    assert cfg.exit_midline is False
    assert "_midline" not in cfg.strategy_id


def test_rsi_midline_exit_closes_long_on_cross_below_50() -> None:
    """RSIStrategy with exit_midline=True must close long when RSI crosses below 50."""
    from engine.research.features.rsi import DEFAULT_RSI_PERIOD
    from engine.research.backtest.strategies.rsi import RSIConfig, RSIStrategy

    # Build a synthetic close series that drives RSI above then below 50.
    close = np.concatenate([
        np.full(DEFAULT_RSI_PERIOD + 2, 100.0, dtype=np.float64),  # RSI ~ 50
        np.linspace(101.0, 115.0, 20, dtype=np.float64),           # RSI > 50
        np.linspace(115.0, 95.0, 20, dtype=np.float64),            # RSI < 50
    ])
    from engine.research.data.mt5_source import BarData
    bars = BarData(
        time=np.arange(close.size, dtype=np.float64),
        open=close, high=close, low=close, close=close,
        tick_volume=np.zeros(close.size, dtype=np.int64),
        spread=np.zeros(close.size, dtype=np.float64),
        real_volume=np.zeros(close.size, dtype=np.int64),
        symbol="TEST", timeframe="H1",
    )

    cfg = RSIConfig(
        period=DEFAULT_RSI_PERIOD, oversold=30.0, overbought=70.0,
        exit_midline=True, volume=0.1, strategy_id="test",
        sl_atr=1.0, tp_atr=2.0,
    )
    strat = RSIStrategy(cfg)

    state: dict[str, object] = {"direction": "long"}
    exit_intent = None
    for i in range(close.size):
        intent = strat.on_bar(i, bars, state)
        if intent is not None and intent.direction == "long" and intent.volume == 0.0:
            exit_intent = intent
            break

    assert exit_intent is not None, "long should have exited on midline cross"
    assert "direction" not in state, "position should be closed after midline exit"
