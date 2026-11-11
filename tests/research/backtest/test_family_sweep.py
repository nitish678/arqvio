"""PART 1: Family slow-engine sweep — 500 DNAs per family, zero exceptions.

The slow engine (``real_backtest_factory``) must accept every signal family
with zero crashes across a representative random sample. This is the safety
net that catches a strategy bug the parity fixtures — each pinned to 20 DNAs
— cannot: a degenerate gene combination that the strategy's on_bar() never
reaches on a hand-chosen DNA but appears in the wild.

One test per family, each running 500 random DNAs through the slow engine on
a real USDCADm H1 snapshot slice. Assertions:

  - zero exceptions (no crash, no hang in the engine loop);
  - fired% (fraction with n_trades > 0) > 10% — a family that never fires
    on a liquid pair is dead weight;
  - deterministic — two identical runs produce the same n_trades and final
    equity (the engine is seeded, not stochastic across runs).

Spec change (2026-09-28): SLICE_BARS reduced 4000→1200 to bring total
budget from ~30h to ≤2h. 500 DNAs/family and determinism replay=10 kept.
Rationale: sweep's job is exception/fire-rate sanity, not deep signal
coverage. The first full run (4000 bars) caught the MACD NaN-seed class;
1200 bars preserves that power at ~1/3 cost. Full-cell behavior is covered
by the canary's full profile. This is a background soak, not a blocking
gate — the soak only gates PART 8 (W4); it must be green there.
"""
from __future__ import annotations

import traceback
from collections import Counter

import numpy as np
import pytest

from engine.research.data.dataset_snapshot import find_snapshot, load_snapshot
from engine.research.data.mt5_source import BarData
from engine.research.search.dna import DNA
from engine.research.search.real_backtest import (
    CellBars,
    real_backtest_factory,
    spec_for,
)

SYMBOL = "USDCADm"
DATASET_ID = "aed3bf314539ec5a"
N_PER_FAMILY = 500
SLICE_BARS = 1_200

_FAMILIES = (
    "MA_CROSS", "RSI_REVERSION",
    "STOCH_CROSS", "MACD_CROSS", "DONCHIAN_BREAKOUT",
)

_FAMILY_SEEDS = {
    "MA_CROSS": 20260901,
    "RSI_REVERSION": 20260902,
    "STOCH_CROSS": 20260903,
    "MACD_CROSS": 20260904,
    "DONCHIAN_BREAKOUT": 20260905,
}


@pytest.fixture(scope="module")
def cell_slice():
    if find_snapshot(DATASET_ID) is None:
        pytest.skip(f"snapshot {DATASET_ID} not taken on this machine")
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


def _dna_with_signal(rng: np.random.Generator, signal: str) -> DNA:
    from engine.research.search.dna import (
        BIAS_VALUES, EXEC_VALUES, FILTER_VALUES,
        RISK_VALUES, SIGNAL2_VALUES, DNA_SCHEMA_VERSION,
    )
    from engine.research.search.sampler import (
        RSI_PERIOD_VALUES, STOCH_K_PERIOD_VALUES, STOCH_D_PERIOD_VALUES,
        STOCH_SMOOTHING_VALUES, MACD_FAST_VALUES, MACD_SLOW_VALUES,
        MACD_SIGNAL_PERIOD_VALUES, DONCHIAN_PERIOD_VALUES,
        _macd_slow_choices,
    )
    macd_fast = int(rng.choice(MACD_FAST_VALUES))
    macd_slow = int(rng.choice(_macd_slow_choices(macd_fast)))
    return DNA(
        bias=str(rng.choice(BIAS_VALUES)),
        filter=str(rng.choice(FILTER_VALUES)),
        signal=signal,
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
        rsi_period=int(rng.choice(RSI_PERIOD_VALUES)),
        rsi_oversold=float(rng.uniform(10.0, 35.0)),
        rsi_overbought=float(rng.uniform(65.0, 90.0)),
        rsi_exit_midline=bool(rng.random() < 0.5),
        stoch_k_period=int(rng.choice(STOCH_K_PERIOD_VALUES)),
        stoch_d_period=int(rng.choice(STOCH_D_PERIOD_VALUES)),
        stoch_smoothing=int(rng.choice(STOCH_SMOOTHING_VALUES)),
        stoch_oversold=float(rng.uniform(5.0, 25.0)),
        stoch_overbought=float(rng.uniform(75.0, 95.0)),
        macd_fast=macd_fast,
        macd_slow=macd_slow,
        macd_signal_period=int(rng.choice(MACD_SIGNAL_PERIOD_VALUES)),
        donchian_period=int(rng.choice(DONCHIAN_PERIOD_VALUES)),
        uses_future_bar=False,
        schema_version=DNA_SCHEMA_VERSION,
    )


def _bucket(exc: BaseException) -> str:
    """Crash bucket = exception type + innermost engine frame."""
    frames = [
        f for f in traceback.extract_tb(exc.__traceback__)
        if "engine" in f.filename.replace("\\", "/")
    ]
    where = f"{frames[-1].filename.split('engine')[-1]}:{frames[-1].lineno}" if frames else "?"
    return f"{type(exc).__name__}@{where}"


def _metrics(r) -> tuple:
    ppr = r.per_period_returns
    return (
        r.outcome, r.n_trades, r.sharpe, r.total_return, r.max_dd,
        None if ppr is None else np.asarray(ppr).tobytes(),
    )


@pytest.mark.parametrize("signal", _FAMILIES)
def test_family_sweep_zero_exceptions(cell_slice, signal: str) -> None:
    """Run 500 random DNAs of one family through the slow engine with zero
    exceptions, fired% > 10%, and determinism on a 10-DNA replay."""
    rng = np.random.default_rng(_FAMILY_SEEDS[signal])
    bt_fn = real_backtest_factory(cell_slice, flat_cost=False)

    buckets: Counter[str] = Counter()
    first_tb: str | None = None
    fired = 0
    replay: list[tuple[DNA, tuple]] = []

    for idx in range(N_PER_FAMILY):
        dna = _dna_with_signal(rng, signal)
        try:
            result = bt_fn(dna)
        except Exception as exc:
            buckets[_bucket(exc)] += 1
            if first_tb is None:
                first_tb = (
                    f"candidate {idx} {dna.dna_hash}\n"
                    +"".join(traceback.format_exception(exc))
                )
            continue
        if (result.n_trades or 0) > 0:
            fired += 1
        if len(replay) < 10:
            replay.append((dna, _metrics(result)))

    crashes = sum(buckets.values())
    fired_pct = fired / N_PER_FAMILY * 100
    print(
        f"\nSWEEP {signal}: n={N_PER_FAMILY} bars={cell_slice.n_bars} "
        f"crashes={crashes} fired={fired} ({fired_pct:.1f}%) "
        f"zero_trade={N_PER_FAMILY - fired - crashes} buckets={dict(buckets)}"
    )
    assert crashes == 0, (
        f"{signal}: {crashes} crashes; buckets={dict(buckets)}\nfirst:\n{first_tb}"
    )
    assert fired_pct > 10.0, (
        f"{signal}: fired {fired}/{N_PER_FAMILY} ({fired_pct:.1f}%) <= 10%"
    )

    # Determinism: replay the first 10 DNAs on a FRESH factory; every metric
    # (including the per-period return series) must be identical.
    bt_fn2 = real_backtest_factory(cell_slice, flat_cost=False)
    for dna, first in replay:
        again = _metrics(bt_fn2(dna))
        assert again == first, f"{signal}: non-deterministic replay for {dna.dna_hash}"
    print(f"SWEEP {signal}: replay-10 deterministic OK")
