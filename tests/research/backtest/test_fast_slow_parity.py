"""S2-E: 200 real DNAs, one real cell slice, slow vs fast.

The synthetic step tests prove each feature in isolation. This proves
the combination on data the engine will actually see -- real spreads,
real gaps, real session structure, and MA windows drawn from the real
sampler rather than chosen to be convenient.

A single divergence fails. A SYSTEMATIC divergence -- the same sign or
the same magnitude across the population -- would mean a semantic
mismatch rather than a rounding difference, so the summary reports the
worst case and the distribution, not just a pass.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from engine.research.backtest.engine import BacktestConfig, BacktestEngine
from engine.research.backtest.fast_path import (
    expressible,
    record_exit_reasons,
    record_to_dict,
    run_fast,
)
from engine.research.backtest.prep import prep_cell
from engine.research.backtest.strategies.ma_crossover import MACrossStrategy
from engine.research.data.dataset_snapshot import find_snapshot, load_snapshot
from engine.research.search.real_backtest import dna_to_ma_config, spec_for
from engine.research.search.sampler import random_dna

SYMBOL = "USDCADm"
DATASET_ID = "aed3bf314539ec5a"
N_CANDIDATES = 200
# ~6 months of H1 bars. Enough for real structure; small enough that the
# slow engine can run 200 candidates in a test.
SLICE_BARS = 4_000


@pytest.fixture(scope="module")
def cell_slice():
    if find_snapshot(DATASET_ID) is None:
        pytest.skip(f"snapshot {DATASET_ID} not taken on this machine")
    bars = load_snapshot(DATASET_ID)
    n = min(SLICE_BARS, int(np.asarray(bars.close).size))
    from engine.research.data.mt5_source import BarData

    return BarData(
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


# The production cost settings, so this is the path a campaign uses.
# Uses a synthetic hourly spread profile that varies by hour so the W0
# path is exercised -- a flat tuple would make cost_spread identical
# whether or not the hourly lookup fires.
_HOURLY_SPREAD_POINTS = tuple(float(h % 20 + 1) for h in range(24))

_CFG = {
    "charge_swap": True,
    "charge_commission": True,
    "commission_per_lot_per_side": 0.0,
    "slippage_mean_points": 0.3,
    "slippage_std_points": 0.2,
    "no_trade_hours": (),
    "max_spread_points": None,
    "hourly_spread_points": _HOURLY_SPREAD_POINTS,
}


def test_fast_slow_parity_on_200_real_dnas(cell_slice) -> None:
    spec = spec_for(SYMBOL)
    prepped = prep_cell(
        cell_slice, spec,
        no_trade_hours=_CFG["no_trade_hours"],
        max_spread_points=_CFG["max_spread_points"],
        hourly_spread_points=_CFG["hourly_spread_points"],
    )

    rng = np.random.default_rng(20260924)
    worst_equity_rel = 0.0
    worst_equity_case = ""
    worst_fee_rel = 0.0
    signed_rels: list[float] = []
    mismatches: list[str] = []
    n_fallback = 0
    n_with_trades = 0

    for idx in range(N_CANDIDATES):
        dna = random_dna(rng)
        # Only MA_CROSS is kernel-expressible; pin the signal so the parity
        # comparison exercises the fast path on every iteration.
        dna = dataclasses.replace(dna, signal="MA_CROSS")
        cfg = BacktestConfig(
            strategy_id=f"parity-{idx}", symbol=SYMBOL, timeframe="H1",
            initial_equity=10_000.0, **_CFG,
        )
        if not expressible(dna, cfg).ok:
            n_fallback += 1
            continue

        ma = dna_to_ma_config(dna)
        slow = BacktestEngine(spec=spec, config=cfg, rng_seed=0).run(
            cell_slice, MACrossStrategy(ma)
        )
        record, _, status = run_fast(
            prepped,
            fast=ma.fast, slow=ma.slow, volume=ma.volume,
            sl_atr=ma.sl_atr, tp_atr=ma.tp_atr,
            initial_equity=10_000.0,
            max_spread_points=_CFG["max_spread_points"],
            charge_swap=_CFG["charge_swap"],
            charge_commission=_CFG["charge_commission"],
            commission_per_lot_per_side=_CFG["commission_per_lot_per_side"],
            slippage_mean_points=_CFG["slippage_mean_points"],
            slippage_std_points=_CFG["slippage_std_points"],
            rng_seed=0,
            want_equity=False,
        )
        assert status == 0, f"candidate {idx} ({dna.dna_hash}) status {status}"
        fast = record_to_dict(record)

        if slow.n_trades > 0:
            n_with_trades += 1

        # Structure must be EXACT.
        for name, slow_value in (
            ("n_trades", slow.n_trades),
            ("n_round_trips", slow.n_round_trips),
            ("n_open", slow.n_open),
        ):
            if fast[name] != slow_value:
                mismatches.append(
                    f"[{idx} {dna.dna_hash}] {name}: fast={fast[name]} "
                    f"slow={slow_value} (ma {ma.fast}/{ma.slow} "
                    f"sl={ma.sl_atr:.3f} tp={ma.tp_atr:.3f})"
                )
        fast_reasons = record_exit_reasons(record)
        if fast_reasons != dict(slow.exit_reasons):
            mismatches.append(
                f"[{idx} {dna.dna_hash}] exit_reasons: fast={fast_reasons} "
                f"slow={dict(slow.exit_reasons)}"
            )

        # Money within 1e-6 relative.
        denom = abs(slow.final_equity) or 1.0
        rel = (fast["final_equity"] - slow.final_equity) / denom
        signed_rels.append(rel)
        if abs(rel) > worst_equity_rel:
            worst_equity_rel = abs(rel)
            worst_equity_case = (
                f"[{idx} {dna.dna_hash}] fast={fast['final_equity']!r} "
                f"slow={slow.final_equity!r} rel={rel:.3e}"
            )

        slow_fees = (
            slow.cost_commission + slow.cost_spread
            + slow.cost_slippage + slow.cost_swap
        )
        fast_fees = (
            fast["cost_commission"] + fast["cost_spread"]
            + fast["cost_slippage"] + fast["cost_swap"]
        )
        fee_denom = abs(slow_fees) or 1.0
        worst_fee_rel = max(worst_fee_rel, abs(fast_fees - slow_fees) / fee_denom)

    print(
        f"\nparity over {N_CANDIDATES} DNAs on {int(np.asarray(cell_slice.close).size)}"
        f" real bars:\n"
        f"  fallbacks              : {n_fallback}\n"
        f"  candidates with trades : {n_with_trades}\n"
        f"  worst |equity rel|     : {worst_equity_rel:.3e}  {worst_equity_case}\n"
        f"  worst |fee rel|        : {worst_fee_rel:.3e}\n"
        f"  mean signed equity rel : "
        f"{(sum(signed_rels) / len(signed_rels) if signed_rels else 0.0):.3e}"
    )

    assert not mismatches, (
        f"{len(mismatches)} structural mismatch(es):\n  "
        + "\n  ".join(mismatches[:10])
    )
    assert n_with_trades > N_CANDIDATES // 2, (
        f"only {n_with_trades} candidates traded; the slice is too quiet "
        "for this to be evidence"
    )
    assert worst_equity_rel < 1e-6, (
        f"worst equity divergence {worst_equity_rel:.3e} exceeds 1e-6: "
        f"{worst_equity_case}"
    )
    assert worst_fee_rel < 1e-6, f"worst fee divergence {worst_fee_rel:.3e}"

    # A systematic offset would mean a semantic mismatch, not rounding.
    if signed_rels:
        mean_rel = abs(sum(signed_rels) / len(signed_rels))
        assert mean_rel < 1e-9, (
            f"mean signed divergence {mean_rel:.3e} is systematic, not noise"
        )
