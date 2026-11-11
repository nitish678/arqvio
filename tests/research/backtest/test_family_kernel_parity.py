"""W1.F1-F4: 60 real DNAs per family, slow engine (oracle) vs kernel.

Pinned USDCADm H1 snapshot, production-shaped costs (hourly spread,
slippage stream, swap) plus BOTH entry guardrails live -- a no-trade hour
(deferral) and a max-spread gate -- so the event stream is compared under
the paths a campaign actually takes.

Per DNA:
  * trade list EXACT: signal/fill/exit bars, side, entry/exit prices,
    volume, SL/TP levels, exit reason, realized P&L
  * counters EXACT: trades, round trips, open, exit partition, blocked /
    deferred / gated events
  * money: final equity and total fees <= 1e-6 rel

Across the population: mean signed Sharpe bias < 1e-9, >= 50% of DNAs
fired (else the slice proves nothing), zero-trade rate reported. The
stats table is printed verbatim for the report (run with -s).
"""
from __future__ import annotations

import dataclasses
import math
import time

import numpy as np
import pytest

from engine.research.backtest.engine import BacktestConfig, BacktestEngine
from engine.research.backtest.fast_path import (
    record_exit_reasons,
    record_to_dict,
    run_fast,
)
from engine.research.backtest.kernel import TR_SIZE
from engine.research.backtest.prep import family_arrays, prep_cell
from engine.research.data.dataset_snapshot import find_snapshot, load_snapshot
from engine.research.data.mt5_source import BarData
from engine.research.search.real_backtest import dna_to_strategy, spec_for
from engine.research.search.sampler import random_dna

from tests.research.backtest.test_kernel_parity_steps import (
    slow_trade_rows,
    trade_list_diffs,
)

SYMBOL = "USDCADm"
DATASET_ID = "aed3bf314539ec5a"  # USDCADm H1
N_CANDIDATES = 60
SLICE_BARS = 800
BUDGET_S = 600.0
PPY_H1 = 6048

FAMILY_SEEDS = {
    "RSI_REVERSION": 20261001,
    # Reseeded from 20261002 (rule: +100*k, first k whose 60 DNAs fire the
    # gate): only 5 bars of this slice are above the p90 gate and tradeable
    # (hour 22; the other 18 are the no-trade hour 21), and 20261002 and
    # 20261102 never filled on one -- exact parity, but the gate unexercised.
    "STOCH_CROSS": 20261202,
    "MACD_CROSS": 20261003,
    "DONCHIAN_BREAKOUT": 20261004,
}

_HOURLY = tuple(float(h % 20 + 1) for h in range(24))
_NO_TRADE_HOURS = (21,)
_COUNTERS = (
    "n_trades", "n_round_trips", "n_open", "blocked_zero_spread_events",
    "deferred_zero_spread_fills", "sltp_deferred_zero_spread",
    "blocked_max_spread", "blocked_daily_loss", "friday_flattens",
)


@pytest.fixture(scope="module")
def bars() -> BarData:
    if find_snapshot(DATASET_ID) is None:
        pytest.skip(f"snapshot {DATASET_ID} not taken on this machine")
    b = load_snapshot(DATASET_ID)
    s = slice(-SLICE_BARS, None)
    return BarData(
        time=np.asarray(b.time)[s], open=np.asarray(b.open)[s],
        high=np.asarray(b.high)[s], low=np.asarray(b.low)[s],
        close=np.asarray(b.close)[s], tick_volume=np.asarray(b.tick_volume)[s],
        spread=np.asarray(b.spread)[s], real_volume=np.asarray(b.real_volume)[s],
        symbol=b.symbol, timeframe=b.timeframe,
    )


def _sharpe(eq: np.ndarray) -> float:
    """real_backtest_factory's Sharpe arithmetic, verbatim."""
    if eq.size < 2 or not np.all(eq[:-1] > 0):
        return 0.0
    r = np.diff(eq) / eq[:-1]
    if r.size < 2 or np.std(r) <= 0:
        return 0.0
    return float(np.mean(r) / np.std(r, ddof=1)) * math.sqrt(PPY_H1)


def run_parity(bars: BarData, family: str) -> dict:
    spec = spec_for(SYMBOL)
    gate = float(np.percentile(np.asarray(bars.spread, dtype=np.float64), 90))
    base = dict(
        charge_swap=True, charge_commission=True, commission_per_lot_per_side=0.0,
        slippage_mean_points=0.3, slippage_std_points=0.2,
        no_trade_hours=_NO_TRADE_HOURS, max_spread_points=gate,
        hourly_spread_points=_HOURLY,
    )
    prepped = prep_cell(
        bars, spec, no_trade_hours=_NO_TRADE_HOURS, max_spread_points=gate,
        hourly_spread_points=_HOURLY,
    )
    rng = np.random.default_rng(FAMILY_SEEDS[family])
    stats = {
        "family": family, "n": N_CANDIDATES, "fired": 0, "zero_trade": 0,
        "trades": 0, "blocked": 0, "deferred": 0, "gated": 0,
        "worst_equity_rel": 0.0, "worst_fee_rel": 0.0, "mismatches": [],
        "sharpe_bias": [], "signed_equity_rel": [],
    }
    started = time.time()
    for idx in range(N_CANDIDATES):
        dna = dataclasses.replace(random_dna(rng), signal=family)
        cfg, strategy = dna_to_strategy(dna)
        slow = BacktestEngine(
            spec=spec,
            config=BacktestConfig(
                strategy_id=cfg.strategy_id, symbol=SYMBOL, timeframe="H1",
                initial_equity=10_000.0, **base,
            ),
            rng_seed=0,
        ).run(bars, strategy)

        trades_out = np.zeros((prepped.n_bars, TR_SIZE), dtype=np.float64)
        record, eq, status = run_fast(
            prepped, volume=cfg.volume, sl_atr=cfg.sl_atr, tp_atr=cfg.tp_atr,
            initial_equity=10_000.0, max_spread_points=gate,
            charge_swap=True, charge_commission=True,
            commission_per_lot_per_side=0.0,
            slippage_mean_points=0.3, slippage_std_points=0.2, rng_seed=0,
            want_equity=True, family=family_arrays(prepped, cfg),
            trades_out=trades_out,
        )
        tag = f"[{idx} {dna.dna_hash}]"
        if status != 0:
            stats["mismatches"].append(f"{tag} kernel status {status}")
            continue
        fast = record_to_dict(record)
        for name in _COUNTERS:
            if fast[name] != getattr(slow, name):
                stats["mismatches"].append(
                    f"{tag} {name}: fast={fast[name]} slow={getattr(slow, name)}"
                )
        if record_exit_reasons(record) != dict(slow.exit_reasons):
            stats["mismatches"].append(
                f"{tag} exit_reasons fast={record_exit_reasons(record)} "
                f"slow={dict(slow.exit_reasons)}"
            )
        fast_rows = [tuple(r) for r in trades_out[: fast["n_trades"]]]
        stats["mismatches"] += [
            f"{tag} {d}" for d in trade_list_diffs(fast_rows, slow_trade_rows(slow))[:3]
        ]

        rel = (fast["final_equity"] - slow.final_equity) / (abs(slow.final_equity) or 1.0)
        stats["signed_equity_rel"].append(rel)
        stats["worst_equity_rel"] = max(stats["worst_equity_rel"], abs(rel))
        slow_fees = slow.cost_commission + slow.cost_spread + slow.cost_slippage + slow.cost_swap
        fast_fees = (fast["cost_commission"] + fast["cost_spread"]
                     + fast["cost_slippage"] + fast["cost_swap"])
        stats["worst_fee_rel"] = max(
            stats["worst_fee_rel"], abs(fast_fees - slow_fees) / (abs(slow_fees) or 1.0)
        )
        stats["sharpe_bias"].append(_sharpe(eq) - _sharpe(slow.equity_curve))
        stats["fired" if slow.n_trades > 0 else "zero_trade"] += 1
        stats["trades"] += slow.n_trades
        stats["blocked"] += slow.blocked_zero_spread_events
        stats["deferred"] += slow.deferred_zero_spread_fills
        stats["gated"] += slow.blocked_max_spread
    stats["elapsed_s"] = time.time() - started
    stats["mean_sharpe_bias"] = float(np.mean(stats["sharpe_bias"])) if stats["sharpe_bias"] else 0.0
    stats["mean_signed_equity_rel"] = (
        float(np.mean(stats["signed_equity_rel"])) if stats["signed_equity_rel"] else 0.0
    )
    return stats


def _table(s: dict) -> str:
    return (
        f"\nPARITY {s['family']} n={s['n']} bars={SLICE_BARS} "
        f"elapsed={s['elapsed_s']:.1f}s\n"
        f"  fired                  : {s['fired']}/{s['n']} "
        f"(zero-trade rate {s['zero_trade'] / s['n']:.1%})\n"
        f"  trades (slow)          : {s['trades']}\n"
        f"  blocked/deferred/gated : {s['blocked']}/{s['deferred']}/{s['gated']}\n"
        f"  worst |equity rel|     : {s['worst_equity_rel']:.3e}\n"
        f"  mean signed equity rel : {s['mean_signed_equity_rel']:.3e}\n"
        f"  worst |fee rel|        : {s['worst_fee_rel']:.3e}\n"
        f"  mean signed Sharpe bias: {s['mean_sharpe_bias']:.3e}\n"
        f"  mismatches             : {len(s['mismatches'])}"
    )


@pytest.mark.parametrize("family", list(FAMILY_SEEDS))
def test_family_kernel_matches_oracle_on_60_real_dnas(bars: BarData, family: str) -> None:
    s = run_parity(bars, family)
    print(_table(s))
    assert not s["mismatches"], (
        f"{len(s['mismatches'])} mismatch(es):\n  " + "\n  ".join(s["mismatches"][:10])
    )
    assert s["fired"] >= N_CANDIDATES // 2, (
        f"{family}: only {s['fired']}/{N_CANDIDATES} DNAs traded"
    )
    assert s["blocked"] > 0 and s["gated"] > 0, "guardrails never fired on this slice"
    assert s["worst_equity_rel"] <= 1e-6
    assert s["worst_fee_rel"] <= 1e-6
    assert abs(s["mean_sharpe_bias"]) < 1e-9
    assert s["elapsed_s"] <= BUDGET_S, f"{family}: {s['elapsed_s']:.0f}s > {BUDGET_S:.0f}s"
