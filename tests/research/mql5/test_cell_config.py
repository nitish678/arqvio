"""The cell configs are exactly what the oracle runs under: the battery's own and run_slow_trial's."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

from engine.research.backtest.engine import BacktestConfig, BacktestEngine
from engine.research.mql5 import ga_subjects
from engine.research.mql5.cell_config import battery_cell, campaign_cell
from engine.research.search.real_backtest import (
    dna_to_strategy,
    load_cell_bars_from_snapshot,
    run_slow_trial,
    slow_cell_setup,
)
from engine.research.search.sampler import random_dna


def test_the_battery_cell_is_the_battery_config(pinned_home: Path) -> None:
    cell = battery_cell()
    full = load_cell_bars_from_snapshot("USDCADm", "H1", "aed3bf314539ec5a").bars
    assert cell.n_bars == 800 and cell.first_time == int(full.time[-800]) and cell.last_time == int(full.time[-1])
    gate = float(np.percentile(np.asarray(full.spread[-800:], dtype=np.float64), 90))
    battery = BacktestConfig(  # tests/research/backtest/test_family_kernel_parity.py, run_parity, verbatim
        strategy_id="x", symbol="USDCADm", timeframe="H1", initial_equity=10_000.0,
        charge_swap=True, charge_commission=True, commission_per_lot_per_side=0.0,
        slippage_mean_points=0.3, slippage_std_points=0.2, no_trade_hours=(21,), max_spread_points=gate,
        hourly_spread_points=tuple(float(h % 20 + 1) for h in range(24)),
    )
    assert cell.engine_config("x", risk_pct=0.01) == battery


def test_the_campaign_cell_is_slow_cell_setup(pinned_home: Path) -> None:
    for sym, did in (("USDCADm", "aed3bf314539ec5a"), ("XAUUSDm", "447b843f87438a9a")):
        cell = campaign_cell(sym, did)
        setup = slow_cell_setup(load_cell_bars_from_snapshot(sym, "H1", did))
        assert (cell.hourly_spread_points, cell.no_trade_hours, cell.max_spread_points,
                cell.commission_per_lot_per_side) == (setup.hourly_spread_points, setup.no_trade_hours,
                                                      setup.max_spread_points, setup.commission)
        assert cell.spec == setup.spec and cell.n_bars == {"USDCADm": 59957, "XAUUSDm": 57011}[sym]
    assert campaign_cell("XAUUSDm", "447b843f87438a9a").no_trade_hours == (21,)


def test_a_campaign_oracle_run_equals_run_slow_trial(pinned_home: Path) -> None:
    """The engine config the parity harness uses reproduces the campaign's own slow trial, bar for bar.
    MA_CROSS: the slow oracle recomputes RSI / MACD / Donchian series on the whole prefix every bar
    (quadratic on a 60k-bar window); MA_CROSS reads windowed means and runs in seconds."""
    cell = campaign_cell("USDCADm", "aed3bf314539ec5a")
    dna = dataclasses.replace(random_dna(np.random.default_rng(20261005)), signal="MA_CROSS")
    cfg, strategy = dna_to_strategy(dna)
    ours = BacktestEngine(spec=cell.spec, config=cell.engine_config(cfg.strategy_id, risk_pct=dna.risk_pct),
                          rng_seed=0).run(cell.bars, strategy)
    raw = load_cell_bars_from_snapshot("USDCADm", "H1", "aed3bf314539ec5a")
    cfg2, strategy2 = dna_to_strategy(dna)
    trial = run_slow_trial(raw, slow_cell_setup(raw), dna, cfg2, strategy2, notes="")
    eq = np.asarray(ours.equity_curve)
    assert ours.n_trades == trial.n_trades > 0
    assert float(eq[-1] / eq[0] - 1.0) == trial.total_return
    assert {"spread": ours.cost_spread, "commission": ours.cost_commission, "slippage": ours.cost_slippage,
            "swap": ours.cost_swap} == trial.costs
    assert dict(ours.exit_reasons) == trial.exit_reasons


def test_the_finalists_run_on_both_campaign_cells(pinned_home: Path, real_artifacts_dir: Path) -> None:
    import shutil

    src = real_artifacts_dir / "ga" / ga_subjects.GA_CAMPAIGN / "manifest_v2.json"
    dest = pinned_home / "artifacts" / "ga" / ga_subjects.GA_CAMPAIGN / "manifest_v2.json"
    dest.parent.mkdir(parents=True)
    shutil.copyfile(src, dest)
    fin = ga_subjects.load_finalists()[0]
    assert fin.datasets == {"usdcadm_h1": campaign_cell("USDCADm", "aed3bf314539ec5a").dataset_id,
                            "xauusdm_h1": campaign_cell("XAUUSDm", "447b843f87438a9a").dataset_id}
