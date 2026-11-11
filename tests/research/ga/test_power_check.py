"""The fills-matched power-check harness: seeding, the plant, the admitted-bar draw, the gates."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from engine.research.ga import power_check as pc
from engine.research.power.context import resolve_power_context
from engine.research.search.sampler import random_dna
from tests.research.ga.test_runner_integration import _seed_isolated_home, real_artifacts  # noqa: F401

# The usdcadm_h1 context of the S4 Stage-2 campaign, resolved from the registry and its recorded manifest values
# (DR-2026-10-08-u2 §2). It equals the retired CELL / PPY_H1 / PRIMARY_N.
CTX = resolve_power_context("usdcadm_h1", {"run": {
    "campaign_id": "ga-usdcadm_h1-s7-cbcb2ae061e70bef", "dsr_n_final": 4760,
    "config": {"symbol": "USDCADm", "timeframe": "H1", "dataset": "aed3bf314539ec5a"}}})


def test_plant_seeds_are_deterministic_and_distinct() -> None:
    assert pc.plant_seed(1.5, 3) == pc.plant_seed(1.5, 3)
    seeds = {pc.plant_seed(s, r) for s in pc.STRENGTHS for r in range(pc.REPLICATES)}
    assert len(seeds) == len(pc.STRENGTHS) * pc.REPLICATES


def test_the_plant_hits_its_target_sharpe_on_average() -> None:
    n = 59956
    for s in (0.0, 1.5, 5.0):
        got = [pc.annual_sharpe(pc.planted_series(s, r, n, 6048), 6048) for r in range(40)]
        # the SE of an annual Sharpe over n H1 bars is sqrt(6048 / n) ~ 0.32; 40 replicates -> ~0.05
        assert abs(float(np.mean(got)) - s) < 0.2, (s, np.mean(got))
    np.testing.assert_array_equal(pc.planted_series(1.0, 7, 100, 6048), pc.planted_series(1.0, 7, 100, 6048))


def test_annual_sharpe_is_the_factories_arithmetic() -> None:
    r = np.array([0.01, -0.005, 0.002, 0.0])
    assert pc.annual_sharpe(r, 6048) == float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(6048))
    assert pc.annual_sharpe(np.zeros(10), 6048) == 0.0 and pc.annual_sharpe(np.ones(1), 6048) == 0.0


def test_admitted_mask_follows_the_engines_fill_rules() -> None:
    hours = np.array([0, 1, 2, 3, 4, 5])
    times = hours * 3600
    spread = np.array([1.0, 9.0, 1.0, 1.0, 1.0, 1.0])
    mask = pc.admitted_entry_mask(times, spread, max_spread_points=5.0, no_trade_hours=(3,))
    # entry at i fills at i+1: bar 1 spread 9 > 5 refuses i=0; hour 3 is no-trade so i=2 is out; last bar never
    assert mask.tolist() == [False, True, False, True, True, False]
    assert pc.admitted_entry_mask(times, spread, max_spread_points=None, no_trade_hours=()).tolist() == [True] * 5 + [False]


def test_admitted_draws_use_only_admitted_bars_and_the_v2_seed() -> None:
    mask = np.zeros(1000, dtype=bool)
    mask[::3] = True
    bars, sides = pc.draw_admitted_entries("ab" * 8, "astra-s3-sham-v1", 10, 900, 50, mask)
    assert len(bars) == 50 and all(mask[b] and 10 <= b <= 900 for b in bars) and list(bars) == sorted(bars)
    assert (bars, sides) == pc.draw_admitted_entries("ab" * 8, "astra-s3-sham-v1", 10, 900, 50, mask)
    assert pc.draw_admitted_entries("ab" * 8, "astra-s3-sham-v1", 10, 900, 50, mask) != \
        pc.draw_admitted_entries("cd" * 8, "astra-s3-sham-v1", 10, 900, 50, mask)
    capped, _ = pc.draw_admitted_entries("ab" * 8, "astra-s3-sham-v1", 0, 8, 50, mask)
    assert capped == (0, 3, 6)  # every admitted bar, no more


def test_judge_applies_the_campaigns_own_gates() -> None:
    prior = list(np.random.default_rng(0).normal(0.0, 0.8, size=50))
    strong = pc.planted_series(30.0, 0, 59956, CTX.ppy)
    rec = pc.judge(strong, 100, -1.0, n_trials=20, prior=prior, ctx=CTX)
    assert rec["detected"] and rec["dsr_ok"] and rec["n_trials"] == 20
    assert rec["sham_margin"] == pytest.approx(rec["sharpe"] + 1.0)
    flat = pc.planted_series(0.0, 0, 59956, CTX.ppy)
    assert pc.judge(flat, 100, -1.0, n_trials=20, prior=prior, ctx=CTX)["detected"] is False  # zero mean: no DSR
    low_trades = pc.judge(strong, 29, -1.0, n_trials=20, prior=prior, ctx=CTX)
    assert low_trades["trades_ok"] is False and low_trades["detected"] is False
    beaten = pc.judge(strong, 100, 1e9, n_trials=20, prior=prior, ctx=CTX)
    assert beaten["sham_ok"] is False and beaten["detected"] is False


def test_the_verdict_reads_the_primary_luck_bar() -> None:
    table = []
    for s in pc.STRENGTHS:
        for g in pc.REGIMES:
            for n in pc.luck_bars(CTX):
                # Below the primary bar every plant is "detected": only N = 4760 tells the truth.
                rate = 0.0 if s == 0 else (1.0 if s >= 3.0 or n != CTX.primary_n else 0.5)
                table.append({"strength": s, "regime": g, "N": n, "detected": rate})
    v = pc.verdict(table, CTX)
    assert v["smallest_powered_strength"] == 3.0 and v["powered_at_1_5"] is False and v["controlled"] is True
    assert v["rate_1_5"] == {"v2": 0.5, "matched": 0.5}


def test_the_luck_bars_are_s4s_for_the_s4_campaign_and_sorted_for_any_n() -> None:
    assert pc.luck_bars(CTX) == (20, 200, 1000, 4760)  # the retired LUCK_BARS, exactly
    assert pc.luck_bars(dataclasses.replace(CTX, primary_n=500)) == (20, 200, 500, 1000)
    assert pc.luck_bars(dataclasses.replace(CTX, primary_n=200)) == (20, 200, 1000)


def test_hosts_are_the_first_survivors_in_admission_order() -> None:
    dna = random_dna(np.random.default_rng(1)).to_dict()
    rows = [
        {"dna": dna, "dna_hash": f"h{i}", "metrics": {"n_trades": 40 + i, "blocked_max_spread": 2, "sharpe": 0.1 * i},
         "fitness": {"killed_by": None if i % 2 else "sharpe_min", "sham_sharpe": -0.5}}
        for i in range(8)
    ]
    hosts = pc.load_hosts(rows, 3)
    assert [h["dna_hash"] for h in hosts] == ["h1", "h3", "h5"] and hosts[0]["attempted"] == 43
    with pytest.raises(ValueError, match="hosts pre-registered"):
        pc.load_hosts(rows, 5)
    assert pc.archive_sharpes(rows) == pytest.approx([0.1 * i for i in range(8)])


def test_regime_a_reproduces_a_recorded_ga_sham_on_the_pinned_cell(
    real_artifacts: Path, isolated_arqvio_home: Path,  # noqa: F811
) -> None:
    """The sanity check, on one host: the v2 sham IS the GA's sham, bit for bit."""
    smoke = Path(__file__).resolve().parents[3] / "artifacts" / "reports" / "s4_stage2" / "smoke_final_run1_out"
    if not (smoke / "individuals.json").is_file():
        pytest.skip("Stage-2 smoke outputs not on this machine")
    _seed_isolated_home(real_artifacts, isolated_arqvio_home)
    from engine.research.search.real_backtest import load_cell_bars_from_snapshot
    from engine.research.search.sham_control import SHAM_SALT, sham_backtest_factory

    host = pc.load_hosts(json.loads((smoke / "individuals.json").read_text(encoding="utf-8")), 1)[0]
    cell = load_cell_bars_from_snapshot(*CTX.cell)
    v2 = sham_backtest_factory(cell)(host["dna"], host["attempted"])
    assert float(v2.sharpe) == host["recorded_sham_sharpe"]
    matched = pc.matched_sham_factory(cell, salt=SHAM_SALT)(host["dna"], host["attempted"])
    assert int(matched.n_trades) >= int(v2.n_trades)  # the reference fills at least as many
