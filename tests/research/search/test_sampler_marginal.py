"""Sampler-marginal saturation baseline: the sampler's own boundary probability, opt-in."""
from __future__ import annotations

import dataclasses
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from engine.research.search import degeneracy as dg
from engine.research.search import gene_pool as gp
from engine.research.search.dna import SIGNAL_VALUES
from engine.research.search.runner import TrialResult
from engine.research.search.sampler import MACD_FAST_VALUES, MACD_SLOW_VALUES, _macd_slow_choices, random_dna


def _pool(genes: dict) -> gp.GenePool:
    return gp.canonicalise({"families": {f: {"enabled": True, "weight": 1.0} for f in SIGNAL_VALUES}, "genes": genes})


def test_the_macd_marginal_is_the_samplers_not_two_over_k(isolated_arqvio_home: Path) -> None:
    exact = sum(
        Fraction(1, len(MACD_FAST_VALUES)) * Fraction(
            sum(1 for s in _macd_slow_choices(f) if s in (MACD_SLOW_VALUES[0], MACD_SLOW_VALUES[-1])),
            len(_macd_slow_choices(f)),
        )
        for f in MACD_FAST_VALUES
    )
    assert exact == Fraction(3327799, 57769740)  # c6_macd_marginal.log, S4 stage 1
    got = dg.sampler_marginal_baselines()
    se = (float(exact) * (1 - float(exact)) / dg.SAMPLER_MARGINAL_DRAWS) ** 0.5
    assert abs(got["MACD_CROSS.macd_slow"] - float(exact)) < 4 * se
    assert abs(got["MACD_CROSS.macd_slow"] - 2 / 31) > 4 * se  # distinguishable from the uniform baseline
    assert abs(got["STOCH_CROSS.stoch_d_period"] - 2 / 3) < 0.02
    assert abs(got["RSI_REVERSION.rsi_oversold"] - 2 * dg.FLOAT_EDGE_FRACTION) < 0.01
    assert "RSI_REVERSION.rsi_exit_midline" not in got  # bool: no boundary


def test_pool_fixed_genes_are_exactly_zero_or_one(isolated_arqvio_home: Path) -> None:
    got = dg.sampler_marginal_baselines(
        _pool({"DONCHIAN_BREAKOUT": {"donchian_period": {"fixed": 55}}, "STOCH_CROSS": {"stoch_d_period": {"fixed": 4}}}),
        n_draws=500,
    )
    assert got["DONCHIAN_BREAKOUT.donchian_period"] == 1.0
    assert got["STOCH_CROSS.stoch_d_period"] == 0.0


def test_a_fixed_macd_side_constrains_its_partner(isolated_arqvio_home: Path) -> None:
    got = dg.sampler_marginal_baselines(_pool({"MACD_CROSS": {"macd_slow": {"fixed": 24}}}), n_draws=6000)
    # fast is drawn from 5..19 (gap >= 5 below 24): only 5 is a range end of 5..21 -> 1/15
    assert abs(got["MACD_CROSS.macd_fast"] - 1 / 15) < 0.015
    assert got["MACD_CROSS.macd_slow"] == 0.0  # 24 is not a range end of 22..52


def test_the_baselines_are_seeded(isolated_arqvio_home: Path) -> None:
    assert dg.sampler_marginal_baselines(n_draws=300) == dg.sampler_marginal_baselines(n_draws=300)
    assert dg.sampler_marginal_baselines(n_draws=300) != dg.sampler_marginal_baselines(n_draws=300, seed=1)


def _trials() -> tuple[list[TrialResult], list[str]]:
    rng = np.random.default_rng(3)
    trials = []
    for i in range(12):
        dna = dataclasses.replace(random_dna(rng), signal="STOCH_CROSS", stoch_d_period=(3, 5)[i % 2])
        trials.append(TrialResult(dna=dna, sharpe=1.0 + i, n_trades=50))
    return trials, ["EVALUATED"] * len(trials)


def test_the_default_census_is_unchanged(isolated_arqvio_home: Path) -> None:
    trials, outcomes = _trials()
    entry = dg.boundary_saturation(trials, outcomes, min_trades=30)["per_gene"]["STOCH_CROSS.stoch_d_period"]
    assert set(entry) == {"n", "at_boundary", "share", "raw_share", "distinct_values", "value_set_size",
                          "uniform_baseline", "adjusted_margin", "small_value_set_caveat"}


def test_opted_in_the_census_carries_the_marginal_beside_each_entry(isolated_arqvio_home: Path) -> None:
    trials, outcomes = _trials()
    marginals = dg.sampler_marginal_baselines(n_draws=500)
    rep = dg.degeneracy_report(trials, outcomes, min_trades=30, trials_raw=12, trials_effective=12,
                               marginal_baselines=marginals)
    entry = rep["boundary_saturation"]["per_gene"]["STOCH_CROSS.stoch_d_period"]
    assert entry["sampler_marginal_baseline"] == marginals["STOCH_CROSS.stoch_d_period"]
    assert entry["sampler_marginal_margin"] == pytest.approx(entry["share"] - entry["sampler_marginal_baseline"])
