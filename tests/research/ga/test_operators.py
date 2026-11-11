"""GA operators: tournament, family-rule crossover, sampler-primitive mutation."""
from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from engine.research.backtest.strategies.switch import READY_MGMT
from engine.research.ga import operators as ops
from engine.research.ga.population import Individual, checked
from engine.research.search import gene_pool as gp
from engine.research.search.dna import DNA, SIGNAL_VALUES
from engine.research.search.effective_hash import effective_gene_hash
from engine.research.search.gene_pool import FAMILY_GENES
from engine.research.search.sampler import (
    RISK_PCT_RANGE,
    SL_ATR_RANGE,
    TP_ATR_RANGE,
    random_dna,
)


def _dna(seed: int, **genes) -> DNA:
    return dataclasses.replace(random_dna(np.random.default_rng(seed)), **genes)


def _ind(dna: DNA, fitness: float) -> Individual:
    ind = Individual(dna=dna, effective_hash=effective_gene_hash(dna), generation=0)
    ind.fitness = SimpleNamespace(fitness=fitness)
    return ind


def _pool(families: dict[str, tuple[bool, float]] | None = None, genes: dict | None = None) -> gp.GenePool:
    fams = {f: {"enabled": True, "weight": 1.0} for f in SIGNAL_VALUES}
    for name, (on, w) in (families or {}).items():
        fams[name] = {"enabled": on, "weight": w}
    return gp.canonicalise({"families": fams, "genes": genes or {}})


# --- tournament -------------------------------------------------------------

def test_tournament_takes_the_fittest_of_its_draws() -> None:
    scored = [_ind(_dna(i), float(i)) for i in range(10)]
    rng = np.random.default_rng(4)
    picks = np.random.default_rng(4).integers(0, 10, size=3)
    assert ops.tournament(rng, scored, 3) is scored[int(max(picks))]


def test_tournament_ties_go_to_the_lower_dna_hash() -> None:
    tied = [_ind(_dna(i), 1.0) for i in range(6)]
    lowest = min(tied, key=lambda i: i.dna_hash)
    # k large enough that every individual is drawn at least once for this seed
    rng = np.random.default_rng(0)
    assert ops.tournament(rng, tied, 200) is lowest


def test_tournament_is_seeded() -> None:
    scored = [_ind(_dna(i), float(i % 4)) for i in range(20)]
    a = [ops.tournament(np.random.default_rng(3), scored, 3).dna_hash for _ in range(1)]
    b = [ops.tournament(np.random.default_rng(3), scored, 3).dna_hash for _ in range(1)]
    assert a == b


# --- crossover: the family rule --------------------------------------------

def _pair(fam_a: str, fam_b: str, seed: int) -> tuple[DNA, DNA]:
    return _dna(seed, signal=fam_a), _dna(seed + 1000, signal=fam_b)


@pytest.mark.parametrize(("fam_a", "fam_b"), [("RSI_REVERSION", "STOCH_CROSS"), ("MACD_CROSS", "MA_CROSS")])
def test_different_families_take_a_family_genes_wholesale(fam_a: str, fam_b: str) -> None:
    donors_seen: set[str] = set()
    for seed in range(60):
        a, b = _pair(fam_a, fam_b, seed)
        child = ops.crossover(np.random.default_rng(seed), a, b)
        checked(child)
        assert child.signal == a.signal
        for fam in SIGNAL_VALUES:  # own family AND every latent family: all from A
            for spec in FAMILY_GENES[fam]:
                assert getattr(child, spec.name) == getattr(a, spec.name)
        for name in ops.CROSSOVER_COMMON_GENES:
            value = getattr(child, name)
            assert value in (getattr(a, name), getattr(b, name))
            if getattr(a, name) != getattr(b, name):
                donors_seen.add("a" if value == getattr(a, name) else "b")
    assert donors_seen == {"a", "b"}  # common genes really are uniform


def test_same_family_mixes_family_genes_per_gene() -> None:
    mixed = False
    for seed in range(60):
        a, b = _pair("STOCH_CROSS", "STOCH_CROSS", seed)
        child = ops.crossover(np.random.default_rng(seed), a, b)
        checked(child)
        from_a = {s.name for s in FAMILY_GENES["STOCH_CROSS"] if getattr(child, s.name) == getattr(a, s.name)}
        for spec in FAMILY_GENES["STOCH_CROSS"]:
            assert getattr(child, spec.name) in (getattr(a, spec.name), getattr(b, spec.name))
        mixed |= 0 < len(from_a) < len(FAMILY_GENES["STOCH_CROSS"])
    assert mixed


def test_the_macd_pair_crosses_as_one_gene() -> None:
    for seed in range(200):
        a, b = _pair("MACD_CROSS", "MACD_CROSS", seed)
        child = ops.crossover(np.random.default_rng(seed), a, b)
        pair = (child.macd_fast, child.macd_slow)
        assert pair in ((a.macd_fast, a.macd_slow), (b.macd_fast, b.macd_slow))
        checked(child)  # the gap holds


def test_crossover_never_touches_pinned_genes() -> None:
    for seed in range(40):
        a, b = _pair("RSI_REVERSION", "RSI_REVERSION", seed)
        child = ops.crossover(np.random.default_rng(seed), a, b)
        for name in ops.PINNED_GENES:
            assert getattr(child, name) == getattr(a, name)


# --- mutation ---------------------------------------------------------------

def test_p_zero_changes_nothing() -> None:
    dna = _dna(1)
    child, changed = ops.mutate(np.random.default_rng(1), dna, gp.default_gene_pool(), p=0.0, scale=1.0)
    assert child == dna and changed == ()


def _in_range(child: DNA) -> None:
    checked(child)  # family ranges, MACD gap, categorical domains
    assert SL_ATR_RANGE[0] <= child.sl_atr <= SL_ATR_RANGE[1]
    assert TP_ATR_RANGE[0] <= child.tp_atr <= TP_ATR_RANGE[1]
    assert RISK_PCT_RANGE[0] <= child.risk_pct <= RISK_PCT_RANGE[1]
    assert 10 <= child.max_feature_lookback <= 199
    assert child.rsi_oversold < child.rsi_overbought  # threshold ordering
    assert child.stoch_oversold < child.stoch_overbought
    assert child.mgmt == "NONE" and child.exec_mode == "MARKET2" and not child.uses_future_bar


@pytest.mark.parametrize("scale", [1.0, 0.5, 0.1])
def test_every_mutant_stays_in_the_sampler_space(scale: float) -> None:
    pool = gp.default_gene_pool()
    rng = np.random.default_rng(12)
    for seed in range(400):
        child, changed = ops.mutate(rng, _dna(seed), pool, p=1.0, scale=scale)
        _in_range(child)
        allowed = {"signal", *ops.family_genes(child.signal), *ops.COMMON_ENGINE_GENES}
        assert set(changed) <= allowed


@pytest.mark.parametrize("scale", [0.2, 0.1, 0.05])
def test_a_narrow_integer_gene_still_mutates_at_a_small_scale(scale: float) -> None:
    """stoch_d_period spans 3..5: a plain rounded blend would freeze it below 0.25."""
    rng = np.random.default_rng(3)
    moved = 0
    for seed in range(300):
        dna = _dna(seed, signal="STOCH_CROSS")
        child, changed = ops.mutate(rng, dna, gp.default_gene_pool(), p=1.0, scale=scale)
        _in_range(child)
        if "stoch_d_period" in changed:
            moved += 1
            assert abs(child.stoch_d_period - dna.stoch_d_period) == 1  # one step, toward the draw
    assert moved > 0


def test_mutation_only_touches_engine_reaching_genes() -> None:
    rng = np.random.default_rng(8)
    for seed in range(200):
        dna = _dna(seed)
        child, changed = ops.mutate(rng, dna, gp.default_gene_pool(), p=0.5, scale=1.0)
        for f in dataclasses.fields(DNA):
            if f.name not in changed:
                assert getattr(child, f.name) == getattr(dna, f.name), f.name


def test_pool_fixed_genes_never_mutate_and_disabled_families_are_never_drawn() -> None:
    pool = _pool(
        {"MACD_CROSS": (False, 1.0)},
        {"RSI_REVERSION": {"rsi_period": {"fixed": 9}}, "STOCH_CROSS": {"stoch_k_period": {"fixed": 7}}},
    )
    rng = np.random.default_rng(21)
    for seed in range(400):
        dna = _dna(seed, signal="RSI_REVERSION", rsi_period=9, stoch_k_period=7)
        child, changed = ops.mutate(rng, dna, pool, p=1.0, scale=1.0)
        assert child.signal != "MACD_CROSS"
        assert child.rsi_period == 9 and child.stoch_k_period == 7
        assert "rsi_period" not in changed and "stoch_k_period" not in changed


def test_a_fixed_macd_side_leaves_the_other_inside_the_gap() -> None:
    pool = _pool(genes={"MACD_CROSS": {"macd_slow": {"fixed": 24}}})
    rng = np.random.default_rng(2)
    for seed in range(300):
        dna = _dna(seed, signal="MACD_CROSS", macd_slow=24, macd_fast=10)
        child, _ = ops.mutate(rng, dna, pool, p=1.0, scale=1.0)
        if child.signal == "MACD_CROSS":
            assert child.macd_slow == 24 and child.macd_slow - child.macd_fast >= 5
        checked(child)


def test_mutation_is_seeded() -> None:
    dna, pool = _dna(4), gp.default_gene_pool()
    a = ops.mutate(np.random.default_rng(6), dna, pool, p=0.3, scale=0.7)
    b = ops.mutate(np.random.default_rng(6), dna, pool, p=0.3, scale=0.7)
    assert a == b


def test_the_gene_draws_cover_exactly_the_sampler_ranges() -> None:
    """Drift pin: the operator's per-gene draws are the sampler's ranges (the entry gates' are the DNA's sets)."""
    from engine.research.search.dna import SESSION_VALUES, TREND_EMA_VALUES

    assert set(ops._DRAWS) == {
        s.name for specs in FAMILY_GENES.values() for s in specs
    } | set(ops.COMMON_ENGINE_GENES) | set(ops.GATE_GENES) | {"mgmt_switches", "signal_switches"}  # v4 pool-gated switches
    rng = np.random.default_rng(0)
    base = _dna(0, macd_fast=10, macd_slow=40)
    assert {ops._DRAWS["session"](rng, base) for _ in range(500)} == set(SESSION_VALUES)
    assert {ops._DRAWS["trend_ema"](rng, base) for _ in range(500)} == set(TREND_EMA_VALUES)
    for fam, specs in FAMILY_GENES.items():
        for spec in specs:
            values = [ops._DRAWS[spec.name](rng, base) for _ in range(3000)]
            if spec.kind == "bool":
                assert set(values) == {False, True}
            elif spec.kind == "int":
                legal = set(range(int(spec.lo), int(spec.hi) + 1))
                if spec.name == "macd_fast":
                    legal = {v for v in legal if 40 - v >= 5}
                elif spec.name == "macd_slow":
                    legal = {v for v in legal if v - 10 >= 5}
                assert set(values) == legal, (fam, spec.name)
            else:
                assert all(spec.lo <= v < spec.hi for v in values), (fam, spec.name)


def test_mutate_redraws_mgmt_switches_at_any_scale():
    """2026-11-01: with the mgmt_switches gate on, mutate at scale < 1 blended the switch tuple and raised
    TypeError; a switch tuple is a category, redrawn whole like the gates."""
    pool = gp.canonicalise({"families": {f: {"enabled": True, "weight": 1.0} for f in SIGNAL_VALUES},
                            "genes": {}, "gates": {"mgmt_switches": True}})
    rng = np.random.default_rng(4)
    dna = dataclasses.replace(_dna(1), mgmt_switches=("breakeven",))
    for _ in range(50):
        child, _ = ops.mutate(rng, dna, pool, p=1.0, scale=0.5)
        assert isinstance(child.mgmt_switches, tuple)
        assert set(child.mgmt_switches) <= set(READY_MGMT)
