"""GA population: seeded sampler init honouring the pool, dedupe by effective hash."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from engine.research.ga import population as pop
from engine.research.search import gene_pool as gp
from engine.research.search.dna import SIGNAL_VALUES
from engine.research.search.effective_hash import effective_gene_hash
from engine.research.search.sampler import random_dna


def _pool(families: dict[str, tuple[bool, float]] | None = None, genes: dict | None = None) -> gp.GenePool:
    fams = {f: {"enabled": True, "weight": 1.0} for f in SIGNAL_VALUES}
    for name, (on, w) in (families or {}).items():
        fams[name] = {"enabled": on, "weight": w}
    return gp.canonicalise({"families": fams, "genes": genes or {}})


def test_generation_zero_is_the_sampler_stream(isolated_arqvio_home: Path) -> None:
    """With no collision, gen 0 is exactly random search's first N draws."""
    pool = gp.default_gene_pool()
    reg = pop.Registry()
    got = pop.initial_population(np.random.default_rng(7), pool, 60, reg)
    rng = np.random.default_rng(7)
    expected = [random_dna(rng, gene_pool=pool) for _ in range(60)]
    assert reg.collisions == 0
    assert [i.dna for i in got] == expected
    assert all(i.generation == 0 and i.operators == ("init",) and i.parents == () for i in got)


def test_every_dna_is_unique_by_effective_hash(isolated_arqvio_home: Path) -> None:
    reg = pop.Registry()
    got = pop.initial_population(np.random.default_rng(3), gp.default_gene_pool(), 200, reg)
    hashes = [i.effective_hash for i in got]
    assert len(set(hashes)) == 200
    assert set(hashes) == reg.seen
    assert all(h == effective_gene_hash(i.dna) for h, i in zip(hashes, got, strict=True))


def test_the_pool_holds_disabled_families_weights_and_fixed_genes(isolated_arqvio_home: Path) -> None:
    pool = _pool(
        {"MACD_CROSS": (False, 1.0), "MA_CROSS": (True, 3.0)},
        {"STOCH_CROSS": {"stoch_d_period": {"fixed": 4}}, "DONCHIAN_BREAKOUT": {"donchian_period": {"fixed": 21}}},
    )
    got = pop.initial_population(np.random.default_rng(11), pool, 400, pop.Registry())
    signals = [i.dna.signal for i in got]
    assert "MACD_CROSS" not in signals
    assert signals.count("MA_CROSS") > signals.count("RSI_REVERSION")  # weight 3/7 vs 1/7
    assert {i.dna.stoch_d_period for i in got} == {4}
    assert {i.dna.donchian_period for i in got} == {21}


def test_a_collision_takes_the_next_sampler_draw_and_is_counted(isolated_arqvio_home: Path) -> None:
    pool = gp.default_gene_pool()
    rng = np.random.default_rng(5)
    first, second = random_dna(rng, gene_pool=pool), random_dna(rng, gene_pool=pool)
    reg = pop.Registry()
    assert reg.admit(effective_gene_hash(first))  # the run already evaluated `first`
    got = pop.initial_population(np.random.default_rng(5), pool, 1, reg)
    assert reg.collisions == 1
    assert got[0].dna == second


def test_an_exhausted_space_is_refused_not_looped(
    isolated_arqvio_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    same = random_dna(np.random.default_rng(1))
    monkeypatch.setattr(pop, "random_dna", lambda rng, gene_pool=None: same)
    monkeypatch.setattr(pop, "MAX_RESAMPLES_PER_SLOT", 5)
    reg = pop.Registry()
    with pytest.raises(pop.DedupeExhausted):
        pop.initial_population(np.random.default_rng(1), gp.default_gene_pool(), 2, reg)
    assert reg.collisions == 5


def test_an_invalid_dna_is_refused_not_repaired() -> None:
    dna = random_dna(np.random.default_rng(2))
    with pytest.raises(ValueError, match="macd gap"):
        pop.checked(dataclasses.replace(dna, macd_fast=21, macd_slow=22))
    with pytest.raises(ValueError, match="invalid signal"):
        pop.checked(dataclasses.replace(dna, signal="BOGUS"))


def test_the_same_seed_draws_the_same_population(isolated_arqvio_home: Path) -> None:
    pool = gp.default_gene_pool()
    a = pop.initial_population(np.random.default_rng(9), pool, 50, pop.Registry())
    b = pop.initial_population(np.random.default_rng(9), pool, 50, pop.Registry())
    assert [i.dna_hash for i in a] == [i.dna_hash for i in b]
