"""ARQVIO entry gates: oracle and kernel agree on gated DNAs, the gates fire, and nothing bred before them moves."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from engine.research.backtest.gates import SESSIONS, TrendEma, gate_arrays, session_open
from engine.research.data.dataset_snapshot import find_snapshot
from engine.research.ga import operators as op
from engine.research.search import gene_pool as gp
from engine.research.search import sampler
from engine.research.search.dna import DNA, SESSION_VALUES, TREND_EMA_VALUES, validate_gene_space

DATASET_ID = "aed3bf314539ec5a"  # USDCADm H1, pinned
SLICE_BARS = 1500


def test_sessions_and_the_trend_ema_are_causal_and_agree_bar_by_bar() -> None:
    assert [session_open(h, "LONDON") for h in (6, 7, 15, 16)] == [False, True, True, False]
    assert all(session_open(h, "") for h in range(24))
    rng = np.random.default_rng(1)
    close = 1.0 + np.cumsum(rng.normal(0, 0.001, 600))
    hours = np.arange(600) % 24
    long_ok, short_ok = gate_arrays(hours, close, "NEWYORK", 50)
    ema = TrendEma(50)
    for i in range(600):
        ema.update(close[i])
        inside = SESSIONS["NEWYORK"][0] <= hours[i] < SESSIONS["NEWYORK"][1]
        assert long_ok[i] == (inside and ema.allows(1, close[i]))
        assert short_ok[i] == (inside and ema.allows(-1, close[i]))
    assert not long_ok[:49].any() and not short_ok[:49].any()  # no entry before the EMA has its period


def test_an_ungated_genome_keeps_its_hash_and_the_sampler_its_draws() -> None:
    d = sampler.random_dna(np.random.default_rng(2))
    assert "session" not in d.to_dict() and "trend_ema" not in d.to_dict() and not d.gated
    g = dataclasses.replace(d, session="ASIA", trend_ema=100)
    assert g.gated and g.dna_hash != d.dna_hash and DNA(**g.to_dict()) == g and g.ungated() == d
    pool = gp.default_gene_pool()
    assert "gates" not in pool.canonical() and pool.gates == ()
    a = [sampler.random_dna(r, gene_pool=pool) for r in [np.random.default_rng(3)] for _ in range(40)]
    b = [sampler.random_dna(r) for r in [np.random.default_rng(3)] for _ in range(40)]
    assert [x.dna_hash for x in a] == [x.dna_hash for x in b]
    assert set(op.GATE_GENES).isdisjoint(op.CROSSOVER_COMMON_GENES)
    for bad in ({"session": "TOKYO"}, {"trend_ema": 75}):
        with pytest.raises(ValueError):
            validate_gene_space({**d.to_dict(), **bad})


def test_a_gated_pool_draws_crosses_and_mutates_the_gates() -> None:
    pool = gp.merge(gp.default_gene_pool(), {"gates": {"session": True, "trend_ema": True}})
    assert pool.gates == ("session", "trend_ema") and pool.canonical()["gates"] == {"session": True, "trend_ema": True}
    assert gp.canonicalise(pool.to_dict()).hash == pool.hash
    rng = np.random.default_rng(4)
    pop = [sampler.random_dna(rng, gene_pool=pool) for _ in range(300)]
    assert {d.session for d in pop} == set(SESSION_VALUES) and {d.trend_ema for d in pop} == set(TREND_EMA_VALUES)
    kids = [op.mutate(rng, d, pool, p=1.0, scale=0.5)[0] for d in pop[:50]]
    assert any((k.session, k.trend_ema) != (d.session, d.trend_ema) for k, d in zip(kids, pop[:50], strict=True))
    child = op.crossover(rng, dataclasses.replace(pop[0], session="ASIA"), dataclasses.replace(pop[1], session="LONDON"))
    assert child.session in {"ASIA", "LONDON"}
    with pytest.raises(gp.GenePoolError):
        gp.canonicalise({**gp.default_gene_pool().to_dict(), "gates": {"moon_phase": True}})


@pytest.fixture(scope="module")
def factories():
    if find_snapshot(DATASET_ID) is None:
        pytest.skip(f"snapshot {DATASET_ID} not taken on this machine")
    from engine.research.search.real_backtest import (
        fast_backtest_factory,
        load_cell_bars_from_snapshot,
        real_backtest_factory,
        slice_cell_bars,
    )

    cell = slice_cell_bars(load_cell_bars_from_snapshot("USDCADm", "H1", DATASET_ID), SLICE_BARS)
    return real_backtest_factory(cell), fast_backtest_factory(cell)


def test_oracle_and_kernel_agree_on_gated_genomes_and_the_gates_fire(factories) -> None:
    slow, fast = factories
    rng = np.random.default_rng(20261022)
    fired = 0
    for k in range(10):
        d = dataclasses.replace(sampler.random_dna(rng), session=SESSION_VALUES[1 + k % 4], trend_ema=(50, 100)[k % 2])
        a, b = slow(d), fast(d)
        assert (a.sharpe, a.n_trades) == (b.sharpe, b.n_trades), d.dna_hash
        assert np.array_equal(a.per_period_returns, b.per_period_returns), d.dna_hash
        fired += fast(d.ungated()).n_trades != b.n_trades
    assert fast.fallback_state["fallbacks"] == 0  # every gated genome ran on the kernel, none fell back
    assert fired > 0  # the gates refused openings the ungated genome took
