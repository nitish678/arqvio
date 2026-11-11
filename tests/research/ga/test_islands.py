"""The island model: split, migration, cryo, revolution, war, determinism, off by default."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from engine.research.campaign.manifest import config_hash
from engine.research.ga import islands as il
from engine.research.ga import runner as ga
from engine.research.ga.config import CAMPAIGN_GATES, GAConfigError, IslandConfig, Selection, ga_config
from engine.research.ga.determinism import canonical, individuals_table, run_hash
from engine.research.search import gene_pool as gp
from engine.research.search.run_campaign_cli import campaign_config_for
from tests.research.ga.test_runner_integration import SyntheticCell


def _cfg(**kw):
    base = dict(seed=11, gates=CAMPAIGN_GATES, gene_pool=gp.default_gene_pool(), population=40, generations=7,
                trials_budget=2000, final_n=6, selection=Selection(elitism=8), islands=IslandConfig(k=4))
    base.update(kw)
    return ga_config(**base)


def _search(cfg):
    return il.island_search(cfg, SyntheticCell())


# --- off by default, hash-compatible ------------------------------------------

def test_islands_are_off_by_default_and_absent_from_the_hash(isolated_arqvio_home: Path) -> None:
    plain = ga_config(seed=7, gates=CAMPAIGN_GATES, gene_pool=gp.default_gene_pool())
    assert plain.islands is None and "islands" not in plain.block()
    assert ga.searcher_for(plain) is ga.search
    assert ga.searcher_for(_cfg()) is il.island_search
    assert _cfg().block()["islands"]["k"] == 4


def test_the_stage_2_config_hashes_are_reproduced(isolated_arqvio_home: Path) -> None:
    """cbcb2ae061e70bef (search) and c8a71c501836b932 (cross), from the full run's inputs."""
    cfg = ga_config(seed=7, gates=CAMPAIGN_GATES, gene_pool=gp.default_gene_pool(), population=200,
                    generations=25, trials_budget=8000, final_n=20)
    search = campaign_config_for(symbol="USDCADm", timeframe="H1", population=200, target_sharpe=1.5, seed=7,
                                 dataset_id="aed3bf314539ec5a", gene_pool=cfg.gene_pool, min_trades=30, ga=cfg.block())
    assert config_hash(search) == "cbcb2ae061e70bef"
    cross = campaign_config_for(
        symbol="XAUUSDm", timeframe="H1", population=20, target_sharpe=1.5, seed=7, dataset_id="447b843f87438a9a",
        gene_pool=cfg.gene_pool, min_trades=30,
        ga={**cfg.block(), "role": "cross_check", "search_campaign_id": "ga-usdcadm_h1-s7-cbcb2ae061e70bef",
            "search_config_hash": "cbcb2ae061e70bef"})
    assert config_hash(cross) == "c8a71c501836b932"


@pytest.mark.parametrize(("isl", "needle"), [
    (IslandConfig(k=1), "islands.k must be an int >= 2"),
    (IslandConfig(k=30), "needs a population of at least 60"),
    (IslandConfig(k=4, war_every=0), "islands.war_every must be an int >= 1"),
    (IslandConfig(k=4, revolution_fraction=0.0), "revolution_fraction must be in (0, 1]"),
    (IslandConfig(k=4, migrants=-1), "islands.migrants must be an int >= 0"),
])
def test_invalid_island_configs_are_refused(isl: IslandConfig, needle: str) -> None:
    with pytest.raises(GAConfigError) as err:
        _cfg(islands=isl)
    assert any(needle in p for p in err.value.problems), err.value.problems


# --- the pure pieces -----------------------------------------------------------

def _scored(n: int, seed: int = 0):
    cfg = ga_config(seed=seed, gates=CAMPAIGN_GATES, gene_pool=gp.default_gene_pool(), population=n,
                    generations=1, trials_budget=n, final_n=1, selection=Selection(elitism=1))
    return ga.search(cfg, SyntheticCell()).archive


def test_split_keeps_admission_order_and_balances_sizes() -> None:
    pop = _scored(10)
    parts = il.split(pop, 4)
    assert [len(p) for p in parts] == [3, 3, 2, 2] and [x for p in parts for x in p] == pop


def test_weakest_is_the_lowest_mean_and_ties_go_low() -> None:
    pop = _scored(8)
    parts = [pop[:4], pop[4:]]
    means = [np.mean([x.fitness.fitness for x in p]) for p in parts]
    assert il.weakest(parts) == int(np.argmin(means))
    assert il.weakest([pop[:4], pop[:4]]) == 0


def test_cryo_keeps_only_the_fittest_survivors() -> None:
    pop = _scored(40)
    cryo = il.update_cryo([], pop, 5)
    alive = sorted((x for x in pop if x.fitness.alive), key=ga.rank_key)
    assert cryo == alive[:5] and all(x.fitness.alive for x in cryo)


def test_migration_moves_the_sources_picks_into_its_ring_successor() -> None:
    pop = _scored(24)
    parts = il.split(pop, 3)
    out, moved = il.migrate(np.random.default_rng(5), parts, IslandConfig(k=3, migrants=2), 3)
    assert moved > 0 and [len(p) for p in out] == [len(p) for p in parts]
    for i in range(3):
        src = {x.effective_hash for x in parts[(i - 1) % 3]}
        arrived = {x.effective_hash for x in out[i]} - {x.effective_hash for x in parts[i]}
        assert arrived <= src


# --- the mechanisms fire on their triggers --------------------------------------

def test_migration_runs_on_its_schedule(isolated_arqvio_home: Path) -> None:
    logs = _search(_cfg(islands=IslandConfig(k=4, migration_every=2, revolution_after=99, war_every=99,
                                             defrost_after=99))).generations
    assert [g["islands"]["migrated"] > 0 for g in logs] == [g["generation"] > 0 and g["generation"] % 2 == 0
                                                             for g in logs]


def test_revolution_reseeds_a_stagnant_island_with_fresh_draws(isolated_arqvio_home: Path) -> None:
    res = _search(_cfg(islands=IslandConfig(k=4, revolution_after=1, war_every=99, defrost_after=99)))
    events = [r for g in res.generations for r in g["islands"]["revolutions"]]
    assert events
    fresh = [x for x in res.archive if x.operators == ("revolution",)]
    assert len(fresh) == sum(r["reseeded"] for r in events) and all(x.parents == () for x in fresh)


def test_war_culls_the_weakest_island_on_its_schedule(isolated_arqvio_home: Path) -> None:
    res = _search(_cfg(islands=IslandConfig(k=4, war_every=3, revolution_after=99, defrost_after=99)))
    wars = {g["generation"]: g["islands"]["war"] for g in res.generations}
    assert [g for g, w in wars.items() if w is not None] == [3, 6]
    assert all(w["reseeded"] > 0 for w in wars.values() if w is not None)
    assert any(x.operators == ("war",) for x in res.archive)


def test_defrost_injects_cryo_copies_when_the_search_stagnates(isolated_arqvio_home: Path) -> None:
    res = _search(_cfg(islands=IslandConfig(k=4, defrost_after=1, revolution_after=99, war_every=99)))
    assert any(g["islands"]["defrosted"] > 0 for g in res.generations)


def test_every_new_trial_counts_against_the_budget(isolated_arqvio_home: Path) -> None:
    res = _search(_cfg(trials_budget=150, islands=IslandConfig(k=4, revolution_after=1, war_every=2)))
    assert res.trials_used <= 150 and res.trials_used == len(res.archive)
    assert len({x.effective_hash for x in res.archive}) == len(res.archive)  # one registry for all islands


def test_island_runs_are_seeded(isolated_arqvio_home: Path) -> None:
    a = individuals_table(_search(_cfg()).archive)
    b = individuals_table(_search(_cfg()).archive)
    assert canonical(a) == canonical(b) and canonical(_search(_cfg()).generations) == canonical(
        _search(_cfg()).generations)
    assert run_hash(individuals_table(_search(_cfg(seed=12)).archive)) != run_hash(a)


def test_per_island_stats_are_logged(isolated_arqvio_home: Path) -> None:
    for g in _search(_cfg()).generations:
        isl = g["islands"]
        assert len(isl["per_island"]) == 4 and sum(x["size"] for x in isl["per_island"]) == 40
        assert {"best", "mean", "alive", "stagnation"} <= set(isl["per_island"][0])
        assert {"cryo_size", "global_stagnation", "migrated", "revolutions", "war", "defrosted"} <= set(isl)
