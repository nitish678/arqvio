"""The GA config block: validated, plain JSON, linked to the gene pool, in the hash."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from engine.research.campaign.manifest import config_hash
from engine.research.ga.config import (
    CAMPAIGN_GATES,
    PINNED_CELLS,
    GAConfigError,
    Gates,
    Selection,
    cross_cell_for,
    ga_config,
)
from engine.research.search import gene_pool as gp
from engine.research.search.dna import SIGNAL_VALUES
from engine.research.search.run_campaign_cli import campaign_config_for


def _cfg(**kw):
    base = dict(seed=7, gates=CAMPAIGN_GATES, gene_pool=gp.default_gene_pool())
    base.update(kw)
    return ga_config(**base)


def _campaign(ga: dict | None, pool: gp.GenePool | None = None) -> dict:
    return campaign_config_for(
        symbol="USDCADm", timeframe="H1", population=200, target_sharpe=1.5, seed=7,
        dataset_id="aed3bf314539ec5a", gene_pool=pool or gp.default_gene_pool(), ga=ga,
    )


def test_the_spec_defaults(isolated_arqvio_home: Path) -> None:
    block = _cfg().block()
    assert (block["population"], block["generations"], block["trials_budget"]) == (200, 25, 8000)
    assert block["kernel_share_min"] == 0.95 and block["final_n"] == 20 and block["seed"] == 7
    assert block["selection"] == {
        "tournament_k": 3, "elitism": 10, "crossover_p": 0.7, "mutation_p": 0.1, "mutation_scale": 1.0,
    }
    assert block["gates"] == {"min_trades": 30, "sharpe_min": 0.0, "dsr_min": 0.5, "sham_margin_min": 0.0}


def test_the_block_is_plain_json(isolated_arqvio_home: Path) -> None:
    block = _cfg().block()
    assert json.loads(json.dumps(block)) == block


def test_families_come_from_the_pool_and_are_linked_by_hash(isolated_arqvio_home: Path) -> None:
    pool = gp.canonicalise({"families": {
        f: {"enabled": f != "MACD_CROSS", "weight": 1.0} for f in SIGNAL_VALUES
    }})
    block = _cfg(gene_pool=pool).block()
    assert block["families"] == pool.families
    assert block["families"]["MACD_CROSS"] == {"enabled": False, "weight": 0.0}
    assert block["gene_pool_hash"] == pool.hash


@pytest.mark.parametrize(
    ("kw", "needle"),
    [
        ({"population": 0}, "population must be an int > 0"),
        ({"generations": -1}, "generations must be an int > 0"),
        ({"trials_budget": 0}, "trials_budget must be an int > 0"),
        ({"final_n": 0}, "final_n must be an int > 0"),
        ({"seed": -3}, "seed must be an int >= 0"),
        ({"seed": 1.5}, "seed must be an int >= 0"),
        ({"kernel_share_min": 1.01}, "kernel_share_min must be in [0, 1]"),
        ({"kernel_share_min": -0.1}, "kernel_share_min must be in [0, 1]"),
        ({"trials_budget": 100}, "cannot evaluate one population of 200"),
        ({"selection": Selection(elitism=200)}, "must leave room for offspring"),
        ({"selection": Selection(tournament_k=0)}, "tournament_k must be an int >= 1"),
        ({"selection": Selection(crossover_p=1.5)}, "crossover_p must be in [0, 1]"),
        ({"selection": Selection(mutation_p=-0.1)}, "mutation_p must be in [0, 1]"),
        ({"selection": Selection(mutation_scale=0.0)}, "mutation_scale must be in (0, 1]"),
        ({"gates": Gates(min_trades=-1, sharpe_min=0.0, dsr_min=0.5, sham_margin_min=0.0)},
         "gates.min_trades must be an int >= 0"),
        ({"gates": Gates(min_trades=30, sharpe_min=float("nan"), dsr_min=0.5, sham_margin_min=0.0)},
         "gates.sharpe_min must be a finite number"),
        ({"gene_pool": {"families": {}}}, "gene_pool must be a canonical GenePool"),
    ],
)
def test_invalid_configs_are_refused_with_the_reason(kw: dict, needle: str) -> None:
    with pytest.raises(GAConfigError) as err:
        _cfg(**kw)
    assert any(needle in p for p in err.value.problems), err.value.problems


def test_the_config_hash_includes_the_ga_block(isolated_arqvio_home: Path) -> None:
    cfg = _cfg()
    without = _campaign(None)
    assert "ga" not in without  # a random search keeps the hash it always had
    with_ga = _campaign(cfg.block())
    assert with_ga["ga"] == cfg.block()
    assert config_hash(with_ga) != config_hash(without)
    assert config_hash(_campaign(cfg.block())) == config_hash(with_ga)  # deterministic


@pytest.mark.parametrize(
    "change",
    [
        {"seed": 8},
        {"population": 201},
        {"generations": 26},
        {"trials_budget": 8001},
        {"kernel_share_min": 0.9},
        {"final_n": 19},
        {"selection": Selection(mutation_p=0.2)},
        {"selection": Selection(tournament_k=4)},
        {"gates": dataclasses.replace(CAMPAIGN_GATES, dsr_min=0.6)},
        {"gates": dataclasses.replace(CAMPAIGN_GATES, sham_margin_min=0.1)},
    ],
)
def test_every_ga_knob_moves_the_hash(isolated_arqvio_home: Path, change: dict) -> None:
    assert config_hash(_campaign(_cfg(**change).block())) != config_hash(_campaign(_cfg().block()))


def test_a_pool_change_moves_the_ga_block_too(isolated_arqvio_home: Path) -> None:
    pool = gp.canonicalise({"families": {
        f: {"enabled": True, "weight": 2.0 if f == "MA_CROSS" else 1.0} for f in SIGNAL_VALUES
    }})
    assert _cfg(gene_pool=pool).block()["gene_pool_hash"] != _cfg().block()["gene_pool_hash"]


def test_int_and_float_spellings_of_a_knob_hash_alike(isolated_arqvio_home: Path) -> None:
    ints = _cfg(gates=Gates(min_trades=30, sharpe_min=0, dsr_min=0.5, sham_margin_min=0),
                kernel_share_min=1, selection=Selection(mutation_scale=1, crossover_p=0.7))
    assert ints.block() == _cfg(kernel_share_min=1.0).block()
    assert config_hash(_campaign(ints.block())) == config_hash(_campaign(_cfg(kernel_share_min=1.0).block()))


def test_random_search_keeps_its_min_trades_and_the_ga_states_its_own(isolated_arqvio_home: Path) -> None:
    assert _campaign(None)["min_trades"] == 30
    ga = campaign_config_for(symbol="USDCADm", timeframe="H1", population=200, target_sharpe=1.5, seed=7,
                             dataset_id="aed3bf314539ec5a", gene_pool=gp.default_gene_pool(), min_trades=50,
                             ga=_cfg().block())
    assert ga["min_trades"] == 50


def test_each_pinned_cell_is_cross_checked_on_the_other() -> None:
    assert PINNED_CELLS["usdcadm_h1"] == ("USDCADm", "H1", "aed3bf314539ec5a")
    assert PINNED_CELLS["xauusdm_h1"] == ("XAUUSDm", "H1", "447b843f87438a9a")
    assert cross_cell_for("usdcadm_h1") == "xauusdm_h1"
    assert cross_cell_for("xauusdm_h1") == "usdcadm_h1"
    with pytest.raises(GAConfigError):
        cross_cell_for("eurusdm_h1")
