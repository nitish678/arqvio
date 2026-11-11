"""Gene registry (worker4 W4-1): it says only what the code proves, and both engines read it."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from engine.api.server import create_app
from engine.research.ga import operators
from engine.research.genes import registry
from engine.research.genes.registry import RESEARCH, Status
from engine.research.search.dna import SIGNAL_VALUES


def test_registry_agrees_with_the_code() -> None:
    assert registry.problems() == []


def test_research_switch_counts_match_part_4_1() -> None:
    # CLAUDE_BUILD_INSTRUCTIONS.md PART 4.1: 63 switches = 13 bias + 21 signals + 14 filters + 7 management + 8 risk.
    research = {g.group: 0 for g in registry.GENES}
    for g in registry.genes(engine="v1"):
        if g.kind == "switch":
            research[g.group] += 1
    assert {k: v for k, v in research.items() if v} == {
        "bias": 13, "signal": 21, "filter": 14, "management": 7, "risk": 8}
    assert sum(research.values()) == 63


def test_research_hidden_genes_are_the_seven() -> None:
    hidden = {g.name for g in registry.GENES if g.research_hidden}
    assert hidden == {"management.breakeven", "bias.ema", "filter.adx", "management.sl_lock", "filter.bb",
                      "exec.swing", "exec.scalein"}


def test_engine_v1_is_research_only_and_v2_is_everything() -> None:
    assert all(g.origin == RESEARCH for g in registry.genes(engine="v1"))
    assert registry.genes(engine="v2") == registry.GENES
    assert "MA_CROSS" not in registry.breedable_families("v1")
    assert registry.breedable_families("v2") == SIGNAL_VALUES


def test_every_dna_family_is_one_ready_gene() -> None:
    families = [g.family for g in registry.GENES if g.family]
    assert sorted(families) == sorted(SIGNAL_VALUES)
    assert all(registry.BY_NAME[g.name].status is Status.READY for g in registry.GENES if g.family)


def test_planned_genes_carry_no_ranges() -> None:
    for g in registry.genes(status=Status.PLANNED):
        assert "ranges" not in g.public(), g.name


def test_ga_pins_the_registry_locked_fields() -> None:
    assert operators.PINNED_GENES is registry.LOCKED_DNA_FIELDS


def test_unknown_engine_is_refused() -> None:
    with pytest.raises(ValueError, match="v3"):
        registry.genes(engine="v3")


def test_registry_endpoint_serves_the_registry(isolated_arqvio_home: Path) -> None:
    body = TestClient(create_app()).get("/api/v1/gene-pool/registry").json()
    assert body["hash"] == registry.registry_hash()
    assert body["counts"]["total"] == len(registry.GENES)
    assert body["counts"]["engines"]["v1"]["families"] == list(registry.breedable_families("v1"))
    macd = next(g for g in body["genes"] if g["name"] == "signal.macd")
    assert macd["status"] == "READY"
    assert {r["name"] for r in macd["ranges"]} == {"macd_fast", "macd_slow", "macd_signal_period"}
