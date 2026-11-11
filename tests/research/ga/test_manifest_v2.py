"""Manifest v2: every required field per candidate, checked before anything is written."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from engine.research.ga import manifest as mv2

CELLS = ("usdcadm_h1", "xauusdm_h1")


def _fitness(killed_by: str | None) -> dict:
    return {"fitness": 0.0 if killed_by else 2.7, "killed_by": killed_by, "n_trades": 50, "sharpe": 2.0}


def _entry() -> dict:
    per_cell = lambda v: {c: v for c in CELLS}  # noqa: E731
    return {
        "dna": {"signal": "RSI_REVERSION"},
        "dna_hash": "a" * 16,
        "effective_gene_hash": "b" * 16,
        "lineage": {"generation_found": 3, "parents": ["c" * 16, "d" * 16], "operators": ["crossover"]},
        "metrics": per_cell({"fitness": _fitness(None), "n_trades": 50}),
        "sham_margin": per_cell(0.8),
        "dsr": {"primary_at_eval": {"dsr": 0.7}, "primary_final": {"dsr": 0.6}, "cross": {"dsr": 0.4}},
        "cross_cell_delta": {"sharpe": -0.5},
        "degeneracy_census": {"trials_per_family": {"RSI_REVERSION": 3}},
        "controls_id": {"kind": "sham_random_entry", "salt": "astra-s3-sham-v1", "regime": "v2",
                        "engine": "slow", "seed": "sha256(dna_hash:salt)"},
        "audit": per_cell({"as_of": "2026-09-24T18:39:39", "time_basis": "utc", "source": "MT5_LIVE"}),
        "spread_sha256": per_cell("e" * 64),
        "schema_versions": {"dna": 3, "effective_hash": 4, "spread_calibrator": 2, "ga_manifest": 2},
        "config_hash": "f" * 16,
        "run_hash": "0" * 64,
        "dataset_id": {"usdcadm_h1": "aed3bf314539ec5a", "xauusdm_h1": "447b843f87438a9a"},
        "dataset_provenance": per_cell("pinned_cell"),
    }


def test_the_required_fields_are_the_spec_list() -> None:
    """Pinned as a literal, so dropping a field cannot also drop its test."""
    assert set(mv2.REQUIRED_CANDIDATE_FIELDS) == {
        "dna", "dna_hash", "effective_gene_hash", "lineage", "metrics", "sham_margin", "dsr",
        "cross_cell_delta", "degeneracy_census", "controls_id", "audit", "spread_sha256",
        "schema_versions", "config_hash", "run_hash", "dataset_id", "dataset_provenance",
    }
    assert set(mv2.PER_CELL_FIELDS) == {
        "metrics", "sham_margin", "audit", "spread_sha256", "dataset_id", "dataset_provenance",
    }


def test_a_complete_entry_has_nothing_missing() -> None:
    assert mv2.missing_candidate_fields(_entry(), CELLS) == []


@pytest.mark.parametrize("name", mv2.REQUIRED_CANDIDATE_FIELDS)
def test_each_required_field_is_required(name: str) -> None:
    entry = _entry()
    del entry[name]
    assert name in mv2.missing_candidate_fields(entry, CELLS)


@pytest.mark.parametrize("name", mv2.PER_CELL_FIELDS)
def test_each_per_cell_field_must_name_both_cells(name: str) -> None:
    entry = _entry()
    del entry[name]["xauusdm_h1"]
    assert f"{name}.xauusdm_h1" in mv2.missing_candidate_fields(entry, CELLS)


def test_a_sham_margin_may_be_absent_only_where_the_gates_killed() -> None:
    entry = _entry()
    entry["sham_margin"]["xauusdm_h1"] = None
    assert "sham_margin.xauusdm_h1" in mv2.missing_candidate_fields(entry, CELLS)
    entry["metrics"]["xauusdm_h1"]["fitness"] = _fitness("min_trades")
    assert mv2.missing_candidate_fields(entry, CELLS) == []


@pytest.mark.parametrize("value", ["", "MT5_LIVE", "pinned", None])
def test_dataset_provenance_must_be_a_legal_value(value) -> None:
    entry = _entry()
    entry["dataset_provenance"]["usdcadm_h1"] = value
    assert "dataset_provenance.usdcadm_h1" in mv2.missing_candidate_fields(entry, CELLS)


@pytest.mark.parametrize(("path", "needle"), [
    (("lineage", "operators"), "lineage.operators"),
    (("controls_id", "salt"), "controls_id.salt"),
    (("dsr", "primary_final"), "dsr.primary_final"),
    (("audit", "usdcadm_h1", "time_basis"), "audit.usdcadm_h1.time_basis"),
])
def test_nested_requirements(path: tuple[str, ...], needle: str) -> None:
    entry = _entry()
    node = entry
    for key in path[:-1]:
        node = node[key]
    del node[path[-1]]
    assert needle in mv2.missing_candidate_fields(entry, CELLS)


def test_build_manifest_refuses_an_incomplete_candidate() -> None:
    good, bad = _entry(), copy.deepcopy(_entry())
    del bad["run_hash"]
    bad["dna_hash"] = "9" * 16
    with pytest.raises(ValueError, match="9999999999999999"):
        mv2.build_manifest(run={}, candidates=[good, bad], cells=CELLS)
    doc = mv2.build_manifest(run={"config_hash": "f" * 16}, candidates=[good], cells=CELLS)
    assert (doc["schema"], doc["version"]) == ("arqvio.ga.manifest", 2)
    assert doc["cells"] == {"primary": "usdcadm_h1", "cross": "xauusdm_h1"}


def test_write_json_is_real_json_and_keeps_timestamps(tmp_path: Path) -> None:
    out = mv2.write_json(tmp_path / "a" / "m.json",
                         {"x": float("nan"), "completed_at_utc": "2026-09-30T00:00:00Z", "t": (1, 2)})
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data == {"x": "nan", "completed_at_utc": "2026-09-30T00:00:00Z", "t": [1, 2]}
