"""The cell registry: the real file holds the two migrated pinned cells exactly and the U4 wizard's appended cell, and
every SL4 tripwire refuses."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import re
from pathlib import Path

import pytest

from engine.core import paths
from engine.core.paths import get_artifacts_dir
from engine.research.registry import cells
from engine.research.registry.cells import (
    AmbiguousCellError,
    PinnedCellsView,
    RegistryIntegrityError,
    RegistryMissing,
    RegistrySchemaError,
    RegistrySchemaVersionError,
    UnknownCellError,
    annualisation,
    chain_hash,
    cross_of,
    entry_hash,
    get_cell,
    list_cells,
    load,
    seal,
    validated_cells,
)

# The table U1 replaced (engine/research/ga/config.py at 3d20de8). SL5: these do not move.
OLD_PINNED = {
    "usdcadm_h1": ("USDCADm", "H1", "aed3bf314539ec5a"),
    "xauusdm_h1": ("XAUUSDm", "H1", "447b843f87438a9a"),
}
# U4 W4 appended the wizard's cell (DR-2026-10-09-u4-wizard §4; the v2 package supersedes U1-SL3 for U4).
REGISTERED = {**OLD_PINNED, "gbpjpym_h1": ("GBPJPYm", "H1", "5979a019aa6b3e0c")}


def _resealed(doc, mutate):
    """A copy of ``doc`` with ``mutate`` applied to its entries and everything re-sealed (hashes valid)."""
    entries = copy.deepcopy(doc["entries"])
    mutate(entries)
    return seal(entries)


# ---- the real registry -------------------------------------------------------------------------------------------

def test_the_registry_holds_the_two_pinned_cells_and_the_wizards_cell() -> None:
    entries = list_cells()
    assert [e["cell_id"] for e in entries] == ["usdcadm_h1", "xauusdm_h1", "gbpjpym_h1"]
    assert validated_cells() == ["gbpjpym_h1", "usdcadm_h1", "xauusdm_h1"]
    assert dict(PinnedCellsView()) == REGISTERED
    assert cross_of("usdcadm_h1") == "xauusdm_h1"
    assert cross_of("xauusdm_h1") == "usdcadm_h1"
    assert cross_of("gbpjpym_h1") == "usdcadm_h1"  # #3-Q1
    assert annualisation("usdcadm_h1") == annualisation("xauusdm_h1") == annualisation("gbpjpym_h1") == 6048


def test_the_registry_lives_in_configs_and_resolves_at_call_time(monkeypatch, tmp_path, registry_doc) -> None:
    assert cells.registry_path() == paths.PROJECT_ROOT / "configs" / "cells_registry.json"
    other = tmp_path / "configs" / "cells_registry.json"
    other.parent.mkdir()
    other.write_text(json.dumps(_resealed(registry_doc, lambda es: es.reverse())), encoding="utf-8")
    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path)
    assert [e["cell_id"] for e in list_cells()] == ["gbpjpym_h1", "xauusdm_h1", "usdcadm_h1"]


def test_the_chain_is_the_fold_of_the_entry_hashes(registry_doc) -> None:
    hashes = [e["entry_hash"] for e in registry_doc["entries"]]
    assert hashes == [entry_hash(e) for e in registry_doc["entries"]]
    h = "0" * 64
    for eh in hashes:
        h = hashlib.sha256((h + eh).encode("ascii")).hexdigest()
    assert registry_doc["chain_hash"] == h == chain_hash(hashes)


def test_the_migrated_provenance_carries_no_invented_field(registry_doc) -> None:
    for e in registry_doc["entries"][:2]:  # the U1-migrated cells; the U4 entry has its own test below
        assert e["entered_by_stage"] == "U1"
        # The U3 worker_status ceremony (DR-2026-10-08-u3-worker §5) replaced U1's UNKNOWN with the proof's status.
        assert re.fullmatch(r"PROVEN@[0-9a-f]{7}", e["worker_status"]["status"]) and e["worker_status"]["fills_in"] == "U3"
        assert "UNKNOWN" in e["specs_artifact_hash"]["source"]
        assert "docs/decisions/DR-2026-10-07-u1-cell-registry.md" in e["evidence"]
    usd, xau = registry_doc["entries"][:2]
    assert usd["power_check"]["status"] == "COMPUTED"
    assert usd["power_check"]["ref"].startswith("artifacts/reports/s4_stage3_report.md:276 ")
    # The U2 power ceremony (DR-2026-10-08-u2 §6) replaced U1's DEFERRED with the computed curve's ref.
    assert xau["power_check"]["status"] == "COMPUTED"
    assert xau["power_check"]["ref"].startswith("artifacts/reports/u2/xau_power_run.log:")
    # The U4 v2 migration (DR-2026-10-09-u4-wizard §3) added campaign_status: each cell's registry-cited campaign.
    assert registry_doc["schema_version"] == usd["schema_version"] == xau["schema_version"] == 2
    assert usd["campaign_status"] == "RUN@ga-usdcadm_h1-s7-cbcb2ae061e70bef"
    assert xau["campaign_status"] == "RUN@ga-xauusdm_h1-s7-f32e0d4defd9cf07"
    for e in (usd, xau):
        assert f"<ARQVIO_HOME>/artifacts/ga/{e['campaign_status'][4:]}/manifest_v2.json" in \
            e["spread_artifact_hash"]["source"]


def test_the_wizards_entry_rests_where_the_ceremony_left_it(registry_doc) -> None:
    """U4's states through the ceremony (DR-2026-10-09-u4-wizard §5): after W6S, BLOCKED:no-campaign /
    PROVEN-PRECAMPAIGN@<sha> / UNRUN -- "ready for first search"."""
    e = registry_doc["entries"][2]
    assert (e["cell_id"], e["entered_by_stage"], e["schema_version"], e["status"]) == \
        ("gbpjpym_h1", "U4", 2, "VALIDATED")
    assert e["dataset_window"] == {"start": "2017-03-01T00:00:00+00:00", "end": "2026-09-01T00:00:00+00:00",
                                   "bars": 59034}
    assert e["campaign_status"] == "UNRUN"
    assert e["power_check"] == {"status": "BLOCKED:no-campaign", "ref": "docs/cells/gbpjpym_h1.md#power"}
    assert re.fullmatch(r"PROVEN-PRECAMPAIGN@[0-9a-f]{7}", e["worker_status"]["status"])
    assert e["worker_status"]["fills_in"] == "U4"
    assert e["worker_status"]["note"].startswith(  # the #3-Q3 citation, built from the proof's machine results
        "battery: global tier R4(a) == banked S6 (704/1096/2967/1994) verified @")
    assert ", era S6-bank; audit_row: csv:38 recorded-unchanged; proven " in e["worker_status"]["note"]
    assert "manifest_v2.json" not in e["spread_artifact_hash"]["source"]  # a never-searched cell cites no campaign
    assert e["evidence"] == ["docs/decisions/DR-2026-10-09-u4-wizard.md",
                             *(f"<ARQVIO_HOME>/artifacts/cells/gbpjpym_h1/{n}"
                               for n in ("w1_audit.json", "w2_manifest.json", "w3_pin.json")),
                             "configs/cells/gbpjpym_h1.accepted_holes.json",
                             "<ARQVIO_HOME>/artifacts/cells/gbpjpym_h1/w5_power.json",
                             "artifacts/reports/u4/worker/gbpjpym_h1.proof.json",
                             "docs/cells/gbpjpym_h1.md"]  # W7S: the runbook


def test_the_provenance_matches_the_pinned_sources(pinned_sources: Path) -> None:
    art = get_artifacts_dir()
    audit = (art / "history_depth_audit_target.csv").read_text(encoding="utf-8").splitlines()
    for e in list_cells():
        sym, tf, d = e["symbol"], e["timeframe"], e["dataset_hash16"]
        sidecar = json.loads((art / "datasets" / f"{sym}_{tf}_{d}.json").read_text(encoding="utf-8"))
        assert sidecar["dataset_id"] == d
        assert e["dataset_window"] == {"start": sidecar["start_utc"], "end": sidecar["end_utc"],
                                       "bars": sidecar["n_bars"]}
        for field, sub in (("spread_artifact_hash", "spread"), ("specs_artifact_hash", "specs")):
            assert e[field]["sha256"] == hashlib.sha256((art / sub / f"{sym}.json").read_bytes()).hexdigest()
        ref = e["audit_row_ref"]
        assert audit[ref["line"] - 1] == ref["row"]
        assert next(csv.reader([ref["row"]]))[:2] == [sym, tf]
        assert f"# AS_OF {ref['as_of']}" in audit


# ---- SL4 tripwires: every one refuses with its named error -------------------------------------------------------

def test_a_missing_registry_is_refused(tmp_path) -> None:
    with pytest.raises(RegistryMissing):
        load(tmp_path / "absent.json")


def test_a_value_edited_without_resealing_is_refused(registry_doc, write_registry) -> None:
    registry_doc["entries"][0]["dataset_window"]["bars"] += 1
    with pytest.raises(RegistryIntegrityError, match="entry_hash"):
        load(write_registry(registry_doc))


def test_a_dataset_hash_swapped_without_resealing_is_refused(registry_doc, write_registry) -> None:
    registry_doc["entries"][0]["dataset_hash16"] = "0123456789abcdef"
    with pytest.raises(RegistryIntegrityError):
        load(write_registry(registry_doc))


def test_an_entry_rehashed_with_a_stale_chain_is_refused(registry_doc, write_registry) -> None:
    e = registry_doc["entries"][1]
    e["dataset_window"]["bars"] += 1
    e["entry_hash"] = entry_hash(e)
    with pytest.raises(RegistryIntegrityError, match="chain_hash"):
        load(write_registry(registry_doc))


def test_reordered_entries_break_the_chain(registry_doc, write_registry) -> None:
    registry_doc["entries"].reverse()
    with pytest.raises(RegistryIntegrityError, match="chain_hash"):
        load(write_registry(registry_doc))


@pytest.mark.parametrize("version", [1, 3], ids=["v1-no-longer-loads", "v3"])
@pytest.mark.parametrize("where", ["top", "entry"])
def test_a_schema_version_other_than_2_is_refused(registry_doc, write_registry, where, version) -> None:
    if where == "top":
        registry_doc["schema_version"] = version
    else:
        registry_doc = _resealed(registry_doc, lambda es: es[0].update(schema_version=version))
    with pytest.raises(RegistrySchemaVersionError):
        load(write_registry(registry_doc))


@pytest.mark.parametrize("bad", ["", "UNPROVEN", "RUN@", "RUN@cbcb2ae061e70bef", "run@ga-x", "RUN@ga-X", "UNRUN ",
                                 "RUN@ga-a/b", None])
def test_a_campaign_status_off_its_grammar_is_refused_even_when_sealed(registry_doc, write_registry, bad) -> None:
    with pytest.raises(RegistrySchemaError, match="campaign_status"):
        load(write_registry(_resealed(registry_doc, lambda es: es[0].update(campaign_status=bad))))


def test_a_foreign_schema_is_refused(registry_doc, write_registry) -> None:
    registry_doc["schema"] = "arqvio.something_else"
    with pytest.raises(RegistrySchemaVersionError):
        load(write_registry(registry_doc))


@pytest.mark.parametrize("mutate", [
    lambda es: es[0].pop("cross_cell"),
    lambda es: es[0].update(extra_field=1),
    lambda es: es[0]["dataset_window"].pop("bars"),
    lambda es: es[0]["power_check"].update(extra="x"),
    lambda es: es[0].update(dataset_hash16="AED3BF314539EC5A"),
    lambda es: es[0].update(dataset_hash16="aed3bf31"),
    lambda es: es[0].update(cell_id="usdcad_h1"),
    lambda es: es[0].update(status="MAYBE"),
    lambda es: es[0].update(timeframe="W1", cell_id="usdcadm_w1"),
    lambda es: es[0]["dataset_window"].update(bars=True),
    lambda es: es[0].update(evidence=[]),
], ids=["missing-field", "extra-field", "missing-nested", "extra-nested", "upper-hex", "short-hash",
        "cell-id-convention", "unknown-status", "unknown-timeframe", "bool-count", "no-evidence"])
def test_a_malformed_entry_is_refused_even_when_sealed(registry_doc, write_registry, mutate) -> None:
    with pytest.raises(RegistrySchemaError):
        load(write_registry(_resealed(registry_doc, mutate)))


def test_an_annualisation_off_the_law_is_refused(registry_doc, write_registry) -> None:
    doc = _resealed(registry_doc, lambda es: es[0].update(annualisation_ppy=252))
    with pytest.raises(RegistrySchemaError, match="off the law"):
        load(write_registry(doc))


@pytest.mark.parametrize("mutate", [
    lambda es: es[0].update(cross_cell="usdcadm_h1"),
    lambda es: es[0].update(cross_cell="eurusdm_h1"),
    lambda es: es[1].update(status="RETIRED"),
], ids=["self", "unknown", "retired"])
def test_a_cross_cell_that_is_not_another_validated_cell_is_refused(registry_doc, write_registry, mutate) -> None:
    with pytest.raises(RegistrySchemaError, match="cross_cell"):
        load(write_registry(_resealed(registry_doc, mutate)))


def test_two_entries_on_one_cell_are_refused(registry_doc, write_registry) -> None:
    doc = _resealed(registry_doc, lambda es: es.append(copy.deepcopy(es[0])))
    with pytest.raises(AmbiguousCellError):
        load(write_registry(doc))


def test_an_unknown_cell_is_refused_by_every_lookup() -> None:
    for lookup in (get_cell, cross_of, annualisation):
        with pytest.raises(UnknownCellError, match="eurusdm_h1"):
            lookup("eurusdm_h1")
    with pytest.raises(KeyError):
        PinnedCellsView()["eurusdm_h1"]
    assert "eurusdm_h1" not in PinnedCellsView()


# ---- the view -----------------------------------------------------------------------------------------------------

def test_the_view_is_read_only() -> None:
    view = PinnedCellsView()
    with pytest.raises(TypeError):
        view["eurusdm_h1"] = ("EURUSDm", "H1", "0123456789abcdef")  # type: ignore[index]
    with pytest.raises(TypeError):
        del view["usdcadm_h1"]  # type: ignore[attr-defined]


def test_the_view_resolves_every_access_and_refuses_a_tampered_file(registry_doc, write_registry) -> None:
    path = write_registry(registry_doc)
    view = PinnedCellsView(path)
    assert sorted(view) == sorted(REGISTERED)
    path.write_text(json.dumps(_resealed(registry_doc, lambda es: es.append(
        dict(copy.deepcopy(es[0]), cell_id="usdcadm_d1", timeframe="D1", annualisation_ppy=252,
             status="RETIRED")))), encoding="utf-8")
    assert sorted(view) == sorted(REGISTERED)  # a RETIRED entry stays in the chain, never resolves
    assert [e["cell_id"] for e in list_cells(path)][-1] == "usdcadm_d1"
    registry_doc["entries"][0]["dataset_hash16"] = "0123456789abcdef"
    path.write_text(json.dumps(registry_doc), encoding="utf-8")
    with pytest.raises(RegistryIntegrityError):
        "usdcadm_h1" in view  # noqa: B015 -- the membership test itself must refuse
