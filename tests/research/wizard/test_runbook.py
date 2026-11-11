"""W7 RUNBOOK on a COPY of the registry with a proven UNRUN cell: the page is rendered from data (pins, holes, drift,
power, worker citation, prior searches on disk), written, and appended to evidence once by the guarded ceremony; a
cell without an admitted proof or a product refuses."""
from __future__ import annotations

import copy
import json
import sqlite3
from pathlib import Path

import pytest

from engine.core import paths
from engine.research.registry import cells as registry
from engine.research.wizard import runbook as w7
from engine.research.wizard.common import WizardRefused, cell_dir, write_product

HOLE = {"missing_from": "2024-01-09T05:00:00+00:00", "missing_to": "2024-01-09T05:00:00+00:00", "bars_missing": 1,
        "seen_in": [], "note": "Tue; in neither trusted pin", "category": "owner_accepted_hole", "citation": "AMD-X"}


@pytest.fixture
def setup(wizard_home: Path, tmp_path: Path, monkeypatch) -> dict:
    doc = json.loads(registry.registry_path().read_text(encoding="utf-8"))
    new = copy.deepcopy(doc["entries"][0])
    proof_ref = "artifacts/reports/u4/worker/aaam_h1.proof.json"
    new.update(cell_id="aaam_h1", symbol="AAAm", campaign_status="UNRUN", cross_cell="usdcadm_h1",
               power_check={"status": "BLOCKED:no-campaign", "ref": "docs/cells/aaam_h1.md#power"},
               worker_status={"status": "PROVEN-PRECAMPAIGN@abcdef0", "fills_in": "U4", "note": "battery: CITED"},
               evidence=["docs/decisions/DR-x.md", proof_ref])
    reg = tmp_path / "configs" / "cells_registry.json"
    reg.parent.mkdir()
    reg.write_text(json.dumps(registry.seal([*doc["entries"], new]), indent=2), encoding="utf-8")
    monkeypatch.setattr(registry, "registry_path", lambda: reg)
    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(w7, "_ceremony_commits", lambda cid: ["abc1234 data(registry): the aaam_h1 ceremony"])
    spread = wizard_home / "artifacts" / "spread"
    spread.mkdir(parents=True)
    for e in [*doc["entries"], new]:
        (spread / f"{e['symbol']}.json").write_text(json.dumps({"calibrated_utc": "2026-09-24T08:52:29+00:00"}),
                                                    encoding="utf-8")
    (tmp_path / proof_ref).parent.mkdir(parents=True)
    (tmp_path / proof_ref).write_text(json.dumps({
        "tier": "pre-campaign", "checks": {"dataset": {"ok": True}, "canary": {"ok": True}},
        "not_applicable": {"power": "no campaign"}}), encoding="utf-8")
    base = {"cell_id": "aaam_h1", "symbol": "AAAm", "timeframe": "H1"}
    write_product(cell_dir("aaam_h1") / "w1_audit.json", {
        **base, "read_utc": "2026-10-09T05:16:02+00:00", "sessions": {"badge": "NOT-EXPOSED", "note": "no call"},
        "pinned_specs": {"drift": ["swap_short"]},
        "fields": {"swap_short": {"pinned_value": -17.7, "value": -17.3, "badge": "DETECTED"}}})
    write_product(cell_dir("aaam_h1") / "w2_manifest.json", {
        **base, "n_bars": 1000, "files_sha256": {"<ARQVIO_HOME>/artifacts/datasets/AAAm_H1_x.npz": "1" * 64},
        "threshold": {"bars": 900, "rule": "R4"},
        "gaps": {"counts": {"weekend": 4, "pin_explained": 1, "owner_accepted_hole": 1, "unexplained": 0},
                 "trusted_pins": ["usdcadm_h1"], "owner_accepted_holes": [HOLE], "owner_accepted_bars": 1,
                 "acceptance": {"file": "configs/cells/aaam_h1.accepted_holes.json", "canonical_sha256": "2" * 64,
                                "citation": "AMD-X"}}})
    write_product(cell_dir("aaam_h1") / "w3_pin.json", {
        **base, "spread": {"file": "<ARQVIO_HOME>/artifacts/spread/AAAm.json",
                           "provenance": {"time_basis": None, "calibrated_utc": "2026-09-24", "sample_count": 7}},
        "specs": {"file": "<ARQVIO_HOME>/artifacts/specs/AAAm.json"}})
    write_product(cell_dir("aaam_h1") / "w5_power.json", {
        **base, "decision": "BLOCKED:no-campaign", "ruling": "R1", "why": "needs a campaign",
        "control_at_n": "not applicable", "discharge": "the first R-chain campaign flips it"})
    db = wizard_home / "artifacts" / "rerun-aaam-h1.sqlite"
    with sqlite3.connect(db) as con:
        con.execute("CREATE TABLE campaigns (campaign_id TEXT, symbol TEXT, timeframe TEXT, dataset_hash TEXT, "
                    "created_at TEXT)")
        con.execute("CREATE TABLE trials (campaign_id TEXT, op_type TEXT, source TEXT)")
        con.execute("INSERT INTO campaigns VALUES ('rerun-aaam-h1', 'AAAm', 'H1', 'live-x', '2026-09-23')")
        con.executemany("INSERT INTO trials VALUES ('rerun-aaam-h1', 'SEARCH_TRIAL', 'MT5_LIVE')", [()] * 3)
    con.close()
    return {"reg": reg, "root": tmp_path}


def test_the_page_is_rendered_from_data(setup) -> None:
    cid, text = w7.render("AAAm", "H1")
    assert cid == "aaam_h1" and text.startswith("# aaam_h1 -- cell runbook")
    for want in ("## Known Holes", "1 owner-accepted hole(s), 1 bars of 1000 (0.10%)", "| 2024-01-09T05:00:00+00:00",
                 "Citation: AMD-X.", "no time_basis (absent, null)", "| swap_short | -17.7 | -17.3 |",
                 f"{len(registry.load())} of {len(registry.load())} VALIDATED cells' spread files carry no time_basis, "
                 "calibrated on 2026-09-24 (one batch",
                 "## Power", "Discharge: the first R-chain campaign flips it.", "Citation: battery: CITED",
                 "`<ARQVIO_HOME>/artifacts/rerun-aaam-h1.sqlite`: campaign `rerun-aaam-h1`", "3 SEARCH_TRIAL (MT5_LIVE)",
                 "- `abc1234 data(registry): the aaam_h1 ceremony`", "\"ready for first search\"",
                 "python -m engine.research.reproof --cell aaam_h1"):
        assert want in text, want


def test_the_page_is_written_and_appended_to_evidence_once(setup) -> None:
    before = registry.load(setup["reg"])
    out = w7.run("AAAm", "H1")
    after = registry.load(setup["reg"])
    assert (setup["root"] / "docs" / "cells" / "aaam_h1.md").read_text(encoding="utf-8") == w7.render("AAAm", "H1")[1]
    assert after[:-1] == before[:-1] and out["changed"]["aaam_h1"] == ["evidence"]
    assert after[-1]["evidence"] == [*before[-1]["evidence"], "docs/cells/aaam_h1.md"]
    w7.run("AAAm", "H1")  # a re-render never appends twice
    assert registry.load(setup["reg"])[-1]["evidence"] == after[-1]["evidence"]


def test_a_missing_product_refuses(setup) -> None:
    (cell_dir("aaam_h1") / "w5_power.json").unlink()
    with pytest.raises(WizardRefused, match="product-missing"):
        w7.render("AAAm", "H1")


def test_an_unknown_cell_refuses(setup) -> None:
    with pytest.raises(WizardRefused, match="unknown-cell"):
        w7.render("BBBm", "H1")


def test_a_cell_without_an_admitted_proof_refuses(setup) -> None:
    doc = json.loads(setup["reg"].read_text(encoding="utf-8"))
    doc["entries"][-1]["worker_status"]["status"] = "UNKNOWN"
    setup["reg"].write_text(json.dumps(registry.seal(doc["entries"])), encoding="utf-8")
    with pytest.raises(WizardRefused, match="no-proof"):
        w7.render("AAAm", "H1")
