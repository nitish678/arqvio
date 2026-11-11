"""The tier surgery (DR-2026-10-09-u4-wizard §3; #2-R2, #3-Q3): the tier comes from the registry entry alone; the
pre-campaign tier runs exactly its seven checks and records power / rederive as not applicable (never as passes);
the full tier's proof keeps U3's exact keys; a tier contradiction refuses before any check, with an audit row."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from engine.research.registry import cells
from engine.research.reproof import checks as rc
from engine.research.reproof.__main__ import main

U3_KEYS = {"schema", "version", "cell", "head", "registry", "entry", "checks", "verdict", "reason", "failures",
           "status"}
CHECKS = {"dataset": "check_dataset", "specs_spread": "check_specs_spread", "audit_row": "check_audit_row",
          "annualisation": "check_annualisation", "power": "check_power", "evidence": "check_evidence",
          "rederive": "check_rederive", "battery": "check_battery", "canary": "check_canary"}


def unrun_entry(base: dict, **over) -> dict:
    e = copy.deepcopy(base)
    e.update(cell_id="aaam_h1", symbol="AAAm", campaign_status="UNRUN", cross_cell="usdcadm_h1",
             power_check={"status": "BLOCKED:no-campaign", "ref": "docs/cells/aaam_h1.md#power"},
             worker_status={"status": "UNKNOWN", "fills_in": "U4", "note": "W6 fills this in"},
             spread_artifact_hash={"sha256": base["spread_artifact_hash"]["sha256"], "source": "sha256 of a file"})
    e.update(over)
    return e


@pytest.fixture
def with_entry(tmp_path: Path, monkeypatch):
    """with_entry(**over) -> a sealed copy of the real registry with an UNRUN aaam_h1 appended, made the registry."""
    def make(**over) -> dict:
        doc = json.loads(cells.registry_path().read_text(encoding="utf-8"))
        sealed = cells.seal([*doc["entries"], unrun_entry(doc["entries"][0], **over)])
        path = tmp_path / "cells_registry.json"
        path.write_text(json.dumps(sealed), encoding="utf-8")
        monkeypatch.setattr(cells, "registry_path", lambda: path)
        return sealed["entries"][-1]
    return make


@pytest.fixture
def ran(monkeypatch) -> list[str]:
    """Every check stubbed to pass, recording its name: the tier logic is under test, not the checks (test_checks)."""
    names: list[str] = []
    for name, fn in CHECKS.items():
        monkeypatch.setattr(rc, fn, lambda *a, _n=name, **k: names.append(_n) or rc._ok())
    return names


def test_the_tier_comes_from_campaign_status(with_entry) -> None:
    assert rc.select_tier(cells.get_cell("usdcadm_h1")) == "full"
    assert rc.select_tier(with_entry()) == "pre-campaign"


def test_the_pre_campaign_tier_runs_its_seven_checks_and_names_the_two_it_cannot(with_entry, ran, tmp_path,
                                                                                 capsys) -> None:
    with_entry()
    proof = rc.reproof("aaam_h1", out_dir=tmp_path / "out", battery_log=None)
    assert ran == list(rc.PRECAMPAIGN_CHECKS) and list(proof["checks"]) == list(rc.PRECAMPAIGN_CHECKS)
    assert proof["tier"] == "pre-campaign" and proof["not_applicable"] == rc.NOT_APPLICABLE_PRECAMPAIGN
    assert set(proof["not_applicable"]) == {"power", "rederive"} and not {"power", "rederive"} & set(proof["checks"])
    assert proof["verdict"] == "PROVEN" and proof["status"] == "PROVEN-PRECAMPAIGN@" + proof["head"][:7]
    assert rc.STATUS_GRAMMAR.fullmatch(proof["status"])
    assert main(["--cell", "aaam_h1", "--out", str(tmp_path / "out"), "--battery-log", "x.log"]) == 0
    out = capsys.readouterr().out
    assert "power          NOT APPLICABLE (pre-campaign)" in out and "VERDICT PROVEN-PRECAMPAIGN@" in out


def test_a_failing_pre_campaign_check_blocks(with_entry, ran, monkeypatch, tmp_path) -> None:
    with_entry()
    monkeypatch.setattr(rc, "check_canary", lambda entry: rc._fail("battery-fail:canary"))
    proof = rc.reproof("aaam_h1", out_dir=tmp_path, battery_log=None)
    assert proof["status"] == "BLOCKED:battery-fail:canary" and proof["tier"] == "pre-campaign"


def test_the_full_tier_proof_keeps_u3s_exact_keys(ran, tmp_path) -> None:
    proof = rc.reproof("usdcadm_h1", out_dir=tmp_path, battery_log=None)
    assert set(proof) == U3_KEYS and list(proof["checks"]) == list(CHECKS)
    assert proof["status"] == "PROVEN@" + proof["head"][:7]


@pytest.mark.parametrize("over, ghost_dir", [
    ({"power_check": {"status": "UNKNOWN", "ref": "U4 W5"}}, False),
    ({"power_check": {"status": "COMPUTED", "ref": "x.log:1"}}, False),
    ({}, True),
    ({"spread_artifact_hash": {"sha256": "0" * 64,
                               "source": "<ARQVIO_HOME>/artifacts/ga/ga-aaam_h1-s9-x/manifest_v2.json"}}, False),
], ids=["power-unknown", "power-computed", "campaign-dir", "cited-manifest"])
def test_a_tier_contradiction_refuses_before_any_check(isolated_arqvio_home: Path, with_entry, ran, tmp_path, over,
                                                        ghost_dir) -> None:
    with_entry(**over)
    if ghost_dir:
        (isolated_arqvio_home / "artifacts" / "ga" / "ga-aaam_h1-s9-x").mkdir(parents=True)
    out = tmp_path / "out"
    with pytest.raises(rc.ReproofRefused) as exc:
        rc.reproof("aaam_h1", out_dir=out, battery_log=None)
    assert exc.value.reason == "tier-contradiction" and ran == []
    assert main(["--cell", "aaam_h1", "--out", str(out), "--battery-log", "x.log"]) == 2
    rows = [json.loads(line) for line in (out / "audit.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["reason"] for r in rows] == ["tier-contradiction"] * 2 and not list(out.glob("*.proof.json"))
