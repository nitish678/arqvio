"""W5 POWER-CHECK on a COPY of the registry with an UNRUN cell appended: the decision BLOCKED:no-campaign is pure and
deterministic, it is recorded through the guarded ceremony (power_check + an evidence append only), and a cell that
has a campaign, is unknown, or is already decided refuses with the registry unchanged."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from engine.research.registry import cells as registry
from engine.research.wizard import power as w5
from engine.research.wizard.common import WizardRefused, canonical, cell_dir


@pytest.fixture
def reg(wizard_home: Path, tmp_path: Path, monkeypatch) -> Path:
    doc = json.loads(registry.registry_path().read_text(encoding="utf-8"))
    new = copy.deepcopy(doc["entries"][0])
    new.update(cell_id="aaam_h1", symbol="AAAm", campaign_status="UNRUN", cross_cell="usdcadm_h1",
               power_check={"status": "UNKNOWN", "ref": "U4 W5"},
               worker_status={"status": "UNKNOWN", "fills_in": "U4", "note": "W6 fills this in"})
    path = tmp_path / "cells_registry.json"
    path.write_text(json.dumps(registry.seal([*doc["entries"], new]), indent=2), encoding="utf-8")
    monkeypatch.setattr(registry, "registry_path", lambda: path)
    return path


def test_the_decision_is_blocked_no_campaign_and_deterministic(reg) -> None:
    doc = w5.decide("AAAm", "H1")
    assert (doc["decision"], doc["ref"]) == ("BLOCKED:no-campaign", "docs/cells/aaam_h1.md#power")
    assert doc["checked"] == {"campaign_status": "UNRUN", "campaign_dirs": []}
    assert doc["discharge"] == ("the first R-chain campaign on this cell flips power_check to COMPUTED (computing "
                                "stage named in ref), via U2's machinery as-is")
    assert canonical(w5.decide("AAAm", "H1")) == canonical(doc)


def test_the_decision_is_recorded_by_the_guarded_ceremony(reg) -> None:
    before = registry.load(reg)
    out = w5.run("AAAm", "H1")
    after = registry.load(reg)
    assert after[:-1] == before[:-1] and out["changed"]["aaam_h1"] == ["evidence", "power_check"]
    assert after[-1]["power_check"] == {"status": "BLOCKED:no-campaign", "ref": "docs/cells/aaam_h1.md#power"}
    assert after[-1]["evidence"] == [*before[-1]["evidence"], "<ARQVIO_HOME>/artifacts/cells/aaam_h1/w5_power.json"]
    assert json.loads((cell_dir("aaam_h1") / "w5_power.json").read_text(encoding="utf-8")) == out["decision"]
    with pytest.raises(WizardRefused, match="power-decided"):
        w5.run("AAAm", "H1")
    assert registry.load(reg) == after


def test_a_cell_with_a_campaign_refuses(reg, wizard_home: Path) -> None:
    (wizard_home / "artifacts" / "ga" / "ga-aaam_h1-s9-x").mkdir(parents=True)
    with pytest.raises(WizardRefused, match="campaign-run"):
        w5.run("AAAm", "H1")
    with pytest.raises(WizardRefused, match="campaign-run"):
        w5.decide("USDCADm", "H1")  # RUN@: U2's machinery applies
    assert registry.load(reg)[-1]["power_check"]["status"] == "UNKNOWN"
    assert not (cell_dir("aaam_h1") / "w5_power.json").exists()


def test_a_refused_ceremony_leaves_no_product(reg, monkeypatch) -> None:
    def refuse(*a, **k):
        raise WizardRefused("ceremony-race", "x")
    monkeypatch.setattr(w5, "registry_ceremony", refuse)
    with pytest.raises(WizardRefused, match="ceremony-race"):
        w5.run("AAAm", "H1")
    assert not (cell_dir("aaam_h1") / "w5_power.json").exists()


def test_the_document_names_the_registrys_cell_whatever_the_input_case(reg) -> None:
    doc = w5.decide("aaam", "H1")
    assert (doc["cell_id"], doc["symbol"]) == ("aaam_h1", "AAAm")


def test_an_unknown_cell_refuses(reg) -> None:
    with pytest.raises(WizardRefused, match="unknown-cell"):
        w5.decide("BBBm", "H1")
