"""W1 AUDIT on a fake terminal: the badged inventory, and every refusal (unknown timeframe, absent symbol, disabled
trade mode, wrong account, tampered registry) before anything is written."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.research.registry import cells as registry
from engine.research.wizard import audit as w1
from engine.research.wizard.__main__ import main
from engine.research.wizard.common import WizardRefused, cell_dir, cell_id
from tests.research.wizard.conftest import FakeMT5

SYMBOL = {"name": "AAAm", "digits": 3, "point": 0.001, "trade_mode": 4, "volume_min": 0.01, "swap_long": -1.5}


def test_the_cell_id_is_the_u1_convention_and_needs_a_law_row() -> None:
    assert cell_id("AAAm", "H1") == "aaam_h1"
    with pytest.raises(WizardRefused, match="unknown-timeframe"):
        cell_id("AAAm", "W1")


def test_the_audit_writes_a_badged_inventory(wizard_home: Path) -> None:
    specs = wizard_home / "artifacts" / "specs"
    specs.mkdir(parents=True)
    (specs / "AAAm.json").write_text(json.dumps({"digits": 3, "point": 0.002}), encoding="utf-8")
    fake = FakeMT5({"AAAm": SYMBOL})
    doc = w1.run("AAAm", "H1", mt5=fake)
    f = doc["fields"]
    assert f["digits"]["badge"] == "DETECTED" and f["digits"]["vs_pinned_specs"] == "MATCH"
    assert f["point"]["vs_pinned_specs"] == "DRIFT" and f["point"]["pinned_value"] == 0.002
    assert f["trade_tick_size"]["badge"] == "ABSENT"
    assert doc["sessions"]["badge"] == "NOT-EXPOSED" and doc["time_basis"]["basis"] == "unresolved"
    assert doc["attach"]["login_verified"] and fake.kw["login"] == 900001 and fake.shut
    assert json.loads((cell_dir("aaam_h1") / "w1_audit.json").read_text(encoding="utf-8")) == doc


@pytest.mark.parametrize("fake, reason", [
    (FakeMT5({}), "symbol-absent"),
    (FakeMT5({"AAAm": dict(SYMBOL, trade_mode=0)}), "trade-disabled"),
    (FakeMT5({"AAAm": SYMBOL}, login=1), "wrong-account"),
    (FakeMT5({"AAAm": SYMBOL}, mode=2), "wrong-account"),
], ids=["absent", "disabled", "other-login", "non-demo"])
def test_the_audit_refuses_and_writes_nothing(wizard_home: Path, fake, reason: str) -> None:
    with pytest.raises(WizardRefused, match=reason):
        w1.run("AAAm", "H1", mt5=fake)
    assert not (cell_dir("aaam_h1") / "w1_audit.json").exists()


def test_a_tampered_registry_refuses_before_any_attach(wizard_home: Path, tmp_path: Path, monkeypatch) -> None:
    doc = json.loads(registry.registry_path().read_text(encoding="utf-8"))
    doc["entries"][0]["dataset_window"]["bars"] += 1
    bad = tmp_path / "cells_registry.json"
    bad.write_text(json.dumps(doc), encoding="utf-8")
    monkeypatch.setattr(registry, "registry_path", lambda: bad)
    fake = FakeMT5({"AAAm": SYMBOL})
    with pytest.raises(WizardRefused, match="registry-RegistryIntegrityError"):
        w1.run("AAAm", "H1", mt5=fake)
    assert not hasattr(fake, "kw")  # never attached


def test_the_cli_refuses_with_an_audit_row(wizard_home: Path, tmp_path: Path, capsys) -> None:
    assert main(["audit", "--symbol", "AAAm", "--tf", "W1", "--out", str(tmp_path)]) == 2
    row = json.loads((tmp_path / "audit.jsonl").read_text(encoding="utf-8"))
    assert (row["step"], row["outcome"], row["reason"]) == ("audit", "REFUSED", "unknown-timeframe")
    assert "REFUSED audit AAAm W1: unknown-timeframe" in capsys.readouterr().out
