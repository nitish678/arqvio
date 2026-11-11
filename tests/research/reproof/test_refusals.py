"""The five refuse-paths (ARQVIO_U3_PROMPT_v1.md Part 1.4; DR-2026-10-08-u3-worker §4): unknown cell and tampered
registry REFUSE before any check, with an audit row; missing dataset, battery mismatch and missing evidence BLOCK
with their reason, never a fabricated pass."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.research.registry import cells
from engine.research.reproof import checks as rc
from engine.research.reproof.__main__ import main


def _audit(out: Path) -> list[dict]:
    return [json.loads(line) for line in (out / "audit.jsonl").read_text(encoding="utf-8").splitlines()]


def test_an_unknown_cell_is_refused_with_an_audit_row(tmp_path: Path, capsys) -> None:
    with pytest.raises(rc.ReproofRefused) as exc:
        rc.reproof("eurusdm_h1", out_dir=tmp_path, battery_log=None)
    assert exc.value.reason == "unknown-cell"
    assert main(["--cell", "eurusdm_h1", "--out", str(tmp_path), "--battery-log", "x.log"]) == 2
    assert "REFUSED eurusdm_h1: unknown-cell" in capsys.readouterr().out
    rows = _audit(tmp_path)
    assert [(r["cell"], r["outcome"], r["reason"]) for r in rows] == [("eurusdm_h1", "REFUSED", "unknown-cell")] * 2
    assert not list(tmp_path.glob("*.proof.json"))


def test_a_tampered_registry_aborts_before_any_check(tmp_path: Path, monkeypatch) -> None:
    doc = json.loads(cells.registry_path().read_text(encoding="utf-8"))
    doc["entries"][1]["dataset_window"]["bars"] += 1  # edited without resealing
    tampered = tmp_path / "cells_registry.json"
    tampered.write_text(json.dumps(doc), encoding="utf-8")
    monkeypatch.setattr(cells, "registry_path", lambda: tampered)
    ran = []
    monkeypatch.setattr(rc, "check_dataset", lambda entry: ran.append("dataset"))
    out = tmp_path / "out"
    with pytest.raises(rc.ReproofRefused) as exc:
        rc.reproof("xauusdm_h1", out_dir=out, battery_log=None)
    assert exc.value.reason == "registry-RegistryIntegrityError" and ran == []
    assert _audit(out)[0]["reason"] == "registry-RegistryIntegrityError"
    assert not list(out.glob("*.proof.json"))


def test_a_missing_dataset_blocks_never_fakes(isolated_arqvio_home: Path, tmp_path: Path, write_battery_log) -> None:
    proof = rc.reproof("usdcadm_h1", out_dir=tmp_path / "out", battery_log=write_battery_log())
    assert proof["verdict"] == "BLOCKED" and proof["status"] == "BLOCKED:dataset-missing"
    assert proof["checks"]["dataset"]["missing"] == ["USDCADm_H1_aed3bf314539ec5a.npz",
                                                     "USDCADm_H1_aed3bf314539ec5a.json"]
    assert not proof["checks"]["rederive"]["ok"] and not proof["checks"]["canary"]["ok"]
    assert _audit(tmp_path / "out")[-1]["status"] == "BLOCKED:dataset-missing"


def test_a_battery_mismatch_blocks_with_its_counts(write_battery_log) -> None:
    counts = dict(rc.BANKED_BATTERY, RSI_REVERSION=("47/60", 703, 0))
    got = rc.check_battery("usdcadm_h1", write_battery_log(counts=counts))
    assert got["reason"] == "battery-fail:RSI_REVERSION"
    assert (got["observed"]["RSI_REVERSION"]["trades"], got["banked"]["RSI_REVERSION"]["trades"]) == (703, 704)


def test_missing_evidence_blocks_with_its_count() -> None:
    entry = dict(cells.get_cell("xauusdm_h1"), evidence=["artifacts/reports/gone_1.md", "artifacts/reports/gone_2.md"])
    got = rc.check_evidence(entry)
    assert got["reason"] == f"evidence-missing:{len(got['missing'])}"
    assert {"artifacts/reports/gone_1.md", "artifacts/reports/gone_2.md"} <= set(got["missing"])
    assert rc.verdict({"evidence": got})[1] == got["reason"]
