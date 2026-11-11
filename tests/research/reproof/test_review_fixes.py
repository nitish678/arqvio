"""Regression tests for the adversarial review of the re-proof package (artifacts/reports/u3/review_reproof.json;
U3 DEV-3): each confirmed defect has a test that failed before its fix."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from engine.research.registry import cells
from engine.research.reproof import checks as rc
from tests.research.reproof.conftest import battery_text, head


def test_the_battery_closure_reaches_lazy_and_test_helper_imports() -> None:
    closure = rc.battery_closure()
    assert "tests/research/backtest/test_kernel_parity_steps.py" in closure  # computes the mismatches
    assert "engine/research/data/symbol_specs.py" in closure  # imported inside spec_for
    assert rc.BATTERY_TEST in closure and all(not p.startswith("/") for p in closure)


@pytest.mark.parametrize("summary, exit_line", [("4 passed, 1 error in 300.00s", "EXIT=0"),
                                                ("4 passed in 300.00s", "EXIT=1 WALL=301s")])
def test_a_failing_battery_run_is_not_a_pass(tmp_path: Path, summary: str, exit_line: str) -> None:
    text = battery_text(head()).replace("4 passed in 300.00s (0:05:00)", summary).replace("EXIT=0 WALL=301s",
                                                                                          exit_line)
    p = tmp_path / "battery.log"
    p.write_text(text, encoding="utf-8")
    assert rc.check_battery("usdcadm_h1", p)["reason"] == "battery-fail:log"


def test_a_present_but_tampered_snapshot_is_a_mismatch_not_missing(real_artifacts: Path,
                                                                   isolated_arqvio_home: Path) -> None:
    stem = "XAUUSDm_H1_447b843f87438a9a"
    src = real_artifacts / "datasets"
    if not (src / f"{stem}.npz").is_file():
        pytest.skip("the pinned XAU snapshot is not on this machine")
    dst = isolated_arqvio_home / "artifacts" / "datasets"
    dst.mkdir(parents=True)
    shutil.copy2(src / f"{stem}.json", dst / f"{stem}.json")
    (dst / f"{stem}.npz").write_bytes(b"not a zip archive")
    entry = cells.get_cell("xauusdm_h1")
    got = rc.check_dataset(entry)
    assert not got["ok"] and got["reason"] == "hash-mismatch" and "error" in got
    assert rc._snapshot_reason(entry) == "hash-mismatch"
    (dst / f"{stem}.npz").unlink()
    assert rc._snapshot_reason(entry) == "dataset-missing"


def test_a_non_utf8_registry_is_refused_with_an_audit_row(tmp_path: Path, monkeypatch) -> None:
    bad = tmp_path / "cells_registry.json"
    bad.write_bytes(b"\xff\xfe{garbage")
    monkeypatch.setattr(cells, "registry_path", lambda: bad)
    with pytest.raises(rc.ReproofRefused) as exc:
        rc.reproof("usdcadm_h1", out_dir=tmp_path / "out", battery_log=None)
    assert exc.value.reason == "registry-UnicodeDecodeError"
    row = json.loads((tmp_path / "out" / "audit.jsonl").read_text(encoding="utf-8"))
    assert row["outcome"] == "REFUSED" and not list((tmp_path / "out").glob("*.proof.json"))


def test_a_check_that_raises_is_recorded_and_the_proof_still_written(tmp_path: Path, monkeypatch) -> None:
    def boom(entry):
        raise RuntimeError("synthetic")
    for name in ("check_dataset", "check_specs_spread", "check_audit_row", "check_evidence", "check_rederive",
                 "check_canary"):
        monkeypatch.setattr(rc, name, boom)
    monkeypatch.setattr(rc, "check_power", lambda cell_id: boom(None))
    monkeypatch.setattr(rc, "check_battery", lambda cell_id, log: boom(None))
    proof = rc.reproof("usdcadm_h1", out_dir=tmp_path, battery_log=None)
    assert proof["checks"]["power"] == {"ok": False, "reason": "power-unproven", "error": "RuntimeError: synthetic"}
    assert proof["checks"]["canary"]["reason"] == "battery-fail:canary"
    assert proof["status"] == "BLOCKED:hash-mismatch" and (tmp_path / "usdcadm_h1.proof.json").is_file()


def test_evidence_ties_resolve_to_the_full_count() -> None:
    checks = {"audit_row": {"ok": False, "reason": "evidence-missing:1"},
              "evidence": {"ok": False, "reason": "evidence-missing:3"}}
    assert rc.verdict(checks)[1] == "evidence-missing:3"
