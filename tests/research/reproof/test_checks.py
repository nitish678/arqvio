"""The re-proof's checks in isolation: path resolution, the battery tier, the verdict precedence, the status grammar,
the evidence set, and the read-only contract (AST)."""
from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

from engine.core import paths
from engine.research.registry.cells import get_cell
from engine.research.reproof import checks as rc
from tests.research.reproof.conftest import battery_text


def test_refs_resolve_against_exactly_one_root(isolated_arqvio_home: Path) -> None:
    assert rc.resolve_ref("<ARQVIO_HOME>/artifacts/x.csv") == isolated_arqvio_home / "artifacts" / "x.csv"
    assert rc.resolve_ref("artifacts/reports/x.md") == paths.PROJECT_ROOT / "artifacts" / "reports" / "x.md"


def test_the_battery_cell_is_read_from_the_battery_module() -> None:
    assert rc.battery_cell() == ("usdcadm_h1", "USDCADm", "aed3bf314539ec5a")


def test_the_battery_log_parses_in_the_closing_chain_format() -> None:
    at, fams, passed = rc.parse_battery_log(battery_text("a" * 40))
    assert at == "a" * 40 and passed
    assert fams == {k: {"fired": f, "trades": t, "mismatches": m} for k, (f, t, m) in rc.BANKED_BATTERY.items()}


def test_a_banked_battery_at_head_passes_with_its_tier(write_battery_log) -> None:
    usd, xau = (rc.check_battery(c, write_battery_log()) for c in ("usdcadm_h1", "xauusdm_h1"))
    assert usd["ok"] and xau["ok"] and usd["closure_unchanged_since_log_head"]
    assert usd["tier"].startswith("own cell") and xau["tier"].startswith("global tier R4(a)")


def test_a_battery_count_mismatch_blocks_with_the_counts(write_battery_log) -> None:
    counts = dict(rc.BANKED_BATTERY, STOCH_CROSS=("60/60", 1095, 0))
    got = rc.check_battery("xauusdm_h1", write_battery_log(counts=counts))
    assert not got["ok"] and got["reason"] == "battery-fail:STOCH_CROSS"
    assert got["observed"]["STOCH_CROSS"] == {"fired": "60/60", "trades": 1095, "mismatches": 0}
    assert got["banked"]["STOCH_CROSS"]["trades"] == 1096


def test_a_missing_failed_or_stale_battery_log_blocks(write_battery_log, tmp_path: Path) -> None:
    assert rc.check_battery("usdcadm_h1", tmp_path / "absent.log")["reason"] == "battery-fail:log"
    assert rc.check_battery("usdcadm_h1", None)["reason"] == "battery-fail:log"
    assert rc.check_battery("usdcadm_h1", write_battery_log(passed=False))["reason"] == "battery-fail:log"
    root = subprocess.run(["git", "-C", str(paths.PROJECT_ROOT), "rev-list", "--max-parents=0", "HEAD"],
                          check=True, capture_output=True, text=True).stdout.split()[0]
    stale = rc.check_battery("usdcadm_h1", write_battery_log(at=root))
    assert stale["reason"] == "battery-fail:log" and "battery inputs changed" in stale["stale"]


def test_the_verdict_takes_the_first_reason_by_precedence() -> None:
    ok = {"ok": True, "reason": None}
    assert rc.verdict({"a": ok, "b": ok}) == ("PROVEN", None, [])
    checks = {"a": {"ok": False, "reason": "evidence-missing:2"}, "b": {"ok": False, "reason": "battery-fail:log"},
              "c": {"ok": False, "reason": "power-unproven"}, "d": ok}
    assert rc.verdict(checks) == ("BLOCKED", "power-unproven",
                                  ["battery-fail:log", "evidence-missing:2", "power-unproven"])
    checks["e"] = {"ok": False, "reason": "dataset-missing"}
    assert rc.verdict(checks)[1] == "dataset-missing"


@pytest.mark.parametrize("value, ok", [
    ("UNKNOWN", True), ("PROVEN@e4abf20", True), ("BLOCKED:power-unproven", True), ("BLOCKED:battery-fail:canary", True),
    ("BLOCKED:battery-fail:STOCH_CROSS", True), ("BLOCKED:evidence-missing:3", True), ("UNPROVEN", False),
    ("PROVEN", False), ("PROVEN@E4ABF20", False), ("BLOCKED:evidence-missing:0", False), ("BLOCKED:other", False),
])
def test_the_status_grammar(value: str, ok: bool) -> None:
    assert (rc.STATUS_GRAMMAR.fullmatch(value) is not None) is ok


def test_the_evidence_set_covers_every_cited_path() -> None:
    refs = rc.evidence_refs(get_cell("xauusdm_h1"))
    assert "<ARQVIO_HOME>/artifacts/history_depth_audit_target.csv" in refs
    assert "artifacts/reports/u2/xau_power_run.log" in refs
    assert "<ARQVIO_HOME>/artifacts/ga/ga-xauusdm_h1-s7-f32e0d4defd9cf07/manifest_v2.json" in refs
    assert "docs/decisions/DR-2026-10-07-u1-cell-registry.md" in refs


def test_missing_evidence_is_counted_never_defaulted() -> None:
    entry = dict(get_cell("usdcadm_h1"), evidence=["docs/decisions/DR-2026-10-07-u1-cell-registry.md",
                                                     "artifacts/reports/no_such_report.md"])
    got = rc.check_evidence(entry)
    assert not got["ok"] and got["reason"] == "evidence-missing:" + str(len(got["missing"]))
    assert "artifacts/reports/no_such_report.md" in got["missing"]


def test_the_annualisation_is_the_law_table() -> None:
    assert rc.check_annualisation(get_cell("usdcadm_h1"))["ppy"] == {"registry": 6048, "law": 6048, "cell_dsr": 6048}
    assert rc.check_annualisation(dict(get_cell("usdcadm_h1"), annualisation_ppy=252))["reason"] == "hash-mismatch"


FORBIDDEN_CALLS = {"initialize", "order_send", "order_check", "symbol_select", "copy_rates_from_pos",
                   "copy_rates_range", "copy_rates_from", "write_json_atomic", "flatten", "flatten_own", "trigger"}


def test_the_reproof_package_is_read_only_toward_mt5_and_the_live_lane() -> None:
    pkg = paths.PROJECT_ROOT / "engine" / "research" / "reproof"
    for f in sorted(pkg.glob("*.py")):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
                assert name not in FORBIDDEN_CALLS, f"{f.name}:{node.lineno} calls {name}"
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("engine.research.live"), f"{f.name} imports the live lane"
