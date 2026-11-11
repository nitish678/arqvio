"""W6 worker ceremony on a COPY of the registry with an UNRUN cell: only a pre-campaign proof that RAN its seven checks,
for this cell, at this HEAD, against this registry, is admitted; the note is the #3-Q3 citation built from the proof's
machine results; the guarded ceremony changes worker_status + an evidence append only."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from engine.core import paths
from engine.research.registry import cells as registry
from engine.research.reproof import checks as rc
from engine.research.wizard import worker as w6
from engine.research.wizard.common import WizardRefused

HEAD = "abcdef0123456789abcdef0123456789abcdef01"
BANKED = {k: {"fired": f, "trades": t, "mismatches": m} for k, (f, t, m) in rc.BANKED_BATTERY.items()}
LOG = "artifacts/reports/u3/closing_parity_battery.log"
LOG_TEXT = b"HEAD " + b"b" * 40 + b"\n4 passed in 1.0s\nEXIT=0\n"
LOG_SHA = hashlib.sha256(LOG_TEXT).hexdigest()


@pytest.fixture
def setup(wizard_home: Path, tmp_path: Path, monkeypatch) -> dict:
    doc = json.loads(registry.registry_path().read_text(encoding="utf-8"))
    new = copy.deepcopy(doc["entries"][0])
    new.update(cell_id="aaam_h1", symbol="AAAm", campaign_status="UNRUN", cross_cell="usdcadm_h1",
               power_check={"status": "BLOCKED:no-campaign", "ref": "docs/cells/aaam_h1.md#power"},
               worker_status={"status": "UNKNOWN", "fills_in": "U4", "note": "W6 fills this in"})
    reg = tmp_path / "configs" / "cells_registry.json"
    reg.parent.mkdir()
    sealed = registry.seal([*doc["entries"], new])
    reg.write_text(json.dumps(sealed, indent=2), encoding="utf-8")
    monkeypatch.setattr(registry, "registry_path", lambda: reg)
    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(rc, "git_head", lambda: HEAD)
    (tmp_path / LOG).parent.mkdir(parents=True)
    (tmp_path / LOG).write_bytes(LOG_TEXT)
    return {"reg": reg, "chain": sealed["chain_hash"], "root": tmp_path}


def proof(chain: str, **over) -> dict:
    checks = {n: {"ok": True, "reason": None} for n in rc.PRECAMPAIGN_CHECKS}
    checks["battery"].update(tier="global tier R4(a) (the banked battery runs on usdcadm_h1)", observed=BANKED,
                             four_passed=True, closure_unchanged_since_log_head=True,
                             log={"path": LOG, "sha256": LOG_SHA, "head": "b" * 40})
    checks["audit_row"].update(cell_rows=[38])
    p = {"schema": rc.PROOF_SCHEMA, "version": rc.PROOF_VERSION, "cell": "aaam_h1", "head": HEAD,
         "registry": {"entry_hash": "e" * 64, "chain_hash": chain}, "entry": {}, "checks": checks,
         "verdict": "PROVEN", "reason": None, "failures": [], "status": f"PROVEN-PRECAMPAIGN@{HEAD[:7]}",
         "tier": "pre-campaign", "not_applicable": rc.NOT_APPLICABLE_PRECAMPAIGN}
    p.update(over)
    return p


def write(root: Path, doc: dict) -> Path:
    """The proof written exactly as reproof writes it: canonical JSON (sorted keys), LF."""
    p = root / "artifacts" / "reports" / "u4" / "worker" / "aaam_h1.proof.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(rc.canonical(doc) + "\n", encoding="utf-8", newline="\n")
    return p


def test_an_admissible_proof_is_recorded_with_its_citation(setup) -> None:
    before = registry.load(setup["reg"])
    out = w6.run("AAAm", "H1", proof_path=write(setup["root"], proof(setup["chain"])))
    after = registry.load(setup["reg"])
    assert after[:-1] == before[:-1] and out["changed"]["aaam_h1"] == ["evidence", "worker_status"]
    ws = after[-1]["worker_status"]
    assert ws["status"] == f"PROVEN-PRECAMPAIGN@{HEAD[:7]}" and ws["fills_in"] == "U4"
    assert ws["note"].startswith("battery: global tier R4(a) == banked S6 (704/1096/2967/1994) verified @"
                                 + LOG_SHA + ", era S6-bank; audit_row: csv:38 recorded-unchanged; proven ")
    assert after[-1]["evidence"] == [*before[-1]["evidence"], "artifacts/reports/u4/worker/aaam_h1.proof.json"]
    chain_now = json.loads(setup["reg"].read_text(encoding="utf-8"))["chain_hash"]  # a proof against the new chain
    with pytest.raises(WizardRefused, match="worker-decided"):
        w6.run("AAAm", "H1", proof_path=write(setup["root"], proof(chain_now)))


def _bad_checks(chain: str) -> dict:
    p = proof(chain)
    p["checks"]["canary"] = {"ok": False, "reason": "battery-fail:canary"}
    return p


@pytest.mark.parametrize("make", [
    lambda c: proof(c, cell="bbbm_h1"),
    lambda c: proof(c, head="0" * 40),
    lambda c: proof("0" * 64),
    lambda c: {k: v for k, v in proof(c, status=f"PROVEN@{HEAD[:7]}").items() if k not in ("tier", "not_applicable")},
    lambda c: _bad_checks(c),
    lambda c: proof(c, not_applicable={}),
    lambda c: proof(c, checks={**proof(c)["checks"], "power": {"ok": True, "reason": None}}),
    lambda c: proof(c, checks={**proof(c)["checks"], "battery": {**proof(c)["checks"]["battery"], "observed": {}}}),
    lambda c: proof(c, status="BLOCKED:battery-fail:canary"),
], ids=["other-cell", "stale-head", "stale-chain", "full-tier", "failed-check", "no-not-applicable",
        "power-as-a-pass", "battery-not-run", "blocked"])
def test_an_inadmissible_proof_refuses_and_writes_nothing(setup, make) -> None:
    raw = setup["reg"].read_bytes()
    with pytest.raises(WizardRefused, match="proof-not-admissible"):
        w6.run("AAAm", "H1", proof_path=write(setup["root"], make(setup["chain"])))
    assert setup["reg"].read_bytes() == raw


def _no_cell_rows(chain: str) -> dict:
    p = proof(chain)
    del p["checks"]["audit_row"]["cell_rows"]
    return p


@pytest.mark.parametrize("make, mutate_log", [
    (lambda c: proof(c), True),
    (_no_cell_rows, False),
    (lambda c: proof(c, checks={**proof(c)["checks"], "battery": {"ok": True, "reason": None}}), False),
    (lambda c: proof(c, registry=None), False),
], ids=["battery-log-changed", "no-cell-rows", "no-battery-detail", "registry-not-a-dict"])
def test_a_stale_or_malformed_proof_refuses_never_crashes(setup, make, mutate_log) -> None:
    raw = setup["reg"].read_bytes()
    if mutate_log:
        (setup["root"] / LOG).write_bytes(LOG_TEXT + b"edited\n")
    with pytest.raises(WizardRefused, match="proof-not-admissible"):
        w6.run("AAAm", "H1", proof_path=write(setup["root"], make(setup["chain"])))
    assert setup["reg"].read_bytes() == raw


def test_a_missing_proof_refuses(setup) -> None:
    with pytest.raises(WizardRefused, match="no-proof"):
        w6.run("AAAm", "H1", proof_path=setup["root"] / "absent.proof.json")


def test_the_cli_worker_needs_a_proof(wizard_home: Path, tmp_path: Path) -> None:
    from engine.research.wizard.__main__ import main

    assert main(["worker", "--symbol", "AAAm", "--tf", "H1", "--out", str(tmp_path)]) == 2
    row = json.loads((tmp_path / "audit.jsonl").read_text(encoding="utf-8"))
    assert (row["step"], row["outcome"], row["reason"]) == ("worker", "REFUSED", "no-proof")
