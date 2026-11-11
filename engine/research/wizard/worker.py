"""W6 WORKER RE-PROOF ceremony (DR-2026-10-09-u4-wizard §4; #2-R2, #3-Q3; ARQVIO_U4_PROMPT_v2_EN §2 Worker, §3 W6).

The proof comes from `python -m engine.research.reproof` (the registry selects the pre-campaign tier for an UNRUN
cell). This step admits it into the registry, and REFUSES unless the proof is:
- for this cell, at the current HEAD, against the registry being written (its chain_hash);
- the pre-campaign tier, verdict PROVEN, status PROVEN-PRECAMPAIGN@<sha> under the status grammar;
- every tier check ok, with power and rederive recorded as not applicable (never as passes);
- a battery check that RAN against a log whose observed families equal the banked S6 counts.
The note is the #3-Q3 citation, built from the proof's machine results only (a citation without a run is rejected):
  "battery: <tier> == banked S6 (<trades>) verified @<battery-log sha256>, era S6-bank; audit_row: csv:<line>
  recorded-unchanged". The guarded ceremony changes worker_status (only from UNKNOWN; fills_in kept) and appends the
proof to evidence, nothing else.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from engine.research.registry import cells as registry
from engine.research.wizard.common import WizardRefused, cell_id, registry_ceremony, require_registry


def citation(proof: dict[str, Any]) -> str:
    from engine.research.reproof.checks import BANKED_BATTERY

    bat, row = proof["checks"]["battery"], proof["checks"]["audit_row"]
    trades = "/".join(str(t) for _, t, _ in BANKED_BATTERY.values())
    return (f"battery: {bat['tier'].split(' (')[0]} == banked S6 ({trades}) verified @{bat['log']['sha256']}, "
            f"era S6-bank; audit_row: csv:{row['cell_rows'][0]} recorded-unchanged")


def admit(proof: dict[str, Any], cid: str, head: str, chain: str) -> None:
    from engine.core import paths
    from engine.research.reproof.checks import (
        BANKED_BATTERY,
        NOT_APPLICABLE_PRECAMPAIGN,
        PRECAMPAIGN_CHECKS,
        STATUS_GRAMMAR,
        sha256_file,
    )

    checks = proof.get("checks", {})
    bat = checks.get("battery", {})
    log = paths.PROJECT_ROOT / bat.get("log", {}).get("path", "<no battery log>")
    banked = {k: {"fired": f, "trades": t, "mismatches": m} for k, (f, t, m) in BANKED_BATTERY.items()}
    tests = {
        "cell": proof.get("cell") == cid, "head": proof.get("head") == head,
        "chain": proof.get("registry", {}).get("chain_hash") == chain,
        "tier": proof.get("tier") == "pre-campaign",
        "verdict": proof.get("verdict") == "PROVEN",
        "status": STATUS_GRAMMAR.fullmatch(proof.get("status", "")) is not None
        and proof["status"] == f"PROVEN-PRECAMPAIGN@{head[:7]}",
        # the proof file is canonical JSON (sorted keys): the set of checks, not their order, is the contract
        "checks": sorted(checks) == sorted(PRECAMPAIGN_CHECKS) and all(c.get("ok") is True for c in checks.values()),
        "not_applicable": proof.get("not_applicable") == NOT_APPLICABLE_PRECAMPAIGN,
        "battery_ran": bat.get("observed") == banked and bat.get("four_passed") is True
        and bat.get("closure_unchanged_since_log_head") is True,
        "battery_log_current": log.is_file() and sha256_file(log) == bat["log"]["sha256"],
        "citable": isinstance(bat.get("tier"), str) and checks["audit_row"].get("cell_rows", [None])[0] is not None,
    }
    if not all(tests.values()):
        raise WizardRefused("proof-not-admissible", f"{cid}: {tests}")


def run(symbol: str, tf: str, *, proof_path: Path) -> dict[str, Any]:
    """Admit (symbol, tf)'s pre-campaign proof and record worker_status. Refuses (raises) otherwise."""
    from engine.core import paths
    from engine.research.reproof.checks import git_head

    cid = cell_id(symbol, tf)
    require_registry()
    if not proof_path.is_file():
        raise WizardRefused("no-proof", f"no proof at {proof_path}")
    try:
        proof = json.loads(proof_path.read_text(encoding="utf-8"))
        ref = proof_path.resolve().relative_to(paths.PROJECT_ROOT.resolve()).as_posix()
    except ValueError as exc:  # unreadable JSON, or a proof outside the repo (its evidence ref must resolve)
        raise WizardRefused("proof-not-admissible", f"{proof_path}: {exc}") from exc
    chain = json.loads(registry.registry_path().read_text(encoding="utf-8"))["chain_hash"]
    try:
        admit(proof, cid, git_head(), chain)
        note = citation(proof)
    except WizardRefused:
        raise
    except (KeyError, TypeError, AttributeError, IndexError) as exc:  # a malformed proof refuses, never crashes
        raise WizardRefused("proof-not-admissible", f"{cid}: {type(exc).__name__}: {exc}") from exc
    date = datetime.now(tz=UTC).date().isoformat()
    note = f"{note}; proven {date} by python -m engine.research.reproof at {proof['head'][:7]} ({ref})"

    def change(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if registry.chain_hash([e["entry_hash"] for e in entries]) != proof["registry"]["chain_hash"]:
            raise WizardRefused("proof-not-admissible", f"{cid}: the registry being written is not the proof's")
        for e in entries:
            if e["cell_id"] == cid:
                if e["worker_status"]["status"] != "UNKNOWN":
                    raise WizardRefused("worker-decided", f"{cid}: worker_status is already {e['worker_status']}")
                e["worker_status"] = {"status": proof["status"], "fills_in": e["worker_status"]["fills_in"],
                                      "note": note}
                e["evidence"] = [*e["evidence"], ref]
        return entries

    result = registry_ceremony(change, allowed={cid: frozenset({"worker_status", "evidence"})})
    return {"cell_id": cid, "status": proof["status"], "note": note, "proof": ref, **result}
