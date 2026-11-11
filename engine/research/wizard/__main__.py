"""python -m engine.research.wizard <step> --symbol <S> --tf <T> --out <dir> [step options]

The new-cell wizard's steps (DR-2026-10-09-u4-wizard §4). Every run appends one audit row to <out>/audit.jsonl.
Exit 0 done; 2 REFUSED (the reason is printed and audited; nothing else is written).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from engine.research.wizard.common import WizardRefused, audit_row


def _audit(a: argparse.Namespace) -> dict[str, Any]:
    import MetaTrader5 as mt5

    from engine.research.wizard.audit import run

    doc = run(a.symbol, a.tf, mt5=mt5)
    return {"product": f"<ARQVIO_HOME>/artifacts/cells/{doc['cell_id']}/w1_audit.json",
            "drift": doc["pinned_specs"]["drift"], "time_basis": doc["time_basis"]["basis"]}


def _pin(a: argparse.Namespace) -> dict[str, Any]:
    import MetaTrader5 as mt5

    from engine.research.wizard.pin import run

    if not (a.start and a.end):
        raise WizardRefused("no-window", "the pin step needs --start and --end (YYYY-MM-DD, UTC)")
    doc = run(a.symbol, a.tf, a.start, a.end, mt5=mt5)
    return {"product": f"<ARQVIO_HOME>/artifacts/cells/{doc['cell_id']}/w2_manifest.json",
            "dataset_hash16": doc["dataset_hash16"], "n_bars": doc["n_bars"],
            "threshold": doc["threshold"]["bars"], "gap_counts": doc["gaps"]["counts"]}


def _calibrate(a: argparse.Namespace) -> dict[str, Any]:
    from engine.research.wizard.calibrate import run

    if not (a.expect_spread_sha256 and a.expect_specs_sha256):
        raise WizardRefused("no-expected-hash", "the calibrate step needs --expect-spread-sha256 and "
                                                "--expect-specs-sha256 (the hashes recorded before the stage)")
    doc = run(a.symbol, a.tf, spread_sha256=a.expect_spread_sha256, specs_sha256=a.expect_specs_sha256)
    return {"product": f"<ARQVIO_HOME>/artifacts/cells/{doc['cell_id']}/w3_pin.json",
            "spread_sha256": doc["spread"]["sha256"], "specs_sha256": doc["specs"]["sha256"],
            "spread_evaluate": doc["spread"]["evaluate"]["status"]}


def _promote(a: argparse.Namespace) -> dict[str, Any]:
    from engine.research.wizard.promote import run

    if not a.cross_cell:
        raise WizardRefused("no-cross-cell", "the promote step needs --cross-cell (ceremony data)")
    out = run(a.symbol, a.tf, cross_cell=a.cross_cell)
    return {"cell_id": out["cell_id"], "chain_before": out["chain_before"], "chain_after": out["chain_after"],
            "entry_hash": out["entry_hash_after"][out["cell_id"]], "existing_changed": out["changed"]}


def _power(a: argparse.Namespace) -> dict[str, Any]:
    from engine.research.wizard.power import run

    out = run(a.symbol, a.tf)
    return {"product": f"<ARQVIO_HOME>/artifacts/cells/{out['cell_id']}/w5_power.json",
            "power_check": out["decision"]["decision"], "ref": out["decision"]["ref"],
            "chain_before": out["chain_before"], "chain_after": out["chain_after"], "changed": out["changed"]}


def _worker(a: argparse.Namespace) -> dict[str, Any]:
    from engine.research.wizard.worker import run

    if not a.proof:
        raise WizardRefused("no-proof", "the worker step needs --proof (a python -m engine.research.reproof proof)")
    out = run(a.symbol, a.tf, proof_path=a.proof)
    return {"cell_id": out["cell_id"], "worker_status": out["status"], "proof": out["proof"],
            "chain_before": out["chain_before"], "chain_after": out["chain_after"], "changed": out["changed"]}


def _runbook(a: argparse.Namespace) -> dict[str, Any]:
    from engine.research.wizard.runbook import run

    out = run(a.symbol, a.tf)
    return {"runbook": out["runbook"], "chain_before": out["chain_before"], "chain_after": out["chain_after"],
            "changed": out["changed"]}


STEPS = {"audit": _audit, "pin": _pin, "calibrate": _calibrate, "promote": _promote, "power": _power,
         "worker": _worker, "runbook": _runbook}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=sorted(STEPS))
    p.add_argument("--symbol", required=True)
    p.add_argument("--tf", required=True)
    p.add_argument("--out", required=True, type=Path, help="where audit.jsonl is appended")
    p.add_argument("--start", help="pin: the window's first UTC day, YYYY-MM-DD")
    p.add_argument("--end", help="pin: the window's last UTC day (00:00), YYYY-MM-DD")
    p.add_argument("--expect-spread-sha256", help="calibrate: the spread file's sha256 recorded before the stage")
    p.add_argument("--expect-specs-sha256", help="calibrate: the specs file's sha256 recorded before the stage")
    p.add_argument("--cross-cell", help="promote: the VALIDATED cell a GA on this cell re-tests on (ceremony data)")
    p.add_argument("--proof", type=Path, help="worker: the cell's pre-campaign proof.json")
    a = p.parse_args(argv)
    base = {"step": a.step, "symbol": a.symbol, "tf": a.tf}
    try:
        result = STEPS[a.step](a)
    except WizardRefused as exc:
        audit_row(a.out, {**base, "outcome": "REFUSED", "reason": exc.reason})
        print(f"REFUSED {a.step} {a.symbol} {a.tf}: {exc.reason}: {exc.detail}")
        return 2
    audit_row(a.out, {**base, "outcome": "DONE", **result})
    print(f"DONE {a.step} {a.symbol} {a.tf}: {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
