"""S6 artifact records the read-only API serves, written atomically under ARQVIO_HOME/artifacts (call time).

* ``mql5/ea_registry.json``: magic -> the generated EA (range, slot, dna_hash, family, strategy_id,
  template version and digest, cell, source sha256);
* ``mql5/parity/<subject id>.json``: one subject's verdict, level by level;
* ``live/worker_status.json``: the DEMO worker's latest state and reconciliation (written by the worker).
"""
from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from engine.core.paths import get_artifacts_dir
from engine.research.mql5.magic import owner_of


def ea_registry_path() -> Path:
    return get_artifacts_dir() / "mql5" / "ea_registry.json"


def parity_dir() -> Path:
    return get_artifacts_dir() / "mql5" / "parity"


def worker_status_path() -> Path:
    return get_artifacts_dir() / "live" / "worker_status.json"


def write_json_atomic(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path


def record_ea(ea: Any) -> Path:
    """Add (or refresh) one generated EA in the registry, keyed by its magic and cell."""
    path = ea_registry_path()
    reg = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"schema": "arqvio.s6.ea_registry", "eas": {}}
    owner = owner_of(ea.magic)
    reg["eas"][f"{ea.magic}:{ea.cell_name}"] = {
        "magic": ea.magic, "range": None if owner is None else owner[0], "slot": None if owner is None else owner[1],
        "dna_hash": ea.dna_hash, "family": ea.family, "strategy_id": ea.strategy_id, "cell": ea.cell_name,
        "template_version": ea.template_version, "template_digest": ea.template_digest,
        "source_sha256": ea.source_sha256, "name": ea.name,
    }
    return write_json_atomic(path, reg)


def record_parity(sid: str, note: str, cell_name: str, run_id: str, verdict: Any) -> Path:
    return write_json_atomic(parity_dir() / f"{sid}.json", {
        "schema": "arqvio.s6.parity_subject", "subject": sid, "note": note, "cell": cell_name, "run_id": run_id,
        "ok": verdict.ok, "recorded_utc": datetime.now(tz=UTC).isoformat(),
        "levels": [{"level": lv.name, "ok": lv.ok, "summary": lv.summary, "mismatches": len(lv.lines),
                    "compared": lv.compared, "bit_exact": lv.exact, "within_tolerance": len(lv.inexact)}
                   for lv in verdict.levels],
    })


__all__ = ["ea_registry_path", "parity_dir", "record_ea", "record_parity", "worker_status_path", "write_json_atomic"]
