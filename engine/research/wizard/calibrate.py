"""W3 CALIBRATE = verify + pin (DR-2026-10-09-u4-wizard §4; AMENDMENT-U4-#2 R5: REUSE + PIN, ZERO recalibration).

The cell's existing spread matrix and specs file are re-hashed and must byte-equal the hashes the caller recorded
before the stage (passed in, never written here). Nothing is re-measured. Then:
- the specs file passes load_spec;
- the spread matrix loads (calibrator_version >= MIN, enforced by the loader), its 24 hourly mean buckets are
  non-degenerate (>= 2 distinct positive means), and the spread gate's evaluate() does not block;
- the spread-gate override setting (max_spread_points_by_symbol) is empty.
The provenance is cited from the files themselves. The product, w3_pin.json, carries no clock: it is deterministic.
Refuses: a missing file, a hash mismatch, an invalid specs or spread file, degenerate buckets, a gate block, an
override.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from engine.research.wizard.common import WizardRefused, cell_dir, cell_id, require_registry, write_product

SCHEMA = "arqvio.wizard.w3_pin"
HOURS = 24


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(symbol: str, tf: str, *, spread_sha256: str, specs_sha256: str) -> dict[str, Any]:
    """Verify and pin (symbol, tf)'s spread matrix and specs against the expected sha256s. Refuses (raises)."""
    from engine.config.settings import get_settings
    from engine.data import spread_calibration as sc
    from engine.research.data.symbol_specs import load_spec, specs_dir

    cid = cell_id(symbol, tf)
    if any(e["cell_id"] == cid for e in require_registry()):
        raise WizardRefused("cell-exists", f"{cid} is already in the registry")
    files = {"spread": sc.spread_dir() / f"{symbol}.json", "specs": specs_dir() / f"{symbol}.json"}
    absent = sorted(k for k, p in files.items() if not p.is_file())
    if absent:
        raise WizardRefused("artifact-missing", f"no {absent} file for {symbol}; W3 never calibrates (#2-R5)")
    got = {k: sha256_file(p) for k, p in files.items()}
    want = {"spread": spread_sha256, "specs": specs_sha256}
    if got != want:
        raise WizardRefused("hash-mismatch", f"sha256 {got} != the recorded {want}")
    # the loaders raise bare KeyError / AttributeError / OverflowError on a malformed file: every one is a refusal
    try:
        spec = load_spec(symbol)
    except Exception as exc:
        raise WizardRefused("specs-invalid", f"{type(exc).__name__}: {exc}") from exc
    if spec is None:
        raise WizardRefused("specs-invalid", f"load_spec({symbol!r}) found nothing")
    try:
        text = files["spread"].read_text(encoding="utf-8")
        matrix = sc.SpreadMatrix.from_json(text)
    except Exception as exc:
        raise WizardRefused("spread-invalid", f"{type(exc).__name__}: {exc}") from exc
    means = [float(m) for m in matrix.mean_buckets]
    distinct = sorted({m for m in means if m > 0.0})
    if len(means) != HOURS or len(distinct) < 2:
        raise WizardRefused("degenerate-buckets", f"{len(means)} mean buckets, {len(distinct)} distinct positive; "
                                                  "an owner ruling is needed")
    check = sc.evaluate(matrix)
    if check.blocks:
        raise WizardRefused("spread-gate-blocks", check.line())
    override = dict(get_settings().max_spread_points_by_symbol)
    if override:
        raise WizardRefused("spread-gate-override", f"max_spread_points_by_symbol is set: {override}")
    raw_spread, raw_specs = json.loads(text), json.loads(files["specs"].read_text(encoding="utf-8"))
    doc = {
        "schema": SCHEMA, "version": 1, "cell_id": cid, "symbol": symbol, "timeframe": tf,
        "recalibration": "none: REUSE + PIN (#2-R5); the files are verified byte-equal and pinned, never re-measured",
        "spread": {"file": f"<ARQVIO_HOME>/artifacts/spread/{symbol}.json", "sha256": got["spread"],
                   "provenance": {k: raw_spread.get(k) for k in ("calibrated_utc", "sample_count", "source",
                                                                  "calibrator_version", "time_basis")},
                   "evaluate": {"status": check.status.value, "blocks": check.blocks, "reason": check.reason},
                   "no_trade_hours": sorted(matrix.no_trade_hours), "mean_buckets_distinct_positive": len(distinct),
                   "spread_gate_override": override},
        "specs": {"file": f"<ARQVIO_HOME>/artifacts/specs/{symbol}.json", "sha256": got["specs"],
                  "provenance": {k: raw_specs.get(k) for k in ("snapshot_utc", "server", "account_login", "source")},
                  "load_spec": "ok"},
        "carried_question": "recalibration cadence policy (#2-R5; the S9/R1 ledger)",
    }
    write_product(cell_dir(cid) / "w3_pin.json", doc)
    return doc
