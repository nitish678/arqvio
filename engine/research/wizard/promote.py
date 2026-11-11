"""W4 REGISTRY PROMOTE (DR-2026-10-09-u4-wizard §4-§5; ARQVIO_U4_PROMPT_v2_EN §3 W4-b).

The new VALIDATED entry is built from the cell's W1-W3 products, each re-verified against the files now on disk:
- the W2 pin: load_snapshot re-derives its hash; the window is the sidecar's; no gap unexplained;
- the W3 pin: the spread and specs files still hash to what W3 pinned;
- the target-audit row for (symbol, timeframe) is unique, with its AS_OF line;
- the cross cell (ceremony data, --cross-cell) is another VALIDATED cell.
The resting states are the plan page's (§5): power_check UNKNOWN (W5 decides), worker_status UNKNOWN (W6 fills it),
campaign_status UNRUN. The entry is appended through the registry ceremony: every existing entry stays byte-equal.
Refuses: a registered cell, an invalid cross cell, a missing or contradictory product, an audit row not unique.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from engine.research.wizard.common import (
    WizardRefused,
    cell_dir,
    cell_id,
    registry_ceremony,
    require_registry,
    utc_now,
)

PRODUCTS = ("w1_audit.json", "w2_manifest.json", "w3_pin.json")
AUDIT_FILE = "history_depth_audit_target.csv"


def sha256_file(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_row(symbol: str, tf: str) -> dict[str, Any]:
    """The unique target-audit row for (symbol, tf), as the registry cites it."""
    from engine.core.paths import get_artifacts_dir

    p = get_artifacts_dir() / AUDIT_FILE
    if not p.is_file():
        raise WizardRefused("audit-row", f"no target audit at {p}")
    lines = p.read_text(encoding="utf-8").splitlines()
    rows = [n for n, line in enumerate(lines, 1) if line.startswith(f"{symbol},{tf},")]
    as_of = [line[len("# AS_OF "):] for line in lines if line.startswith("# AS_OF ")]
    if len(rows) != 1 or len(as_of) != 1:
        raise WizardRefused("audit-row", f"{len(rows)} rows for {symbol},{tf} and {len(as_of)} AS_OF lines; "
                                         "the registry cites exactly one of each")
    return {"file": f"<ARQVIO_HOME>/artifacts/{AUDIT_FILE}", "line": rows[0], "as_of": as_of[0],
            "row": lines[rows[0] - 1]}


def products(cid: str, symbol: str, tf: str) -> dict[str, dict[str, Any]]:
    out = {}
    for name in PRODUCTS:
        p = cell_dir(cid) / name
        if not p.is_file():
            raise WizardRefused("product-missing", f"{p} is absent; run the wizard's earlier steps first")
        doc = json.loads(p.read_text(encoding="utf-8"))
        if (doc.get("cell_id"), doc.get("symbol"), doc.get("timeframe")) != (cid, symbol, tf):
            raise WizardRefused("product-contradiction", f"{name} is for {doc.get('cell_id')!r}, not {cid!r}")
        out[name] = doc
    return out


def build_entry(symbol: str, tf: str, cross_cell: str, entries: list[dict[str, Any]]) -> dict[str, Any]:
    from engine.core.errors import DataError
    from engine.data.spread_calibration import spread_dir
    from engine.research.data.dataset_snapshot import find_snapshot, load_meta, load_snapshot
    from engine.research.data.symbol_specs import specs_dir
    from engine.research.registry.annualisation import TF_MINUTES, bars_per_year
    from engine.research.registry.cells import SCHEMA_VERSION, _ENTRY_FIELDS
    from engine.research.wizard.pin import iso, load_accepted

    cid = cell_id(symbol, tf)
    if any(e["cell_id"] == cid for e in entries):
        raise WizardRefused("cell-exists", f"{cid} is already in the registry")
    validated = {e["cell_id"] for e in entries if e["status"] == "VALIDATED"}
    if cross_cell not in validated:
        raise WizardRefused("cross-cell-invalid", f"{cross_cell!r} is not a VALIDATED cell ({sorted(validated)})")
    prod = products(cid, symbol, tf)
    w2, w3 = prod["w2_manifest.json"], prod["w3_pin.json"]
    h16 = w2["dataset_hash16"]
    try:
        bars = load_snapshot(h16)
    except DataError as exc:
        raise WizardRefused("product-contradiction", f"the W2 pin {h16} does not verify: {exc}") from exc
    meta, npz = load_meta(h16), find_snapshot(h16)
    t = bars.time
    window = {"start": iso(t[0]), "end": iso(t[-1]), "bars": int(len(t))}  # from the verified bars themselves
    pinned_files = {f"<ARQVIO_HOME>/artifacts/datasets/{p.name}": sha256_file(p)
                    for p in (npz, npz.with_suffix(".json"))}
    if window != {"start": meta.start_utc, "end": meta.end_utc, "bars": meta.n_bars} \
            or (meta.symbol, meta.timeframe, window["bars"]) != (symbol, tf, w2["n_bars"]) \
            or (w2["window"]["first_bar"], w2["window"]["last_bar"]) != (window["start"], window["end"]) \
            or w2["gaps"]["counts"]["unexplained"] != 0 or pinned_files != w2["files_sha256"]:
        raise WizardRefused("product-contradiction", f"the W2 manifest, the pin {h16} and its sidecar disagree")
    acceptance = w2["gaps"]["acceptance"]
    if acceptance is not None:
        listed = load_accepted(cid, TF_MINUTES[tf] * 60)
        if listed is None or (listed["file"], listed["canonical_sha256"]) != (acceptance["file"],
                                                                              acceptance["canonical_sha256"]):
            raise WizardRefused("product-contradiction", "the accepted-holes list is not the one W2 applied")
    files = {"spread": spread_dir() / f"{symbol}.json", "specs": specs_dir() / f"{symbol}.json"}
    now = {k: sha256_file(p) if p.is_file() else None for k, p in files.items()}
    if now != {"spread": w3["spread"]["sha256"], "specs": w3["specs"]["sha256"]}:
        raise WizardRefused("product-contradiction", f"the files now hash {now}, not what W3 pinned")
    cells_ref = f"<ARQVIO_HOME>/artifacts/cells/{cid}"
    entry = {
        "schema_version": SCHEMA_VERSION, "cell_id": cid, "status": "VALIDATED", "symbol": symbol, "timeframe": tf,
        "dataset_hash16": h16, "dataset_window": window, "cross_cell": cross_cell,
        "annualisation_ppy": bars_per_year(tf),
        "spread_artifact_hash": {"sha256": now["spread"], "source": (
            f"sha256 of <ARQVIO_HOME>/artifacts/spread/{symbol}.json, verified against the hash recorded before U4 and "
            f"pinned by the wizard's W3 step ({cells_ref}/w3_pin.json; calibrated_utc "
            f"{w3['spread']['provenance']['calibrated_utc']}); reused, not recalibrated")},
        "specs_artifact_hash": {"sha256": now["specs"], "source": (
            f"sha256 of <ARQVIO_HOME>/artifacts/specs/{symbol}.json, verified against the hash recorded before U4 and "
            f"pinned by the wizard's W3 step ({cells_ref}/w3_pin.json; snapshot_utc "
            f"{w3['specs']['provenance']['snapshot_utc']}); reused, not re-read")},
        "audit_row_ref": audit_row(symbol, tf),
        "power_check": {"status": "UNKNOWN", "ref": "U4 W5"},
        "worker_status": {"status": "UNKNOWN", "fills_in": "U4",
                          "note": "the U4 W6 pre-campaign re-proof fills this in"},
        "campaign_status": "UNRUN", "entered_by_stage": "U4", "entered_at": utc_now(),
        "evidence": ["docs/decisions/DR-2026-10-09-u4-wizard.md", *(f"{cells_ref}/{n}" for n in PRODUCTS)],
    }
    if acceptance is not None:
        entry["evidence"].append(acceptance["file"])
    return {k: entry[k] for k in _ENTRY_FIELDS if k != "entry_hash"}


def run(symbol: str, tf: str, *, cross_cell: str) -> dict[str, Any]:
    """Append (symbol, tf)'s VALIDATED entry through the guarded ceremony. Refuses (raises) otherwise."""
    entries = require_registry()
    try:
        entry = build_entry(symbol, tf, cross_cell, entries)
    except WizardRefused:
        raise
    except Exception as exc:  # a malformed product, sidecar or audit file is a refusal, never a crash
        raise WizardRefused("product-contradiction", f"{type(exc).__name__}: {exc}") from exc
    result = registry_ceremony(lambda es: [*es, entry], allowed={}, new=frozenset({entry["cell_id"]}))
    return {"cell_id": entry["cell_id"], "entry": entry, **result}
