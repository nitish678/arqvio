"""W2 DATASET PIN (DR-2026-10-09-u4-wizard §4; AMENDMENT-U4-#2 R4, #3 Q2, #4 Q5).

It pulls the cell's bars on acct1 (explicit login) with the S2-D0 builder's own call (copy_rates_range over a fixed
window), classifies every gap, and pins the bar set with the unchanged save_snapshot. Refuses:
- the time basis is not server_equals_utc;
- fewer bars than the R4 threshold (the minimum over the registered cells' dataset bars, read from the registry
  now, never hard-coded);
- no trusted pin to explain gaps against, or any UNEXPLAINED gap;
- a malformed acceptance list, or a listed hole the pull does not show;
- an existing pin (the hash is computed in memory first; nothing is overwritten).

The gap rule (#3-Q2), with step = the timeframe's minutes. The window's two edges count as gaps too:
- STEP1: one step.
- WEEKEND, by rule: the bar before the gap is a Friday (UTC), the bar after is a Sunday or Monday, and the span is
  <= 72 h.
- EXPLAINED: any other gap whose missing interval overlaps a gap present in EVERY trusted pin, i.e. every other
  VALIDATED registered cell of the same timeframe. The calendar source is the registry's own pins: the MetaTrader5
  package has no broker-calendar call.
- OWNER-ACCEPTED HOLE (#4-Q5): a gap otherwise UNEXPLAINED that EXACTLY equals (missing_from, missing_to,
  bars_missing, seen_in) an entry of the cell's closed acceptance list, configs/cells/<cell_id>.accepted_holes.json:
  ceremony data written from an owner amendment, never a rate or a tolerance.
- UNEXPLAINED: anything else; the step refuses.
"""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from engine.core import paths
from engine.core.schema import same_schema
from engine.research.registry.annualisation import TF_MINUTES
from engine.research.wizard.common import (
    WizardRefused,
    attach_acct1,
    canonical,
    cell_dir,
    cell_id,
    require_registry,
    utc_now,
    write_product,
)

SCHEMA = "arqvio.wizard.w2_manifest"
HOLES_SCHEMA = "arqvio.wizard.accepted_holes"
HOLES_KEYS = {"schema", "version", "cell_id", "citation", "evidence", "holes"}
HOLE_KEYS = {"missing_from", "missing_to", "bars_missing", "seen_in", "note"}
WEEKEND_MAX_S = 72 * 3600
GAP_RULE = ("WEEKEND by rule (Friday -> Sunday/Monday UTC, <= 72 h); any other gap EXPLAINED iff it overlaps a gap "
            "in EVERY trusted pin (the registry's own pins: the MetaTrader5 package has no calendar call); a gap "
            "otherwise unexplained is an owner_accepted_hole iff it EXACTLY equals an entry of the cell's closed "
            "acceptance list (an owner amendment; no rate-based acceptance); else UNEXPLAINED and the pin refuses")


def iso(t: int) -> str:
    return datetime.fromtimestamp(int(t), tz=UTC).isoformat()


def parse_day(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=UTC)


def gaps(time: np.ndarray, step_s: int, start: int, end: int) -> list[tuple[int, int]]:
    """(last bar before, first bar after) for every jump > one step, the window's two edges included."""
    t = np.asarray(time, dtype=np.int64)
    if t.size == 0:
        return [(start - step_s, end + step_s)]
    out = [(int(a), int(b)) for a, b in zip(t[:-1], t[1:]) if b - a > step_s]
    if t[0] - start >= step_s:
        out.insert(0, (start - step_s, int(t[0])))
    if end - t[-1] >= step_s:
        out.append((int(t[-1]), end + step_s))
    return out


def is_weekend(prev_t: int, next_t: int) -> bool:
    a, b = datetime.fromtimestamp(prev_t, tz=UTC), datetime.fromtimestamp(next_t, tz=UTC)
    return a.weekday() == 4 and b.weekday() in (6, 0) and next_t - prev_t <= WEEKEND_MAX_S


def missing(gap: tuple[int, int], step_s: int) -> tuple[int, int]:
    """The interval of absent bar times, inclusive."""
    return gap[0] + step_s, gap[1] - step_s


def classify(time: np.ndarray, step_s: int, trusted: dict[str, list[tuple[int, int]]], start: int, end: int
             ) -> dict[str, Any]:
    """The weekend tally and the EXPLAINED / UNEXPLAINED gaps against the trusted pins' gaps."""
    explained, unexplained, weekend = [], [], 0
    for gap in gaps(time, step_s, start, end):
        if is_weekend(*gap):
            weekend += 1
            continue
        lo, hi = missing(gap, step_s)
        hits = {cid: [g for g in pin if missing(g, step_s)[0] <= hi and lo <= missing(g, step_s)[1]]
                for cid, pin in trusted.items()}
        rec: dict[str, Any] = {"missing_from": iso(lo), "missing_to": iso(hi), "bars_missing": (hi - lo) // step_s + 1}
        if all(hits.values()):
            rec["explained_by"] = {cid: [{"missing_from": iso(missing(g, step_s)[0]),
                                          "missing_to": iso(missing(g, step_s)[1])} for g in h]
                                   for cid, h in sorted(hits.items())}
            explained.append(rec)
        else:
            rec["seen_in"] = sorted(cid for cid, h in hits.items() if h)
            unexplained.append(rec)
    return {"weekend": weekend, "explained": explained, "unexplained": unexplained}


def accepted_holes_path(cid: str) -> Path:
    """configs/cells/<cell_id>.accepted_holes.json in the project root, resolved now."""
    return paths.PROJECT_ROOT / "configs" / "cells" / f"{cid}.accepted_holes.json"


def hole_key(rec: dict[str, Any]) -> tuple:
    return rec["missing_from"], rec["missing_to"], rec["bars_missing"], tuple(rec["seen_in"])


def load_accepted(cid: str, step_s: int) -> dict[str, Any] | None:
    """The cell's closed acceptance list, validated, or None when the cell has none."""
    path = accepted_holes_path(cid)
    if not path.is_file():
        return None
    raw = path.read_bytes()

    def bad(why: str) -> WizardRefused:
        return WizardRefused("accepted-holes-malformed", f"{path.name}: {why}")

    try:
        doc = json.loads(raw)
    except ValueError as exc:
        raise bad(f"not JSON ({exc})") from exc
    if not isinstance(doc, dict) or set(doc) != HOLES_KEYS:
        raise bad(f"top-level keys must be exactly {sorted(HOLES_KEYS)}")
    if type(doc["version"]) is not int or not same_schema(doc["schema"], HOLES_SCHEMA) or (doc["version"], doc["cell_id"]) != (1, cid):
        raise bad(f"schema/version/cell_id must be {HOLES_SCHEMA!r}/1/{cid!r}")
    if not isinstance(doc["citation"], str) or not doc["citation"].strip():
        raise bad("no citation")
    holes = doc["holes"]
    if not isinstance(holes, list) or not holes:
        raise bad("holes must be a non-empty list")
    for h in holes:
        if not isinstance(h, dict) or set(h) != HOLE_KEYS:
            raise bad(f"each hole needs exactly {sorted(HOLE_KEYS)}: {h!r}")
        s = h["seen_in"]
        if not (isinstance(s, list) and all(isinstance(c, str) for c in s) and s == sorted(set(s))):
            raise bad(f"seen_in must be a sorted list of distinct cell ids: {h!r}")
        if type(h["bars_missing"]) is not int:
            raise bad(f"bars_missing must be an integer: {h!r}")
        try:
            lo, hi = (int(datetime.fromisoformat(h[k]).timestamp()) for k in ("missing_from", "missing_to"))
        except (TypeError, ValueError) as exc:
            raise bad(f"unreadable hole times {h!r}") from exc
        if (iso(lo), iso(hi)) != (h["missing_from"], h["missing_to"]) or hi < lo:
            raise bad(f"hole times must be canonical UTC with from <= to: {h!r}")
        if h["bars_missing"] != (hi - lo) // step_s + 1 or (hi - lo) % step_s:
            raise bad(f"bars_missing does not match the interval at this timeframe: {h!r}")
    if len({hole_key(h) for h in holes}) != len(holes):
        raise bad("duplicate holes")
    # the canonical form's digest: the same on an LF or a CRLF checkout of the file
    return {"doc": doc, "file": f"configs/cells/{path.name}",
            "canonical_sha256": hashlib.sha256(canonical(doc).encode("utf-8")).hexdigest()}


def accept(unexplained: list[dict[str, Any]], accepted: dict[str, Any] | None
           ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """(owner-accepted, still unexplained, listed but absent): an unexplained gap is accepted only when it EXACTLY
    equals a listed hole."""
    if accepted is None:
        return [], unexplained, []
    listed = {hole_key(h): h for h in accepted["doc"]["holes"]}
    seen = {hole_key(r) for r in unexplained}
    owned = [{**r, "category": "owner_accepted_hole", "note": listed[hole_key(r)]["note"],
              "citation": accepted["doc"]["citation"]} for r in unexplained if hole_key(r) in listed]
    rest = [r for r in unexplained if hole_key(r) not in listed]
    return owned, rest, [h for k, h in listed.items() if k not in seen]


def trusted_pins(entries: list[dict[str, Any]], cid: str, tf: str, step_s: int, start: int, end: int
                 ) -> dict[str, list[tuple[int, int]]]:
    """The gaps inside the window of every other VALIDATED cell of this timeframe, from its verified snapshot."""
    from engine.core.errors import DataError
    from engine.research.data.dataset_snapshot import load_snapshot

    out = {}
    for e in entries:
        if e["status"] != "VALIDATED" or e["cell_id"] == cid or e["timeframe"] != tf:
            continue
        try:
            t = np.asarray(load_snapshot(e["dataset_hash16"]).time, dtype=np.int64)  # re-derives the pin's hash
        except DataError as exc:
            raise WizardRefused("trusted-pin-unverifiable", f"{e['cell_id']}: {exc}") from exc
        out[e["cell_id"]] = gaps(t[(t >= start) & (t <= end)], step_s, start, end)
    if not out:
        raise WizardRefused("no-trusted-pin", f"no other VALIDATED {tf} cell to explain gaps against")
    return out


def run(symbol: str, tf: str, start_day: str, end_day: str, *, mt5: Any) -> dict[str, Any]:
    """Pin (symbol, tf) over [start_day, end_day] (UTC days, both at 00:00). Refuses (raises) otherwise."""
    from engine.research.data.dataset_snapshot import dataset_hash, find_snapshot, save_snapshot
    from engine.research.data.time_basis import probe_basis

    cid = cell_id(symbol, tf)
    entries = require_registry()
    if any(e["cell_id"] == cid for e in entries):
        raise WizardRefused("cell-exists", f"{cid} is already in the registry")
    threshold = min(e["dataset_window"]["bars"] for e in entries if e["status"] == "VALIDATED")
    step_s = TF_MINUTES[tf] * 60
    start, end = parse_day(start_day), parse_day(end_day)
    t0, t1 = int(start.timestamp()), int(end.timestamp())
    trusted = trusted_pins(entries, cid, tf, step_s, t0, t1)
    accepted = load_accepted(cid, step_s)
    acct = attach_acct1(mt5)
    try:
        if not mt5.symbol_select(symbol, True):
            raise WizardRefused("symbol-absent", f"{symbol!r} is not selectable: {mt5.last_error()}")
        probe = probe_basis(mt5, symbol)
        if probe.basis != "server_equals_utc":
            raise WizardRefused("time-basis", f"probe_basis gave {probe.basis!r}; a pin needs server_equals_utc")
        rates = mt5.copy_rates_range(symbol, getattr(mt5, f"TIMEFRAME_{tf}"), start, end)
        pulled_utc = utc_now()
    finally:
        mt5.shutdown()
    n = 0 if rates is None else len(rates)
    if n < threshold:
        raise WizardRefused("small-dataset", f"{n} bars < the R4 threshold {threshold} (min over registered cells)")
    a = {k: np.asarray(rates[k]) for k in ("time", "open", "high", "low", "close", "spread", "tick_volume",
                                           "real_volume")}
    tally = classify(a["time"], step_s, trusted, t0, t1)
    owned, rest, absent = accept(tally["unexplained"], accepted)
    if rest:
        raise WizardRefused("unexplained-gap", f"{len(rest)} gap(s) neither explained by the trusted pins nor on an "
                                               f"acceptance list, first: {rest[:3]}")
    if absent:
        raise WizardRefused("accepted-hole-unmatched", f"{len(absent)} listed hole(s) not an unexplained gap of this "
                                                       f"pull: {absent}")
    bars = {"time": a["time"], "open_": a["open"], "high": a["high"], "low": a["low"], "close": a["close"],
            "spread": a["spread"]}
    h16 = dataset_hash(symbol=symbol, timeframe=tf, **bars)
    if find_snapshot(h16) is not None:
        raise WizardRefused("pin-exists", f"snapshot {h16} already exists; a pin is never rewritten")
    meta = save_snapshot(symbol=symbol, timeframe=tf, **bars, tick_volume=a["tick_volume"],
                         real_volume=a["real_volume"], server=acct["server"], time_basis="utc")
    npz = find_snapshot(h16)
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (npz, npz.with_suffix(".json"))}
    doc = {
        "schema": SCHEMA, "version": 1, "cell_id": cid, "symbol": symbol, "timeframe": tf,
        "window": {"requested_start": start.isoformat(), "requested_end": end.isoformat(),
                   "first_bar": meta.start_utc, "last_bar": meta.end_utc},
        "n_bars": meta.n_bars, "dataset_hash16": h16, "server": acct["server"], "pulled_utc": pulled_utc,
        "attach": {"account": acct["id"], "login_verified": True, "demo": True},
        "time_basis": probe.basis, "source": "MetaTrader5.copy_rates_range (the S2-D0 builder's pull)",
        "threshold": {"bars": threshold, "rule": "R4: min over the registered VALIDATED cells' dataset_window.bars"},
        "files_sha256": {f"<ARQVIO_HOME>/artifacts/datasets/{k}": v for k, v in sorted(files.items())},
        "gaps": {"rule": GAP_RULE, "trusted_pins": sorted(trusted),
                 "counts": {"weekend": tally["weekend"], "pin_explained": len(tally["explained"]),
                            "owner_accepted_hole": len(owned), "unexplained": 0},
                 "explained": tally["explained"], "owner_accepted_holes": owned,
                 "owner_accepted_bars": sum(r["bars_missing"] for r in owned),
                 "acceptance": None if accepted is None else {
                     "file": accepted["file"], "canonical_sha256": accepted["canonical_sha256"],
                     "citation": accepted["doc"]["citation"], "evidence": accepted["doc"]["evidence"]}},
    }
    write_product(cell_dir(cid) / "w2_manifest.json", doc)
    return doc
