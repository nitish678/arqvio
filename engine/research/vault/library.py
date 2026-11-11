"""The strategy vault: every bred strategy worth keeping, scored, graded and tiered from its own record.

Entries come from completed GA campaigns (``<ARQVIO_HOME>/artifacts/ga/<campaign>/manifest_v2.json``): importing
a campaign copies each finalist alive on its primary cell, with the campaign id, the candidate index and the
manifest's sha256, so every number in the vault traces to the record it was read from.

Two tiers, decided by the gates below and never by hand:

    CANDIDATE   alive on its cell; fast to get -- the score says how good it looks
    CERTIFIED   every strict gate passes: DSR, sham, cross cell, trade count, drawdown and the three
                robustness checks (engine/research/vault/robustness.py) -- the bar for trusting the edge

The tier is a label, not a gate: the owner may deploy either (the deploy records which it was).

Score v2 (0-100) = Sharpe 20 + drawdown 10 + trades 10 + sham margin 10 + cross-cell Sharpe 10 + robustness 15 +
held-out persistence 20 + profit spread 5. Robustness 15 = neighbours that make money 3 + plateau 2 + time segments
3 + bootstrap P(Sharpe <= 0) 2 + Monte Carlo p95 drawdown 1 + beats random-entry twins 4. A part counts only once its
check has run (robustness and profit spread with the robustness run, persistence with the held-out test; a
robustness run from before v4 has no plateau or random-twin part), so the score says out of how much it is.
Grade: A >= 80, B >= 65, C >= 50, D >= 35, else F.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from engine.core import atomic
from engine.core.paths import get_artifacts_dir
from engine.core.schema import same_schema
from engine.research.ga.config import CAMPAIGN_GATES

SCHEMA = "arqvio.vault"
VERSION = 1
TIERS = ("CANDIDATE", "CERTIFIED", "UNCERTIFIED")  # UNCERTIFIED: an engine V1 (research engine) strategy
MIN_TRADES_CERTIFIED = 100
MAX_DD_CERTIFIED = 0.30
# v4 (ARQVIO_V4.md 10.2): no single month may carry more than half the profit, and the strategy must hold a
# position in at least 40% of the months of its window.
MAX_MONTH_SHARE = 0.50
MIN_MONTHS_IN_MARKET = 0.40
MIN_HELD_OUT_TRADES = 15  # engine/research/vault/held_out.py: fewer held-out trades judge nothing


def min_held_out_trades() -> int:
    """SETTINGS OOS TRADES (engine/core/app_settings.py campaign.held_out_min_trades; default MIN_HELD_OUT_TRADES)."""
    from engine.core import app_settings

    return int(app_settings.load()["campaign"]["held_out_min_trades"])
NEIGHBOUR_SHARE_MIN = 0.70
BOOTSTRAP_P_MAX = 0.10
# v4 (ARQVIO_V4.md 10.3). Calibrated on the owner's 5 robustness-checked strategies (2026-11-01): plateau 0.67-1.0,
# Monte Carlo p95 drawdown 0.11-0.40 for a realised 0.09-0.21. The p95 bar is the realised bar x 4/3: a resampled
# path may go deeper than the one history dealt, not unboundedly so. Beating 95 of 100 random-entry twins is the
# conventional one-sided 5% level.
PLATEAU_MIN = 0.50
MC_DD_P95_MAX = 0.40
BEAT_RANDOM_MIN = 0.95
GRADES = ((80.0, "A"), (65.0, "B"), (50.0, "C"), (35.0, "D"))


class VaultError(ValueError):
    """A vault operation refused; nothing was written."""


def vault_dir() -> Path:
    return get_artifacts_dir() / "vault"


def vault_path() -> Path:
    return vault_dir() / "strategies.json"


def ga_dir() -> Path:
    return get_artifacts_dir() / "ga"


def utc_now() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds")


def load(path: Path | None = None) -> list[dict[str, Any]]:
    p = path or vault_path()
    if not p.is_file():
        return []
    doc = json.loads(p.read_text(encoding="utf-8"))
    if not same_schema(doc.get("schema"), SCHEMA) or doc.get("version") != VERSION or not isinstance(doc.get("strategies"), list):
        raise VaultError(f"{p} is not a {SCHEMA} v{VERSION} document")
    return doc["strategies"]


def save(rows: list[dict[str, Any]], path: Path | None = None) -> None:
    p = path or vault_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"schema": SCHEMA, "version": VERSION, "strategies": rows}, indent=2, sort_keys=True)
                   + "\n", encoding="utf-8")
    atomic.replace(tmp, p)


def get(entry_id: str, path: Path | None = None) -> dict[str, Any]:
    for r in load(path):
        if r["id"] == entry_id:
            return r
    raise VaultError(f"no strategy {entry_id!r} in the vault")


def completed_campaigns() -> list[dict[str, Any]]:
    """Every GA campaign directory with a manifest v2 (completed runs only write one)."""
    out = []
    for m in sorted(ga_dir().glob("*/manifest_v2.json")) if ga_dir().is_dir() else ():
        try:
            doc = json.loads(m.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        out.append({"campaign_id": m.parent.name, "cells": doc.get("cells"), "candidates": len(doc.get("candidates", [])),
                    "completed_at_utc": (doc.get("run") or {}).get("completed_at_utc")})
    return out


def _finite(x: Any) -> float | None:
    return float(x) if isinstance(x, int | float) and not isinstance(x, bool) and math.isfinite(x) else None


def entry_from_candidate(campaign_id: str, doc: dict[str, Any], index: int, manifest_sha256: str) -> dict[str, Any]:
    from engine.research.ga.config import PINNED_CELLS

    c = doc["candidates"][index]
    primary, cross = doc["cells"]["primary"], doc["cells"]["cross"]
    mp, mc = c["metrics"][primary], c["metrics"][cross]
    fp, fc = mp["fitness"], mc["fitness"]
    symbol, timeframe, dataset_id = PINNED_CELLS.get(primary, (None, None, None))
    final = (c.get("dsr") or {}).get("primary_final") or {}
    metrics = {
        "sharpe": _finite(fp.get("sharpe")), "n_trades": int(mp.get("n_trades") or 0), "max_dd": _finite(mp.get("max_dd")),
        "total_return": _finite(mp.get("total_return")), "killed_by": fp.get("killed_by"),
        "fitness": _finite(fp.get("fitness")), "sham_margin": _finite((c.get("sham_margin") or {}).get(primary)),
        "sham_passed": fp.get("sham_passed"), "dsr_final": _finite(final.get("dsr")), "dsr_status": final.get("status"),
        "dsr_n_trials": final.get("n_trials"), "cross_sharpe": _finite(fc.get("sharpe")),
        "cross_killed_by": fc.get("killed_by"), "cross_n_trades": int(mc.get("n_trades") or 0),
        "cross_sham_margin": _finite((c.get("sham_margin") or {}).get(cross)),
        "alive_on_both": bool((c.get("cross_cell_delta") or {}).get("alive_on_both")),
        "engine": mp.get("engine"), "costs": mp.get("costs"),
    }
    entry = {"id": f"{c['dna_hash']}@{primary}", "dna": c["dna"], "dna_hash": c["dna_hash"], "family": c["dna"]["signal"],
             "cell": primary, "cross_cell": cross, "symbol": symbol, "timeframe": timeframe, "dataset_id": dataset_id,
             "source": {"campaign_id": campaign_id, "candidate_index": index, "manifest_sha256": manifest_sha256,
                        "run_hash": c.get("run_hash"), "config_hash": c.get("config_hash")},
             "metrics": metrics, "robustness": None, "added_utc": utc_now(),
             "cost_model": cost_model(doc, primary)}
    return {**entry, **assess(entry)}


def cost_model(doc: dict[str, Any], cell: str) -> str:
    """"bar" when the campaign charged fills the bar's own spread (v4 1.2), else "hourly" (every campaign before)."""
    src = str((((doc.get("run") or {}).get("cells") or {}).get(cell) or {}).get("spread_cost_source", ""))
    return "bar" if src.startswith("bar_spread") else "hourly"


def recorded_cost_model(entry: dict[str, Any]) -> str:
    """The cost model a vault strategy's record was made with; entries from before v4 1.2 carry none: hourly."""
    return entry.get("cost_model") or "hourly"


def import_campaign(campaign_id: str, path: Path | None = None) -> dict[str, Any]:
    """Add every finalist of ``campaign_id`` alive on its primary cell; already-vaulted ones are kept as they are."""
    mpath = ga_dir() / campaign_id / "manifest_v2.json"
    if "/" in campaign_id or "\\" in campaign_id or ".." in campaign_id or not mpath.is_file():
        raise VaultError(f"no completed campaign {campaign_id!r} (no manifest_v2.json)")
    raw = mpath.read_bytes()
    doc = json.loads(raw)
    sha = hashlib.sha256(raw).hexdigest()
    rows = load(path)
    have = {r["id"] for r in rows}
    added, skipped = [], []
    for i, c in enumerate(doc.get("candidates", [])):
        primary = doc["cells"]["primary"]
        if c["metrics"][primary]["fitness"].get("killed_by") is not None:
            skipped.append({"index": i, "reason": f"killed by {c['metrics'][primary]['fitness']['killed_by']}"})
            continue
        e = entry_from_candidate(campaign_id, doc, i, sha)
        if e["id"] in have:
            skipped.append({"index": i, "reason": "already in the vault"})
            continue
        rows.append(e)
        have.add(e["id"])
        added.append(e["id"])
    save(rows, path)
    return {"campaign_id": campaign_id, "added": added, "skipped": skipped}


def entry_from_v1(campaign_id: str, doc: dict[str, Any], gate: dict[str, Any], sha: str) -> dict[str, Any]:
    """One passing V1 finalist as a vault entry (engine "v1"; the fields ARQVIO's gates read are None)."""
    camp = doc["campaign"]
    f = next(x for x in doc["run"]["finalists"] if x["dna_hash"] == gate["dna_hash"])
    full, ho = gate["full"], gate["held_out"]
    metrics = {
        "sharpe": _finite(full["sharpe"]), "n_trades": int(full["n_trades"]), "max_dd": _finite(full["max_dd"]),
        "total_return": _finite(full["total_return"]), "killed_by": None, "fitness": _finite(f["fit_a"] + f["fit_b"]),
        "sham_margin": None, "sham_passed": None, "dsr_final": None, "dsr_status": None, "dsr_n_trials": None,
        "cross_sharpe": None, "cross_killed_by": None, "cross_n_trades": 0, "cross_sham_margin": None,
        "alive_on_both": False, "engine": f["engine"], "costs": None, "profit_factor": _finite(full["profit_factor"]),
    }
    entry = {"id": f"{f['dna_hash']}@{camp['cell']}", "engine": "v1", "dna": f["dna"], "dna_hash": f["dna_hash"],
             "family": f["dna"]["signal"], "cell": camp["cell"], "cross_cell": None, "symbol": camp["symbol"],
             "timeframe": camp["timeframe"], "dataset_id": camp["dataset_id"], "tribe_origin": f["origin"] or "",
             "lineage": {"generation": f["generation"], "parents": f["parents"], "operators": f["operators"],
                         "source": f["source"]},
             "source": {"campaign_id": campaign_id, "engine": "v1", "manifest_sha256": sha},
             "metrics": metrics, "robustness": None, "added_utc": utc_now(),
             "held_out": {"state": "DONE", "split_time": gate.get("split_time"), "return": ho["total_return"],
                          "sharpe": ho["sharpe"], "max_dd": ho["max_dd"], "trades": ho["n_trades"]},
             "cost_model": "bar" if str(camp.get("spread_cost_source") or "").startswith("bar_spread") else "hourly",
             "v1": {k: gate[k] for k in ("checks", "score", "verdict", "parts", "in_sample", "random", "reshuffled",
                                         "nudged")}}
    return {**entry, **assess(entry)}


def import_v1(campaign_id: str, path: Path | None = None) -> dict[str, Any]:
    """Add every V1 finalist that passed the research vault gates (ga_dir()/<campaign_id>/v1_result.json)."""
    rpath = ga_dir() / campaign_id / "v1_result.json"
    if "/" in campaign_id or "\\" in campaign_id or ".." in campaign_id or not rpath.is_file():
        raise VaultError(f"no completed V1 campaign {campaign_id!r} (no v1_result.json)")
    raw = rpath.read_bytes()
    doc = json.loads(raw)
    sha = hashlib.sha256(raw).hexdigest()
    rows = load(path)
    have = {r["id"] for r in rows}
    added, skipped = [], []
    for g in doc.get("gates", []):
        if not g.get("passed"):
            skipped.append({"dna_hash": g["dna_hash"], "reason": "failed " + ",".join(g.get("failed") or [])})
            continue
        e = entry_from_v1(campaign_id, doc, g, sha)
        if e["id"] in have:
            skipped.append({"dna_hash": g["dna_hash"], "reason": "already in the vault"})
            continue
        rows.append(e)
        have.add(e["id"])
        added.append(e["id"])
    save(rows, path)
    return {"campaign_id": campaign_id, "added": added, "skipped": skipped}


def v1_trial_sharpes(entry: dict[str, Any]) -> list[float | None]:
    """The Sharpe of every strategy the V1 campaign that bred ``entry`` tested (its checkpoint archive): the trials
    a CERTIFY's DSR is set against. CERTIFY itself is a VALIDATION_RUN and adds none (worker4/02_PLAN.md §6)."""
    p = ga_dir() / entry["source"]["campaign_id"] / "v1_checkpoint.json"
    doc = json.loads(p.read_text(encoding="utf-8"))
    return [a.get("engine_sharpe", (a.get("metrics") or {}).get("sharpe")) for a in doc["archive"]]


def certify_metrics(entry: dict[str, Any], *, primary: Any, cross: Any, dsr: Any, sham_sharpe: float,
                    n_trials: int, cross_cell: str) -> dict[str, Any]:
    """An engine V1 strategy run through engine V2's funnel: the bred-window numbers, the cross market, the sham and
    the DSR against its own campaign's trials. ``primary`` / ``cross`` are Evaluations; the entry's assess then uses
    ARQVIO's gates (robustness and held-out included), so CERTIFIED needs the robustness run as for any V2 strategy."""
    from engine.research.ga.fitness import early_kill

    pr, xr = primary.result, cross.result
    p_kill = early_kill(pr, crashed=primary.crashed, gates=CAMPAIGN_GATES)
    x_kill = early_kill(xr, crashed=cross.crashed, gates=CAMPAIGN_GATES)
    sharpe = _finite(pr.sharpe)
    m = {**entry["metrics"],
         "sharpe": sharpe, "n_trades": int(pr.n_trades or 0), "max_dd": _finite(pr.max_dd),
         "total_return": _finite(pr.total_return), "killed_by": None if p_kill is None else p_kill.killed_by,
         "sham_margin": None if sharpe is None else _finite(sharpe - sham_sharpe),
         "sham_passed": None if sharpe is None else (sharpe - sham_sharpe) >= CAMPAIGN_GATES.sham_margin_min,
         "dsr_final": _finite(dsr.dsr), "dsr_status": dsr.status, "dsr_n_trials": int(n_trials),
         "cross_sharpe": _finite(xr.sharpe), "cross_killed_by": None if x_kill is None else x_kill.killed_by,
         "cross_n_trades": int(xr.n_trades or 0), "alive_on_both": p_kill is None and x_kill is None,
         "engine": primary.engine}
    out = {**entry, "metrics": m, "cross_cell": cross_cell,
           "certify": {"state": "DONE", "utc": utc_now(), "kind": "VALIDATION_RUN", "dsr_n_trials": int(n_trials),
                       "cross_cell": cross_cell}}
    return {**out, **assess(out)}


def set_certify(entry_id: str, updated: dict[str, Any] | None, *, state: str | None = None,
                reason: str | None = None, path: Path | None = None) -> dict[str, Any]:
    rows = load(path)
    for i, r in enumerate(rows):
        if r["id"] == entry_id:
            if updated is not None:
                rows[i] = updated
            else:
                r["certify"] = {"state": state, "reason": reason, "utc": utc_now()}
                r.update(assess(r))
            save(rows, path)
            return rows[i]
    raise VaultError(f"no vault strategy {entry_id!r}")


RECYCLE_DAYS = 30


def recycle_path() -> Path:
    return vault_dir() / "recycle.json"


def recycled() -> list[dict[str, Any]]:
    """Strategies deleted in the last RECYCLE_DAYS days, newest first; older ones are gone for good."""
    p = recycle_path()
    if not p.is_file():
        return []
    rows = json.loads(p.read_text(encoding="utf-8")).get("strategies", [])
    cutoff = datetime.now(tz=UTC).timestamp() - RECYCLE_DAYS * 86400
    keep = [r for r in rows if datetime.fromisoformat(r["deleted_utc"]).timestamp() >= cutoff]
    return sorted(keep, key=lambda r: r["deleted_utc"], reverse=True)


def _save_recycle(rows: list[dict[str, Any]]) -> None:
    p = recycle_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"schema": "arqvio.vault_recycle", "version": 1, "strategies": rows}, indent=2,
                              sort_keys=True) + "\n", encoding="utf-8")
    atomic.replace(tmp, p)


def audit_path() -> Path:
    return vault_dir() / "audit.jsonl"


def audit(action: str, ids: list[str], reason: str | None = None) -> None:
    """One line per vault change that removes or brings back strategies (worker4 W4-5 5.3: delete audit log).
    Append-only; a failed write never blocks the change itself."""
    line = json.dumps({"utc": utc_now(), "action": action, "ids": list(ids), "reason": reason}, sort_keys=True)
    try:
        audit_path().parent.mkdir(parents=True, exist_ok=True)
        with audit_path().open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def audit_log(limit: int = 200) -> list[dict[str, Any]]:
    try:
        lines = audit_path().read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out = []
    for ln in lines[-limit:]:
        try:
            out.append(json.loads(ln))
        except ValueError:
            continue  # a torn last line from a crash mid-write: skipped, never fatal
    return out


def address(entry: dict[str, Any]) -> dict[str, Any]:
    """Where every number of ``entry`` comes from (W4-5 5.7): the campaign's folder, its bars and when it was made."""
    src = entry.get("source") or {}
    cid = src.get("campaign_id")
    return {"campaign_dir": str(ga_dir() / cid) if cid else None, "campaign_id": cid,
            "file": "v1_result.json" if entry.get("engine") == "v1" else "manifest_v2.json",
            "sha256": src.get("manifest_sha256"), "dataset_id": entry.get("dataset_id"), "cell": entry.get("cell"),
            "as_of": entry.get("added_utc"), "cost_model": recorded_cost_model(entry)}


def export(ids: list[str] | None = None, path: Path | None = None) -> dict[str, Any]:
    """The vault (or the named strategies) as one JSON document (W4-5 5.3 GET JSON)."""
    rows = load(path)
    if ids:
        want = set(ids)
        rows = [r for r in rows if r["id"] in want]
        missing = sorted(want - {r["id"] for r in rows})
        if missing:
            raise VaultError(f"no strategy {missing[0]!r} in the vault")
    return {"schema": "arqvio.vault_export", "version": 1, "exported_utc": utc_now(), "count": len(rows),
            "strategies": [{**r, "address": address(r)} for r in rows]}


def remove(entry_id: str, path: Path | None = None, *, reason: str = "deleted") -> None:
    """Move a strategy to the recycle bin (restorable for RECYCLE_DAYS days)."""
    remove_many([entry_id], path, reason=reason)


def remove_many(ids: list[str], path: Path | None = None, *, reason: str) -> list[str]:
    rows = load(path)
    want = set(ids)
    gone = [r for r in rows if r["id"] in want]
    if len(gone) != len(want):
        missing = sorted(want - {r["id"] for r in gone})
        raise VaultError(f"no strategy {missing[0]!r} in the vault")
    bin_rows = [r for r in recycled() if r["id"] not in want]
    now = utc_now()
    _save_recycle([*[{**r, "deleted_utc": now, "deleted_reason": reason} for r in gone], *bin_rows])
    save([r for r in rows if r["id"] not in want], path)
    audit("recycle", [r["id"] for r in gone], reason)
    return [r["id"] for r in gone]


def restore(entry_id: str, path: Path | None = None) -> dict[str, Any]:
    bin_rows = recycled()
    hit = next((r for r in bin_rows if r["id"] == entry_id), None)
    if hit is None:
        raise VaultError(f"no strategy {entry_id!r} in the recycle bin")
    rows = load(path)
    if any(r["id"] == entry_id for r in rows):
        raise VaultError(f"{entry_id!r} is in the vault already")
    back = {k: v for k, v in hit.items() if k not in ("deleted_utc", "deleted_reason")}
    save([*rows, back], path)
    _save_recycle([r for r in bin_rows if r["id"] != entry_id])
    audit("restore", [entry_id])
    return back


# Purge rules: each names what it removes; purge previews by default and moves to the recycle bin when applied.
PURGE_RULES: dict[str, str] = {
    "grade_f": "grade F",
    "held_out_failed": "failed the held-out test (return <= 0 or under 15 held-out trades)",
    "copies": "copies: made exactly the same trades as a better-ranked strategy",
}


def _purge_match(rule: str, rows: list[dict[str, Any]]) -> list[str]:
    if rule == "grade_f":
        return [r["id"] for r in rows if r.get("grade") == "F"]
    if rule == "held_out_failed":
        return [r["id"] for r in rows if (r.get("held_out") or {}).get("state") == "DONE"
                and ((r["held_out"].get("return") or 0.0) <= 0 or (r["held_out"].get("trades") or 0) < min_held_out_trades())]
    if rule == "copies":
        ranked = sorted(rows, key=lambda e: (-(e["tier"] == "CERTIFIED"), -e["score"], e["id"]))
        dup = copies(ranked)
        return [i for i, d in dup.items() if d["copy_of"]]
    raise VaultError(f"unknown purge rule {rule!r}; one of {sorted(PURGE_RULES)}")


def purge(rule: str, *, apply: bool = False, path: Path | None = None) -> dict[str, Any]:
    rows = load(path)
    ids = _purge_match(rule, rows)
    if apply and ids:
        remove_many(ids, path, reason=f"purge: {PURGE_RULES[rule]}")
    return {"rule": rule, "what": PURGE_RULES[rule], "ids": ids, "count": len(ids), "applied": bool(apply and ids)}


def set_robustness(entry_id: str, result: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
    rows = load(path)
    for r in rows:
        if r["id"] == entry_id:
            r["robustness"] = {**result, "utc": utc_now()}
            r.update(assess(r))
            save(rows, path)
            return r
    raise VaultError(f"no strategy {entry_id!r} in the vault")


def set_parity(entry_id: str, result: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
    """The strategy's Strategy Tester parity test (vault/parity_job.py): grade A-F, levels, run id."""
    rows = load(path)
    for r in rows:
        if r["id"] == entry_id:
            r["parity"] = {**result, "utc": result.get("utc") or utc_now()}
            save(rows, path)
            return r
    raise VaultError(f"no strategy {entry_id!r} in the vault")


def set_held_out(entry_id: str, result: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
    rows = load(path)
    for r in rows:
        if r["id"] == entry_id:
            r["held_out"] = {**result, "utc": utc_now()}
            r.update(assess(r))
            save(rows, path)
            return r
    raise VaultError(f"no strategy {entry_id!r} in the vault")


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def gates(entry: dict[str, Any]) -> list[dict[str, Any]]:
    """The CERTIFIED gates, each with its value and its bar; passed is None when it cannot be judged yet."""
    m = entry["metrics"]
    rb = entry.get("robustness") or {}
    done = rb.get("state") == "DONE"
    nb, seg, bs = rb.get("neighbourhood") or {}, rb.get("segments") or {}, rb.get("bootstrap") or {}
    conc = (rb.get("concentration") or {}) if done else {}
    ho = entry.get("held_out") or {}
    ho_done = ho.get("state") == "DONE"
    has_conc = bool(conc) and conc.get("months", 0) > 0  # a check run before v4 has none: re-run robustness
    plateau = nb.get("plateau_share") if done else None
    mc_dd = _finite(bs.get("max_dd_p95")) if done else None
    rnd = (rb.get("random") or {}).get("beaten_share") if done else None

    def g(name: str, value: Any, need: str, passed: bool | None) -> dict[str, Any]:
        return {"gate": name, "value": value, "need": need, "passed": passed}

    out = [
        g("alive on both cells", m["alive_on_both"], "true", m["alive_on_both"]),
        g("Sharpe", m["sharpe"], "> 0", m["sharpe"] is not None and m["sharpe"] > 0),
        g("trades", m["n_trades"], f">= {MIN_TRADES_CERTIFIED}", m["n_trades"] >= MIN_TRADES_CERTIFIED),
        g("max drawdown", m["max_dd"], f"<= {MAX_DD_CERTIFIED:.0%}", m["max_dd"] is not None and m["max_dd"] <= MAX_DD_CERTIFIED),
        g("deflated Sharpe (DSR)", m["dsr_final"], f">= {CAMPAIGN_GATES.dsr_min}",
          m["dsr_final"] is not None and m["dsr_final"] >= CAMPAIGN_GATES.dsr_min),
        g("beats its random-entry sham", m["sham_margin"], f">= {CAMPAIGN_GATES.sham_margin_min}",
          m["sham_margin"] is not None and m["sham_margin"] >= CAMPAIGN_GATES.sham_margin_min),
        g("cross-cell Sharpe", m["cross_sharpe"], "> 0", m["cross_sharpe"] is not None and m["cross_sharpe"] > 0),
        g("reproduces exactly", (rb.get("reproduction") or {}).get("identical") if done else None, "true",
          bool((rb.get("reproduction") or {}).get("identical")) if done else None),
        g("neighbours that make money", nb.get("positive_share") if done else None, f">= {NEIGHBOUR_SHARE_MIN:.0%}",
          (nb.get("positive_share", 0.0) >= NEIGHBOUR_SHARE_MIN) if done else None),
        g("every time segment positive", seg.get("sharpe") if done else None, "all > 0",
          all(s > 0 for s in seg.get("sharpe", [])) if done and seg.get("sharpe") else (False if done else None)),
        g("bootstrap P(Sharpe <= 0)", bs.get("p_sharpe_le_0") if done else None, f"<= {BOOTSTRAP_P_MAX:.0%}",
          (bs.get("p_sharpe_le_0", 1.0) <= BOOTSTRAP_P_MAX) if done else None),
        g("plateau: neighbours keeping half the Sharpe", plateau, f">= {PLATEAU_MIN:.0%}",
          None if plateau is None else plateau >= PLATEAU_MIN),
        g("Monte Carlo p95 drawdown", mc_dd, f"<= {MC_DD_P95_MAX:.0%}", None if mc_dd is None else mc_dd <= MC_DD_P95_MAX),
        g("beats random-entry twins", rnd, f">= {BEAT_RANDOM_MIN:.0%}", None if rnd is None else rnd >= BEAT_RANDOM_MIN),
        g("best month's share of the profit", conc.get("largest_month_share") if has_conc else None,
          f"<= {MAX_MONTH_SHARE:.0%}",
          (conc["largest_month_share"] is not None and conc["largest_month_share"] <= MAX_MONTH_SHARE)
          if has_conc else None),
        g("held-out trades (bars breeding never saw)", ho.get("trades") if ho_done else None,
          f">= {min_held_out_trades()}", ho["trades"] >= min_held_out_trades() if ho_done else None),
        g("held-out return", ho.get("return") if ho_done else None, "> 0", ho["return"] > 0 if ho_done else None),
        g("months with a position", conc.get("months_in_market_share") if has_conc else None,
          f">= {MIN_MONTHS_IN_MARKET:.0%}",
          (conc["months_in_market_share"] or 0.0) >= MIN_MONTHS_IN_MARKET if has_conc else None),
    ]
    return out


def score(entry: dict[str, Any]) -> tuple[float, float, dict[str, float]]:
    """(score, maximum reachable now, components)."""
    m = entry["metrics"]
    rb = entry.get("robustness") or {}
    done = rb.get("state") == "DONE"
    parts = {
        "sharpe": 20.0 * _clamp01((m["sharpe"] or 0.0) / 2.0),
        "drawdown": 10.0 * (1.0 - _clamp01((m["max_dd"] if m["max_dd"] is not None else 1.0) / 0.30)),
        "trades": 10.0 * _clamp01(m["n_trades"] / 300.0),
        "sham": 10.0 * _clamp01((m["sham_margin"] or 0.0) / 1.0),
        "cross": 10.0 * _clamp01((m["cross_sharpe"] or 0.0) / 1.0),
    }
    reachable = 60.0
    if done:
        nb, seg, bs = rb["neighbourhood"], rb["segments"]["sharpe"], rb["bootstrap"]
        rob = (3.0 * nb.get("positive_share", 0.0)
               + 3.0 * (sum(s > 0 for s in seg) / len(seg) if seg else 0.0)
               + 2.0 * (1.0 - bs.get("p_sharpe_le_0", 1.0)))
        reachable += 8.0
        if nb.get("plateau_share") is not None:
            rob += 2.0 * nb["plateau_share"]
            reachable += 2.0
        if _finite(bs.get("max_dd_p95")) is not None:
            rob += 1.0 * (1.0 - _clamp01(bs["max_dd_p95"] / MC_DD_P95_MAX))
            reachable += 1.0
        if (rb.get("random") or {}).get("beaten_share") is not None:
            # 4 at beating every twin, 0 at beating half of them (a coin)
            rob += 4.0 * _clamp01((rb["random"]["beaten_share"] - 0.5) / 0.5)
            reachable += 4.0
        parts["robustness"] = rob
        share = (rb.get("concentration") or {}).get("largest_month_share")
        if (rb.get("concentration") or {}).get("months"):
            # 5 at a best month of <= 20% of the profit, 0 at >= 80% (or no profit)
            parts["profit_spread"] = 5.0 * (1.0 - _clamp01(((share if share is not None else 1.0) - 0.2) / 0.6))
            reachable += 5.0
    ho = entry.get("held_out") or {}
    if ho.get("state") == "DONE":
        # 20 when the held-out Sharpe holds the in-sample Sharpe (ratio >= 1), 0 at a held-out Sharpe <= 0
        ref = max(m["sharpe"] or 0.0, 0.1)
        parts["persistence"] = 20.0 * _clamp01((ho.get("sharpe") or 0.0) / ref)
        reachable += 20.0
    return round(sum(parts.values()), 1), reachable, {k: round(v, 2) for k, v in parts.items()}


def grade(points: float) -> str:
    for bar, letter in GRADES:
        if points >= bar:
            return letter
    return "F"


def assess_v1(entry: dict[str, Any]) -> dict[str, Any]:
    """An engine V1 strategy keeps the research score and the research gates it passed at breeding (v1/gates.py);
    none of ARQVIO's validation core judged it, so it is UNCERTIFIED until engine V2 certifies it."""
    v1 = entry["v1"]
    gs = [{"gate": name.replace("_", " "), "value": c["value"], "need": "research gate", "passed": c["passed"]}
          for name, c in (v1.get("checks") or {}).items()]
    return {"tier": "UNCERTIFIED", "gates": gs, "score": float(v1["score"]), "score_max": 100.0,
            "score_parts": v1["parts"], "grade": v1["verdict"]}


def assess(entry: dict[str, Any]) -> dict[str, Any]:
    if entry.get("engine") == "v1" and (entry.get("certify") or {}).get("state") != "DONE":
        return assess_v1(entry)
    gs = gates(entry)
    points, reachable, parts = score(entry)
    certified = all(x["passed"] is True for x in gs)
    return {"tier": "CERTIFIED" if certified else "CANDIDATE", "gates": gs, "score": points, "score_max": reachable,
            "score_parts": parts, "grade": grade(points)}


def behaviour_key(entry: dict[str, Any]) -> tuple:
    """Strategies with the same cell, trade count, return and Sharpe made the same trades: copies of one behaviour
    (owner's vault, 2026-10-26: 17 near-identical rows). Genes that never reach the engine differ between copies."""
    m = entry["metrics"]

    def r(x: Any) -> Any:
        return None if x is None else float(f"{float(x):.10g}")

    return (entry["cell"], m.get("n_trades"), r(m.get("total_return")), r(m.get("sharpe")))


def copies(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """id -> {"copies": size of its behaviour cluster, "copy_of": the cluster's lead id or None}; ``rows`` in display
    order, so the lead is the best-ranked copy."""
    lead: dict[tuple, str] = {}
    size: dict[tuple, int] = {}
    for e in rows:
        k = behaviour_key(e)
        lead.setdefault(k, e["id"])
        size[k] = size.get(k, 0) + 1
    out = {}
    for e in rows:
        k = behaviour_key(e)
        out[e["id"]] = {"copies": size[k], "copy_of": None if lead[k] == e["id"] else lead[k]}
    return out


__all__ = ["TIERS", "VaultError", "assess", "behaviour_key", "copies", "completed_campaigns", "cost_model",
           "recorded_cost_model", "entry_from_candidate", "gates", "get", "grade",
           "import_campaign", "load", "purge", "recycled", "remove", "remove_many", "restore", "save", "score", "set_held_out", "set_robustness", "vault_path"]
