"""S5-APP read API: what the Windows app renders, each number with its address (DR-2026-10-10-s5-app §7).

Additive and GET-only (the campaign and vault verbs live in their own modules). Every rendered value is an L1 payload
``{value, unit, artifact_path, sha256, computed_utc, state}``: ``artifact_path`` is the file the value was read from,
``sha256`` that file's digest at read time, ``computed_utc`` the SOURCE's own timestamp (never the read clock), and
``state`` one of OK / NO DATA / UNRELIABLE / NOT VALIDATED YET / STALE / UNAVAILABLE. Nothing is computed that the
engine did not record; a missing source is a state with its reason, never a filled-in value. Paths resolve at CALL time
(engine/core/paths).

ld-yes (DR-2026-10-16-s9-community §1): the sidecar performs a READ-ONLY MT5 probe at the /environment endpoint and on
explicit user refresh. Permitted calls only: initialize/shutdown/terminal_info/account_info/symbols_get/symbol_info (static
allowlist scan in tests). DETECTED != VALIDATED (ENV4) is preserved verbatim: a terminal that answered the probe is
DETECTED, not validated for trading. The frontend never touches MT5 directly; it reads the probe result here.

* /environment          -- build stamp, ARQVIO_HOME, accounts (+ DEMO from the last health snapshot), VALIDATED vs
                           DETECTED-ONLY symbols, pinned datasets, the machine line (stdlib/ctypes, owner ruling D-7),
                           mt5_live (read-only probe result);
* /cells                -- the registry entries with the triple, the D2 render state and their badges;
* /cells/{cell}/stats   -- the stats card from the cell's registry-cited campaign (owner rulings D-1, D-2, D-3, D-8);
* /wizard               -- each cell's ceremony status derived from the records that exist (D-4);
* /refusals             -- every REFUSED record of the audit files and the app's own event log, verbatim;
* /logs?source=&after=  -- real log lines (timestamp, source, level, origin path:line) per source (LG1).
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
import platform
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from engine.core import paths
from engine.core.build_stamp import BUILD_COMMIT, BUILD_UTC
from engine.core.paths import get_artifacts_dir, get_arqvio_home
from engine.research.registry import cells as registry
from engine.research.registry.annualisation import bars_per_year

router = APIRouter(prefix="/api/v2/app", tags=["app"])

STALE_AFTER_S = 24 * 3600
NOT_RECORDED = "not recorded by campaign machinery; derivation deferred to R-stage — owner ledger"
MC_FAN_NO_DATA = "Monte Carlo equity fan — NO DATA (engine records no MC artifact)"
EXIT_LABELS = {  # owner ruling D-2: the engine's own exit vocabulary (engine/research/backtest/order.py:22-27)
    "SIGNAL": "SIGNAL", "SL": "SL", "TP": "TP", "EOD": "EOD", "FRIDAY_CLOSE": "FRIDAY_CLOSE",
    "SLTP_DEFERRED_ZERO_SPREAD": "SLTP deferred (zero spread)",
}
_SHA_CACHE: dict[tuple[str, int, int], str] = {}


# ---- L1 helpers ----------------------------------------------------------------------------------------------------

def sha256_file(p: Path) -> str | None:
    try:
        st = p.stat()
    except OSError:
        return None
    key = (str(p), st.st_size, st.st_mtime_ns)
    if key not in _SHA_CACHE:
        _SHA_CACHE[key] = hashlib.sha256(p.read_bytes()).hexdigest()
    return _SHA_CACHE[key]


def display(p: Path) -> str:
    """'<ARQVIO_HOME>/x' for a home file, a repo-relative path for a repo file, else the absolute path."""
    for root, label in ((get_arqvio_home(), "<ARQVIO_HOME>"), (paths.PROJECT_ROOT, None)):
        try:
            rel = p.resolve().relative_to(Path(root).resolve()).as_posix()
        except ValueError:
            continue
        return f"{label}/{rel}" if label else rel
    return str(p)


def addressed(value: Any, src: Path | None, *, unit: str | None = None, computed_utc: str | None = None,
              state: str = "OK", field: str | None = None, reason: str | None = None) -> dict[str, Any]:
    out = {"value": value, "unit": unit, "artifact_path": display(src) if src else None,
           "sha256": sha256_file(src) if src else None, "computed_utc": computed_utc, "state": state}
    if field:
        out["field"] = field
    if reason:
        out["reason"] = reason
    return out


def no_data(reason: str, src: Path | None = None, field: str | None = None) -> dict[str, Any]:
    return addressed(None, src, state="NO DATA", field=field, reason=reason)


def read_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def age_state(stamp: str | None, now: datetime) -> str:
    if not stamp:
        return "UNAVAILABLE"
    try:
        t = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return "UNRELIABLE"
    return "STALE" if (now - t).total_seconds() > STALE_AFTER_S else "OK"


def repo_path(rel: str) -> Path:
    return paths.PROJECT_ROOT / rel


def config(name: str) -> dict[str, Any]:
    p = repo_path(f"configs/app/{name}")
    return read_json(p) if p.is_file() else {}


# ---- environment ---------------------------------------------------------------------------------------------------

def _mt5_live_block(now_dt: datetime) -> dict[str, Any]:
    """ld-yes (DR §2): the read-only MT5 probe result on the environment endpoint.

    ``DETECTED != VALIDATED`` (ENV4 fence) is preserved verbatim: a terminal that
    answered the probe is DETECTED, not validated for trading.
    """
    from engine.research.data.mt5_live_probe import probe_mt5_live

    r = probe_mt5_live()
    block: dict[str, Any] = {
        "state": r.state,
        "source": r.source,
        "as_of": r.as_of,
        "terminal_path": r.terminal_path,
        "terminal_source": r.terminal_source,
    }
    if r.terminal_info is not None:
        block["terminal_info"] = r.terminal_info
    if r.account_info is not None:
        block["account_info"] = r.account_info
    if r.symbol_count is not None:
        block["symbol_count"] = r.symbol_count
    if r.detail is not None:
        block["detail"] = r.detail
    block["candidates_checked"] = list(r.candidates_checked)
    return block


def machine_line(now: str) -> dict[str, Any]:
    """CPU/RAM from the Python standard library (owner ruling D-7); a live local OS read, labelled as such."""
    src = "local OS (python os.cpu_count, platform, ctypes GlobalMemoryStatusEx)"
    ram = None
    if sys.platform == "win32":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        st = MemoryStatus()
        st.dwLength = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            ram = int(st.ullTotalPhys)
    return {"source": src, "as_of": now,
            "cpu_logical": {"value": os.cpu_count(), "unit": "threads", "state": "OK" if os.cpu_count() else "UNAVAILABLE"},
            "processor": {"value": platform.processor() or None, "state": "OK" if platform.processor() else "UNAVAILABLE"},
            "ram_total": {"value": ram, "unit": "bytes", "state": "OK" if ram else "UNAVAILABLE"},
            "os": {"value": platform.platform(), "state": "OK"}}


@router.get("/engine-settings")
def engine_settings() -> dict[str, Any]:
    """The rules the engine judges and costs by (read-only; ARQVIO_* environment overrides them): the SETTINGS tab."""
    from engine.config.settings import get_settings
    from engine.research.vault import library

    s = get_settings()
    return {
        "costs": {"spread_cost_mode": s.spread_cost_mode, "spread_gate_mode": s.spread_gate_mode,
                  "spread_regime_days": s.spread_regime_days, "spread_regime_multiple": s.spread_regime_multiple,
                  "max_spread_gate_percentile": s.max_spread_gate_percentile,
                  "commission_per_lot_per_side": s.commission_per_lot_per_side},
        "certified": {"min_trades": library.MIN_TRADES_CERTIFIED, "max_dd": library.MAX_DD_CERTIFIED,
                      "max_month_share": library.MAX_MONTH_SHARE, "min_months_in_market": library.MIN_MONTHS_IN_MARKET,
                      "min_held_out_trades": library.min_held_out_trades(), "neighbour_share_min": library.NEIGHBOUR_SHARE_MIN,
                      "plateau_min": library.PLATEAU_MIN, "mc_dd_p95_max": library.MC_DD_P95_MAX,
                      "beat_random_min": library.BEAT_RANDOM_MIN, "bootstrap_p_max": library.BOOTSTRAP_P_MAX},
        "grades": [{"min": bar, "grade": g} for bar, g in library.GRADES],
        "v1": _v1_rules(),
    }


def _v1_rules() -> dict[str, Any]:
    """Engine V1's numbers (the research engine, engine/research/v1/config.py), for each depth the app offers."""
    from engine.research.v1.config import PRESETS, v1_config
    from engine.research.v1.score import GOOD_AT, WEIGHTS

    keep = ("proving_grounds", "generations", "tribe_size", "finalists", "held_out_min_trades", "random_n",
            "random_beat_min", "shuffle_n", "shuffle_dd_max", "nudge_hold_min", "floor_max_dd", "floor_pf",
            "floor_trades", "drawdown_limit")
    return {"presets": {p: {k: v1_config(p).block()[k] for k in keep} for p in PRESETS},
            "score_weights": WEIGHTS, "good_at": GOOD_AT}


@router.get("/environment")
async def environment() -> dict[str, Any]:
    from engine.data.spread_calibration import spread_dir
    from engine.research.data.dataset_snapshot import datasets_dir
    from engine.research.health.probe import accounts_path, load_accounts, snapshot_path

    now_dt = datetime.now(tz=UTC)
    now = now_dt.isoformat()
    build_state = "OK" if BUILD_COMMIT not in ("source", "unknown") else ("SOURCE" if BUILD_COMMIT == "source"
                                                                         else "UNAVAILABLE")
    snap_p = snapshot_path()
    snap = read_json(snap_p) if snap_p.is_file() else None
    snap_utc = snap.get("snapshot_utc") if snap else None
    snap_accounts = {a.get("id"): a for a in (snap or {}).get("accounts", [])}
    acc_p = accounts_path()
    accounts = []
    for a in (load_accounts() if acc_p.is_file() else []):
        s = snap_accounts.get(a["id"])
        accounts.append({
            "id": a["id"], "server": a["server"], "terminal": a["terminal"],
            "demo": addressed(s.get("demo"), snap_p, computed_utc=s.get("probed_utc"),
                              state=age_state(s.get("probed_utc"), now_dt), field="accounts[].demo")
            if s else no_data("no health snapshot row for this account", snap_p),
            "time_basis": addressed((s.get("basis") or {}).get("basis"), snap_p, computed_utc=s.get("probed_utc"),
                                    state=age_state(s.get("probed_utc"), now_dt), field="accounts[].basis.basis")
            if s else no_data("no health snapshot row for this account", snap_p)})
    reg_p = registry.registry_path()
    entries = registry.load()
    validated = sorted(e["cell_id"] for e in entries if e["status"] == "VALIDATED")
    validated_symbols = sorted({e["symbol"] for e in entries if e["status"] == "VALIDATED"})
    sd = spread_dir()
    calibrated = sorted(p.stem for p in sd.glob("*.json")) if sd.is_dir() else []
    dd = datasets_dir()
    datasets = []
    for side in sorted(dd.glob("*.json")) if dd.is_dir() else []:
        try:
            m = read_json(side)
        except ValueError:
            continue
        datasets.append({"dataset_id": m.get("dataset_id"), "symbol": m.get("symbol"), "timeframe": m.get("timeframe"),
                         "bars": m.get("n_bars"), "start": m.get("start_utc"), "end": m.get("end_utc"),
                         "artifact_path": display(side), "sha256": sha256_file(side)})
    return {
        "as_of": now,
        "build": {"commit": BUILD_COMMIT, "utc": BUILD_UTC, "state": build_state,
                  "source": "engine/core/build_stamp.py (a source checkout reports 'source'; a packaged build its stamp)"},
        "arqvio_home": str(get_arqvio_home()),
        "accounts": {"artifact_path": display(acc_p), "sha256": sha256_file(acc_p), "count": len(accounts),
                     "items": accounts, "health_snapshot_utc": snap_utc,
                     "health_state": age_state(snap_utc, now_dt) if snap else "UNAVAILABLE"},
        "validated": {"artifact_path": display(reg_p), "sha256": sha256_file(reg_p), "cells": validated,
                      "symbols": validated_symbols},
        "detected_only": {"source": display(sd), "symbols": [s for s in calibrated if s not in validated_symbols],
                          "note": "calibrated spread artifact present, no registry entry: a wizard candidate, NOT VALIDATED YET"},
        "datasets": datasets,
        "machine": machine_line(now),
        "mt5_live": _mt5_live_block(now_dt),
    }


# ---- first-run wizard ------------------------------------------------------------------------------------------------
# DR-2026-10-16-s9-community §3: runs only when ARQVIO_HOME has no config/accounts.json.
# The wizard reads whether MT5 answered during the probe (it does NOT start MT5) and
# scaffolds a placeholder accounts.json the operator edits.


ACCOUNTS_SCAFFOLD = """\
{
  "accounts": [
    {
      "id": "acct1",
      "login": 0,
      "server": "FILL-ME",
      "terminal": "FILL-ME",
      "password_env": "FILL-ME",
      "probe_symbol": "EURUSDm",
      "pinned_symbol": "EURUSDm"
    }
  ]
}
"""


def _first_run_data_check(now_dt: datetime) -> dict[str, Any]:
    """Honest YES/NO for each data presence check."""
    from engine.data.spread_calibration import spread_dir
    from engine.research.data.dataset_snapshot import datasets_dir
    from engine.research.registry import cells as registry

    reg_p = registry.registry_path()
    sd = spread_dir()
    calibrated = sorted(p.stem for p in sd.glob("*.json")) if sd.is_dir() else []
    dd = datasets_dir()
    datasets = sorted(dd.glob("*.json")) if dd.is_dir() else []
    home = get_arqvio_home()
    acc_p = home / "config" / "accounts.json"

    return {
        "accounts_json": {"present": acc_p.is_file(), "path": str(acc_p)},
        "registry": {"present": reg_p.is_file(), "path": display(reg_p),
                      "cells": len([e for e in registry.load() if e["status"] == "VALIDATED"])},
        "calibrated_spreads": {"present": len(calibrated) > 0, "count": len(calibrated),
                                "symbols": calibrated},
        "datasets": {"present": len(datasets) > 0, "count": len(datasets)},
    }


@router.post("/first-run")
async def first_run() -> dict[str, Any]:
    """Run the first-run readiness checklist.

    Returns the state of each check (accounts, MT5 probe, data presence) and the
    overall 'ready' flag.  'ready' is True only when all five gates pass:
    ARQVIO_HOME exists, accounts.json exists, MT5 answered the probe, registry has
    >=1 validated cell, at least one calibrated spread.
    """
    from engine.research.data.mt5_live_probe import probe_mt5_live

    now_dt = datetime.now(tz=UTC)
    home = get_arqvio_home()
    home_exists = home.is_dir()
    acc_p = home / "config" / "accounts.json"
    accounts_exists = acc_p.is_file()

    probe = probe_mt5_live()
    mt5_ok = probe.state == "MT5_PROBE_OK"
    mt5_block: dict[str, Any] = {
        "state": probe.state,
        "terminal_path": probe.terminal_path,
        "terminal_source": probe.terminal_source,
        "detail": probe.detail,
    } if probe.detail is not None else {
        "state": probe.state,
        "terminal_path": probe.terminal_path,
        "terminal_source": probe.terminal_source,
    }
    if probe.symbol_count is not None:
        mt5_block["symbol_count"] = probe.symbol_count
    if probe.account_info is not None:
        mt5_block["demo"] = probe.account_info.get("demo", False)

    data = _first_run_data_check(now_dt)

    checks = {
        "arqvio_home": {"ready": home_exists, "path": str(home)},
        "accounts_json": {"ready": accounts_exists, "path": str(acc_p)},
        "mt5_live": {"ready": mt5_ok, **mt5_block},
        "registry": {"ready": data["registry"]["present"] and data["registry"]["cells"] > 0, **data["registry"]},
        "calibrated_spreads": {"ready": data["calibrated_spreads"]["present"], **data["calibrated_spreads"]},
    }
    ready = all(c["ready"] for c in checks.values())
    not_ready = [k for k, v in checks.items() if not v["ready"]]
    return {"checks": checks, "ready": ready, "not_ready": not_ready, "as_of": now_dt.isoformat()}


@router.post("/first-run/scaffold")
async def first_run_scaffold() -> dict[str, Any]:
    """Write the accounts.json scaffold template.  The operator edits it; the engine never fills credentials."""
    home = get_arqvio_home()
    acc_p = home / "config" / "accounts.json"
    existed = acc_p.is_file()
    if not existed:
        acc_p.parent.mkdir(parents=True, exist_ok=True)
        acc_p.write_text(ACCOUNTS_SCAFFOLD, encoding="utf-8")
    return {"scaffolded": not existed, "path": str(acc_p), "existed": existed}


@router.get("/first-run")
async def first_run_status() -> dict[str, Any]:
    """Lightweight: is this a first-run (no accounts.json)?  GET-only for the shell to gate."""
    home = get_arqvio_home()
    acc_p = home / "config" / "accounts.json"
    return {"first_run": not acc_p.is_file(), "accounts_path": str(acc_p), "home": str(home)}

def render_state(e: dict[str, Any]) -> str:
    """Owner D2: READY FOR FIRST SEARCH := UNRUN and power BLOCKED:no-campaign and worker PROVEN-PRECAMPAIGN."""
    if (e["campaign_status"] == "UNRUN" and e["power_check"]["status"] == "BLOCKED:no-campaign"
            and e["worker_status"]["status"].startswith("PROVEN-PRECAMPAIGN@")):
        return "READY FOR FIRST SEARCH"
    return e["status"]


def known_holes(cid: str) -> dict[str, Any] | None:
    p = get_artifacts_dir() / "cells" / cid / "w2_manifest.json"
    if not p.is_file():
        return None
    g = read_json(p).get("gaps", {})
    holes = g.get("owner_accepted_holes") or []
    if not holes:
        return None
    n = read_json(p).get("n_bars")
    held = g.get("owner_accepted_bars")
    return {"holes": addressed(len(holes), p, field="gaps.owner_accepted_holes"),
            "bars": addressed(held, p, field="gaps.owner_accepted_bars"),
            "of_bars": addressed(n, p, field="n_bars"),
            "items": holes, "acceptance": g.get("acceptance")}


def prior_trials(cid: str) -> dict[str, Any] | None:
    """The U4-Q1 record (owner ruling D3): 'PRIOR TRIALS on <SYM> <TF>: <n>' and the database sha256, from the log."""
    for p in sorted(repo_path("artifacts/reports").glob(f"*/prior_trials_{cid}.log")):
        text = p.read_text(encoding="utf-8")
        n = re.search(r"^PRIOR TRIALS on \S+ \S+: (\d+)$", text, re.M)
        db = re.search(r"sha256 ([0-9a-f]{64})", text)
        if n:
            return {"prior_trials": addressed(int(n.group(1)), p, field="PRIOR TRIALS line"),
                    "database_sha256": db.group(1) if db else None,
                    "caveat": ("deflation-integration mechanics pending R-stage DR; any verdict from this campaign "
                               f"carries the caveat '{n.group(1)} prior unbanked trials not included in deflation'")}
    return None


def known_limitations(cid: str) -> dict[str, Any] | None:
    p = repo_path(f"docs/cells/{cid}.md")
    if not p.is_file():
        return None
    text = p.read_text(encoding="utf-8")
    m = re.search(r"^## Known Limitations\n(.*?)(?=^## )", text, re.S | re.M)
    return {"artifact_path": display(p), "sha256": sha256_file(p), "text": m.group(1).strip()} if m else None


def scope_note(cid: str) -> dict[str, Any] | None:
    """The cell's power-scope caveat (DR §6: XAU carries U2's), read from the cited source, addressed."""
    note = config("campaign_presets.json").get("scope_notes", {}).get(cid)
    if not note:
        return None
    p = repo_path(note["path"])
    if not p.is_file():
        return no_data("scope source missing", None, note["field"])
    return {**addressed(read_json(p).get(note["field"]), p, field=note["field"]), "label": note["label"]}


def cell_card(e: dict[str, Any]) -> dict[str, Any]:
    # a QUICK cell is addressed to the user cells file it lives in, a registry cell to the registry
    reg_p = registry.user_cells_path() if e["status"] == registry.QUICK else registry.registry_path()
    cid = e["cell_id"]
    return {"cell_id": cid, "symbol": e["symbol"], "timeframe": e["timeframe"], "status": e["status"],
            "render_state": render_state(e),
            "campaign_status": addressed(e["campaign_status"], reg_p, field=f"{cid}.campaign_status"),
            "power_check": addressed(e["power_check"], reg_p, field=f"{cid}.power_check"),
            "worker_status": addressed(e["worker_status"], reg_p, field=f"{cid}.worker_status"),
            "dataset_window": addressed(e["dataset_window"], reg_p, field=f"{cid}.dataset_window"),
            "annualisation_ppy": addressed(e["annualisation_ppy"], reg_p, field=f"{cid}.annualisation_ppy"),
            "law_ppy": bars_per_year(e["timeframe"]), "cross_cell": e["cross_cell"],
            "entered_by_stage": e["entered_by_stage"], "entry_hash": e["entry_hash"],
            "known_holes": known_holes(cid), "prior_trials": prior_trials(cid),
            "known_limitations": known_limitations(cid), "scope_note": scope_note(cid)}


@router.get("/cells")
async def cells() -> dict[str, Any]:
    """The research cells: the registry's VALIDATED ones, then the QUICK cells added in the app (each labelled by
    its status; the registry's address and hash describe the registry file)."""
    reg_p = registry.registry_path()
    return {"artifact_path": display(reg_p), "sha256": sha256_file(reg_p),
            "cells": [cell_card(e) for e in registry.research_entries()]}


def validated_entry(cell_id: str) -> dict[str, Any]:
    """L2 fail-closed: only a research cell (VALIDATED in the registry, or QUICK) resolves; anything else is 404."""
    for e in registry.research_entries():
        if e["cell_id"] == cell_id:
            return e
    raise HTTPException(status_code=404, detail="unknown cell")


# ---- stats card ----------------------------------------------------------------------------------------------------

def verdict_artifact(campaign_id: str) -> tuple[Path, dict[str, Any]] | None:
    for p in sorted(get_artifacts_dir().glob("*/verdict.json")):
        try:
            v = read_json(p)
        except ValueError:
            continue
        if v.get("campaign_id") == campaign_id:
            return p, v
    return None


def verdict_block(campaign_id: str) -> dict[str, Any]:
    """Owner D-3: the campaign's verdict artifact if one exists, else its report-text citation, else NO DATA."""
    found = verdict_artifact(campaign_id)
    if found:
        p, v = found
        return {"kind": "artifact", "text": addressed(v.get("verdict"), p, field="verdict"),
                "survivors": addressed(sum(1 for f in v.get("finalists", []) if f.get("survivor")), p,
                                       field="finalists[].survivor (count)")}
    cite = config("verdict_citations.json").get("citations", {}).get(campaign_id)
    if cite:
        p = repo_path(cite["path"])
        lines = p.read_text(encoding="utf-8").splitlines() if p.is_file() else []
        a, b = cite["lines"]
        text = "\n".join(lines[a - 1:b]) if len(lines) >= b else None
        return {"kind": "report_text", "badge": cite["badge"],
                "text": addressed(text, p, field=f"lines {a}-{b}") if text else no_data("cited report missing", p)}
    return {"kind": "none", "text": no_data("no verdict artifact and no citation for this campaign")}


def statistical_context(cid: str, campaign: dict[str, Any], man_p: Path) -> dict[str, Any]:
    """Owner D-8: vs-random p_max + the planted-edge power curve, each labelled; the MC fan is NO DATA."""
    canary = campaign["run"]["cells"].get(cid, {}).get("canary", {})
    out: dict[str, Any] = {
        "vs_random": {"label": "vs-random bootstrap of the campaign canary (engine/validation/vs_random.py)",
                      "p_max": addressed(canary.get("vs_random_p_max"), man_p,
                                         field=f"run.cells.{cid}.canary.vs_random_p_max")
                      if "vs_random_p_max" in canary else no_data("not in this manifest", man_p),
                      "verdict": canary.get("verdict")},
        "mc_fan": {"title": "Monte Carlo equity fan", "state": "NO DATA", "reason": MC_FAN_NO_DATA},
    }
    src = config("statistical_context.json").get("power_tables", {}).get(cid)
    if src and repo_path(src["path"]).is_file():
        p = repo_path(src["path"])
        t = read_json(p)
        out["power_curve"] = {"label": src["label"], "artifact_path": display(p), "sha256": sha256_file(p),
                              "verdict": t.get("verdict"), "rows": t.get("table"), "design": t.get("design")}
    else:
        out["power_curve"] = no_data("no power table registered for this cell")
    return out


@router.get("/machine")
async def machine() -> dict[str, Any]:
    return machine_line(datetime.now(tz=UTC).isoformat())


@router.get("/cells/{cell_id}/campaigns")
async def cell_campaigns(cell_id: str) -> dict[str, Any]:
    """The registry-cited campaign of the cell and the app's own runs on it (state per the campaign verbs)."""
    from engine.api.routes_app_campaigns import runs_root, status_of

    e = validated_entry(cell_id)
    reg_p = registry.registry_path()
    root = runs_root()
    runs = sorted((p.parent.name for p in root.glob(f"*-{cell_id}-s*/request.json")), reverse=True) \
        if root.is_dir() else []
    return {"cell_id": cell_id,
            "cited": addressed(e["campaign_status"], reg_p, field=f"{cell_id}.campaign_status"),
            "app_runs": [{k: v for k, v in status_of(r, root / r).items() if k in ("run", "state", "request")}
                         for r in runs]}


@router.get("/cells/{cell_id}/stats")
async def stats(cell_id: str) -> dict[str, Any]:
    e = validated_entry(cell_id)
    reg_p = registry.registry_path()
    base = {"cell_id": cell_id, "render_state": render_state(e),
            "power_status": addressed(e["power_check"], reg_p, field=f"{cell_id}.power_check")}
    if not e["campaign_status"].startswith("RUN@"):
        return {**base, "state": "NO CAMPAIGN",
                "reason": f"campaign_status {e['campaign_status']}: no campaign has run on this cell"}
    campaign_id = e["campaign_status"][len("RUN@"):]
    man_p = get_artifacts_dir() / "ga" / campaign_id / "manifest_v2.json"
    if not man_p.is_file():
        return {**base, "state": "UNAVAILABLE", "reason": f"the cited campaign manifest is absent: {display(man_p)}"}
    m = read_json(man_p)
    ind_p = man_p.parent / "individuals.json"
    exits: dict[str, dict[str, int]] = {}
    if ind_p.is_file():
        for row in read_json(ind_p):
            if "exit_reasons" in row.get("metrics", {}):
                exits[row["dna_hash"]] = row["metrics"]["exit_reasons"]
    # A finalist's pass/fail is shown only where an artifact records it (the verdict file's finalists[].survivor); it
    # is never re-derived here from the gate thresholds (owner D-3: no verdict object is invented).
    va = verdict_artifact(campaign_id)
    recorded = {f["dna_hash"]: f.get("survivor") for f in va[1].get("finalists", [])} if va else {}
    finalists = []
    for i, c in enumerate(m.get("candidates", [])):
        met = c["metrics"].get(cell_id, {})
        fld = f"candidates[{i}].metrics.{cell_id}"
        ex = exits.get(c["dna_hash"])
        finalists.append({
            "dna_hash": c["dna_hash"], "rank": i + 1, "family": c.get("dna", {}).get("signal"),
            "survivor": addressed(recorded[c["dna_hash"]], va[0], field="finalists[].survivor")
            if va and c["dna_hash"] in recorded else no_data("no artifact records this finalist's gate outcome"),
            "net_return_pct": addressed(met["total_return"] * 100 if "total_return" in met else None, man_p,
                                        unit="%", field=f"{fld}.total_return",
                                        state="OK" if "total_return" in met else "NO DATA"),
            "max_dd_pct": addressed(met["max_dd"] * 100 if "max_dd" in met else None, man_p, unit="%",
                                    field=f"{fld}.max_dd", state="OK" if "max_dd" in met else "NO DATA"),
            "sharpe": addressed(met.get("sharpe"), man_p, field=f"{fld}.sharpe"),
            "n_trades": addressed(met.get("n_trades"), man_p, field=f"{fld}.n_trades"),
            "costs": addressed(met.get("costs"), man_p, field=f"{fld}.costs"),
            "dsr": addressed(c.get("dsr"), man_p, field=f"candidates[{i}].dsr"),
            "sham": addressed(met.get("sham"), man_p, field=f"{fld}.sham"),
            "exit_anatomy": addressed({EXIT_LABELS.get(k, k): v for k, v in ex.items()}, ind_p,
                                      field="individuals[dna_hash].metrics.exit_reasons")
            if ex is not None else no_data("no exit record for this finalist", ind_p),
            "win_rate": no_data(NOT_RECORDED), "profit_factor": no_data(NOT_RECORDED),
            "expectancy": no_data(NOT_RECORDED),
            "dna": c.get("dna")})
    cfg = m["run"]["config"]
    return {**base, "state": "OK", "campaign_id": campaign_id, "manifest": display(man_p),
            "manifest_sha256": sha256_file(man_p),
            "verdict": verdict_block(campaign_id),
            "best": finalists[0] if finalists else None,
            "finalists": finalists,
            "exit_labels_note": "exit anatomy: primary cell only; the cross cell's exits are not recorded (NO DATA)",
            "deflation": {"dsr_n_final": addressed(m["run"].get("dsr_n_final"), man_p, field="run.dsr_n_final"),
                          "gates": cfg.get("ga", {}).get("gates")},
            "preregistered": addressed({"population": cfg.get("population"), "generations": cfg.get("ga", {}).get(
                "generations"), "trials_budget": cfg.get("ga", {}).get("trials_budget"),
                "final_n": cfg.get("ga", {}).get("final_n"), "seed": cfg.get("seed"),
                "gene_pool_hash": cfg.get("ga", {}).get("gene_pool_hash")}, man_p, field="run.config"),
            "trials_used": addressed(m["run"].get("trials_used"), man_p, field="run.trials_used"),
            "statistical_context": statistical_context(cell_id, m, man_p)}


# ---- wizard --------------------------------------------------------------------------------------------------------

WIZARD_PRODUCTS = ("w1_audit.json", "w2_manifest.json", "w3_pin.json", "w5_power.json")


def audit_rows(pattern: str) -> list[dict[str, Any]]:
    """Every row of the repo's audit.jsonl files under artifacts/reports matching pattern, with origin path:line."""
    rows = []
    for p in sorted(repo_path("artifacts/reports").glob(pattern)):
        sha = sha256_file(p)
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            rows.append({**r, "_origin": f"{display(p)}:{n}", "_sha256": sha, "_raw": line})
    return rows


@router.get("/wizard")
async def wizard() -> dict[str, Any]:
    from engine.data.spread_calibration import spread_dir

    entries = registry.load()
    rows = audit_rows("*/wizard/**/audit.jsonl")
    out = []
    for e in entries:
        cid = e["cell_id"]
        cd = get_artifacts_dir() / "cells" / cid
        refused = [r for r in rows if r.get("outcome") == "REFUSED" and str(r.get("symbol", "")).lower() == e[
            "symbol"].lower() and r.get("tf") == e["timeframe"]]
        out.append({"cell_id": cid, "symbol": e["symbol"], "timeframe": e["timeframe"], "state": "VALIDATED",
                    "entered_by_stage": e["entered_by_stage"],
                    "steps": {n: (cd / n).is_file() for n in WIZARD_PRODUCTS},
                    "proof": next((r for r in e["evidence"] if r.endswith(".proof.json")), None),
                    "runbook": f"docs/cells/{cid}.md" if repo_path(f"docs/cells/{cid}.md").is_file() else None,
                    "refusals": [{"utc": r.get("utc"), "step": r.get("step"), "reason": r.get("reason"),
                                  "origin": r["_origin"], "sha256": r["_sha256"]} for r in refused]})
    symbols = {e["symbol"] for e in entries}
    sd = spread_dir()
    for s in sorted(p.stem for p in sd.glob("*.json")) if sd.is_dir() else []:
        if s not in symbols:
            out.append({"cell_id": None, "symbol": s, "timeframe": None, "state": "NOT VALIDATED YET",
                        "instruction": "wizard ceremonies run from the CLI -- docs/cells/<runbook>; the U4 runbook "
                                       "sequence: python -m engine.research.wizard audit|pin|calibrate|promote|power|"
                                       "worker|runbook --symbol <S> --tf <T> --out <dir> (DR-2026-10-09-u4-wizard §4)"})
    return {"rows": out, "audit_sources": sorted({r["_origin"].rsplit(":", 1)[0] for r in rows})}


# ---- refusals ------------------------------------------------------------------------------------------------------

def app_events_path() -> Path:
    return get_artifacts_dir() / "app" / "events.jsonl"


def app_events() -> list[dict[str, Any]]:
    p = app_events_path()
    if not p.is_file():
        return []
    sha = sha256_file(p)
    out = []
    for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        try:
            out.append({**json.loads(line), "_origin": f"{display(p)}:{n}", "_sha256": sha, "_raw": line})
        except ValueError:
            continue
    return out


@router.get("/refusals")
async def refusals() -> dict[str, Any]:
    recs = [r for r in audit_rows("**/audit.jsonl") + app_events() if r.get("outcome") == "REFUSED"]
    recs.sort(key=lambda r: str(r.get("utc", "")), reverse=True)
    return {"count": len(recs), "records": [{"utc": r.get("utc"), "reason": r.get("reason"),
                                             "subject": r.get("cell") or r.get("symbol") or r.get("run"),
                                             "step": r.get("step"), "origin": r["_origin"], "sha256": r["_sha256"],
                                             "raw": r["_raw"]} for r in recs]}


# ---- logs (LG1) ----------------------------------------------------------------------------------------------------

def campaign_log_paths() -> dict[str, Path]:
    root = get_artifacts_dir() / "app" / "campaigns"
    if not root.is_dir():
        return {}
    return {**{f"campaign:{p.parent.name}": p for p in sorted(root.glob("*/runner.log"))},
            **{f"probe:{p.parent.parent.name}": p for p in sorted(root.glob("*/probe/probe.log"))}}


LINE = re.compile(r"^(?P<ts>\S+)\s+\|\s+(?P<level>[A-Z]+)\s+\|\s+(?P<text>.*)$")


@router.get("/logs")
async def logs(source: str = Query("app-events"), after: int = Query(0, ge=0)) -> dict[str, Any]:
    sources = ["app-events", "wizard-audit", "reproof-audit", *campaign_log_paths()]
    if source not in sources:
        raise HTTPException(status_code=404, detail=f"unknown log source; known: {sources}")
    lines: list[dict[str, Any]] = []
    if source == "app-events":
        for r in app_events():
            lines.append({"ts": r.get("utc"), "source": source, "level": r.get("level", "INFO"),
                          "text": r["_raw"], "origin": r["_origin"]})
    elif source in ("wizard-audit", "reproof-audit"):
        rows = audit_rows("*/wizard/**/audit.jsonl" if source == "wizard-audit" else "*/worker/**/audit.jsonl")
        for r in rows:
            lines.append({"ts": r.get("utc"), "source": source,
                          "level": "WARN" if r.get("outcome") == "REFUSED" else "INFO",
                          "text": r["_raw"], "origin": r["_origin"]})
    else:
        p = campaign_log_paths()[source]
        for n, raw in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            m = LINE.match(raw)
            lines.append({"ts": m["ts"] if m else None, "source": source, "level": m["level"] if m else "INFO",
                          "text": m["text"] if m else raw, "origin": f"{display(p)}:{n}"})
    return {"source": source, "sources": sources, "total": len(lines), "after": after, "lines": lines[after:]}
