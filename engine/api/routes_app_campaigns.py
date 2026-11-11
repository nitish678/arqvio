"""S5-APP campaign verbs (DR-2026-10-10-s5-app §10; build rulings 2026-10-10: Decision 1 and the H-START LAW).

The app launches the frozen GA CLI unchanged -- ``engine.research.ga.runner`` with ``--cell --pop --gen --seed
--budget --final-n --engine auto --checkpoint <run>/checkpoint.npz --monitor-snapshot-every 5`` -- through the app's
launcher (engine/api/campaign_launch.py), and adds only app-side guards around it:

* PREFLIGHT runs a discarded timing probe (configs/app/campaign_presets.json "probe") with ARQVIO_HOME on an ISOLATED
  probe home built from copies of the GA inputs, and projects the chosen preset from the probe's own recorded walls;
  the probe's manifest is the estimate's address. The engine's MinBTL for the chosen budget against the cell's audited
  years carries the small-window SMOKE-only warning.
* START is refused unless the measured preflight is OK, and -- the H-START LAW -- whenever the target campaign
  directory ``artifacts/ga/<campaign_id>`` already exists: run_pinned unlinks an existing directory's outputs on a
  fresh start (engine/research/ga/runner.py:735-740) and the id is cell + seed + config (runner.py:650-653), so a fresh
  start there would erase a cited campaign. The sidecar checks before spawning and the launcher re-checks in the
  child, on the code the runner runs. The pre-registration record is written exclusively, once, before the spawn (L7);
  a run starts once (one-attempt law).
* STOP terminates only a process tree this sidecar spawned (never by a recorded pid). Every spawned process dies with
  the sidecar (a kill-on-close job), and a process left alive by an earlier sidecar session blocks every new spawn.
* RESUME is the S4 checkpoint contract only: the SAME recorded runner argv + ``--resume <this run's checkpoint>``
  (licensed by the S4 Stage-3 kill->resume raw-identical proof, artifacts/reports/s4_stage3_report.md:1066, :750-758),
  and only onto the history this run's last process left: the checkpoint and the first ``registry_rows`` trial rows
  must equal what was fingerprinted when that process exited. Anything else is REFUSED.

No file the runner writes is opened while a process of the run is alive (on Windows an open handle makes the runner's
atomic replace fail); live progress is the runner's own stdout lines. Every refusal is returned verbatim and appended
to ``<ARQVIO_HOME>/artifacts/app/events.jsonl``; the
runner's output is written to ``<run>/runner.log`` as ``<utc> | <LEVEL> | <text>`` (served by /api/v2/app/logs).
Nothing here attaches MetaTrader5, starts a worker or runs a wizard step (PC2, PC3).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from engine.api.routes_app import (
    addressed,
    app_events_path,
    config,
    display,
    no_data,
    prior_trials,
    read_json,
    repo_path,
    sha256_file,
    validated_entry,
)
from engine.core import paths
from engine.core.paths import get_artifacts_dir, get_arqvio_home

router = APIRouter(prefix="/api/v2/app/campaigns", tags=["app-campaigns"])

RUNNER_MODULE = "engine.research.ga.runner"
LAUNCHER_MODULE = "engine.api.campaign_launch"
MONITOR_EVERY_S = 5
ARG_KEYS = ("pop", "gen", "budget", "final_n")
ANSI = re.compile(r"\x1b\[[0-9;]*m")
LOGURU = re.compile(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+ \| (?P<level>[A-Z]+)\s*\| (?P<text>.*)$")
CHECKPOINT_LINE = re.compile(r"^\s*CHECKPOINT after generation (?P<g>\d+) -> ")  # runner.py write_checkpoint echo
GEN_LINE = re.compile(r"^\s*gen\s+\d+:")  # runner.py on_generation echo
H_START = ("H-START: {target} already exists; a fresh start would erase its outputs (engine/research/ga/runner.py:"
           "735-740). Every new campaign is a new pre-registered decision: choose another seed or configuration.")
STILL_ACTIVE = 259
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

_procs: dict[str, dict[str, Any]] = {}
_lock = threading.RLock()
_job: Any = None
_ckpt_cache: dict[tuple[str, int, int, str], dict[str, Any]] = {}
_ckpt_lock = threading.Lock()


def now_utc() -> str:
    return datetime.now(tz=UTC).isoformat()


def runs_root() -> Path:
    return get_artifacts_dir() / "app" / "campaigns"


def run_dir(run: str) -> Path:
    if not re.fullmatch(r"[0-9TZ]+-[a-z0-9_]+-s\d+", run):
        raise HTTPException(status_code=404, detail="unknown run")
    d = runs_root() / run
    if not (d / "request.json").is_file():
        raise HTTPException(status_code=404, detail="unknown run")
    return d


def jsonl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()] if p.is_file() else []


def append(p: Path, rec: dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")


def event(outcome: str, run: str | None, cell: str | None, step: str, reason: str | None = None,
          level: str = "INFO", **extra: Any) -> None:
    append(app_events_path(), {"utc": now_utc(), "level": level, "outcome": outcome, "run": run, "cell": cell,
                               "step": step, "reason": reason, **extra})
    if level != "INFO":  # refusals and failures also reach the app's one log
        from engine.core import applog

        applog.write("breed", f"{cell or ''} {step}: {outcome}" + (f" -- {reason}" if reason else ""), level)


def refuse(reason: str, *, run: str | None, cell: str | None, step: str, status: int = 409) -> HTTPException:
    event("REFUSED", run, cell, step, reason, level="WARN")
    return HTTPException(status_code=status, detail=f"REFUSED: {reason}")


def presets() -> dict[str, Any]:
    return config("campaign_presets.json")


# ---- the runner's own derivations, computed without running anything ------------------------------------------------

def campaign_target(cell: str, args: dict[str, int], seed: int, islands: int | None = None) -> dict[str, Any]:
    """The campaign id and directory exactly as runner.main + run_pinned derive them (runner.py:996-1001, 718-730)."""
    from engine.research.campaign.manifest import config_hash
    from engine.research.ga.config import CAMPAIGN_GATES, PINNED_CELLS, IslandConfig, ga_config
    from engine.research.ga.runner import campaign_id_for
    from engine.research.search import gene_pool
    from engine.research.search.run_campaign_cli import CAMPAIGN_TARGET_SHARPE, artifact_path, campaign_config_for

    cfg = ga_config(seed=seed, gates=CAMPAIGN_GATES, gene_pool=gene_pool.load(), population=args["pop"],
                    generations=args["gen"], trials_budget=args["budget"], final_n=args["final_n"],
                    islands=IslandConfig(k=int(islands)) if islands else None)
    symbol, timeframe, dataset_id = PINNED_CELLS[cell]
    cc = campaign_config_for(symbol=symbol, timeframe=timeframe, population=cfg.population,
                             target_sharpe=CAMPAIGN_TARGET_SHARPE, seed=cfg.seed, dataset_id=dataset_id,
                             gene_pool=cfg.gene_pool, min_trades=cfg.gates.min_trades, ga=cfg.block())
    h = config_hash(cc)
    cid = campaign_id_for(cell, cfg, h)
    return {"campaign_id": cid, "config_hash": h, "dir": artifact_path("ga", cid)}


def window(cell: str, budget: int) -> dict[str, Any]:
    """The engine's MinBTL for this budget against the cell's audited years: the small-window SMOKE-only warning."""
    from engine.research.ga.config import PINNED_CELLS
    from engine.research.search.run_campaign_cli import (
        CAMPAIGN_TARGET_SHARPE,
        TARGET_AUDIT_CSV_NAME,
        artifact_path,
        load_target_audit,
    )
    from engine.validation.minbtl import min_backtest_length

    sym, tf, _ds = PINNED_CELLS[cell]
    p = artifact_path(TARGET_AUDIT_CSV_NAME)
    try:
        row = load_target_audit()[1].get((sym, tf))
        years = float(row["years_available"]) if row else None
    except (FileNotFoundError, KeyError, ValueError):
        years = None
    if not years or years <= 0:
        return {"state": "NO DATA", "reason": f"no usable years_available for {sym} {tf} in the target audit",
                "artifact_path": display(p) if p.is_file() else None}
    res = min_backtest_length(n_trials=budget, target_sharpe=CAMPAIGN_TARGET_SHARPE, history_years=years)
    out = {"state": "OK", "years_available": addressed(years, p, field=f"{sym} {tf} years_available"),
           "minbtl_required_years": res.min_years, "n_trials": budget, "target_sharpe": CAMPAIGN_TARGET_SHARPE,
           "satisfied": res.satisfied, "source": "engine.validation.minbtl.min_backtest_length (Bailey et al. 2014)"}
    if not res.satisfied:
        out["warning"] = (f"small window: MinBTL for {budget:,} trials at target Sharpe {CAMPAIGN_TARGET_SHARPE} needs "
                          f"{res.min_years:.2f} years; {sym} {tf} has {years:.2f}: SMOKE-only, no edge claim from this "
                          "run")
    return out


# ---- runner files: opened only when no writer is alive ---------------------------------------------------------------
#
# On Windows any open handle on a file -- even one sharing DELETE -- makes os.replace over it fail (WinError 5), and the
# runner writes its checkpoint and monitor by os.replace (checkpoint.py save, mv2.write_json): a status read during a run
# could crash the campaign. So nothing the runner writes is opened while a process of that run is alive; live progress
# comes from the runner's own stdout lines in runner.log, which the sidecar writes itself.

def writer_alive(run: str) -> bool:
    return bool(live(run)) or any(r == run for r, _pid in unowned_alive())


def checkpoint_state(path: Path, config_hash: str) -> dict[str, Any]:
    """{generation, registry_rows, sha256} of the newest checkpoint, verified against the run's config hash by the
    engine's own loader (CheckpointError otherwise). Refused while a process of the run is alive."""
    from engine.research.ga import checkpoint as ckpt

    if writer_alive(path.parent.name):
        raise RuntimeError(f"{display(path)} is not opened while its writer is alive")
    st = path.stat()
    key = (str(path), st.st_size, st.st_mtime_ns, config_hash)
    with _ckpt_lock:
        if key not in _ckpt_cache:
            data = path.read_bytes()
            state = ckpt.load(path, config_hash=config_hash)
            _ckpt_cache[key] = {"generation": int(state["generation"]),
                                "registry_rows": int(state["extra"]["registry_rows"]),
                                "sha256": hashlib.sha256(data).hexdigest()}
        return _ckpt_cache[key]


def registry_digest(db: Path, campaign_id: str, rows: int) -> str | None:
    """sha256 over the campaign's first ``rows`` trial rows in rowid order -- the history a resume continues from.
    Opened immutable (no WAL/SHM sidecar is created); None when the file or the rows are missing."""
    if not db.is_file():
        return None
    con = sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro&immutable=1", uri=True)
    try:
        got = con.execute("SELECT * FROM trials WHERE campaign_id = ? ORDER BY rowid LIMIT ?",
                          (campaign_id, rows)).fetchall()
    finally:
        con.close()
    if len(got) < rows:
        return None
    h = hashlib.sha256()
    for r in got:
        h.update(json.dumps(r, default=repr).encode("utf-8") + b"\n")
    return h.hexdigest()


def record_exit(d: Path, artifacts: Path, rc: int, stopped: bool) -> None:
    """Fingerprint what this run's process left (checkpoint + the first registry_rows rows): RESUME's evidence.
    ``d`` and ``artifacts`` are the run directory and home captured at spawn time."""
    prereg = read_json(d / "preregistration.json")
    rec: dict[str, Any] = {"utc": now_utc(), "rc": rc, "stopped": stopped, "checkpoint": None, "rows_sha256": None}
    ck = d / "checkpoint.npz"
    if ck.is_file():
        try:
            rec["checkpoint"] = checkpoint_state(ck, prereg["config_hash"])
            db = artifacts / "ga" / prereg["campaign_id"] / "trials.sqlite"
            rec["rows_sha256"] = registry_digest(db, prereg["campaign_id"], rec["checkpoint"]["registry_rows"])
        except Exception as exc:  # recorded as such; RESUME then refuses for want of evidence
            rec["error"] = f"{type(exc).__name__}: {exc}"
    append(d / "exits.jsonl", rec)


# ---- processes ------------------------------------------------------------------------------------------------------

def adopt(proc: subprocess.Popen) -> None:
    """Make a spawned process a member of a kill-on-close job whose only handle the sidecar holds: when the sidecar
    exits, the job closes and its members die, so an orphaned runner cannot outlive its sidecar. The sidecar itself is
    never a member (a member's own exit code is overwritten when the job closes). The venv launcher kills its own
    interpreter child when it dies, which covers the instant before this assignment."""
    global _job
    if os.name != "nt":
        return
    import win32api
    import win32con
    import win32job

    if _job is None:
        job = win32job.CreateJobObject(None, "")
        info = win32job.QueryInformationJobObject(job, win32job.JobObjectExtendedLimitInformation)
        info["BasicLimitInformation"]["LimitFlags"] |= win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        win32job.SetInformationJobObject(job, win32job.JobObjectExtendedLimitInformation, info)
        _job = job
    h = win32api.OpenProcess(win32con.PROCESS_SET_QUOTA | win32con.PROCESS_TERMINATE, False, proc.pid)
    try:
        win32job.AssignProcessToJobObject(_job, h)
    finally:
        win32api.CloseHandle(h)


def process_created(pid: int) -> str | None:
    """The creation time of a live process, or None when it is not running (a liveness read; nothing is killed)."""
    if os.name != "nt":
        return None
    import pywintypes
    import win32api
    import win32process

    try:
        h = win32api.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    except pywintypes.error:
        return None
    try:
        if win32process.GetExitCodeProcess(h) != STILL_ACTIVE:
            return None
        return str(win32process.GetProcessTimes(h)["CreationTime"])
    finally:
        win32api.CloseHandle(h)


def live(run: str | None = None) -> dict[str, dict[str, Any]]:
    return {r: p for r, p in _procs.items() if p["proc"].poll() is None and (run is None or r == run)}


def unowned_alive() -> list[tuple[str, int]]:
    """(run, pid) of spawn records of earlier sidecar sessions whose process (same pid and creation time) still runs."""
    owned = {p["proc"].pid for p in _procs.values()}
    root = runs_root()
    out = []
    for f in sorted(root.glob("*/spawns.jsonl")) if root.is_dir() else []:
        for s in jsonl(f):
            if s["pid"] not in owned and s.get("created") and process_created(s["pid"]) == s["created"]:
                out.append((f.parent.name, s["pid"]))
    return out


def source_sidecar() -> bool:
    """True when ``sys.executable`` is a Python interpreter (``-m`` works), whatever the file is called: the
    installed app ships CPython itself as arqvio-engine.exe. False for a compiled binary (Nuitka marks every compiled
    module with ``__compiled__``; freezers set ``sys.frozen``), which runs modules through ``--run-module``."""
    return "__compiled__" not in globals() and not getattr(sys, "frozen", False)


def guard_spawn(run: str | None, cell: str, step: str) -> None:
    if live():
        raise refuse(f"another campaign process is running ({', '.join(live())})", run=run, cell=cell, step=step)
    alive = "; ".join(f"{r} pid {pid}" for r, pid in unowned_alive())
    if alive:
        raise refuse(f"a process spawned by an earlier sidecar session is still alive ({alive}); it is "
                     "never killed by a recorded pid -- read its command line and close it first", run=run, cell=cell,
                     step=step)


def runner_args(cell: str, args: dict[str, int], seed: int, checkpoint: Path | None, *,
                workers: int | None = None, islands: int | None = None) -> list[str]:
    out = ["--cell", cell, "--pop", str(args["pop"]), "--gen", str(args["gen"]), "--seed", str(seed),
           "--budget", str(args["budget"]), "--final-n", str(args["final_n"]), "--engine", "auto"]
    if islands:  # the tribal search (ga/islands.py): no checkpoint yet, so it cannot pause or resume
        out += ["--islands", str(int(islands))]
        checkpoint = None
    if workers is not None:  # operational (outside the config hash); runs registered before power modes have none
        out += ["--workers", str(workers)]
    if checkpoint is not None:
        out += ["--checkpoint", str(checkpoint), "--monitor-snapshot-every", str(MONITOR_EVERY_S)]
    return out


def launcher_argv(mode: str, campaign_id: str, rargs: list[str]) -> list[str]:
    # A packaged sidecar has no "-m": it runs the same module through its own --run-module dispatch.
    head = [sys.executable, "-u", "-m", LAUNCHER_MODULE] if source_sidecar() else         [sys.executable, "--run-module", LAUNCHER_MODULE]
    return [*head, "--runner", RUNNER_MODULE, "--expect-campaign-id", campaign_id, "--mode", mode, "--", *rargs]


def pump(run: str, kind: str, proc: subprocess.Popen, log: Path, d: Path, artifacts: Path) -> None:
    """Copy the child's merged output to the run log as '<utc> | <LEVEL> | <text>' lines, then its exit line, then
    (for a runner) the exit fingerprint. The runner's own CHECKPOINT and gen lines are remembered with their log line
    number: they are the live progress (runner files are not opened while it runs)."""
    n = len(log.read_text(encoding="utf-8").splitlines()) if log.is_file() else 0
    entry = _procs[run]
    with log.open("a", encoding="utf-8") as f:
        for raw in proc.stdout:
            line = ANSI.sub("", raw.rstrip("\r\n"))
            m = LOGURU.match(line)
            level = (m["level"] if m else "WARN" if line.startswith(("REFUSED", "UNAVAILABLE")) else "INFO")
            text = m["text"] if m else line
            f.write(f"{now_utc()} | {'WARN' if level == 'WARNING' else level} | {text}\n")
            f.flush()
            n += 1
            ck, gen = CHECKPOINT_LINE.match(text), GEN_LINE.match(text)
            if ck:
                entry["last_checkpoint"] = {"generation": int(ck["g"]), "line": n}
            if gen:
                entry["last_generation"] = {"text": text.strip(), "line": n}
        rc = proc.wait()
        f.write(f"{now_utc()} | {'INFO' if rc == 0 else 'ERROR'} | process exited rc={rc}\n")
    if kind == "runner":
        record_exit(d, artifacts, rc, bool(entry.get("stopped")))


def spawn(run: str, kind: str, argv: list[str], home: Path, log: Path) -> subprocess.Popen:
    env = {**os.environ, "ARQVIO_HOME": str(home), "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.Popen(argv, cwd=str(paths.PROJECT_ROOT), env=env, stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                            errors="replace", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        adopt(proc)
    except Exception as exc:  # a child that could outlive the sidecar is not left running
        kill_tree(proc)
        raise OSError(f"could not bind the spawned process to the sidecar's lifetime: {exc}") from exc
    d = runs_root() / run
    t = threading.Thread(target=pump, args=(run, kind, proc, log, d, get_artifacts_dir()), daemon=True)
    _procs[run] = {"kind": kind, "proc": proc, "thread": t, "argv": argv}
    append(d / "spawns.jsonl", {"utc": now_utc(), "kind": kind, "pid": proc.pid,
                                "created": process_created(proc.pid), "argv": argv, "arqvio_home": str(home)})
    t.start()
    return proc


def kill_tree(proc: subprocess.Popen) -> int:
    """Terminate a spawned process and its children (the venv launcher spawns the interpreter as a child)."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True, check=False)
    else:
        proc.terminate()
    rc = proc.wait(timeout=60)
    entry = next((e for e in _procs.values() if e["proc"] is proc), None)
    if entry is not None:
        entry["thread"].join(timeout=30)
    return rc


# ---- preflight ------------------------------------------------------------------------------------------------------

def probe_home(target: Path, cell: str) -> None:
    """An isolated home holding COPIES of the GA inputs (the real home is only read)."""
    from engine.research.ga.config import PINNED_CELLS, cross_cell_for

    src = get_artifacts_dir()
    dst = target / "artifacts"
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("spread", "specs"):
        if (src / name).is_dir():
            shutil.copytree(src / name, dst / name)
    audit = src / "history_depth_audit_target.csv"
    if audit.is_file():  # a fresh app home has none: its QUICK cells carry their own audit rows
        shutil.copy2(audit, dst / "history_depth_audit_target.csv")
    for c in (cell, cross_cell_for(cell)):
        sym, tf, ds = PINNED_CELLS[c]
        for f in (src / "datasets").glob(f"{sym}_{tf}_{ds}.*"):
            (dst / "datasets").mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst / "datasets" / f.name)
    for name in ("gene_pool.json", "user_cells.json"):  # the owner's pool, and the QUICK cells the probe resolves
        f = get_arqvio_home() / "configs" / name
        if f.is_file():
            (target / "configs").mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target / "configs" / name)


def settle_probe(run: str, d: Path) -> None:
    """When the probe has exited, write preflight.json once from the probe's own manifest."""
    with _lock:
        pre = d / "preflight.json"
        entry = _procs.get(run)
        if pre.is_file() or entry is None or entry["kind"] != "probe" or entry["proc"].poll() is None:
            return
        entry["thread"].join(timeout=30)
        req = read_json(d / "request.json")
        log = d / "probe" / "probe.log"
        mans = sorted((d / "probe" / "home" / "artifacts" / "ga").glob("*/manifest_v2.json"))
        rc = entry["proc"].returncode
        doc: dict[str, Any] = {"schema": "arqvio.app.preflight", "version": 1, "run": run, "utc": now_utc(),
                               "probe": {"args": presets()["probe"], "argv": entry["argv"], "exit_code": rc,
                                         "log": display(log), "log_sha256": sha256_file(log)}}
        if rc != 0 or len(mans) != 1:
            doc.update(state="UNAVAILABLE", reason=f"probe exit {rc}, {len(mans)} manifest(s); see {display(log)}")
        else:
            man = mans[0]
            r = read_json(man)["run"]
            w, used, fin = r["wall_seconds"], r["trials_used"], r["finalists"]
            a = req["args"]
            per_trial = w["search"] / used
            est: dict[str, Any] = {"per_trial_s": per_trial, "formula_note": "the budget caps new backtests and a run "
                                                                             "can stop earlier"}
            if fin:
                per_fin = w["cross"] / fin
                est.update(per_finalist_cross_s=per_fin, kind="upper bound",
                           projected_s=w["prepare"] + per_trial * a["budget"] + per_fin * a["final_n"],
                           formula="prepare + (search / trials_used) x budget + (cross / finalists) x final_n")
            else:
                est.update(per_finalist_cross_s=None, kind="lower bound",
                           projected_s=w["prepare"] + per_trial * a["budget"],
                           formula="prepare + (search / trials_used) x budget; the cross-check wall is not measured "
                                   "(the probe had 0 finalists)")
            doc.update(state="OK", manifest=display(man), manifest_sha256=sha256_file(man),
                       measured={"prepare_s": w["prepare"], "search_s": w["search"], "cross_s": w["cross"],
                                 "trials_used": used, "finalists": fin}, estimate=est)
        with pre.open("x", encoding="utf-8") as f:
            f.write(json.dumps(doc, indent=2, sort_keys=True))
        event("PREFLIGHT", run, req["cell"], "preflight", doc.get("reason"), level="INFO" if rc == 0 else "WARN",
              state=doc["state"])


class Plan(BaseModel):
    cell: str
    preset: Literal["SMOKE", "STANDARD", "DEEP", "CUSTOM"]
    seed: int = Field(ge=0)
    pop: int | None = None
    gen: int | None = None
    budget: int | None = None
    final_n: int | None = None
    islands: int | None = Field(None, ge=2, le=4)  # the tribal search: k tribes with revolution and war


@router.get("/presets")
def get_presets() -> dict[str, Any]:
    p = repo_path("configs/app/campaign_presets.json")
    return {"artifact_path": display(p), "sha256": sha256_file(p), **presets()}


@router.post("/preflight")
def preflight(plan: Plan) -> dict[str, Any]:
    validated_entry(plan.cell)  # L2: a non-validated cell is 404
    given = {k: getattr(plan, k) for k in ARG_KEYS}
    if plan.preset == "CUSTOM":
        if any(v is None for v in given.values()):
            raise refuse("CUSTOM needs pop, gen, budget and final_n", run=None, cell=plan.cell, step="preflight",
                         status=422)
        args = {k: int(v) for k, v in given.items()}
    else:
        if any(v is not None for v in given.values()):
            raise refuse(f"preset {plan.preset} is sealed; its values cannot be overridden", run=None,
                         cell=plan.cell, step="preflight", status=422)
        args = {k: presets()["presets"][plan.preset][k] for k in ARG_KEYS}
    from engine.research.ga.config import GAConfigError

    try:
        target = campaign_target(plan.cell, args, plan.seed, plan.islands)
        probe_target = campaign_target(plan.cell, presets()["probe"], plan.seed, plan.islands)
    except GAConfigError as exc:
        raise refuse(f"configuration refused by the engine: {exc}", run=None, cell=plan.cell, step="preflight",
                     status=422) from exc
    if target["dir"].exists():
        raise refuse(H_START.format(target=display(target["dir"])), run=None, cell=plan.cell, step="preflight")
    with _lock:
        guard_spawn(None, plan.cell, "preflight")
        run = f"{datetime.now(tz=UTC):%Y%m%dT%H%M%SZ}-{plan.cell}-s{plan.seed}"
        d = runs_root() / run
        if d.exists():
            raise refuse(f"run {run} exists", run=run, cell=plan.cell, step="preflight")
        (d / "probe" / "home").mkdir(parents=True)
        (d / "request.json").write_text(json.dumps(
            {"schema": "arqvio.app.campaign_request", "version": 1, "run": run, "utc": now_utc(), "cell": plan.cell,
             "preset": plan.preset, "seed": plan.seed, "args": args, "campaign_id": target["campaign_id"],
             "config_hash": target["config_hash"], "window": window(plan.cell, args["budget"]),
             **({"islands": plan.islands} if plan.islands else {})},
            indent=2, sort_keys=True), encoding="utf-8")
        try:
            probe_home(d / "probe" / "home", plan.cell)
            spawn(run, "probe", launcher_argv("fresh", probe_target["campaign_id"],
                                              runner_args(plan.cell, presets()["probe"], plan.seed, None,
                                                          islands=plan.islands)),
                  d / "probe" / "home", d / "probe" / "probe.log")
        except OSError as exc:  # e.g. a path past MAX_PATH: recorded, never a half-made run
            reason = f"the probe could not be started: {type(exc).__name__}: {exc}"
            (d / "preflight.json").write_text(json.dumps(
                {"schema": "arqvio.app.preflight", "version": 1, "run": run, "utc": now_utc(), "state": "UNAVAILABLE",
                 "reason": reason}, indent=2, sort_keys=True), encoding="utf-8")
            raise refuse(reason, run=run, cell=plan.cell, step="preflight") from exc
    event("PROBING", run, plan.cell, "preflight", campaign_id=target["campaign_id"])
    return {"run": run, "state": "PROBING", "campaign_id": target["campaign_id"]}


# ---- start / stop / resume ------------------------------------------------------------------------------------------

class Start(BaseModel):
    confirm: bool
    # Operational: how many CPU cores the run uses. The results are identical for every mode.
    power: Literal["eco", "medium", "high", "extreme"] = "high"


@router.post("/{run}/start")
def start(run: str, body: Start) -> dict[str, Any]:
    d = run_dir(run)
    settle_probe(run, d)
    req = read_json(d / "request.json")
    cell = req["cell"]
    if not body.confirm:
        raise refuse("START needs confirm=true (a run starts once: the one-attempt law)", run=run, cell=cell,
                     step="start", status=422)
    if (d / "preregistration.json").exists():
        raise refuse("this run was already started; a run starts once (one-attempt law) -- use RESUME", run=run,
                     cell=cell, step="start")
    pre = d / "preflight.json"
    if not pre.is_file() or read_json(pre).get("state") != "OK":
        raise refuse("no OK measured preflight for this run", run=run, cell=cell, step="start")
    target = campaign_target(cell, req["args"], req["seed"], req.get("islands"))
    if target["campaign_id"] != req["campaign_id"]:
        raise refuse(f"campaign id drifted since preflight ({req['campaign_id']} -> {target['campaign_id']}); "
                     "make a new preflight", run=run, cell=cell, step="start")
    with _lock:
        guard_spawn(run, cell, "start")
        if target["dir"].exists():
            raise refuse(H_START.format(target=display(target["dir"])), run=run, cell=cell, step="start")
        from engine.core.git_sha import resolve_git_sha
        from engine.research.ga.parallel import cpu_count, workers_for

        workers = workers_for(body.power)
        rargs = runner_args(cell, req["args"], req["seed"], d / "checkpoint.npz", workers=workers,
                            islands=req.get("islands"))
        argv = launcher_argv("fresh", target["campaign_id"], rargs)
        prereg = {"schema": "arqvio.app.preregistration", "version": 1, "run": run, "utc": now_utc(), "cell": cell,
                  "preset": req["preset"], "seed": req["seed"], "args": req["args"], "runner_args": rargs,
                  "power": {"mode": body.power, "workers": workers, "logical_cores": cpu_count()},
                  "argv": argv, "campaign_id": target["campaign_id"], "config_hash": target["config_hash"],
                  "target_dir": display(target["dir"]), "target_absent_at_start": True,
                  "git_head": resolve_git_sha(), "registry_entry_hash": validated_entry(cell)["entry_hash"],
                  "preflight": {"path": display(pre), "sha256": sha256_file(pre)}, "window": req.get("window"),
                  "d3": prior_trials(cell),
                  "one_attempt": "a run starts once; RESUME continues it with the same arguments only"}
        with (d / "preregistration.json").open("x", encoding="utf-8") as f:
            f.write(json.dumps(prereg, indent=2, sort_keys=True))
        try:
            spawn(run, "runner", argv, get_arqvio_home(), d / "runner.log")
        except OSError as exc:
            event("SPAWN FAILED", run, cell, "start", f"{type(exc).__name__}: {exc}", level="ERROR")
            raise HTTPException(status_code=500, detail=f"spawn failed after pre-registration: {exc}") from exc
    event("STARTED", run, cell, "start", campaign_id=target["campaign_id"])
    return {"run": run, "state": "RUNNING", "campaign_id": target["campaign_id"]}


@router.post("/{run}/stop")
def stop(run: str) -> dict[str, Any]:
    d = run_dir(run)
    cell = read_json(d / "request.json")["cell"]
    entry = live(run).get(run)
    if entry is None:
        raise refuse(f"no live process owned by this sidecar session for run {run}", run=run, cell=cell, step="stop")
    entry["stopped"] = True
    rc = kill_tree(entry["proc"])
    gen = checkpoint_generation(d)
    event("STOPPED", run, cell, "stop", rc=rc, checkpoint_generation=gen["value"])
    return {"run": run, "state": "STOPPED", "rc": rc, "checkpoint_generation": gen}


class Resume(BaseModel):
    cell: str | None = None
    seed: int | None = None
    pop: int | None = None
    gen: int | None = None
    budget: int | None = None
    final_n: int | None = None


@router.post("/{run}/resume")
def resume(run: str, body: Resume) -> dict[str, Any]:
    d = run_dir(run)
    req = read_json(d / "request.json")
    cell = req["cell"]
    pr = d / "preregistration.json"
    if not pr.is_file():
        raise refuse("this run was never started; nothing to resume", run=run, cell=cell, step="resume")
    prereg = read_json(pr)
    if req.get("islands"):
        raise refuse("a tribal (island) search keeps no checkpoint yet, so it cannot resume; start a new one",
                     run=run, cell=cell, step="resume")
    recorded = {"cell": prereg["cell"], "seed": prereg["seed"], **prereg["args"]}
    diff = {k: {"recorded": recorded[k], "asked": v} for k, v in body.model_dump(exclude_none=True).items()
            if recorded[k] != v}
    if diff:
        raise refuse(f"resume arguments differ from the pre-registered run: {json.dumps(diff, sort_keys=True)}; "
                     "RESUME continues the SAME run only (S4 checkpoint contract)", run=run, cell=cell, step="resume")
    ck = d / "checkpoint.npz"
    workers = prereg.get("power", {}).get("workers")
    if runner_args(cell, prereg["args"], prereg["seed"], ck, workers=workers) != prereg["runner_args"]:
        raise refuse("the recorded runner arguments are not this run's (values or checkpoint path differ)", run=run,
                     cell=cell, step="resume")
    target = campaign_target(cell, prereg["args"], prereg["seed"])
    if target["campaign_id"] != prereg["campaign_id"]:
        raise refuse(f"campaign id drift: pre-registered {prereg['campaign_id']}, computed now "
                     f"{target['campaign_id']}", run=run, cell=cell, step="resume")
    if (target["dir"] / "manifest_v2.json").is_file():
        raise refuse("this run completed; a completed campaign is never resumed", run=run, cell=cell, step="resume")
    if not target["dir"].is_dir():
        raise refuse(f"{display(target['dir'])} is missing; nothing of this run to continue", run=run, cell=cell,
                     step="resume")
    if not ck.is_file():
        raise refuse("no checkpoint was written for this run", run=run, cell=cell, step="resume")
    with _lock:
        guard_spawn(run, cell, "resume")
        runs = [s for s in jsonl(d / "spawns.jsonl") if s["kind"] == "runner"]
        exits = jsonl(d / "exits.jsonl")
        if len(exits) != len(runs) or exits[-1].get("checkpoint") is None or exits[-1].get("rows_sha256") is None:
            raise refuse("no exit fingerprint of this run's last process (checkpoint + registry rows); the history "
                         "cannot be proven to be this run's", run=run, cell=cell, step="resume")
        from engine.research.ga.checkpoint import CheckpointError

        try:
            now = checkpoint_state(ck, prereg["config_hash"])
        except CheckpointError as exc:
            raise refuse(f"checkpoint refused: {exc}", run=run, cell=cell, step="resume") from exc
        if now["sha256"] != exits[-1]["checkpoint"]["sha256"]:
            raise refuse("the checkpoint changed after this run's last process exited", run=run, cell=cell,
                         step="resume")
        db = target["dir"] / "trials.sqlite"
        if registry_digest(db, prereg["campaign_id"], now["registry_rows"]) != exits[-1]["rows_sha256"]:
            raise refuse(f"the trial registry's first {now['registry_rows']} rows are not the history this run left "
                         "at its last exit", run=run, cell=cell, step="resume")
        spawn(run, "runner", launcher_argv("resume", prereg["campaign_id"], [*prereg["runner_args"], "--resume",
                                                                             str(ck)]),
              get_arqvio_home(), d / "runner.log")
    event("RESUMED", run, cell, "resume", after_generation=now["generation"])
    return {"run": run, "state": "RUNNING", "resumed_after_generation": now["generation"]}


# ---- status ---------------------------------------------------------------------------------------------------------

def checkpoint_generation(d: Path) -> dict[str, Any]:
    ck, pr = d / "checkpoint.npz", d / "preregistration.json"
    if not ck.is_file() or not pr.is_file():
        return no_data("no checkpoint written yet", None, "generation")
    try:
        st = checkpoint_state(ck, read_json(pr)["config_hash"])
    except Exception as exc:  # an unverifiable checkpoint is shown as such, never guessed
        return {"value": None, "unit": None, "artifact_path": display(ck), "sha256": None, "computed_utc": None,
                "state": "UNRELIABLE", "field": "generation", "reason": f"{type(exc).__name__}: {exc}"}
    return {"value": st["generation"], "unit": None, "artifact_path": display(ck), "sha256": st["sha256"],
            "computed_utc": None, "state": "OK", "field": "generation"}


def monitor_view(mon: Path) -> dict[str, Any]:
    if not mon.is_file():
        return no_data("no monitor snapshot yet", None, "monitor_snapshot.json")
    return addressed(read_json(mon), mon, field="monitor_snapshot.json")


def log_line(d: Path, mark: dict[str, Any] | None, value: Any, field: str) -> dict[str, Any]:
    """A live value read from the runner's own stdout line, addressed to its runner.log line."""
    if mark is None:
        return no_data("the runner has not printed it yet", None, field)
    log = d / "runner.log"
    return {"value": value, "unit": None, "artifact_path": f"{display(log)}:{mark['line']}", "sha256": None,
            "computed_utc": None, "state": "OK", "field": field}


LIVE_FILES = ("not opened while the runner writes it: on Windows an open handle makes the runner's atomic replace "
              "fail (WinError 5); live progress is read from the runner's own output lines")


def status_of(run: str, d: Path) -> dict[str, Any]:
    settle_probe(run, d)
    req = read_json(d / "request.json")
    pre, pr = d / "preflight.json", d / "preregistration.json"
    entry = _procs.get(run)
    target = get_artifacts_dir() / "ga" / req["campaign_id"]
    man = target / "manifest_v2.json"
    runs = [s for s in jsonl(d / "spawns.jsonl") if s["kind"] == "runner"]
    exits = jsonl(d / "exits.jsonl")
    own = entry if entry is not None and entry["proc"].poll() is None else None
    writing = own is not None or writer_alive(run)
    if own is not None:
        state = "PROBING" if own["kind"] == "probe" else "RUNNING"
    elif writing:
        state = "NOT OWNED (a process of an earlier sidecar session is still alive; see spawns.jsonl)"
    elif pr.is_file() and man.is_file():
        state = "COMPLETED"
    elif pr.is_file() and runs and len(exits) == len(runs):
        state = "STOPPED" if exits[-1]["stopped"] else f"EXITED rc={exits[-1]['rc']}"
    elif pr.is_file():
        state = "NOT OWNED (no exit record for the last spawn; see spawns.jsonl)"
    elif pre.is_file():
        state = "READY" if read_json(pre).get("state") == "OK" else "PREFLIGHT UNAVAILABLE"
    else:
        state = "PROBE LOST (sidecar restarted; make a new preflight)"
    out: dict[str, Any] = {"run": run, "state": state, "request": req,
                           "preflight": read_json(pre) if pre.is_file() else None,
                           "preregistration": ({"artifact_path": display(pr), "sha256": sha256_file(pr),
                                                **read_json(pr)} if pr.is_file() else None),
                           "exits": exits, "log_source": f"campaign:{run}", "probe_log_source": f"probe:{run}"}
    if writing:
        ck, gen = (own or {}).get("last_checkpoint"), (own or {}).get("last_generation")
        out.update(checkpoint_generation=log_line(d, ck, ck and ck["generation"], "CHECKPOINT after generation"),
                   progress=log_line(d, gen, gen and gen["text"], "gen line"),
                   monitor=no_data(LIVE_FILES, None, "monitor_snapshot.json"))
    else:
        out.update(checkpoint_generation=checkpoint_generation(d), progress=None,
                   monitor=monitor_view(target / "monitor_snapshot.json"))
    if state == "COMPLETED":
        r = read_json(man)["run"]
        out["result"] = {k: (addressed(r[k], man, field=f"run.{k}") if k in r
                             else no_data("not in the manifest", man, f"run.{k}"))
                         for k in ("stop_reason", "trials_used", "survivors", "finalists")}
    note = presets().get("scope_notes", {}).get(req["cell"])
    if note:
        p = repo_path(note["path"])
        out["scope_note"] = (addressed(read_json(p).get(note["field"]), p, field=note["field"])
                             if p.is_file() else no_data("scope source missing", None, note["field"]))
    return out


@router.get("")
def list_runs() -> dict[str, Any]:
    root = runs_root()
    runs = sorted((p.parent.name for p in root.glob("*/request.json")), reverse=True) if root.is_dir() else []
    return {"runs": [{k: v for k, v in status_of(r, root / r).items() if k in ("run", "state", "request")}
                     for r in runs]}


@router.get("/{run}/status")
def get_run(run: str) -> dict[str, Any]:
    return status_of(run, run_dir(run))
