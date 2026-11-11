"""Automations: the saved list, the activity log, and the scheduler that runs them inside the engine.

Research behaviour (worker4/RESEARCH_GAP_AUDIT.md §2.1, §2.4):
- an automation is a name, ON/OFF, a schedule (``every_hours``; 0 = NON-STOP) and a chain of steps;
- due (or RUN NOW) -> queued while the owner's own campaign breeds (an automation without a campaign step RUNS
  ALONGSIDE) -> step by step -> done ("finished all n steps") or failed ("finished with 1 failed step: ...");
- NON-STOP automations take turns (the one that ran longest ago goes first);
- STOP cancels the run and stops only the campaign this automation started;
- the engine closing mid-run leaves it INTERRUPTED; it carries on RESUME_S seconds after the engine is back;
- an edit while it runs is kept as ``next_steps`` and used from the next run.
Files under ``<home>/artifacts/automation/``: automations.json, activity.jsonl, run.json (the run in progress).
"""
from __future__ import annotations

import json
import secrets
import threading
import time
import traceback
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from engine.core import atomic
from engine.core.paths import get_artifacts_dir
from engine.research.automation import steps as S

SCHEMA = "arqvio.automations"
VERSION = 1
RESUME_S = 120
TICK_S = 2.0
HISTORY_KEEP = 40
_lock = threading.RLock()


def _dir() -> Path:
    return get_artifacts_dir() / "automation"


def _now() -> datetime:
    return datetime.now(tz=UTC)


def _iso(d: datetime) -> str:
    return d.isoformat(timespec="seconds")


def _write(p: Path, doc: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    atomic.replace(tmp, p)


# ---- the list -------------------------------------------------------------------------------------------------------

def load() -> list[dict[str, Any]]:
    try:
        doc = json.loads((_dir() / "automations.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return list(doc.get("automations") or []) if doc.get("schema") == SCHEMA else []


def _save(rows: list[dict[str, Any]]) -> None:
    _write(_dir() / "automations.json", {"schema": SCHEMA, "version": VERSION, "automations": rows})


def get(aid: str) -> dict[str, Any]:
    for a in load():
        if a["id"] == aid:
            return a
    raise KeyError(f"no automation {aid!r}")


def save(body: dict[str, Any]) -> dict[str, Any]:
    """Create or update; refused with research's messages. An edit to a running automation waits for its next run."""
    problems = S.validate(body)
    if problems:
        raise ValueError("; ".join(problems))
    with _lock:
        rows = load()
        clean = {"name": str(body["name"]).strip(), "enabled": bool(body.get("enabled", True)),
                 "every_hours": float(body.get("every_hours", 24)), "steps": list(body["steps"]),
                 "terminal": str(body.get("terminal") or "AUTO")}
        aid = body.get("id")
        cur = SCHEDULER.current_id()
        for a in rows:
            if a["id"] == aid:
                if aid == cur and a["steps"] != clean["steps"]:
                    a["next_steps"] = clean.pop("steps")
                a.update(clean)
                _save(rows)
                return a
        a = {"id": f"au{secrets.token_hex(4)}", **clean, "history": [], "created_utc": _iso(_now()),
             "next_at": _iso(_now()), "run_now": False, "last_run_utc": None}
        rows.append(a)
        _save(rows)
        activity(a, "ok", "saved; it starts as soon as ARQVIO is free")
        return a


def _update(aid: str, **kw: Any) -> dict[str, Any]:
    with _lock:
        rows = load()
        for a in rows:
            if a["id"] == aid:
                a.update(kw)
                _save(rows)
                return a
    raise KeyError(f"no automation {aid!r}")


def toggle(aid: str, enabled: bool) -> dict[str, Any]:
    a = _update(aid, enabled=bool(enabled))
    activity(a, "ok", "switched ON" if enabled else "switched OFF: saved, never starts")
    return a


def run_now(aid: str) -> dict[str, Any]:
    a = _update(aid, run_now=True)
    alongside = not any(s.get("kind") == "campaign" for s in a["steps"])
    return {**a, "toast": "STARTING · RUNS ALONGSIDE" if alongside else "QUEUED · STARTS WHEN YOUR CAMPAIGNS ARE DONE"}


def delete(aid: str) -> None:
    """Its schedule and history go; strategies and portfolios it created are not touched."""
    with _lock:
        if SCHEDULER.current_id() == aid:
            SCHEDULER.cancel(aid)
        rows = load()
        keep = [a for a in rows if a["id"] != aid]
        if len(keep) == len(rows):
            raise KeyError(f"no automation {aid!r}")
        _save(keep)


# ---- activity -------------------------------------------------------------------------------------------------------

def activity(auto: dict[str, Any], status: str, detail: str) -> None:
    from engine.core import applog

    rec = {"ts": _iso(_now()), "id": auto["id"], "name": auto["name"], "status": status, "detail": detail}
    p = _dir() / "activity.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    applog.write("automation", f"{auto['name']}: {detail}", {"ok": "INFO", "failed": "ERROR"}.get(status, "WARN"))


def activity_log(limit: int = 80) -> list[dict[str, Any]]:
    try:
        lines = (_dir() / "activity.jsonl").read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out = []
    for line in reversed(lines):
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
        if len(out) >= limit:
            break
    return out


# ---- the scheduler --------------------------------------------------------------------------------------------------

def _has_campaign(a: dict[str, Any]) -> bool:
    return any(s.get("kind") == "campaign" for s in a.get("steps") or [])


class Scheduler:
    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._worker: threading.Thread | None = None
        self._cancel = threading.Event()
        self.state: dict[str, Any] = {}
        self.pending: dict[str, Any] | None = None
        self.resume_at: datetime | None = None
        self._started = False
        self._halt = threading.Event()

    # -- public --
    def current_id(self) -> str | None:
        w = self._worker
        return self.state.get("automation_id") if w is not None and w.is_alive() else None

    def cancel(self, aid: str) -> bool:
        if self.current_id() == aid:
            self._cancel.set()
            return True
        if self.resume_at is not None and self._run_file().is_file():
            run = json.loads(self._run_file().read_text(encoding="utf-8"))
            if run.get("automation_id") == aid:
                self._run_file().unlink(missing_ok=True)
                self.resume_at = None
                a = get(aid)
                activity(a, "paused", "stopped by you while waiting to carry on")
                return True
        return False

    def start(self) -> None:
        with _lock:
            if self._started:
                return
            self._started = True
            if self._run_file().is_file():  # the engine closed mid-run: INTERRUPTED, carries on in RESUME_S
                self.resume_at = _now() + timedelta(seconds=RESUME_S)
            self._thread = threading.Thread(target=self._loop, daemon=True, name="arqvio-automation")
            self._thread.start()

    def snapshot(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.current_id():
            out["running"] = dict(self.state)
        elif self.resume_at is not None and self._run_file().is_file():
            run = json.loads(self._run_file().read_text(encoding="utf-8"))
            left = max(0, int((self.resume_at - _now()).total_seconds()))
            out["running"] = {**run, "phase": "INTERRUPTED", "resume_in": left, "resume_at": _iso(self.resume_at)}
        if self.pending:
            out["pending"] = self.pending
        rows = load()
        due = sorted((a for a in rows if a.get("enabled") and a.get("next_at") and not a.get("run_now")),
                     key=lambda a: a["next_at"])
        if due:
            out["due_next"] = {"id": due[0]["id"], "name": due[0]["name"], "ts": due[0]["next_at"]}
        last = next((a["history"][0] for a in sorted(rows, key=lambda a: a.get("last_run_utc") or "", reverse=True)
                     if a.get("history")), None)
        if last:
            out["last_result"] = last
        return out

    # -- internals --
    @staticmethod
    def _run_file() -> Path:
        return _dir() / "run.json"

    def stop(self) -> None:
        """End the loop (tests; the app's engine runs it for its whole life). A running step finishes first."""
        self._halt.set()
        self._cancel.set()

    def _loop(self) -> None:
        while not self._halt.is_set():
            try:
                self._tick()
            except Exception:  # the loop never dies; the reason reaches the engine log
                from engine.core import applog

                applog.write("automation", f"scheduler: {traceback.format_exc(limit=3)}", "ERROR")
            time.sleep(TICK_S)

    def _tick(self) -> None:
        from engine.api import routes_app_launch as launch

        if self.current_id():
            return
        if not launch.busy():
            auto_purge()
        if self.resume_at is not None:
            if _now() < self.resume_at:
                return
            self.resume_at = None
            if self._run_file().is_file():
                run = json.loads(self._run_file().read_text(encoding="utf-8"))
                try:
                    a = get(run["automation_id"])
                except KeyError:
                    self._run_file().unlink(missing_ok=True)
                    return
                activity(a, "ok", f"ARQVIO restarted: carrying on from step {run.get('step_idx', 0) + 1}")
                self._spawn(a, run)
                return
        now = _iso(_now())
        rows = [a for a in load() if a.get("enabled")]
        due = [a for a in rows if a.get("run_now") or (a.get("next_at") and a["next_at"] <= now)]
        # NON-STOP ones take turns: whoever ran longest ago first; RUN NOW first of all
        due.sort(key=lambda a: (not a.get("run_now"), a.get("last_run_utc") or "", a["id"]))
        self.pending = None
        for a in due:
            if _has_campaign(a) and launch.busy():
                self.pending = {"id": a["id"], "name": a["name"], "why": "waits for your campaign to finish"}
                continue
            self._spawn(a, None)
            return

    def _spawn(self, auto: dict[str, Any], resume: dict[str, Any] | None) -> None:
        self._cancel.clear()
        _update(auto["id"], run_now=False)
        self.state = {"automation_id": auto["id"], "name": auto["name"], "step_idx": 0,
                      "steps_total": len(auto["steps"]), "phase": "STARTING", "progress": None,
                      "started_utc": _iso(_now())}
        self._worker = threading.Thread(target=self._work, args=(auto, resume), daemon=True,
                                        name=f"arqvio-automation-{auto['id']}")
        self._worker.start()

    def _work(self, auto: dict[str, Any], resume: dict[str, Any] | None) -> None:
        def phase(label: str, progress: dict[str, Any] | None) -> None:
            self.state.update(phase=label, progress=progress)

        run = S.Run(auto, cancel=self._cancel, log=lambda m: activity(auto, "ok", m), phase=phase,
                    warn=lambda m: activity(auto, "warn", m))
        start = 0
        if resume:
            start = int(resume.get("step_idx", 0))
            run.run_pool = list(resume.get("run_pool") or [])
            run.portfolio_id = resume.get("portfolio_id")
            run.added = list(resume.get("added") or [])
            run.campaigned = bool(resume.get("campaigned"))
        failed: list[str] = []
        stopped = False
        steps = auto["steps"]
        for i in range(start, len(steps)):
            if self._cancel.is_set():
                stopped = True
                break
            step = steps[i]
            self.state.update(step_idx=i, phase=step["kind"].upper(), progress=None)
            _write(self._run_file(), {"automation_id": auto["id"], "name": auto["name"], "step_idx": i,
                                      "steps_total": len(steps), "run_pool": run.run_pool,
                                      "portfolio_id": run.portfolio_id, "added": run.added,
                                      "campaigned": run.campaigned, "utc": _iso(_now())})
            try:
                detail = S.RUNNERS[step["kind"]](step, run)
                activity(auto, "ok", f"step {i + 1}: {detail}")
            except Exception as exc:
                failed.append(f"step {i + 1}: {type(exc).__name__}: {exc}")
                activity(auto, "failed", f"step {i + 1}: {exc}")
        if self._cancel.is_set():
            stopped = True
        self._run_file().unlink(missing_ok=True)
        n = len(steps)
        if stopped:
            rec = {"ts": _iso(_now()), "ok": False, "paused": True, "detail": "stopped by you"}
        elif failed:
            rec = {"ts": _iso(_now()), "ok": False,
                   "detail": f"finished with {len(failed)} failed step{'s' if len(failed) > 1 else ''}: {failed[0]}"}
        else:
            rec = {"ts": _iso(_now()), "ok": True, "detail": f"finished all {n} step{'s' if n != 1 else ''}"}
        with _lock:
            rows = load()
            for a in rows:
                if a["id"] == auto["id"]:
                    a["history"] = ([rec] + list(a.get("history") or []))[:HISTORY_KEEP]
                    a["last_run_utc"] = rec["ts"]
                    if a.get("next_steps"):
                        a["steps"] = a.pop("next_steps")
                    a["next_at"] = _iso(_now() + timedelta(hours=float(a.get("every_hours") or 0)))
            _save(rows)
        activity(auto, "paused" if stopped else "failed" if failed else "ok", rec["detail"])
        self.state = {}


def auto_purge(now: datetime | None = None) -> dict[str, Any] | None:
    """Research AUTOMATIC PURGE (SETTINGS): every ``every_hours`` (the first sweep one interval after it is switched
    on), strategies older than ``older_than_days`` failing the saved purge criteria go to the recycle bin; above
    ``cap`` strategies the weakest older than 7 days follow. Portfolio members and live bots never. Paused while a
    campaign runs (the caller checks)."""
    from engine.core import app_settings
    from engine.research.automation import portfolios
    from engine.research.vault import library

    cfg = app_settings.load()["auto_purge"]
    p = _dir() / "auto_purge.json"
    now = now or _now()
    try:
        mark = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        mark = {}
    if not cfg["enabled"]:
        if mark:
            p.unlink(missing_ok=True)
        return None
    if not mark.get("next_utc"):
        _write(p, {"next_utc": _iso(now + timedelta(hours=cfg["every_hours"]))})
        return None
    if mark["next_utc"] > _iso(now):
        return None
    rows = library.load()
    protect = portfolios.protected_ids() | S._live_ids()
    old = [r for r in rows if now - datetime.fromisoformat(r["added_utc"]) >= timedelta(days=cfg["older_than_days"])]
    ids = S.purge_matches(S.saved_or({}), old, protect, now)
    left = [r for r in rows if r["id"] not in set(ids)]
    over = len(left) - int(cfg["cap"])
    capped: list[str] = []
    if over > 0:
        weak = sorted((r for r in left if r["id"] not in protect
                       and now - datetime.fromisoformat(r["added_utc"]) >= timedelta(days=7)),
                      key=lambda r: (float(r.get("score") or 0), r["id"]))
        capped = [r["id"] for r in weak[:over]]
    if ids or capped:
        library.remove_many(ids + capped, reason="automatic purge")
    _write(p, {"next_utc": _iso(now + timedelta(hours=cfg["every_hours"])), "last_utc": _iso(now),
               "last_count": len(ids) + len(capped)})
    activity({"id": "auto_purge", "name": "AUTOMATIC PURGE"}, "ok",
             f"{len(ids)} failed the purge criteria, {len(capped)} over the vault cap of {cfg['cap']}: "
             "to the recycle bin for 30 days")
    return {"purged": ids, "capped": capped}


SCHEDULER = Scheduler()


def state() -> dict[str, Any]:
    return SCHEDULER.snapshot()


__all__ = ["SCHEDULER", "activity", "activity_log", "delete", "get", "load", "run_now", "save", "state", "toggle"]
