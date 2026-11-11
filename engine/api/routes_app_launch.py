"""One LAUNCH (``/api/v2/app/launch``): the owner picks a terminal, markets and timeframes; the engine does the rest.

A launch is a queue the sidecar works through in a background thread, writing every step to the app's one log
(engine/core/applog.py):

1. data   each chosen market's specs, spread matrix and bars, read from the chosen terminal and pinned
          (engine/research/cells/add_market.py ``pull``). A timeframe with a single chosen market gets one more,
          a main pair, read only to re-test the finalists on ("cross" market).
2. cells  the markets entered as QUICK cells in one sealed write, each re-tested on another market of its timeframe.
3. breed  per market: the campaign preflight (a short trial run in an isolated home), then the campaign itself
          (routes_app_campaigns: the same pre-registration and one-attempt rules), followed live from its output.
4. vault  a completed campaign's finalists that are alive on their market go to the vault.

PAUSE stops the running campaign at its checkpoint and holds the queue; RESUME continues it (the campaign's own
RESUME); STOP ends the launch (the running campaign stops at its checkpoint, the rest is skipped). One launch at a
time. The state is ``<home>/artifacts/app/launch/<id>/launch.json``.
"""
from __future__ import annotations

import json
import re
import secrets
import threading
import time
import traceback
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from engine.core import applog, atomic
from engine.core.paths import get_artifacts_dir

router = APIRouter(prefix="/api/v2/app/launch", tags=["app-launch"])
log_router = APIRouter(prefix="/api/v2/app/log", tags=["app-log"])

DEPTHS = {"QUICK": "SMOKE", "STANDARD": "STANDARD", "DEEP": "DEEP"}
TIMEFRAMES = ("M5", "M15", "M30", "H1", "H4", "D1")
CROSS_PREFERENCE = ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "XAUUSD")
# A lone market is re-tested on a market of its own kind first (gold on silver, an index on another index): a gold
# strategy re-tested on EURUSD says little. Then the general preference above.
CROSS_RELATED: dict[str, tuple[str, ...]] = {
    "XAUUSD": ("XAGUSD",), "XAGUSD": ("XAUUSD",), "US30": ("US500", "US100"), "US500": ("US100", "US30"),
    "US100": ("US500", "US30"), "GER40": ("US500", "US100"), "USOIL": ("XAUUSD",),
    "EURJPY": ("GBPJPY", "USDJPY"), "GBPJPY": ("EURJPY", "USDJPY"), "USDJPY": ("EURJPY", "EURUSD"),
}
POLL_S = 2.0
GEN_LINE = re.compile(r"gen\s+(?P<g>\d+):.*?trials=(?P<t>\d+)/(?P<b>\d+).*?best=(?P<best>[-\d.]+).*?alive=(?P<alive>\d+)")
END_LINE = re.compile(r"stop=(?P<stop>\w+).*?trials=(?P<t>\d+) survivors=(?P<s>\d+) finalists=(?P<f>\d+)"
                      r" alive_on_both=(?P<both>\d+)")

_lock = threading.Lock()
_ctl: dict[str, Any] = {"thread": None, "pause": threading.Event(), "stop": threading.Event(), "state": None,
                        "skip": threading.Event()}  # skip: an engine V1 run finishes its generation, then Final War


def root() -> Path:
    return get_artifacts_dir() / "app" / "launch"


def now_utc() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds")


def _path(lid: str) -> Path:
    return root() / lid / "launch.json"


def save(st: dict[str, Any]) -> None:
    """Keep the state in memory (what GET answers while the launch runs) and on disk (what survives a restart)."""
    _ctl["state"] = json.loads(json.dumps(st))
    p = _path(st["id"])
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=2) + "\n", encoding="utf-8")
    atomic.replace(tmp, p)  # Windows refuses a replace while a reader holds the file: retried


def latest() -> dict[str, Any] | None:
    t = _ctl["thread"]
    if t and t.is_alive() and _ctl.get("state"):
        return json.loads(json.dumps(_ctl["state"]))
    r = root()
    files = sorted(r.glob("*/launch.json")) if r.is_dir() else []
    if not files:
        return None
    try:
        st = json.loads(files[-1].read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return _ctl.get("state")
    t = _ctl["thread"]
    if st.get("state") in ("RUNNING", "PAUSED") and not (t and t.is_alive()):
        st["state"] = "INTERRUPTED"  # the sidecar stopped while it ran (the app was closed)
    return st


def busy() -> bool:
    t = _ctl["thread"]
    return bool(t and t.is_alive())


class LaunchRequest(BaseModel):
    terminal: str
    symbols: list[str] = Field(min_length=1, max_length=12)
    timeframes: list[str] = Field(min_length=1, max_length=6)
    years: int = Field(6, ge=1, le=20)
    # "tribal": the island GA -- 2 tribes breeding apart, revolution and war (ga/islands.py); it cannot pause.
    # "v1": engine V1, the research engine (engine/research/v1): proving grounds, TRIBE A/B, Final War; PAUSE
    # freezes its process tree, SKIP finishes the generation and goes to the Final War.
    search: Literal["standard", "tribal", "v1"] = "standard"
    depth: Literal["QUICK", "STANDARD", "DEEP"] = "STANDARD"
    power: Literal["eco", "medium", "high", "extreme"] = "high"
    # AUTOMATION chain (research: CAMPAIGN -> PORTFOLIO -> PURGE -> PARITY, REPEAT): steps run when the launch
    # finishes (never when it is stopped). Purges move strategies to the recycle bin (restorable 30 days).
    after: list[Literal["purge_copies", "purge_held_out_failed", "purge_grade_f", "repeat"]] = Field(default_factory=list)
    # who started it: "" = the owner (LAUNCH / QUEUE); "automation:<id>" = an automation, whose STOP stops only it
    origin: str = Field("", max_length=64)


def defaults_path() -> Path:
    return root() / "launch_defaults.json"


def manual_defaults() -> dict[str, Any]:
    """The owner's last hand-started launch: an automation's blank campaign settings take these (research: blank =
    the Launch page value at run time)."""
    try:
        return json.loads(defaults_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def plan(req: LaunchRequest, offered: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The queue: data tasks (chosen markets, then any cross market a timeframe needs), then one breed task per
    chosen market. Pure: ``offered`` is the terminal's symbol list."""
    from engine.research.cells.add_market import cell_id
    from engine.research.data.terminals import main_pairs

    names = {s["name"] for s in offered}
    missing = [s for s in req.symbols if s not in names]
    if missing:
        raise ValueError(f"the terminal does not offer {missing}")
    bad = [t for t in req.timeframes if t not in TIMEFRAMES]
    if bad:
        raise ValueError(f"timeframes {bad}; one of {list(TIMEFRAMES)}")
    symbols = list(dict.fromkeys(req.symbols))
    tfs = [t for t in TIMEFRAMES if t in req.timeframes]
    pairs = {r["pair"]: r["symbol"] for r in main_pairs(offered) if r["symbol"]}
    data, breed = [], []
    for tf in tfs:
        for s in symbols:
            data.append({"kind": "data", "role": "market", "symbol": s, "timeframe": tf, "cell": cell_id(s, tf)})
            breed.append({"kind": "breed", "symbol": s, "timeframe": tf, "cell": cell_id(s, tf)})
        if len(symbols) == 1:
            own = next((p for p, name in pairs.items() if name == symbols[0]), None)
            order = (*CROSS_RELATED.get(own or "", ()), *CROSS_PREFERENCE)
            helper = next((pairs[p] for p in order if pairs.get(p) and pairs[p] != symbols[0]), None)
            if helper:
                data.append({"kind": "data", "role": "cross", "symbol": helper, "timeframe": tf,
                             "cell": cell_id(helper, tf)})
    for t in data + breed:
        t.update(state="WAITING", detail=None)
    return data + breed


def cross_map(tasks: list[dict[str, Any]]) -> dict[str, str]:
    """Each pulled market's cross cell: the next pulled market of its timeframe (a ring), so every cell re-tests its
    finalists on another market. A timeframe with one pulled market has no cross cell (it cannot be bred)."""
    out: dict[str, str] = {}
    for tf in TIMEFRAMES:
        ring = [t["cell"] for t in tasks if t["kind"] == "data" and t["timeframe"] == tf and t["state"] == "DONE"]
        if len(ring) >= 2:
            for i, c in enumerate(ring):
                out[c] = ring[(i + 1) % len(ring)]
    return out


# ---- the worker -------------------------------------------------------------------------------------------------------

def _set(st: dict[str, Any], task: dict[str, Any] | None = None, **kw: Any) -> None:
    (task if task is not None else st).update(kw)
    st["updated_utc"] = now_utc()
    save(st)


def _market(t: dict[str, Any]) -> str:
    return f"{t['symbol']} {t['timeframe']}"


def _wait_resume(st: dict[str, Any]) -> bool:
    """Hold while paused; False when the launch was stopped meanwhile."""
    while _ctl["pause"].is_set() and not _ctl["stop"].is_set():
        time.sleep(0.5)
    return not _ctl["stop"].is_set()


def window_days(tf: str, years: int) -> tuple[int, bool]:
    """Calendar days of history to read for ``tf``: the owner's years, widened when they cannot hold the bars a cell
    needs (MIN_BARS, with a quarter to spare; markets trade about 5 days in 7). Returns (days, widened)."""
    import math

    from engine.research.cells.add_market import MIN_BARS
    from engine.research.registry.annualisation import TF_MINUTES

    asked = int(round(365.25 * years))
    need = int(math.ceil(MIN_BARS * 1.25 * TF_MINUTES[tf] / (60 * 24) * 7 / 5))
    return (need, True) if need > asked else (asked, False)


def _data(st: dict[str, Any], mt5: Any) -> dict[str, dict[str, Any]]:
    from engine.research.cells import add_market

    req = st["request"]
    records: dict[str, dict[str, Any]] = {}
    end = datetime.now(tz=UTC).date()
    for t in [x for x in st["tasks"] if x["kind"] == "data"]:
        if _ctl["stop"].is_set() or not _wait_resume(st):
            _set(st, t, state="SKIPPED", detail="launch stopped")
            continue
        _set(st, t, state="RUNNING")
        days, widened = window_days(t["timeframe"], req["years"])
        start = end - timedelta(days=days)
        role = " (cross-check market)" if t["role"] == "cross" else ""
        if widened:
            applog.write("data", f"{_market(t)}: a cell needs {add_market.MIN_BARS:,} bars, about {days / 365.25:.1f} "
                                 f"years of {t['timeframe']}; reading that instead of {req['years']} year(s)")
        applog.write("data", f"Reading {_market(t)}{role} from MetaTrader 5: {days / 365.25:.1f} years of bars, "
                             "specs, spread")
        try:
            rec = add_market.pull(t["symbol"], t["timeframe"], start.isoformat(), end.isoformat(), mt5=mt5,
                                  terminal=req["terminal_path"])
        except add_market.AddMarketRefused as exc:
            _set(st, t, state="FAILED", detail=str(exc))
            applog.write("data", f"{_market(t)}: {exc}", "WARN")
            continue
        full = rec["window"]
        # SETTINGS TRAINING SPLIT (engine/core/app_settings.py): breeding sees this share, the rest is held out
        from engine.core import app_settings

        rec = add_market.hold_out(rec, frac=round(1 - app_settings.load()["campaign"]["train_split"], 4))
        ho = rec["held_out"]
        if ho["state"] == "KEPT":
            applog.write("data", f"{_market(t)}: breeding sees {rec['window']['bars']:,} bars to "
                                 f"{rec['window']['last_bar'][:10]}; the last {ho['bars']:,} bars (from "
                                 f"{ho['split_time'][:10]}) are held out for the test after breeding")
        else:
            applog.write("data", f"{_market(t)}: no held-out test: {ho['reason']}", "WARN")
        w = {**full, "bars": full["bars"]}
        _set(st, t, state="DONE", detail=f"{w['bars']:,} bars {w['first_bar'][:10]} to {w['last_bar'][:10]}",
             bars=w["bars"], dataset_hash16=rec["dataset_hash16"])
        records[t["cell"]] = rec
        applog.write("data", f"{_market(t)}: {w['bars']:,} bars, {w['first_bar'][:10]} to {w['last_bar'][:10]}", "OK")
        asked = datetime.fromisoformat(w["requested_start"]).date()
        if datetime.fromisoformat(w["first_bar"]).date() - asked > timedelta(days=45):
            applog.write("data", f"{_market(t)}: the terminal has history from {w['first_bar'][:10]} only; "
                                 "that is what was read", "WARN")
    return records


def _cells(st: dict[str, Any], records: dict[str, dict[str, Any]]) -> None:
    from engine.research.cells import add_market

    cross = cross_map(st["tasks"])
    for t in st["tasks"]:
        if t["kind"] == "breed" and t["state"] == "WAITING" and t["cell"] not in cross:
            data = next(x for x in st["tasks"] if x["kind"] == "data" and x["cell"] == t["cell"])
            why = data["detail"] if data["state"] != "DONE" else "no second market of this timeframe to re-test on"
            _set(st, t, state="SKIPPED", detail=why)
    pulls = [(records[c], x) for c, x in cross.items() if c in records]
    if pulls:
        add_market.register(pulls)
        st["cross"] = cross
        save(st)
        ready = [t for t in st["tasks"] if t["kind"] == "breed" and t["state"] == "WAITING"]
        applog.write("data", f"{len(ready)} market(s) ready to breed: " + ", ".join(
            f"{_market(t)} (re-tested on {cross[t['cell']]})" for t in ready), "OK")


def _forward(log: Path, offset: int, t: dict[str, Any], st: dict[str, Any]) -> int:
    """Forward the runner's new generation lines to the app log; returns the new offset."""
    if not log.is_file():
        return offset
    with log.open("r", encoding="utf-8", errors="replace") as f:
        f.seek(offset)
        chunk = f.read()
        offset = f.tell()
    for line in chunk.splitlines():
        m = GEN_LINE.search(line)
        if m:
            prog = {"generation": int(m["g"]), "generations": st["args"]["gen"], "trials": int(m["t"]),
                    "budget": int(m["b"]), "best_sharpe": float(m["best"]), "alive": int(m["alive"]),
                    "utc": now_utc()}
            _set(st, t, progress=prog)
            applog.write("breed", f"{_market(t)} · generation {prog['generation'] + 1}/{prog['generations']} · "
                                  f"{prog['trials']:,}/{prog['budget']:,} strategies tested · best Sharpe "
                                  f"{prog['best_sharpe']:.2f} · {prog['alive']} alive")
            continue
        e = END_LINE.search(line)
        if e:
            applog.write("breed", f"{_market(t)} · search {e['stop']}: {int(e['t']):,} strategies tested, "
                                  f"{e['s']} survived, {e['f']} finalists re-tested on the second market, "
                                  f"{e['both']} alive on both")
    return offset


# v4 early stop (ARQVIO_V4.md 9.6): a market with nothing alive after EARLY_STOP_GEN generations and a best Sharpe
# under EARLY_STOP_SHARPE is stopped, so the launch moves on instead of spending the rest of its budget there.
EARLY_STOP_GEN = 5
EARLY_STOP_SHARPE = 0.1


def hopeless(progress: dict[str, Any] | None) -> bool:
    if not progress:
        return False
    return (int(progress["generation"]) + 1 >= EARLY_STOP_GEN and int(progress["alive"]) == 0
            and float(progress["best_sharpe"]) < EARLY_STOP_SHARPE)


V1_GEN = re.compile(r"GEN (?P<g>\d+)/(?P<n>\d+) · SURVIVORS (?P<s>\d+) · TOP FITNESS (?P<f>[\d.]+) · "
                    r"tested (?P<t>\d+)")
V1_PRESETS = {"QUICK": "quick", "STANDARD": "research", "DEEP": "research"}
V1_RISK_PCT = 0.01


def _v1_argv(cell: str, preset: str, seed: int, power: str, out: Path) -> list[str]:
    import sys

    from engine.api import routes_app_campaigns as campaigns

    head = ([sys.executable, "-u", "-m", "engine.research.v1.cli"] if campaigns.source_sidecar()
            else [sys.executable, "--run-module", "engine.research.v1.cli"])
    return [*head, "--cell", cell, "--preset", preset, "--seed", str(seed), "--risk", str(V1_RISK_PCT),
            "--power", power, "--out", str(out)]


def _forward_v1(log: Path, offset: int, t: dict[str, Any], st: dict[str, Any]) -> int:
    if not log.is_file():
        return offset
    with log.open("r", encoding="utf-8", errors="replace") as f:
        f.seek(offset)
        chunk = f.read()
        offset = f.tell()
    for line in chunk.splitlines():
        text = line.split("|", 1)[-1].strip()
        m = V1_GEN.search(text)
        if m:  # the standard progress shape (0-based generation, trials of a budget) + V1's own numbers
            _set(st, t, progress={"engine": "v1", "generation": int(m["g"]) - 1, "generations": int(m["n"]),
                                  "trials": int(m["t"]), "budget": int(st.get("v1_budget") or int(m["t"])),
                                  "best_sharpe": None, "alive": int(m["s"]), "top_fitness": float(m["f"]),
                                  "utc": now_utc()})
        if text.startswith(("PROVING GROUNDS", "GEN ", "FINAL WAR", "VAULT GATES", "SKIP", "DONE", "REFUSED",
                            "CRASH", "V1 ·")):
            applog.write("breed", f"{_market(t)} · {text}",
                         "ERROR" if text.startswith(("REFUSED", "CRASH")) else "INFO")
    return offset


def _breed_v1(st: dict[str, Any], t: dict[str, Any]) -> None:
    """One market on engine V1: spawn the V1 CLI, follow its feed, PAUSE = hard suspend of its process tree,
    SKIP = finish the generation then the Final War, then add the strategies that passed to the vault."""
    from engine.api import routes_app_campaigns as campaigns
    from engine.core import suspend
    from engine.core.paths import get_arqvio_home
    from engine.research.vault import library

    from engine.research.v1.config import v1_config

    req = st["request"]
    seed = secrets.randbelow(1_000_000)
    preset = V1_PRESETS[req["depth"]]
    c = v1_config(preset)
    st["v1_budget"] = c.proving_grounds + 2 * c.generations * c.tribe_size  # strategies a full run tests, about
    campaign_id = f"v1-{t['cell']}-{seed}"
    out = library.ga_dir() / campaign_id
    run = campaign_id
    campaigns.guard_spawn(run, t["cell"], "start")
    out.mkdir(parents=True, exist_ok=True)
    log = out / "v1.log"
    _set(st, t, state="RUNNING", seed=seed, run=run, campaign_id=campaign_id, engine="v1",
         started_utc=now_utc())
    applog.write("breed", f"{_market(t)}: engine V1 · preset {preset} · power {req['power']}")
    proc = campaigns.spawn(run, "v1", _v1_argv(t["cell"], preset, seed, req["power"], out), get_arqvio_home(),
                           out / "process.log")
    offset, paused = 0, False
    while proc.poll() is None:
        offset = _forward_v1(log, offset, t, st)
        if _ctl["stop"].is_set():
            if paused:
                suspend.resume(proc.pid)
            campaigns.kill_tree(proc)
            _set(st, t, state="STOPPED", detail="stopped")
            applog.write("breed", f"{_market(t)}: stopped", "WARN")
            return
        if _ctl["pause"].is_set() and not paused:
            suspend.suspend(proc.pid)
            paused = True
            _set(st, t, state="PAUSED")
            applog.write("breed", f"{_market(t)}: paused (the engine and its workers are frozen where they stand)")
        elif paused and not _ctl["pause"].is_set():
            suspend.resume(proc.pid)
            paused = False
            _set(st, t, state="RUNNING")
            applog.write("breed", f"{_market(t)}: resumed")
        if _ctl.get("skip") and _ctl["skip"].is_set():
            (out / "skip").write_text("", encoding="utf-8")
            _ctl["skip"].clear()
            applog.write("breed", f"{_market(t)}: SKIP -- this generation finishes, then the Final War")
        time.sleep(POLL_S)
    offset = _forward_v1(log, offset, t, st)
    rc = proc.wait()
    if rc != 0 or not (out / "v1_result.json").is_file():
        tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-3:] if log.is_file() else []
        _set(st, t, state="FAILED", detail=f"engine V1 exited rc={rc}: {' / '.join(x.split('|')[-1].strip() for x in tail)}")
        applog.write("breed", f"{_market(t)}: engine V1 ended rc={rc}", "ERROR")
        return
    res = library.import_v1(campaign_id)
    n = len(res["added"])
    _set(st, t, state="DONE", detail=f"{n} strateg{'y' if n == 1 else 'ies'} to the vault (UNCERTIFIED)", added=n)
    st["vaulted"] = st.get("vaulted", 0) + n
    save(st)
    applog.write("vault", f"{_market(t)}: engine V1 · {n} strateg{'y' if n == 1 else 'ies'} added to the vault, "
                          f"UNCERTIFIED ({len(res['skipped'])} finalist(s) failed a research gate)",
                 "OK" if n else "WARN")


def _breed_one(st: dict[str, Any], t: dict[str, Any]) -> None:
    from engine.api import routes_app_campaigns as campaigns
    from engine.research.vault import library

    req = st["request"]
    if req.get("search") == "v1":
        _breed_v1(st, t)
        return
    seed = secrets.randbelow(1_000_000)
    applog.write("breed", f"{_market(t)}: safety check first (a short trial run on a copy of the data)")
    _set(st, t, state="CHECKING", seed=seed)
    tribal = req.get("search") == "tribal"
    r = campaigns.preflight(campaigns.Plan(cell=t["cell"], preset=DEPTHS[req["depth"]], seed=seed,
                                           islands=2 if tribal else None))
    run = r["run"]
    _set(st, t, run=run, campaign_id=r["campaign_id"])
    d = campaigns.run_dir(run)
    while True:
        s = campaigns.status_of(run, d)
        if s["state"] != "PROBING":
            break
        time.sleep(POLL_S)
    if s["state"] != "READY":
        reason = (s.get("preflight") or {}).get("reason") or s["state"]
        _set(st, t, state="FAILED", detail=f"safety check: {reason}")
        applog.write("breed", f"{_market(t)}: the safety check failed: {reason}", "ERROR")
        return
    campaigns.start(run, campaigns.Start(confirm=True, power=req["power"]))
    _set(st, t, state="RUNNING", started_utc=now_utc())
    a = st["args"]
    applog.write("breed", f"{_market(t)}: breeding started · {a['pop']} strategies x {a['gen']} generations, up to "
                          f"{a['budget']:,} tested · power {req['power']}")
    offset = 0
    while True:
        offset = _forward(d / "runner.log", offset, t, st)
        state = campaigns.status_of(run, d)["state"]
        if state == "RUNNING" and hopeless(t.get("progress")):
            campaigns.stop(run)
            p = t["progress"]
            _set(st, t, state="STOPPED", detail=f"early stop: nothing alive after {p['generation'] + 1} generations, "
                                                f"best Sharpe {p['best_sharpe']:.2f}")
            applog.write("breed", f"{_market(t)}: early stop -- nothing alive after {p['generation'] + 1} generations "
                                  f"(best Sharpe {p['best_sharpe']:.2f} < {EARLY_STOP_SHARPE}); moving on", "WARN")
            return
        if state == "RUNNING":
            if tribal and _ctl["pause"].is_set() and not _ctl["stop"].is_set():
                _ctl["pause"].clear()
                applog.write("breed", f"{_market(t)}: a tribal search cannot pause (no checkpoint yet); "
                                      "it runs on -- STOP ends it", "WARN")
            if _ctl["stop"].is_set() or _ctl["pause"].is_set():
                campaigns.stop(run)
                if _ctl["stop"].is_set():
                    _set(st, t, state="STOPPED", detail="stopped at its last checkpoint")
                    applog.write("breed", f"{_market(t)}: stopped", "WARN")
                    return
                _set(st, t, state="PAUSED")
                applog.write("breed", f"{_market(t)}: paused at its checkpoint")
                if not _wait_resume(st):
                    _set(st, t, state="STOPPED", detail="stopped while paused")
                    return
                campaigns.resume(run, campaigns.Resume())
                _set(st, t, state="RUNNING")
                applog.write("breed", f"{_market(t)}: resumed")
            time.sleep(POLL_S)
            continue
        break
    offset = _forward(d / "runner.log", offset, t, st)
    if state != "COMPLETED":
        tail = [ln.split("|")[-1].strip() for ln in (d / "runner.log").read_text(
            encoding="utf-8", errors="replace").splitlines()[-6:]] if (d / "runner.log").is_file() else []
        _set(st, t, state="FAILED", detail=f"{state}: {' / '.join(tail[-3:])}")
        applog.write("breed", f"{_market(t)}: the campaign ended {state}: {' / '.join(tail[-3:])}", "ERROR")
        return
    res = library.import_campaign(t["campaign_id"])
    n = len(res["added"])
    if n:
        from engine.research.vault import held_out

        for eid in res["added"]:
            try:
                library.set_held_out(eid, held_out.evaluate(library.get(eid)))
            except Exception as exc:  # recorded on the entry: the vault shows why there is no held-out result
                library.set_held_out(eid, {"state": "FAILED", "reason": f"{type(exc).__name__}: {exc}"})
        applog.write("vault", f"{_market(t)}: held-out test run on the {n} new strateg{'y' if n == 1 else 'ies'}")
    _set(st, t, state="DONE", detail=f"{n} strateg{'y' if n == 1 else 'ies'} to the vault", added=n)
    st["vaulted"] = st.get("vaulted", 0) + n
    save(st)
    applog.write("vault", f"{_market(t)}: {n} strateg{'y' if n == 1 else 'ies'} added to the vault "
                          f"({len(res['skipped'])} finalist(s) did not survive)", "OK" if n else "WARN")


def _work(lid: str) -> None:
    st = json.loads(_path(lid).read_text(encoding="utf-8"))
    try:
        import MetaTrader5 as mt5  # noqa: N813

        _cells(st, _data(st, mt5))
        for t in [x for x in st["tasks"] if x["kind"] == "breed"]:
            if t["state"] != "WAITING":
                continue
            if _ctl["stop"].is_set() or not _wait_resume(st):
                _set(st, t, state="SKIPPED", detail="launch stopped")
                continue
            try:
                _breed_one(st, t)
            except HTTPException as exc:
                _set(st, t, state="FAILED", detail=str(exc.detail))
                applog.write("breed", f"{_market(t)}: {exc.detail}", "ERROR")
        stopped = _ctl["stop"].is_set()
        data = [t for t in st["tasks"] if t["kind"] == "data"]
        breeds = [t for t in st["tasks"] if t["kind"] == "breed"]
        completed = any(t["state"] == "DONE" for t in breeds)
        data_refused = any(t["state"] == "FAILED" for t in data)
        if stopped:
            outcome = "STOPPED"
        elif any(t["state"] == "FAILED" for t in breeds):
            outcome = "FAILED"
        elif not completed:
            # A loaded helper/cross market is not an evaluated strategy search.
            outcome = "BLOCKED_DATA" if data_refused else "FAILED"
        elif data_refused:
            outcome = "PARTIAL_DATA"
        elif any(t["state"] != "DONE" for t in breeds):
            outcome = "FAILED"
        else:
            outcome = "DONE" if st.get("vaulted", 0) else "COMPLETED_ZERO_SURVIVORS"
        if outcome in ("DONE", "COMPLETED_ZERO_SURVIVORS"):
            _chain(st)
        detail = "; ".join(dict.fromkeys(
            f"{t.get('symbol', '')} {t.get('timeframe', '')}: {t['detail']}".strip()
            for t in st["tasks"] if t.get("detail") and t["state"] in ("FAILED", "SKIPPED")))
        if not detail and outcome == "FAILED":
            detail = "No strategy search completed; inspect the launch tasks."
        _set(st, state=outcome, detail=detail, finished_utc=now_utc())
        applog.write("engine", f"Launch {outcome}: {st.get('vaulted', 0)} new "
                               "strateg" + ("y" if st.get("vaulted", 0) == 1 else "ies") + " in the vault",
                     "OK" if outcome == "DONE" else "WARN")
    except Exception as exc:
        _set(st, state="FAILED", detail=f"{type(exc).__name__}: {exc}", trace=traceback.format_exc(limit=6),
             finished_utc=now_utc())
        applog.write("engine", f"Launch failed: {type(exc).__name__}: {exc}", "ERROR")
    if not _ctl["stop"].is_set():
        threading.Timer(QUEUE_GAP_S, start_next).start()  # after this thread ends: busy() is false by then


CHAIN_PURGES = {"purge_copies": "copies", "purge_held_out_failed": "held_out_failed", "purge_grade_f": "grade_f"}


def _chain(st: dict[str, Any]) -> None:
    """The launch's ``after`` steps, in order; each one logged with what it did."""
    from engine.research.vault import library

    steps = list((st.get("request") or {}).get("after") or [])
    for step in steps:
        try:
            if step in CHAIN_PURGES:
                r = library.purge(CHAIN_PURGES[step], apply=True)
                applog.write("vault", f"automation: purge {r['what']} -> {r['count']} strateg"
                                      f"{'y' if r['count'] == 1 else 'ies'} to the recycle bin")
            elif step == "repeat":
                items = queue_items()
                req = {k: v for k, v in st["request"].items() if k in LaunchRequest.model_fields}
                items.append({"id": f"q{secrets.token_hex(4)}", "utc": now_utc(), "request": req})
                _save_queue(items)
                applog.write("engine", "automation: REPEAT -- the same launch is queued again")
        except Exception as exc:  # one failed step is named and the rest still run
            applog.write("engine", f"automation step {step} failed: {type(exc).__name__}: {exc}", "ERROR")


# ---- the launch queue (AUTOMATION, ARQVIO_V4.md A19) ------------------------------------------------------------------
# Launches waiting their turn, in a file under the data folder, so the queue outlives the window and the engine:
# when a launch finishes (not when it is stopped) the next one starts; POST /queue/start kicks it after a restart.

QUEUE_GAP_S = 3.0


def queue_path() -> Path:
    return root() / "queue.json"


def queue_items() -> list[dict[str, Any]]:
    try:
        return json.loads(queue_path().read_text(encoding="utf-8")).get("items", [])
    except (OSError, ValueError):
        return []


def _save_queue(items: list[dict[str, Any]]) -> None:
    p = queue_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"schema": "arqvio.launch_queue", "version": 1, "items": items}, indent=2), encoding="utf-8")
    tmp.replace(p)


def start_next() -> dict[str, Any] | None:
    """Start the first queued launch when nothing runs; a refused one leaves the queue with its reason logged."""
    with _qlock:
        if busy():
            return None
        items = queue_items()
        if not items:
            return None
        item, rest = items[0], items[1:]
        _save_queue(rest)
    try:
        out = launch(LaunchRequest(**item["request"]))
        applog.write("engine", f"Queue: started {item['id']} ({len(rest)} still waiting)", "OK")
        return out
    except HTTPException as exc:
        applog.write("engine", f"Queue: {item['id']} could not start: {exc.detail}; removed from the queue", "WARN")
        return None


_qlock = threading.Lock()


# ---- routes ------------------------------------------------------------------------------------------------------------

@router.get("")
def current() -> dict[str, Any]:
    from engine.api.routes_app_campaigns import presets
    from engine.core.paths import get_arqvio_home
    from engine.research.ga.parallel import POWER_MODES, cpu_count, workers_for

    st = latest()
    sealed = presets()["presets"]
    return {"launch": st, "busy": busy(), "paused": _ctl["pause"].is_set() and busy(),
            "depths": {d: {k: sealed[p][k] for k in ("pop", "gen", "budget", "final_n")} for d, p in DEPTHS.items()},
            "timeframes": list(TIMEFRAMES), "home": str(get_arqvio_home()), "logical_cores": cpu_count(),
            "power": {m: workers_for(m) for m in POWER_MODES}}


@router.get("/live")
def live_view() -> dict[str, Any]:
    """The CAMPAIGN tab's generation fan and trade tape: ``live.json`` of the breeding task running now (or of the
    launch's last breeding task), written by the search after each generation (engine/research/search/live_view.py)."""
    from engine.research.search import live_view as lv
    from engine.research.vault import library

    st = latest()
    tasks = [t for t in (st or {}).get("tasks", []) if t.get("kind") == "breed" and t.get("campaign_id")]
    t = next((x for x in tasks if x.get("state") in ("RUNNING", "PAUSED", "CHECKING")), tasks[-1] if tasks else None)
    if t is None:
        return {"task": None, "live": None}
    return {"task": {"symbol": t["symbol"], "timeframe": t["timeframe"], "state": t["state"],
                     "engine": "v1" if t.get("engine") == "v1" else "v2", "campaign_id": t["campaign_id"]},
            "live": lv.read(library.ga_dir() / t["campaign_id"])}


@router.post("")
def launch(req: LaunchRequest) -> dict[str, Any]:
    from engine.api.routes_app_campaigns import live, presets
    from engine.research.data import terminals as mt5_terminals

    with _lock:
        if busy():
            raise HTTPException(status_code=409, detail="a launch is already running; stop it or wait for it")
        if live():
            raise HTTPException(status_code=409, detail="a campaign process is still running")
        try:
            term = mt5_terminals.find(req.terminal)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"no MetaTrader 5 terminal {req.terminal!r}") from None
        acc = mt5_terminals.account(term)
        if acc.get("state") != "OK":
            raise HTTPException(status_code=409, detail=f"MetaTrader 5 ({term.path}): {acc.get('detail')}")
        try:
            tasks = plan(req, mt5_terminals.symbols(term))
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        lid = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
        args = {k: presets()["presets"][DEPTHS[req.depth]][k] for k in ("pop", "gen", "budget", "final_n")}
        st = {"schema": "arqvio.launch", "version": 1, "id": lid, "utc": now_utc(), "state": "RUNNING",
              "request": {**req.model_dump(), "terminal_path": term.path, "server": acc.get("server"),
                          "company": acc.get("company")},
              "args": args, "tasks": tasks, "vaulted": 0, "updated_utc": now_utc()}
        save(st)
        if not req.origin:
            p = defaults_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({k: getattr(req, k) for k in ("years", "depth", "power", "search")}),
                         encoding="utf-8")
        _ctl["pause"].clear()
        _ctl["stop"].clear()
        th = threading.Thread(target=_work, args=(lid,), daemon=True, name="arqvio-launch")
        _ctl["thread"] = th
        markets = ", ".join(f"{t['symbol']} {t['timeframe']}" for t in tasks if t["kind"] == "breed")
        applog.write("engine", f"LAUNCH: {markets} · {req.years} years · depth {req.depth} · power {req.power} · "
                               f"data from {acc.get('company')} ({acc.get('server')})", "OK")
        th.start()
    return {"launch": st, "busy": True}


@router.post("/pause")
def pause() -> dict[str, Any]:
    if not busy():
        raise HTTPException(status_code=409, detail="no launch is running")
    _ctl["pause"].set()
    applog.write("engine", "Pause asked: the running campaign stops at its next checkpoint")
    return {"paused": True}


@router.post("/resume")
def resume() -> dict[str, Any]:
    if not busy():
        raise HTTPException(status_code=409, detail="no launch is running")
    _ctl["pause"].clear()
    applog.write("engine", "Resume asked")
    return {"paused": False}


@router.post("/skip")
def skip() -> dict[str, Any]:
    """Engine V1 only (research SKIP): the current generation finishes, then the Final War and the vault gates."""
    if not busy():
        raise HTTPException(status_code=409, detail="no launch is running")
    st = latest() or {}
    running = next((t for t in st.get("tasks", []) if t.get("state") in ("RUNNING", "PAUSED")), None)
    if not running or running.get("engine") != "v1":
        raise HTTPException(status_code=409, detail="SKIP is for an engine V1 run; this market breeds on another "
                                                    "engine")
    _ctl["skip"].set()
    applog.write("engine", "Skip asked: this generation finishes, then the Final War")
    return {"skipping": True}


@router.post("/stop")
def stop() -> dict[str, Any]:
    if not busy():
        raise HTTPException(status_code=409, detail="no launch is running")
    _ctl["stop"].set()
    _ctl["pause"].clear()
    applog.write("engine", "Stop asked: the running campaign stops at its checkpoint, the rest is skipped", "WARN")
    return {"stopping": True}


@log_router.get("")
def log(after: int = 0, limit: int = 500) -> dict[str, Any]:
    return applog.read(after=after, limit=max(1, min(limit, 2000)))


@router.get("/queue")
def queue() -> dict[str, Any]:
    return {"items": queue_items(), "busy": busy()}


@router.post("/queue")
def enqueue(req: LaunchRequest) -> dict[str, Any]:
    with _qlock:
        items = queue_items()
        qid = f"q{datetime.now(tz=UTC):%Y%m%dT%H%M%S%f}"
        items.append({"id": qid, "utc": now_utc(), "request": req.model_dump()})
        _save_queue(items)
    markets = ", ".join(req.symbols)
    applog.write("engine", f"Queued {qid}: {markets} · {', '.join(req.timeframes)} · depth {req.depth} "
                           f"({len(items)} waiting)")
    return {"queued": qid, "items": items}


@router.delete("/queue")
def clear_queue() -> dict[str, Any]:
    """CLEAR ALL: every waiting launch leaves the queue (the running one is not touched)."""
    with _qlock:
        n = len(queue_items())
        _save_queue([])
    applog.write("engine", f"Queue cleared: {n} waiting launch(es) removed")
    return {"removed": n, "items": []}


@router.delete("/queue/{qid}")
def dequeue(qid: str) -> dict[str, Any]:
    with _qlock:
        items = queue_items()
        keep = [i for i in items if i["id"] != qid]
        if len(keep) == len(items):
            raise HTTPException(status_code=404, detail=f"no queued launch {qid!r}")
        _save_queue(keep)
    return {"removed": qid, "items": keep}


@router.post("/queue/start")
def queue_start() -> dict[str, Any]:
    if busy():
        return {"started": None, "detail": "a launch is running; the queue continues when it finishes"}
    out = start_next()
    return {"started": (out or {}).get("launch", {}).get("id"), "items": queue_items()}
