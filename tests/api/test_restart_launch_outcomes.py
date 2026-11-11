"""Offline restart regressions: prerequisite failure is not a successful empty search."""
from __future__ import annotations

import sys
import threading
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from engine.api import routes_app_launch as launch
from engine.core import applog


@pytest.mark.parametrize("data_states,breed_states,vaulted,expected,chain", [
    (["FAILED"], ["SKIPPED"], 0, "BLOCKED_DATA", False),
    (["DONE", "FAILED"], ["SKIPPED"], 0, "BLOCKED_DATA", False),
    (["DONE"], ["FAILED"], 0, "FAILED", False),
    (["DONE"], ["DONE"], 0, "COMPLETED_ZERO_SURVIVORS", True),
    (["DONE"], ["DONE"], 2, "DONE", True),
    (["DONE", "FAILED"], ["DONE", "SKIPPED"], 2, "PARTIAL_DATA", False),
    (["DONE", "FAILED"], ["DONE", "SKIPPED"], 0, "PARTIAL_DATA", False),
    (["DONE", "FAILED"], ["DONE", "FAILED"], 2, "FAILED", False),
    ([], [], 0, "FAILED", False),
])
def test_launch_classifies_actual_work(isolated_arqvio_home, monkeypatch, data_states, breed_states,
                                     vaulted, expected, chain):
    tasks = [{"kind": kind, "state": state, "symbol": "EURUSD", "timeframe": "H1",
              "detail": "bars: insufficient history" if state in ("FAILED", "SKIPPED") else ""}
             for kind, states in (("data", data_states), ("breed", breed_states)) for state in states]
    st = {"id": "restart-outcome", "request": {}, "tasks": tasks, "vaulted": vaulted}
    launch.save(st)
    monkeypatch.setitem(sys.modules, "MetaTrader5", SimpleNamespace())
    monkeypatch.setattr(launch, "_data", lambda *args: {})
    monkeypatch.setattr(launch, "_cells", lambda *args: None)
    monkeypatch.setitem(launch._ctl, "stop", threading.Event())
    monkeypatch.setitem(launch._ctl, "thread", None)
    monkeypatch.setattr(launch.threading, "Timer", lambda *args: SimpleNamespace(start=lambda: None))
    chained = []
    monkeypatch.setattr(launch, "_chain", lambda current: chained.append(current["id"]))
    launch._work(st["id"])
    result = launch.latest()
    assert result["state"] == expected
    assert result["vaulted"] == vaulted
    assert bool(chained) is chain
    logs = applog.read()["lines"]
    if not chain:
        assert not any(line["level"] == "OK" for line in logs)
    if expected in ("PARTIAL_DATA", "BLOCKED_DATA", "FAILED"):
        assert result["detail"]


def test_breed_failure_keeps_reason_and_never_repeats(isolated_arqvio_home, monkeypatch):
    st = {"id": "breed-refusal", "request": {}, "vaulted": 0,
          "tasks": [{"kind": "data", "state": "DONE"},
                    {"kind": "breed", "state": "WAITING", "symbol": "EURUSD", "timeframe": "H1"}]}
    launch.save(st)
    monkeypatch.setitem(sys.modules, "MetaTrader5", SimpleNamespace())
    monkeypatch.setattr(launch, "_data", lambda *args: {})
    monkeypatch.setattr(launch, "_cells", lambda *args: None)
    monkeypatch.setattr(launch, "_breed_one", lambda *args: (_ for _ in ()).throw(HTTPException(409, "preflight refused")))
    monkeypatch.setattr(launch, "_chain", lambda *args: pytest.fail("failed launch repeated"))
    monkeypatch.setitem(launch._ctl, "stop", threading.Event())
    monkeypatch.setitem(launch._ctl, "thread", None)
    monkeypatch.setattr(launch.threading, "Timer", lambda *args: SimpleNamespace(start=lambda: None))
    launch._work(st["id"])
    assert launch.latest()["state"] == "FAILED"
    assert "preflight refused" in launch.latest()["detail"]


@pytest.mark.parametrize("state", ["PARTIAL_DATA", "FAILED", "BLOCKED_DATA"])
def test_automation_accounts_only_completed_markets(isolated_arqvio_home, monkeypatch, state):
    from engine.research.automation import steps

    monkeypatch.setattr(steps, "plan_campaign", lambda *args: {
        "symbols": ["EURUSD", "GBPUSD"], "timeframes": ["H1"], "years": [3, 6],
        "skipped_dead": [], "missing": [], "terminal": "fake", "depth": "QUICK", "power": "eco",
        "search": "standard"})
    starts = []
    monkeypatch.setattr(launch, "launch", lambda req: starts.append(req.years) or {
        "launch": {"id": "own", "utc": "2026-11-10T00:00:00+00:00"}})
    monkeypatch.setattr(launch, "busy", lambda: False)
    tasks = [{"kind": "breed", "state": "DONE", "symbol": "EURUSD", "timeframe": "H1", "added": 1},
             {"kind": "breed", "state": "SKIPPED", "symbol": "GBPUSD", "timeframe": "H1"}]
    if state == "BLOCKED_DATA":
        tasks[0]["state"] = "SKIPPED"
    monkeypatch.setattr(launch, "latest", lambda: {"id": "own", "state": state, "detail": "GBPUSD history refused",
                                                "vaulted": 1 if state != "BLOCKED_DATA" else 0, "tasks": tasks})
    monkeypatch.setattr(steps, "_vault", lambda: [{"id": "new", "added_utc": "2026-11-10T01:00:00+00:00"}])
    marks, warnings = [], []
    monkeypatch.setattr(steps, "_mark_result", lambda *args: marks.append(args))
    run = steps.Run({"id": "auto"}, cancel=threading.Event(), log=warnings.append, phase=lambda *args: None)
    if state == "PARTIAL_DATA":
        steps.run_campaign({}, run)
        assert starts == [3, 6] and run.campaigned
        assert any("PARTIAL_DATA" in line and "GBPUSD" in line for line in warnings)
    else:
        with pytest.raises(RuntimeError, match=state):
            steps.run_campaign({}, run)
        assert starts == [3]
    assert marks == ([] if state == "BLOCKED_DATA" else [("EURUSD", "H1", 1)] * len(starts))
    # Successful additions survive even when a later task failed.
    if state != "BLOCKED_DATA":
        assert run.run_pool == ["new"]


@pytest.mark.parametrize("failure,fragment", [("stale", "closed"), ("bars", "12 H4 bars"),
                                              ("terminal", "not logged in")])
def test_production_pull_refusal_propagates_without_real_mt5(isolated_arqvio_home, monkeypatch, failure, fragment):
    from datetime import datetime
    from engine.research.cells import add_market
    from tests.research.cells.test_add_market import FakeMt5, NOW

    class FixedTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW

    monkeypatch.setattr(launch, "datetime", FixedTime)
    monkeypatch.setattr(add_market, "datetime", FixedTime)
    fake = FakeMt5(n_bars=12, offset_s=-7 * 86400 if failure == "stale" else 0)
    if failure == "terminal":
        fake.account_info = lambda: None
    monkeypatch.setattr(add_market, "ensure_specs", lambda *args, **kwargs: {})
    monkeypatch.setattr(add_market, "ensure_spread", lambda *args, **kwargs: {})
    monkeypatch.setattr(add_market, "HISTORY_PAUSE_S", 0)
    monkeypatch.setitem(sys.modules, "MetaTrader5", fake)
    monkeypatch.setitem(launch._ctl, "thread", None)
    monkeypatch.setitem(launch._ctl, "stop", threading.Event())
    monkeypatch.setitem(launch._ctl, "pause", threading.Event())
    monkeypatch.setattr(launch.threading, "Timer", lambda *args: SimpleNamespace(start=lambda: None))
    monkeypatch.setattr(launch, "_chain", lambda *args: pytest.fail("refused data ran after-actions"))
    task = {"kind": "data", "role": "market", "cell": "eurusdm_h4", "symbol": "EURUSDm",
            "timeframe": "H4", "state": "WAITING"}
    st = {"id": "pull-refusal", "request": {"years": 1, "terminal_path": "in-memory-fake"},
          "tasks": [task, {**task, "kind": "breed"}], "vaulted": 0}
    launch.save(st)
    launch._work(st["id"])
    result = launch.latest()
    assert result["state"] == "BLOCKED_DATA"
    assert fragment.lower() in result["detail"].lower()
    assert result["tasks"][1]["state"] == "SKIPPED"
    assert "initialize" in fake.calls and "shutdown" in fake.calls


def test_scheduler_keeps_partial_warning_and_continues(isolated_arqvio_home, monkeypatch):
    from engine.research.automation import runner, steps

    def partial(step, run):
        run.run_pool = ["successful-result"]
        run.warn("campaign PARTIAL_DATA: GBPUSD history refused; continuing")
        return "kept EURUSD"

    following = []
    monkeypatch.setitem(steps.RUNNERS, "campaign", partial)
    monkeypatch.setitem(steps.RUNNERS, "purge", lambda step, run: following.append(list(run.run_pool)) or "checked")
    auto = runner.save({"name": "offline-partial", "every_hours": 24,
                        "steps": [steps.default_step("campaign"), steps.default_step("purge")]})
    runner.Scheduler()._work(auto, None)
    assert following == [["successful-result"]]
    assert any(row["status"] == "warn" and "PARTIAL_DATA" in row["detail"] for row in runner.activity_log())
