"""AUTOMATION (research behaviour, worker4/RESEARCH_GAP_AUDIT.md §2): validation messages, the team picker, purge
protection, and the scheduler (run, edit-while-running, stop, resume after a restart)."""
from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from engine.research.automation import portfolios as pf
from engine.research.automation import runner
from engine.research.automation import steps as S

NOW = datetime(2026, 11, 1, 12, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _home(isolated_arqvio_home: Path, monkeypatch):
    sched = runner.Scheduler()
    monkeypatch.setattr(runner, "SCHEDULER", sched)
    monkeypatch.setattr(runner, "TICK_S", 0.05)
    monkeypatch.setattr(runner, "RESUME_S", 0)
    yield isolated_arqvio_home
    sched.stop()
    time.sleep(0.15)  # the loop sees the halt before the next test's home exists


def test_validation_says_what_research_says() -> None:
    assert S.validate({"name": "", "every_hours": 24, "steps": []}) == ["NAME THIS AUTOMATION", "ADD AT LEAST ONE STEP"]
    bad = {"name": "x", "every_hours": 24, "steps": [
        {"kind": "campaign", "preset_groups": [], "extra_symbols": [], "timeframes": []},
        {"kind": "portfolio", "name": ""}, {"kind": "parity", "tests_per_run": 0}, {"kind": "replace"}]}
    msgs = S.validate(bad)
    assert "STEP 1: PICK AT LEAST ONE MARKET GROUP OR EXTRA MARKET" in msgs
    assert "STEP 1: PICK AT LEAST ONE TIMEFRAME" in msgs and "STEP 2: NAME THE PORTFOLIO" in msgs
    assert "STEP 3: TESTS PER RUN MUST BE A WHOLE NUMBER FROM 1 TO 100" in msgs
    assert any(m.startswith("STEP 4: REPLACE IS NOT BUILT YET") for m in msgs)
    assert [S.every_text(h) for h in (0, 168, 336, 72, 36, 24)] == ["NON-STOP", "WEEKLY", "EVERY 2W", "EVERY 3D",
                                                                     "EVERY 36H", "EVERY 24H"]


def _row(i: str, sym: str, tf: str = "H1", score: float = 70, added: datetime = NOW) -> dict:
    return {"id": i, "symbol": sym, "timeframe": tf, "score": score, "added_utc": added.isoformat(),
            "metrics": {"n_trades": 100, "max_dd": 0.1, "total_return": 0.3}, "held_out": {"return": 0.05}}


def test_team_picker_limits_markets_timeframes_and_similar_returns() -> None:
    days = [f"2026-01-{d:02d}" for d in range(1, 29)] + [f"2026-02-{d:02d}" for d in range(1, 28)]
    base = {d: (i % 5 - 2) / 100 for i, d in enumerate(days)}
    other = {d: (2 - i % 5) / 100 for i, d in enumerate(days)}  # the mirror image: correlation -1
    series = {"a": base, "b": base, "c": other, "d": other, "e": base}
    pool = [_row("a", "EURUSD", score=90), _row("b", "GBPUSD", score=85), _row("c", "EURUSD", "H4", score=80),
            _row("d", "EURUSD", "H1", score=75), _row("e", "XAUUSD", score=20)]
    step = {**S.default_step("portfolio"), "name": "T", "target_size": 5, "max_per_symbol": 3, "max_per_tf": 1,
            "max_avg_corr": 0.35, "min_score": 55}
    picks, skips = S.pick_team(step, pool, [], lambda r: series[r["id"]], now=NOW)
    # a first; b moves exactly like a (skipped as similar); c differs; d is a 2nd EURUSD H1 (TF limit); e scores 20
    assert [p["id"] for p in picks] == ["a", "c"]
    assert skips["similar"] == 1 and skips["timeframe"] == 1


def test_purge_spares_portfolio_members_and_reads_the_criteria() -> None:
    rows = [_row("keep", "EURUSD", score=10), _row("weak", "GBPUSD", score=10), _row("good", "XAUUSD", score=80)]
    assert S.purge_matches({"min_score": 50}, rows, {"keep"}, NOW) == ["weak"]
    old = _row("old", "EURUSD", added=NOW - timedelta(days=60))
    assert S.purge_matches({"max_age_days": 30}, [old], set(), NOW) == ["old"]
    p = pf.create("Team Alpha")
    pf.add(p["id"], ["keep"])
    assert pf.protected_ids() == {"keep"}
    with pytest.raises(pf.PortfolioError, match="already exists"):
        pf.create("team alpha")


def _auto(steps: list[dict], every: float = 24) -> dict:
    return runner.save({"name": "Night breed", "every_hours": every, "steps": steps, "terminal": "AUTO"})


def _wait(cond, timeout: float = 5.0) -> None:
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return
        time.sleep(0.02)
    raise AssertionError("timed out")


def test_a_run_goes_step_by_step_and_logs_each(monkeypatch) -> None:
    seen: list[str] = []
    monkeypatch.setitem(S.RUNNERS, "purge", lambda step, run: seen.append("purge") or "3 strategies went")
    a = _auto([{"kind": "purge", "criteria": {}}, {"kind": "purge", "criteria": {"min_score": 1}}])
    runner.SCHEDULER.start()
    _wait(lambda: (runner.get(a["id"]).get("history") or [None])[0] is not None)
    h = runner.get(a["id"])["history"][0]
    assert h["ok"] and h["detail"] == "finished all 2 steps" and seen == ["purge", "purge"]
    details = [x["detail"] for x in runner.activity_log()]
    assert "step 1: 3 strategies went" in details and "step 2: 3 strategies went" in details
    assert runner.get(a["id"])["next_at"] > h["ts"]  # the next run is a schedule away (24 h)


def test_an_edit_while_running_waits_for_the_next_run_and_stop_ends_it(monkeypatch) -> None:
    gate = {"go": False}

    def slow(step, run):
        while not gate["go"] and not run.cancel.is_set():
            time.sleep(0.02)
        return "done"

    monkeypatch.setitem(S.RUNNERS, "purge", slow)
    a = _auto([{"kind": "purge", "criteria": {}}, {"kind": "purge", "criteria": {}}])
    runner.SCHEDULER.start()
    _wait(lambda: runner.SCHEDULER.current_id() == a["id"])
    edited = runner.save({**a, "steps": [{"kind": "purge", "criteria": {"min_score": 9}}]})
    assert edited["next_steps"] == [{"kind": "purge", "criteria": {"min_score": 9}}] and len(edited["steps"]) == 2
    assert runner.SCHEDULER.cancel(a["id"])
    _wait(lambda: (runner.get(a["id"]).get("history") or [None])[0] is not None)
    after = runner.get(a["id"])
    assert after["history"][0]["paused"] and after["history"][0]["detail"] == "stopped by you"
    assert after["steps"] == [{"kind": "purge", "criteria": {"min_score": 9}}] and "next_steps" not in after


def test_a_failed_step_is_named_and_the_rest_still_run(monkeypatch) -> None:
    def boom(step, run):
        raise RuntimeError("no MetaTrader 5 terminal to read data from")

    monkeypatch.setitem(S.RUNNERS, "campaign", boom)
    monkeypatch.setitem(S.RUNNERS, "purge", lambda step, run: "0 strategies went")
    a = _auto([{**S.default_step("campaign")}, {"kind": "purge", "criteria": {}}])
    runner.SCHEDULER.start()
    _wait(lambda: (runner.get(a["id"]).get("history") or [None])[0] is not None)
    h = runner.get(a["id"])["history"][0]
    assert not h["ok"] and h["detail"].startswith("finished with 1 failed step: step 1: RuntimeError")


def test_an_interrupted_run_carries_on_from_its_step(monkeypatch) -> None:
    ran: list[int] = []
    monkeypatch.setitem(S.RUNNERS, "purge", lambda step, run: ran.append(len(run.added)) or "ok")
    a = _auto([{"kind": "purge", "criteria": {}}, {"kind": "purge", "criteria": {}}])
    runner._update(a["id"], next_at=(NOW + timedelta(days=999)).isoformat())  # not due: only the resume runs it
    runner._write(runner.Scheduler._run_file(), {"automation_id": a["id"], "name": a["name"], "step_idx": 1,
                                                 "steps_total": 2, "added": ["x", "y"], "run_pool": []})
    runner.SCHEDULER.start()
    _wait(lambda: (runner.get(a["id"]).get("history") or [None])[0] is not None)
    assert ran == [2]  # only step 2 ran, with the run's state from before the restart
    assert any("carrying on from step 2" in x["detail"] for x in runner.activity_log())


def test_api_round_trip(isolated_arqvio_home) -> None:
    from engine.api.server import create_app

    c = TestClient(create_app())
    assert c.post("/api/v2/app/automation/save", json={"name": "", "steps": []}).status_code == 422
    r = c.post("/api/v2/app/automation/save", json={"name": "Weekly purge", "every_hours": 168,
                                                    "steps": [{"kind": "purge", "criteria": {"min_score": 40}}]})
    assert r.status_code == 200 and r.json()["automation"]["every_text"] == "WEEKLY"
    assert r.json()["automation"]["alongside"] is True
    aid = r.json()["automation"]["id"]
    assert c.post("/api/v2/app/automation/toggle", json={"id": aid, "enabled": False}).json()["automation"]["enabled"] is False
    lst = c.get("/api/v2/app/automation").json()
    assert [a["id"] for a in lst["automations"]] == [aid] and {g["id"] for g in lst["groups"]} >= {"fx_majors", "metals"}
    pv = c.post("/api/v2/app/automation/preview", json={"name": "x", "steps": [{"kind": "purge", "criteria": {}}]}).json()
    assert pv["lines"][0].startswith("STEP 1 PURGE") and pv["note"].startswith("Nothing was started")
    p = c.post("/api/v2/app/portfolios", json={"name": "Auto Best 20"}).json()["portfolio"]
    assert c.post(f"/api/v2/app/portfolios/{p['id']}/add", json={"ids": ["a@x", "b@y"]}).json()["portfolio"]["members"][1]["id"] == "b@y"
    assert c.delete(f"/api/v2/app/portfolios/{p['id']}").json() == {"deleted": p["id"]}
    assert c.post("/api/v2/app/automation/delete", json={"id": aid}).json() == {"deleted": aid}
