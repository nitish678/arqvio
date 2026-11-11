"""One LAUNCH: the queue it plans, the cross-check ring, and the app's one log."""
from __future__ import annotations

import pytest

from engine.api import routes_app_launch as L
from engine.core import applog

OFFERED = [{"name": n, "visible": True} for n in ("EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "US30")]


def _req(**kw) -> L.LaunchRequest:
    base = {"terminal": "t", "symbols": ["EURUSD"], "timeframes": ["H1"], "years": 3, "depth": "QUICK"}
    return L.LaunchRequest(**{**base, **kw})


def test_one_market_gets_a_main_pair_to_be_re_tested_on() -> None:
    tasks = L.plan(_req(), OFFERED)
    data = [(t["symbol"], t["role"]) for t in tasks if t["kind"] == "data"]
    assert data == [("EURUSD", "market"), ("GBPUSD", "cross")]
    assert [t["cell"] for t in tasks if t["kind"] == "breed"] == ["eurusd_h1"]
    assert all(t["state"] == "WAITING" for t in tasks)


def test_several_markets_and_timeframes_breed_each_and_need_no_helper() -> None:
    tasks = L.plan(_req(symbols=["XAUUSD", "US30"], timeframes=["H4", "M15"]), OFFERED)
    assert [t["cell"] for t in tasks if t["kind"] == "breed"] == ["xauusd_m15", "us30_m15", "xauusd_h4", "us30_h4"]
    assert all(t["role"] == "market" for t in tasks if t["kind"] == "data")


def test_a_symbol_the_terminal_does_not_offer_is_refused() -> None:
    with pytest.raises(ValueError, match="does not offer"):
        L.plan(_req(symbols=["BTCUSD"]), OFFERED)
    with pytest.raises(ValueError, match="timeframes"):
        L.plan(_req(timeframes=["W1"]), OFFERED)


def test_the_cross_ring_pairs_each_pulled_market_with_another_of_its_timeframe() -> None:
    tasks = L.plan(_req(symbols=["EURUSD", "GBPUSD", "USDJPY"]), OFFERED)
    for t in tasks:
        if t["kind"] == "data":
            t["state"] = "DONE"
    assert L.cross_map(tasks) == {"eurusd_h1": "gbpusd_h1", "gbpusd_h1": "usdjpy_h1", "usdjpy_h1": "eurusd_h1"}
    tasks[1]["state"] = "FAILED"  # gbpusd did not pull: the ring closes without it
    assert L.cross_map(tasks) == {"eurusd_h1": "usdjpy_h1", "usdjpy_h1": "eurusd_h1"}
    tasks[2]["state"] = "FAILED"  # one market left on H1: nothing to re-test on, so it is not bred
    assert L.cross_map(tasks) == {}


def test_the_app_log_numbers_lines_and_reads_after_a_line(isolated_arqvio_home) -> None:
    for i in range(5):
        applog.write("data", f"line {i}", "OK" if i % 2 else "INFO")
    applog.write("x", "odd level", "SHOUT")
    all_ = applog.read()
    assert [r["seq"] for r in all_["lines"]] == [1, 2, 3, 4, 5, 6]
    assert all_["lines"][-1]["level"] == "INFO"
    after = applog.read(after=4)
    assert [r["msg"] for r in after["lines"]] == ["line 4", "odd level"] and after["last"] == 6
    assert applog.read(after=6)["lines"] == []


def test_the_app_log_keeps_counting_across_a_roll(isolated_arqvio_home, monkeypatch) -> None:
    monkeypatch.setattr(applog, "MAX_BYTES", 200)
    applog._seq.clear()
    for i in range(10):
        applog.write("data", f"a fairly long line number {i} to make the file roll over quickly")
    seqs = [r["seq"] for r in applog.read(limit=100)["lines"]]
    assert seqs == sorted(seqs) and seqs[-1] == 10
    assert applog.log_path().with_name("arqvio.1.log").is_file()


@pytest.mark.parametrize(("tf", "years", "widened"), [("M5", 1, False), ("H1", 1, False), ("H4", 1, True),
                                                      ("D1", 1, True), ("D1", 10, False)])
def test_a_timeframe_gets_at_least_the_history_a_cell_needs(tf, years, widened) -> None:
    days, w = L.window_days(tf, years)
    assert w is widened and days >= round(365.25 * years)
    if widened:  # enough calendar days for 2,000 bars of this timeframe on 5 trading days a week
        from engine.research.registry.annualisation import TF_MINUTES

        assert days * 5 / 7 * 1440 / TF_MINUTES[tf] >= 2000


@pytest.mark.parametrize(("market", "cross"), [("XAUUSD", "XAGUSD"), ("US30", "US500"), ("EURUSD", "GBPUSD"),
                                               ("XAGUSD", "XAUUSD")])
def test_a_lone_market_is_re_tested_on_one_of_its_own_kind(market, cross) -> None:
    offered = [{"name": n, "visible": True} for n in ("EURUSD", "GBPUSD", "XAUUSD", "XAGUSD", "US30", "US500")]
    tasks = L.plan(_req(symbols=[market]), offered)
    assert [t["symbol"] for t in tasks if t["kind"] == "data" and t["role"] == "cross"] == [cross]


def test_early_stop_needs_five_generations_nothing_alive_and_no_sharpe() -> None:
    from engine.api.routes_app_launch import hopeless

    assert not hopeless(None)
    assert not hopeless({"generation": 3, "alive": 0, "best_sharpe": 0.0})   # only 4 generations
    assert hopeless({"generation": 4, "alive": 0, "best_sharpe": 0.05})
    assert not hopeless({"generation": 4, "alive": 1, "best_sharpe": 0.05})  # something alive
    assert not hopeless({"generation": 9, "alive": 0, "best_sharpe": 0.3})   # an edge is forming


def test_the_queue_is_a_file_that_survives_and_starts_the_next_launch(isolated_arqvio_home, monkeypatch) -> None:
    from engine.api import routes_app_launch as L

    req = L.LaunchRequest(terminal="t1", symbols=["EURUSD"], timeframes=["H1"], years=2, depth="QUICK", power="eco")
    a = L.enqueue(req)["queued"]
    b = L.enqueue(req.model_copy(update={"symbols": ["XAUUSD"]}))["queued"]
    assert [i["id"] for i in L.queue_items()] == [a, b]
    L.dequeue(a)
    started = []
    monkeypatch.setattr(L, "busy", lambda: False)
    monkeypatch.setattr(L, "launch", lambda r: started.append(r.symbols) or {"launch": {"id": "x"}})
    assert L.queue_start()["started"] == "x" and started == [["XAUUSD"]] and L.queue_items() == []


def test_the_automation_chain_purges_then_repeats(isolated_arqvio_home, monkeypatch) -> None:
    """AUTOMATION chain (W4-9 9.2): after a finished launch its purge steps run in order, REPEAT queues it again."""
    from engine.research.vault import library

    done = []
    monkeypatch.setattr(library, "purge", lambda rule, apply=False: done.append((rule, apply)) or
                        {"rule": rule, "what": rule, "ids": [], "count": 0, "applied": False})
    req = L.LaunchRequest(terminal="t", symbols=["EURUSD"], timeframes=["H1"], after=["purge_copies", "repeat"])
    L._chain({"request": {**req.model_dump(), "terminal_path": "x"}})
    assert done == [("copies", True)]
    q = L.queue_items()
    assert len(q) == 1 and q[0]["request"]["after"] == ["purge_copies", "repeat"]
    assert "terminal_path" not in q[0]["request"]
