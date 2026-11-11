"""The worker: owner caps enforced, every cycle reconciled, own magic only, named halts, real only when armed."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from engine.research.backtest.order import OrderIntent
from engine.research.accounts import arming
from engine.research.live import guard
from engine.research.live import worker as wk
from engine.research.live.kill_switch import KillSwitch
from engine.research.mql5.magic import live_magic
from tests.research.live.fake_mt5 import FakeMT5

OWN = live_magic("worker")


def _mt5() -> FakeMT5:
    mt5 = FakeMT5()
    mt5.tick_time = int(time.time())
    return mt5


def _worker(mt5: FakeMT5, tmp_path: Path, *, cfg: wk.WorkerConfig | None = None, source=None, lines=None,
            **kw) -> wk.Worker:
    cfg = cfg or wk.WorkerConfig(symbol="USDCADm", source="lifecycle_probe", probe_hold_cycles=2)
    out = lines if lines is not None else []
    return wk.Worker(mt5, cfg, wk.Operational(cycle_seconds=0.0, max_cycles=12),
                     source if source is not None else wk.LifecycleProbeSource(cfg), log=out.append,
                     sleep=lambda s: None, kill=KillSwitch(mt5, flag_dir=tmp_path, log=out.append), **kw)


@pytest.mark.parametrize("bad", [dict(volume=0.0), dict(volume=-0.01), dict(max_positions=0), dict(daily_order_cap=0),
                                 dict(source="martingale"), dict(probe_hold_cycles=0)])
def test_configs_that_cannot_run_are_refused(bad: dict) -> None:
    with pytest.raises(wk.WorkerConfigError):
        wk.WorkerConfig(**{"symbol": "USDCADm", "source": "lifecycle_probe", **bad})


def test_the_owner_sets_the_caps_and_the_defaults_stay_micro() -> None:
    d = wk.WorkerConfig(symbol="USDCADm", source="lifecycle_probe")
    assert (d.volume, d.max_positions, d.daily_order_cap) == (0.01, 1, 6)
    own = wk.WorkerConfig(symbol="USDCADm", source="oracle_signal", volume=0.5, max_positions=3, daily_order_cap=40)
    assert (own.volume, own.max_positions, own.daily_order_cap) == (0.5, 3, 40)


def test_the_config_hash_is_stable_and_covers_the_caps() -> None:
    a = wk.WorkerConfig(symbol="USDCADm", source="lifecycle_probe")
    assert a.config_hash() == wk.WorkerConfig(symbol="USDCADm", source="lifecycle_probe").config_hash()
    assert a.config_hash() != wk.WorkerConfig(symbol="USDCADm", source="lifecycle_probe", daily_order_cap=5).config_hash()


def test_one_full_lifecycle_reconciled_every_cycle(tmp_path: Path) -> None:
    mt5, lines = _mt5(), []
    foreign = mt5.plant_position(symbol="USDCADm", side="long", volume=0.01, magic=424242)
    w = _worker(mt5, tmp_path, lines=lines)
    assert w.run() == "Completed" and w.state == wk.HALTED
    joined = "\n".join(lines)
    assert "ARMED" in joined and joined.count("RECON OK") >= 4
    assert "CAP REFUSED max_positions=1 (no averaging)" in joined
    assert [r["magic"] for r in mt5.sent] == [OWN, OWN]  # one open, one close; the second open never sent
    assert [p.ticket for p in mt5.positions_get()] == [foreign]  # own flat, foreign untouched
    assert "foreign=1" in joined and w.orders_today and list(w.orders_today.values()) == [2]


def test_a_gate_refused_open_never_counts_as_a_lifecycle(tmp_path: Path) -> None:
    """The S6 run-1 false pass: every open refused by the spread gate must not end as Completed."""
    mt5, lines = _mt5(), []
    mt5.bid, mt5.ask = 1.37000, 1.37016  # 16 points, above the 15.7 gate
    w = _worker(mt5, tmp_path, lines=lines, max_spread_points=15.7)
    assert w.run() == "CycleCap"
    assert (mt5.sent, w.fills, w.reconciled_while_open) == ([], [], 0)


def test_an_open_refused_then_filled_completes_from_broker_state(tmp_path: Path) -> None:
    mt5, lines = _mt5(), []
    mt5.bid, mt5.ask = 1.37000, 1.37016

    def narrow(cycle: int, w: wk.Worker) -> None:
        if cycle == 0:
            mt5.bid, mt5.ask = 1.37000, 1.37014  # 14 points: the next open passes the gate

    w = _worker(mt5, tmp_path, lines=lines, max_spread_points=15.7, on_cycle=narrow)
    assert w.run() == "Completed"
    assert [f["action"] for f in w.fills] == ["open", "close"] and w.fills[0]["cycle"] == 1
    assert w.reconciled_while_open == 3  # the cap-check cycle and the two hold cycles
    assert "CYCLE 0 CAP REFUSED spread 16.0" in "\n".join(lines)


def test_a_broker_rejected_open_leaves_no_phantom_and_is_retried(tmp_path: Path) -> None:
    mt5 = _mt5()
    mt5.reject_opens = 1
    w = _worker(mt5, tmp_path)
    assert w.run() == "Completed"  # a phantom ticket in the world-model would MISMATCH here
    assert [f["action"] for f in w.fills] == ["open", "close"] and w.fills[0]["cycle"] == 1


def test_a_refused_close_is_requested_again_and_recorded_as_the_lifecycle_close(tmp_path: Path) -> None:
    mt5 = _mt5()

    def refuse_once(cycle: int, w: wk.Worker) -> None:
        if w.world and not mt5.refuse_close and cycle == 2:
            mt5.refuse_close.update(w.world)  # the close attempt in cycle 3 is refused
        elif cycle == 3:
            mt5.refuse_close.clear()

    w = _worker(mt5, tmp_path, on_cycle=refuse_once)
    assert w.run() == "Completed"
    assert [f["action"] for f in w.fills] == ["open", "close"] and w.fills[1]["cycle"] == 4


def test_a_planted_mismatch_flattens_own_magic_and_halts_by_name(tmp_path: Path) -> None:
    mt5, lines = _mt5(), []
    foreign = mt5.plant_position(symbol="USDCADm", side="long", volume=0.01, magic=424242)

    def tamper(cycle: int, w: wk.Worker) -> None:
        if cycle == 1:  # an own-magic position the worker never opened
            mt5.plant_position(symbol="USDCADm", side="short", volume=0.01, magic=OWN)

    w = _worker(mt5, tmp_path, lines=lines, on_cycle=tamper)
    with pytest.raises(wk.ReconciliationMismatch, match="disagreed with broker truth"):
        w.run()
    assert w.state == wk.HALTED and w.halt_reason == "ReconciliationMismatch"
    assert [p.ticket for p in mt5.positions_get()] == [foreign]
    joined = "\n".join(lines)
    assert "RECON MISMATCH" in joined and "FLATTEN done own_left=[]" in joined


def test_an_unresolved_basis_halts_the_probe(tmp_path: Path) -> None:
    mt5 = _mt5()
    mt5.tick_time = int(time.time()) - 18_774
    w = _worker(mt5, tmp_path)
    assert w.run().startswith("BasisUnresolved") and mt5.sent == []


def test_an_unresolved_basis_mid_run_refuses_that_cycle(tmp_path: Path) -> None:
    mt5, lines = _mt5(), []

    def drift(cycle: int, w: wk.Worker) -> None:
        mt5.tick_time = int(time.time()) - 18_774 if cycle == 0 else int(time.time())

    w = _worker(mt5, tmp_path, lines=lines, on_cycle=drift)
    assert w.run() == "Completed" and w.refused_cycles == 1
    assert "CYCLE 1 REFUSED basis unresolved" in "\n".join(lines)


def test_the_kill_flag_flattens_own_and_halts(tmp_path: Path) -> None:
    mt5 = _mt5()
    holder = {}

    def press(cycle: int, w: wk.Worker) -> None:
        if cycle == 0:
            holder["k"] = w.kill.trigger("test")

    w = _worker(mt5, tmp_path, on_cycle=press)
    assert w.run() == "KillSwitch" and mt5.positions_get() == ()


def test_a_switch_to_real_stops_the_worker_before_any_further_order(tmp_path: Path) -> None:
    mt5 = _mt5()

    def flip(cycle: int, w: wk.Worker) -> None:
        if cycle == 0:
            mt5.trade_mode = mt5.ACCOUNT_TRADE_MODE_REAL

    w = _worker(mt5, tmp_path, on_cycle=flip)
    with pytest.raises(guard.RealAccountNotArmed):
        w.run()
    assert len(mt5.sent) == 1  # only the cycle-0 open


def test_an_armed_real_account_runs_the_full_lifecycle(tmp_path: Path, isolated_arqvio_home: Path) -> None:
    mt5, lines = _mt5(), []
    mt5.trade_mode = mt5.ACCOUNT_TRADE_MODE_REAL
    arming.arm(mt5.login, mt5.server)
    w = _worker(mt5, tmp_path, lines=lines)
    assert w.run() == "Completed"
    assert "PROBE account=REAL" in "\n".join(lines) and [r["magic"] for r in mt5.sent] == [OWN, OWN]


class _OpenEveryCycle:
    done = False

    def next(self, cycle, world):
        return wk.Intent("close", reason="C") if world else wk.Intent("open", "long", "O")


def test_the_daily_order_cap_refuses_opens_but_never_a_close(tmp_path: Path) -> None:
    mt5, lines = _mt5(), []
    cfg = wk.WorkerConfig(symbol="USDCADm", source="lifecycle_probe", daily_order_cap=3)
    w = _worker(mt5, tmp_path, cfg=cfg, source=_OpenEveryCycle(), lines=lines)
    assert w.run() == "CycleCap"
    assert len(mt5.sent) == 4  # open, close, open (cap 3 reached), close (closes are never refused)
    assert "CAP REFUSED daily_order_cap=3" in "\n".join(lines) and mt5.positions_get() == ()


def test_the_spread_gate_and_no_trade_hours_refuse_entries(tmp_path: Path) -> None:
    mt5, lines = _mt5(), []
    mt5.bid, mt5.ask = 1.37000, 1.37100  # 100 points
    w = _worker(mt5, tmp_path, lines=lines, max_spread_points=15.7)
    w.run()
    assert mt5.sent == [] and "CAP REFUSED spread 100.0 > max_spread_points 15.7" in "\n".join(lines)
    mt5b, lines_b = _mt5(), []
    hour = time.gmtime().tm_hour
    _worker(mt5b, tmp_path, lines=lines_b, no_trade_hours=(hour, (hour + 1) % 24)).run()
    assert mt5b.sent == [] and "CAP REFUSED no_trade_hour=" in "\n".join(lines_b)


def test_the_status_snapshot_is_opt_in_and_tracks_reconciliation(tmp_path: Path) -> None:
    import json

    mt5 = _mt5()
    mt5.plant_position(symbol="USDCADm", side="long", volume=0.01, magic=424242)
    status = tmp_path / "live" / "worker_status.json"
    w = _worker(mt5, tmp_path, status_path=status)
    assert w.run() == "Completed"
    doc = json.loads(status.read_text(encoding="utf-8"))
    assert (doc["state"], doc["halt_reason"], doc["magic"]) == ("HALTED", "Completed", OWN)
    assert doc["last_reconciliation"]["ok"] is True and doc["last_reconciliation"]["foreign"] == 1
    assert [f["action"] for f in doc["fills"]] == ["open", "close"] and doc["config_hash"] == w.cfg.config_hash()
    plain = _worker(_mt5(), tmp_path / "x")
    plain.run()
    assert not (tmp_path / "x" / "live").exists()  # no status_path: nothing written anywhere


def _rates(n: int) -> np.ndarray:
    a = np.zeros(n, dtype=FakeMT5.RATES_DTYPE)
    a["time"] = 1_790_000_000 - 1_790_000_000 % 3600 + 3600 * np.arange(n)
    a["open"] = a["high"] = a["low"] = a["close"] = 1.37 + 1e-5 * np.arange(n)
    return a


class _Recorder:
    strategy_id = "rec"

    def __init__(self):
        self.calls = []

    def on_bar(self, i, bars, state):
        self.calls.append((i, len(bars.close), int(bars.time[-1])))
        return OrderIntent(direction="short", volume=0.01, bar_index=i) if len(self.calls) == 2 else None


def test_the_oracle_source_sees_closed_bars_once_each_on_a_growing_anchored_window() -> None:
    mt5 = FakeMT5()
    mt5.rates = _rates(30)  # 29 closed + 1 forming
    strat = _Recorder()
    src = wk.OracleSignalSource(mt5, "USDCADm", strat, history_bars=10)
    assert src.next(0, {}) is None
    assert src.next(1, {}) is None  # no new closed bar: on_bar not called again
    mt5.rates = _rates(31)
    intent = src.next(2, {})
    assert intent == wk.Intent("open", "short", "ORACLE_ENTRY")
    assert [c[1] for c in strat.calls] == [10, 11]  # anchored: the window grew by the one new bar
    assert strat.calls[-1][2] == int(_rates(31)["time"][-2])  # the forming bar is never seen
