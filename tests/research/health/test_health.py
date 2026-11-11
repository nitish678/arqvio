"""Multi-account health (DR §7) against a fake MetaTrader5: two DEMO accounts in one snapshot, REAL is a hard stop,
G12 basis labels (the SIMULATED EET/EEST fixture), the refusal of unlabeled hour-keyed input, the retry rule,
staleness, spec drift, call-time paths, alerts, and no order path anywhere in the S7 packages."""
from __future__ import annotations

import ast
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from engine.research.health.alerting import from_health
from engine.research.health.basis import BasisLabel, BasisRefused, basis_hour, probe_with_retry, to_utc
from engine.research.health.probe import (HealthError, accounts_path, load_accounts, probe_account, snapshot,
                                          snapshot_path)
from engine.research.accounts import arming

NOW = datetime(2026, 1, 15, 10, 30, 0, tzinfo=UTC)  # a winter date: EET = UTC+2
SPEC = SimpleNamespace(digits=5, point=1e-05, contract_size=100000.0, trade_tick_size=1e-05,
                       trade_tick_value_profit=0.7, trade_tick_value_loss=0.7, volume_min=0.01, volume_step=0.01,
                       volume_max=200.0)


class FakeMT5:
    """One fake terminal per path: account, tick offset from UTC (the server clock), symbols."""
    ACCOUNT_TRADE_MODE_DEMO, ACCOUNT_TRADE_MODE_REAL = 0, 2

    def __init__(self, terminals: dict[str, dict]) -> None:
        self.terminals = terminals
        self.cur: dict | None = None
        self.calls: list[tuple] = []

    def initialize(self, **kw):
        self.calls.append(("initialize", kw))
        t = self.terminals.get(kw["path"])
        if t is None or t.get("fail"):
            return False
        self.cur = t
        return True

    def last_error(self):
        return (-1, "fake")

    def shutdown(self):
        self.calls.append(("shutdown",))
        self.cur = None

    def account_info(self):
        a = self.cur["account"]
        return SimpleNamespace(login=a["login"], server=a["server"], trade_mode=a.get("trade_mode", 0), margin_mode=2,
                               currency="USD", balance=a.get("balance", 1000.0), equity=a.get("balance", 1000.0),
                               margin=0.0, margin_free=a.get("balance", 1000.0))

    def positions_total(self):
        return self.cur.get("positions", 0)

    def orders_total(self):
        return self.cur.get("orders", 0)

    def symbol_select(self, sym, on):
        return True

    def symbol_info(self, sym):
        return SimpleNamespace(**{"digits": 5, "point": 1e-05, "trade_contract_size": 100000.0,
                                  "trade_tick_size": 1e-05, "trade_tick_value_profit": self.cur.get("tvp", 0.7),
                                  "trade_tick_value_loss": 0.7, "volume_min": 0.01, "volume_step": 0.01,
                                  "volume_max": 200.0})

    def symbol_info_tick(self, sym):
        offsets = self.cur["tick_offsets"]
        k = min(self.cur.setdefault("_ticks", 0), len(offsets) - 1)
        self.cur["_ticks"] += 1
        return SimpleNamespace(time=int(NOW.timestamp()) + offsets[k] - self.cur.get("age", 2))


def _accts() -> list[dict]:
    return [{"id": "acct1", "login": 1001, "server": "S", "terminal": "T1", "password_env": "PW1",
             "probe_symbol": "USDCADm", "pinned_symbol": "USDCADm"},
            {"id": "acct2", "login": 1002, "server": "S", "terminal": "T2", "password_env": "PW2",
             "probe_symbol": "USDCAD", "pinned_symbol": "USDCADm"}]


def _mt5(off1=(0,), off2=(0,), **extra) -> FakeMT5:
    return FakeMT5({"T1": {"account": {"login": 1001, "server": "S", "balance": 1000.0}, "tick_offsets": list(off1)},
                    "T2": {"account": {"login": 1002, "server": "S", "balance": 2000.0}, "tick_offsets": list(off2),
                           **extra}})


def _snap(m: FakeMT5, **kw) -> dict:
    return snapshot(m, accounts=_accts(), pinned_spec=lambda s: SPEC, env=kw.pop("env", {}),
                    sleep=kw.pop("sleep", lambda s: None), now=lambda: NOW, arming=kw.pop("arming", None))


def test_two_demo_accounts_land_in_one_snapshot_with_basis_labels() -> None:
    snap = _snap(_mt5())
    assert snap["schema"] == "arqvio.health.status" and [r["id"] for r in snap["accounts"]] == ["acct1", "acct2"]
    for r, bal in zip(snap["accounts"], (1000.0, 2000.0)):
        assert r["demo"] is True and r["status"] == "OK" and r["balance"] == bal
        assert r["basis"]["basis"] == "server_equals_utc" and r["basis"]["attempts"] == 1
        assert r["positions"] == 0 and r["orders"] == 0 and r["last_tick_age_s"] == 2.0


def test_the_simulated_eet_eest_fixture_is_labelled_and_converted() -> None:
    """G12 on a SIMULATED account: server clock UTC+2 (EET winter) and UTC+3 (EEST summer)."""
    snap = _snap(_mt5(off1=(7200,), off2=(10800,)))
    winter, summer = snap["accounts"][0]["basis"], snap["accounts"][1]["basis"]
    print(f"\nSIMULATED EET acct1 basis {winter}\nSIMULATED EEST acct2 basis {summer}")
    assert winter["basis"] == "broker_offset" and winter["measured_offset_s"] == 7200
    assert summer["basis"] == "broker_offset" and summer["measured_offset_s"] == 10800
    server_ts = int(NOW.timestamp()) + 7200  # 12:30 on the EET server clock
    assert basis_hour(server_ts, winter) == 10 and basis_hour(int(NOW.timestamp()) + 10800, summer) == 10
    assert all(r["status"] == "OK" and r["last_tick_age_s"] == 2.0 for r in snap["accounts"])  # not 7200 s stale


def test_an_unlabeled_or_unresolved_hour_keyed_input_is_refused() -> None:
    with pytest.raises(BasisRefused, match="NO time-basis label"):
        basis_hour(1_700_000_000, None)
    with pytest.raises(BasisRefused, match="'unresolved'"):
        basis_hour(1_700_000_000, BasisLabel("unresolved", 5000, 5000.0, 3, (5000.0,) * 3))
    assert to_utc(100_000, {"basis": "server_equals_utc", "measured_offset_s": 0}) == 100_000


def test_the_retry_rule_probes_three_times_then_marks_the_account_unavailable() -> None:
    slept = []
    snap = _snap(_mt5(off2=(5000, 5100, 4900)), sleep=slept.append)
    r = snap["accounts"][1]
    assert slept == [30.0, 60.0] and r["basis"]["basis"] == "unresolved" and r["basis"]["attempts"] == 3
    assert r["basis"]["deltas_s"] == [4998.0, 5098.0, 4898.0] and r["status"] == "UNAVAILABLE"
    assert "G12" in r["reason"]
    late = _snap(_mt5(off2=(5000, 0)), sleep=slept.append)["accounts"][1]
    assert late["basis"]["basis"] == "server_equals_utc" and late["basis"]["attempts"] == 2


def test_an_unarmed_real_account_is_unavailable_with_the_arm_command(tmp_path: Path) -> None:
    m = _mt5()
    m.terminals["T2"]["account"]["trade_mode"] = 2
    r = _snap(m, arming=tmp_path / "arming.json")["accounts"][1]
    assert (r["status"], r["real"], r["armed"]) == ("UNAVAILABLE", True, False)
    assert "not armed (login 1002, server S)" in r["reason"] and "arming arm --login 1002 --server S" in r["reason"]


def test_an_armed_real_account_is_probed_read_only(tmp_path: Path) -> None:
    m, path = _mt5(), tmp_path / "arming.json"
    m.terminals["T2"]["account"]["trade_mode"] = 2
    arming.arm(1002, "S", path=path)
    r = _snap(m, arming=path)["accounts"][1]
    assert (r["demo"], r["real"], r["status"], r["balance"]) == (False, True, "OK", 2000.0)


def test_staleness_unavailability_and_spec_drift() -> None:
    """S7 DEV-4: a tick 900 s behind now on a UTC server reads as a -900 s clock delta -- the clock is unmeasurable
    without a fresh tick -- so the account is STALE with an 'unresolved' label (hour-keyed rules still refuse)."""
    m = _mt5(age=900, tvp=0.69)
    m.terminals["T1"]["fail"] = True
    snap = _snap(m)
    a1, a2 = snap["accounts"]
    assert a1["status"] == "UNAVAILABLE" and "initialize failed" in a1["reason"]
    assert a2["status"] == "STALE" and a2["last_tick_age_s"] is None and a2["basis"]["basis"] == "unresolved"
    assert "clock unmeasurable: the last tick is 900 s behind now" in a2["reason"]
    assert a2["spec_drift"] == {"trade_tick_value_profit": {"live": 0.69, "pinned": 0.7}}
    with pytest.raises(BasisRefused):
        basis_hour(1_700_000_000, a2["basis"])
    kinds = [x["kind"] for x in from_health(snap, "status.json", clock=lambda: "c")]
    assert kinds == ["HEALTH_FAIL", "HEALTH_STALE"]


def test_an_odd_positive_offset_is_unavailable_and_a_basis_refusal_alert() -> None:
    snap = _snap(_mt5(off2=(5000, 5100, 4900)))
    kinds = [x["kind"] for x in from_health(snap, "status.json", clock=lambda: "c")]
    assert snap["accounts"][1]["status"] == "UNAVAILABLE" and kinds == ["BASIS_REFUSED", "HEALTH_FAIL"]


def test_the_password_comes_only_from_its_env_var() -> None:
    m = _mt5()
    _snap(m, env={"PW2": "secret"})
    inits = [kw for name, *rest in m.calls if name == "initialize" for kw in rest]
    assert "password" not in inits[0] and inits[1]["password"] == "secret"
    assert [c[0] for c in m.calls].count("shutdown") == 2


def test_the_config_refuses_literal_passwords_duplicates_and_gaps(tmp_path: Path) -> None:
    p = tmp_path / "accounts.json"
    for bad, msg in ((_accts()[:1] + [dict(_accts()[1], password="x")], "literal password"),
                     ([_accts()[0], dict(_accts()[1], login=1001)], "share a login"),
                     ([{k: v for k, v in _accts()[0].items() if k != "pinned_symbol"}], "pinned_symbol")):
        p.write_text(json.dumps({"accounts": bad}), encoding="utf-8")
        with pytest.raises(HealthError, match=msg):
            load_accounts(p)


def test_paths_resolve_at_call_time_and_create_nothing(isolated_arqvio_home: Path) -> None:
    assert accounts_path() == isolated_arqvio_home / "config" / "accounts.json"
    assert snapshot_path() == isolated_arqvio_home / "artifacts" / "health" / "status.json"
    assert not (isolated_arqvio_home / "artifacts" / "health").exists()


def test_probe_account_reads_one_account_and_always_shuts_down() -> None:
    m = _mt5()
    r = probe_account(m, _accts()[0], pinned_spec=lambda s: SPEC, env={}, sleep=lambda s: None, now=lambda: NOW)
    assert r["login"] == 1001 and m.calls[-1] == ("shutdown",)


ORDER_NAMES = {"order_send", "order_check", "send_order", "positions_get", "orders_get", "order_calc_margin"}
S7_PACKAGES = ("engine/research/portfolio", "engine/research/alerts", "engine/research/health")


def test_no_s7_module_references_an_order_path_or_the_live_package() -> None:
    root = Path(__file__).resolve().parents[3]
    hits = []
    for pkg in S7_PACKAGES:
        for f in sorted((root / pkg).glob("*.py")):
            tree = ast.parse(f.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and node.attr in ORDER_NAMES:
                    hits.append(f"{f.name}:{node.lineno} .{node.attr}")
                if isinstance(node, ast.Name) and node.id in ORDER_NAMES:
                    hits.append(f"{f.name}:{node.lineno} {node.id}")
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    mod = node.module if isinstance(node, ast.ImportFrom) else node.names[0].name
                    if mod and mod.startswith("engine.research.live"):
                        hits.append(f"{f.name}:{node.lineno} import {mod}")
    assert hits == []
