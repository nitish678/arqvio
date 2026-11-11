"""The order guard: the account re-read at every send, real accounts open only when armed, foreign magic refused,
own exposure always closable, one door to order_send."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from engine.research.accounts import arming
from engine.research.live import guard
from engine.research.mql5.magic import live_magic
from tests.research.live.fake_mt5 import FakeMT5

REPO = Path(__file__).resolve().parents[3]


def _req(mt5: FakeMT5, magic: int) -> dict:
    return {"action": mt5.TRADE_ACTION_DEAL, "symbol": "USDCADm", "volume": 0.01, "type": mt5.ORDER_TYPE_BUY,
            "price": mt5.ask, "magic": magic, "comment": "t"}


def test_the_fake_carries_the_real_package_constants() -> None:
    mt5 = pytest.importorskip("MetaTrader5", reason="MetaTrader5 package absent: constant check needs it")
    for name in ("ACCOUNT_TRADE_MODE_DEMO", "ACCOUNT_TRADE_MODE_CONTEST", "ACCOUNT_TRADE_MODE_REAL",
                 "TRADE_ACTION_DEAL", "TRADE_ACTION_PENDING", "TRADE_ACTION_REMOVE", "ORDER_TYPE_BUY",
                 "ORDER_TYPE_SELL", "ORDER_TYPE_BUY_LIMIT", "POSITION_TYPE_BUY", "POSITION_TYPE_SELL",
                 "ORDER_FILLING_FOK", "ORDER_FILLING_IOC", "ORDER_FILLING_RETURN", "ORDER_TIME_GTC",
                 "TRADE_RETCODE_DONE", "SYMBOL_TRADE_MODE_FULL"):
        assert getattr(FakeMT5, name) == getattr(mt5, name), name


def test_demo_passes_and_reports_the_account(tmp_path: Path) -> None:
    facts = guard.assert_tradable(FakeMT5(), arming=tmp_path / "arming.json")
    assert (facts.login, facts.server, facts.trade_mode, facts.margin_mode, facts.real) == (
        463965333, "Exness-MT5Trial17", 0, 2, False)


def test_an_unarmed_real_account_cannot_open_and_is_told_how_to_arm(tmp_path: Path) -> None:
    with pytest.raises(guard.RealAccountNotArmed,
                       match=r"login 463965333 server Exness-MT5Trial17 is not armed.*arming arm --login 463965333"):
        guard.assert_tradable(FakeMT5(trade_mode=2), arming=tmp_path / "arming.json")


def test_an_armed_real_account_opens(tmp_path: Path) -> None:
    path = tmp_path / "arming.json"
    arming.arm(463965333, "Exness-MT5Trial17", path=path)
    mt5 = FakeMT5(trade_mode=2)
    assert guard.assert_tradable(mt5, arming=path).real is True
    guard.send_order(mt5, _req(mt5, live_magic("worker")), own_range="s6-live-demo", arming=path)
    assert len(mt5.sent) == 1


def test_arming_is_per_login_and_server(tmp_path: Path) -> None:
    path = tmp_path / "arming.json"
    arming.arm(463965333, "Exness-MT5Real8", path=path)  # same login, another server
    with pytest.raises(guard.RealAccountNotArmed):
        guard.assert_tradable(FakeMT5(trade_mode=2), arming=path)


def test_closing_own_exposure_never_needs_arming(tmp_path: Path) -> None:
    mt5 = FakeMT5(trade_mode=2)
    guard.send_order(mt5, _req(mt5, live_magic("worker")), own_range="s6-live-demo", closing=True,
                     arming=tmp_path / "arming.json")
    assert len(mt5.sent) == 1


def test_contest_and_unknown_modes_are_refused(tmp_path: Path) -> None:
    for mode in (1, 7):
        with pytest.raises(guard.UnknownAccountMode, match="unknown or contest account mode"):
            guard.assert_tradable(FakeMT5(trade_mode=mode), arming=tmp_path / "arming.json")
        with pytest.raises(guard.UnknownAccountMode):
            guard.account_facts(FakeMT5(trade_mode=mode))


def test_every_send_rereads_the_account_and_a_switch_to_an_unarmed_real_is_caught(tmp_path: Path) -> None:
    mt5 = FakeMT5()
    own, path = live_magic("worker"), tmp_path / "arming.json"
    guard.send_order(mt5, _req(mt5, own), own_range="s6-live-demo", arming=path)
    reads = mt5.account_reads
    mt5.trade_mode = mt5.ACCOUNT_TRADE_MODE_REAL
    with pytest.raises(guard.RealAccountNotArmed):
        guard.send_order(mt5, _req(mt5, own), own_range="s6-live-demo", arming=path)
    assert mt5.account_reads == reads + 1  # re-read at the second send, not cached
    assert len(mt5.sent) == 1               # the second order never reached order_send


def test_a_foreign_magic_never_reaches_order_send() -> None:
    mt5 = FakeMT5()
    for magic in (0, 77789, 424242, 610_100_000):
        with pytest.raises(guard.ForeignMagicRefused, match="not in own range"):
            guard.send_order(mt5, _req(mt5, magic), own_range="s6-live-demo")
    assert mt5.sent == []


def test_order_send_is_called_only_inside_guard() -> None:
    callers = []
    for path in sorted((REPO / "engine").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "order_send":
                callers.append(path.relative_to(REPO).as_posix())
    assert callers == ["engine/research/live/guard.py"]
