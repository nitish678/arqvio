"""Tests for the read-only MT5 live-detect probe (DR-2026-10-16-s9-community §2, §7).

Three concern planes:

1. ALLOWLIST -- the probe module only ever references the six permitted MT5
   calls.  A static source scan asserts the forbidden set (order_send,
   positions_get, CopyTicks, trade*, history*, ...) never appears.

2. TIMEOUT -- the 5 s ceiling is enforced; a probe that hangs reports
   MT5_PROBE_TIMEOUT, not a half-made run.

3. HONEST-STATE MATRIX -- every failure path returns its verbatim state and
   reason; MT5_PROBE_OK carries terminal_info, account_info (with DEMO from
   trade_mode == 0) and symbol_count, never an invented number.
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from engine.research.data.mt5_live_probe import (
    ALLOWED_CALLS,
    Mt5LiveResult,
    PROBE_TIMEOUT_S,
    clear_cache,
    probe_mt5_live,
)

REPO = Path(__file__).resolve().parents[2]
PROBE_SRC = REPO / "engine" / "research" / "data" / "mt5_live_probe.py"

FORBIDDEN_CALLS = [
    "order_send", "order_check", "order_close", "order_modify", "order_send",
    "positions_get", "position_get", "position_select", "position_close",
    "history_deals", "history_orders", "history_deal_", "history_order_",
    "CopyTicks", "CopyTicksRange", "CopyRates", "copy_ticks", "copy_rates",
    "trade", "Trade", "deal_send", "chart_", "symbol_select",
]


class _StubTerminal:
    """A fake terminal_info object with a couple of attributes."""
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class _StubAccount:
    """A fake account_info object; trade_mode 0 == DEMO."""
    def __init__(self, trade_mode=0, login=12345, **kw):
        self.trade_mode = trade_mode
        self.login = login
        for k, v in kw.items():
            setattr(self, k, v)


class _FakeMt5:
    """Injectible stand-in for the MetaTrader5 module."""

    def __init__(self, *, terminal_path="C:/mt5/1/terminal64.exe",
                 terminal_info=None, account_info=None, symbols=None,
                 init_ok=True, last_err=(0, ""), hang_init=False,
                 hang_terminal=False, hang_symbols=False):
        self._terminal_path = terminal_path
        self._ti = terminal_info
        self._ai = account_info
        self._syms = symbols if symbols is not None else []
        self._init_ok = init_ok
        self._err = last_err
        self.hang_init = hang_init
        self.hang_terminal = hang_terminal
        self.hang_symbols = hang_symbols
        self.calls: list[str] = []

    def initialize(self, path=None, **kw):
        self.calls.append("initialize")
        if self.hang_init:
            time.sleep(30)
        return self._init_ok

    def shutdown(self):
        self.calls.append("shutdown")

    def terminal_info(self):
        self.calls.append("terminal_info")
        if self.hang_terminal:
            time.sleep(30)
        return self._ti

    def account_info(self):
        self.calls.append("account_info")
        return self._ai

    def symbols_get(self):
        self.calls.append("symbols_get")
        if self.hang_symbols:
            time.sleep(30)
        return self._syms

    def symbol_info(self, name):
        self.calls.append("symbol_info")
        return None

    def last_error(self):
        self.calls.append("last_error")
        return self._err


# ---- 1. ALLOWLIST ---------------------------------------------------------------------------------------------------

def test_probe_source_contains_only_allowlisted_calls():
    """Static scan: the probe module references only the six permitted calls."""
    src = PROBE_SRC.read_text(encoding="utf-8")
    # Strip comments so a comment naming a forbidden call (to document the ban)
    # does not trigger a false positive.
    code_only = re.sub(r"#.*", "", src)
    code_only = re.sub(r'""".*?"""', "", code_only, flags=re.S)
    code_only = re.sub(r"'''.*?'''", "", code_only, flags=re.S)
    for bad in FORBIDDEN_CALLS:
        assert not re.search(rf"\bmt5\.{bad}\b|\bmt5\['{bad}'\]", code_only), \
            f"probe module references forbidden call mt5.{bad}"


def test_probe_source_has_no_bare_mt5_calls_outside_the_shim():
    """Every mt5.* call in the probe body is one of the six allowed."""
    src = PROBE_SRC.read_text(encoding="utf-8")
    code_only = re.sub(r"#.*", "", src)
    calls = re.findall(r"\bmt5\.(\w+)\b", code_only)
    for c in calls:
        assert c in ALLOWED_CALLS, f"probe module calls mt5.{c} which is not in ALLOWED_CALLS"


def test_allowed_calls_constant_is_complete():
    """The mutation-safe calls named in DR §2, plus last_error (read-only inquiry)."""
    assert ALLOWED_CALLS == frozenset(
        {"initialize", "shutdown", "terminal_info", "account_info",
         "symbols_get", "symbol_info", "last_error"}
    )


# ---- 2. TIMEOUT -----------------------------------------------------------------------------------------------------

def test_timeout_returns_mt5_probe_timeout(monkeypatch):
    """A probe that hangs on initialize reports MT5_PROBE_TIMEOUT, not a value."""
    clear_cache()
    fake = _FakeMt5(hang_init=True)
    monkeypatch.setattr("engine.research.data.mt5_live_probe.resolve_terminal", lambda: MagicMock(
        path="C:/mt5/1/terminal64.exe", source="DISCOVERED", candidates_checked=("C:/mt5/1/terminal64.exe",)))
    result = probe_mt5_live(mt5=fake)
    assert result.state == "MT5_PROBE_TIMEOUT"
    assert "detail" in result.to_dict()
    assert "5s" in result.detail or "5" in result.detail
    assert result.terminal_path is None  # honest: we never connected


def test_timeout_circuit_breaks_within_ceiling():
    """The real timeout must not exceed PROBE_TIMEOUT_S by more than a small margin."""
    assert PROBE_TIMEOUT_S == 5.0
    fake = _FakeMt5(hang_init=True)
    t0 = time.monotonic()
    result = probe_mt5_live(mt5=fake)
    elapsed = time.monotonic() - t0
    assert result.state == "MT5_PROBE_TIMEOUT"
    assert elapsed < PROBE_TIMEOUT_S + 2.0


# ---- 3. HONEST-STATE MATRIX -----------------------------------------------------------------------------------------

def test_terminal_not_found_reports_honest_state(monkeypatch):
    """No terminal on disk => MT5_NOT_RUNNING, terminal_path None."""
    clear_cache()
    monkeypatch.setattr("engine.research.data.mt5_live_probe.resolve_terminal", lambda: MagicMock(
        path=None, source="NOT_FOUND", candidates_checked=("C:/nowhere",)))
    result = probe_mt5_live(mt5=_FakeMt5())
    assert result.state == "MT5_NOT_RUNNING"
    assert result.terminal_path is None
    assert result.terminal_source == "NOT_FOUND"
    assert "not found" in result.detail.lower()


def test_initialize_failure_reports_honest_state(monkeypatch):
    """initialize returns False => MT5_NOT_RUNNING, not MT5_PROBE_OK."""
    clear_cache()
    monkeypatch.setattr("engine.research.data.mt5_live_probe.resolve_terminal", lambda: MagicMock(
        path="C:/mt5/1/terminal64.exe", source="DISCOVERED", candidates_checked=("C:/mt5/1/terminal64.exe",)))
    fake = _FakeMt5(init_ok=False, last_err=(1, "no connection"))
    result = probe_mt5_live(mt5=fake)
    assert result.state == "MT5_NOT_RUNNING"
    assert result.terminal_path == "C:/mt5/1/terminal64.exe"
    assert "initialize failed" in result.detail
    assert "no connection" in result.detail


def test_terminal_info_none_is_error(monkeypatch):
    clear_cache()
    monkeypatch.setattr("engine.research.data.mt5_live_probe.resolve_terminal", lambda: MagicMock(
        path="C:/mt5/1/terminal64.exe", source="DISCOVERED", candidates_checked=()))
    fake = _FakeMt5(terminal_info=None,
                    account_info=_StubAccount(trade_mode=0),
                    symbols=[], last_err=(2, "bad handle"))
    result = probe_mt5_live(mt5=fake)
    assert result.state == "MT5_PROBE_ERROR"
    assert "terminal_info" in result.detail
    assert "bad handle" in result.detail


def test_symbols_none_is_error(monkeypatch):
    clear_cache()
    monkeypatch.setattr("engine.research.data.mt5_live_probe.resolve_terminal", lambda: MagicMock(
        path="C:/mt5/1/terminal64.exe", source="DISCOVERED", candidates_checked=()))

    class _BadMt5(_FakeMt5):
        def symbols_get(self):
            self.calls.append("symbols_get")
            return None
        def last_error(self):
            return (3, "symbol group error")

    fake = _BadMt5(terminal_info=_StubTerminal(), account_info=_StubAccount(trade_mode=0))
    result = probe_mt5_live(mt5=fake)
    assert result.state == "MT5_PROBE_ERROR"
    assert "symbols_get" in result.detail


def test_probe_ok_carries_terminal_info_account_info_and_symbol_count(monkeypatch):
    """MT5_PROBE_OK: terminal_info, account_info (with demo flag), symbol_count."""
    clear_cache()
    monkeypatch.setattr("engine.research.data.mt5_live_probe.resolve_terminal", lambda: MagicMock(
        path="C:/mt5/1/terminal64.exe", source="DISCOVERED", candidates_checked=("C:/mt5/1/terminal64.exe",)))
    fake = _FakeMt5(
        terminal_info=_StubTerminal(path="C:/mt5/1/terminal64.exe", company="MetaQuotes"),
        account_info=_StubAccount(trade_mode=0, login=111, balance=1000.0),
        symbols=[MagicMock(), MagicMock(), MagicMock()],
    )
    result = probe_mt5_live(mt5=fake)
    assert result.state == "MT5_PROBE_OK"
    assert result.terminal_path == "C:/mt5/1/terminal64.exe"
    assert result.terminal_source == "DISCOVERED"
    assert result.terminal_info is not None
    assert result.terminal_info["company"] == "MetaQuotes"
    assert result.symbol_count == 3
    assert result.account_info is not None
    assert result.account_info["demo"] is True  # trade_mode == 0
    assert result.account_info["login"] == 111
    assert result.account_info["balance"] == 1000.0


def test_probe_ok_real_account_flag_is_false(monkeypatch):
    """trade_mode == 2 is REAL; demo flag must be False (no claim it is demo)."""
    clear_cache()
    monkeypatch.setattr("engine.research.data.mt5_live_probe.resolve_terminal", lambda: MagicMock(
        path="C:/mt5/1/terminal64.exe", source="DISCOVERED", candidates_checked=()))
    fake = _FakeMt5(
        terminal_info=_StubTerminal(),
        account_info=_StubAccount(trade_mode=2, login=222),  # REAL
        symbols=[],
    )
    result = probe_mt5_live(mt5=fake)
    assert result.state == "MT5_PROBE_OK"
    assert result.account_info is not None
    assert result.account_info["demo"] is False
    assert result.account_info["trade_mode"] == 2


def test_probe_calls_shutdown_after_ok():
    """Every successful or failed probe must call shutdown (no held terminal)."""
    clear_cache()
    fake = _FakeMt5(
        terminal_info=_StubTerminal(),
        account_info=_StubAccount(trade_mode=0),
        symbols=[],
    )
    import engine.research.data.mt5_live_probe as mod
    monkeypatch_terminal = MagicMock(
        path="C:/mt5/1/terminal64.exe", source="DISCOVERED", candidates_checked=("C:/mt5/1/terminal64.exe",))
    orig = mod.resolve_terminal
    mod.resolve_terminal = lambda: monkeypatch_terminal
    try:
        probe_mt5_live(mt5=fake)
    finally:
        mod.resolve_terminal = orig
    assert "shutdown" in fake.calls


def test_exception_in_account_info_is_verbatim_error(monkeypatch):
    """An exception mid-probe is recorded verbatim, MT5_PROBE_ERROR, not swallowed."""
    clear_cache()
    monkeypatch.setattr("engine.research.data.mt5_live_probe.resolve_terminal", lambda: MagicMock(
        path="C:/mt5/1/terminal64.exe", source="DISCOVERED", candidates_checked=()))

    class _ExplodingMt5(_FakeMt5):
        def account_info(self):
            self.calls.append("account_info")
            raise RuntimeError("connection reset by peer")

    fake = _ExplodingMt5(terminal_info=_StubTerminal(), symbols=[])
    result = probe_mt5_live(mt5=fake)
    assert result.state == "MT5_PROBE_ERROR"
    assert "RuntimeError" in result.detail
    assert "connection reset by peer" in result.detail


def test_cache_returns_stale_result_without_reprobing(monkeypatch):
    """Within the 15 s TTL, a second call returns the cached result and does not re-call initialize."""
    clear_cache()
    monkeypatch.setattr("engine.research.data.mt5_live_probe.resolve_terminal", lambda: MagicMock(
        path="C:/mt5/1/terminal64.exe", source="DISCOVERED", candidates_checked=("C:/mt5/1/terminal64.exe",)))
    fake = _FakeMt5(
        terminal_info=_StubTerminal(), account_info=_StubAccount(trade_mode=0), symbols=[],
    )
    r1 = probe_mt5_live(mt5=fake)
    calls_after_first = list(fake.calls)
    r2 = probe_mt5_live(mt5=fake)
    assert r1.to_dict() == r2.to_dict()
    assert fake.calls == calls_after_first  # no re-probe within TTL


def test_cache_ttl_is_15s(monkeypatch):
    """The cache TTL must be 15 s (DR §2)."""
    from engine.research.data.mt5_live_probe import CACHE_TTL_S
    assert CACHE_TTL_S == 15.0


def test_mt5_live_to_dict_shape_on_ok():
    """Serialised shape of an OK result is stable for the frontend."""
    r = Mt5LiveResult(
        state="MT5_PROBE_OK", source="mt5_live_probe", as_of="2026-01-01T00:00:00+00:00",
        terminal_path="/x/t64.exe", terminal_source="DISCOVERED",
        terminal_info={"company": "TestCo"}, account_info={"demo": True, "trade_mode": 0},
        symbol_count=42, candidates_checked=("/x/t64.exe",),
    )
    d = r.to_dict()
    assert d["state"] == "MT5_PROBE_OK"
    assert d["terminal_info"]["company"] == "TestCo"
    assert d["account_info"]["demo"] is True
    assert d["symbol_count"] == 42
    assert "detail" not in d  # detail only present on failure


def test_probe_ok_with_named_tuples_like_metatrader5_serialises(monkeypatch):
    """MetaTrader5 answers with named tuples: their count/index methods are not data and must not reach the payload
    (with a live terminal the environment route used to fail to serialize them)."""
    import json
    from collections import namedtuple

    clear_cache()
    monkeypatch.setattr("engine.research.data.mt5_live_probe.resolve_terminal", lambda: MagicMock(
        path="C:/mt5/1/terminal64.exe", source="DISCOVERED", candidates_checked=("C:/mt5/1/terminal64.exe",)))
    terminal = namedtuple("TerminalInfo", "company connected build")("MetaQuotes", True, 5000)
    account = namedtuple("AccountInfo", "login trade_mode balance")(111, 0, 1000.0)
    result = probe_mt5_live(mt5=_FakeMt5(terminal_info=terminal, account_info=account, symbols=[MagicMock()]))
    assert result.state == "MT5_PROBE_OK"
    assert result.terminal_info == {"company": "MetaQuotes", "connected": True, "build": 5000}
    assert result.account_info == {"login": 111, "trade_mode": 0, "balance": 1000.0, "demo": True}
    json.dumps(result.to_dict())  # the route serialises exactly this
