"""Every MT5 terminal on the machine, read live: installs, data directories, the running one's account, main pairs."""
from __future__ import annotations

import os
from collections import namedtuple
from pathlib import Path

import pytest

from engine.research.data import terminals as T


def _install(root: Path, name: str, *, portable: bool = False) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "terminal64.exe").write_bytes(b"MZ")
    if portable:
        (d / "MQL5").mkdir()
    return d


def _appdata_dir(appdata: Path, hash_: str, install: Path, server: str | None = None) -> Path:
    d = appdata / "MetaQuotes" / "Terminal" / hash_
    (d / "config").mkdir(parents=True)
    (d / "origin.txt").write_bytes(str(install).encode("utf-16"))
    if server:
        (d / "config" / "common.ini").write_bytes(f"[Common]\r\nLogin=1\r\nServer={server}\r\n".encode("utf-16"))
    return d


@pytest.fixture()
def machine(tmp_path, monkeypatch):
    appdata = tmp_path / "appdata"
    pf = tmp_path / "pf"
    monkeypatch.setenv("APPDATA", str(appdata))
    monkeypatch.setenv("ProgramFiles", str(pf))
    monkeypatch.delenv("ProgramFiles(x86)", raising=False)
    monkeypatch.setattr(T, "chosen_path", lambda: None)
    monkeypatch.setattr(T, "running_terminals", lambda: {})
    monkeypatch.setattr(T, "PORTABLE_ROOTS", (tmp_path / "portable",))
    return tmp_path, appdata, pf


def test_a_standard_install_reads_its_appdata_dir_and_remembered_server(machine) -> None:
    _, appdata, pf = machine
    inst = _install(pf, "Exness MT5")
    data = _appdata_dir(appdata, "ABC", inst, server="Exness-MT5Trial17")
    (t,) = [x for x in T.discover() if x.install_dir == str(inst.resolve())]
    assert t.data_dir == str(data) and not t.portable and not t.running
    assert t.remembered_server == "Exness-MT5Trial17"


def test_a_portable_install_is_its_own_data_dir_and_running_is_read_from_processes(machine, monkeypatch) -> None:
    root, _, _ = machine
    inst = _install(root / "portable", "mt5p", portable=True)
    exe = (inst / "terminal64.exe").resolve()
    import os

    monkeypatch.setattr(T, "running_terminals", lambda: {os.path.normcase(str(exe)): [4242]})
    (t,) = T.discover()
    assert t.portable and t.data_dir == str(inst.resolve()) and t.running and t.pids == (4242,)


def test_a_removed_install_is_gone_from_the_next_read(machine) -> None:
    _, appdata, pf = machine
    inst = _install(pf, "Broker MT5")
    _appdata_dir(appdata, "H1", inst)
    assert len(T.discover()) == 1
    (inst / "terminal64.exe").unlink()
    assert T.discover() == []


Acc = namedtuple("Acc", "login server company name currency balance equity leverage trade_mode")
Info = namedtuple("Info", "connected trade_allowed maxbars")


class FakeMt5:
    def __init__(self, ok=True, acc=None):
        self.ok, self.acc, self.calls, self.shutdowns = ok, acc, [], 0

    def initialize(self, path=None, timeout=None):
        self.calls.append(path)
        return self.ok

    def account_info(self):
        return self.acc

    def terminal_info(self):
        return Info(True, True, 100000)

    def symbols_get(self):
        return ()

    def shutdown(self):
        self.shutdowns += 1

    def last_error(self):
        return (-10005, "IPC timeout")


def _t(running: bool) -> T.Terminal:
    return T.Terminal("id", "C:/x/terminal64.exe", "C:/x", None, False, running, (1,) if running else (), None)


def test_a_closed_terminal_is_never_started_to_read_its_account() -> None:
    fake = FakeMt5()
    assert T.account(_t(False), mt5=fake)["state"] == "CLOSED"
    assert fake.calls == []


def test_the_running_terminal_account_is_read_now() -> None:
    acc = Acc(123, "Exness-MT5Trial17", "Exness Technologies Ltd", "Pro", "USD", 2008.28, 2008.28, 2000, 0)
    a = T.account(_t(True), mt5=FakeMt5(acc=acc))
    assert a["state"] == "OK" and a["server"] == "Exness-MT5Trial17" and a["mode"] == "DEMO" and a["max_bars"] == 100000
    assert T.account(_t(True), mt5=FakeMt5(acc=None))["state"] == "LOGGED_OUT"
    assert T.account(_t(True), mt5=FakeMt5(ok=False))["state"] == "UNREACHABLE"


ACC = Acc(123, "Exness-MT5Trial17", "Exness Technologies Ltd", "Pro", "USD", 2008.28, 2008.28, 2000, 0)


def _children(monkeypatch, before: list[int], after: list[int]) -> list[int]:
    """started_by_us answers ``before`` then ``after``; returns the list close_started is handed."""
    answers = iter([before, after])
    monkeypatch.setattr(T, "started_by_us", lambda path: next(answers))
    closed: list[int] = []
    monkeypatch.setattr(T, "close_started", closed.extend)
    return closed


@pytest.mark.parametrize("ok", [True, False])
def test_a_terminal_closed_meanwhile_is_not_left_open_by_the_read(monkeypatch, ok) -> None:
    """Owner's machine, 2026-10-26: MT5 closed while the app listed terminals came back 23 s later, started by the
    engine's mt5.initialize. The terminal the call started is closed again and the read says CLOSED."""
    closed = _children(monkeypatch, [], [7777])
    fake = FakeMt5(ok=ok, acc=ACC)
    a = T.account(_t(True), mt5=fake)
    assert a["state"] == "CLOSED" and "closed while ARQVIO read it" in a["detail"]
    assert closed == [7777] and fake.shutdowns == (1 if ok else 0)


def test_symbols_of_a_terminal_closed_meanwhile_are_refused_and_it_is_closed_again(monkeypatch) -> None:
    closed = _children(monkeypatch, [], [7777])
    with pytest.raises(RuntimeError, match="closed while ARQVIO read it"):
        T.symbols(_t(True), mt5=FakeMt5())
    assert closed == [7777]


def test_a_terminal_this_engine_started_earlier_is_read_and_left_alone(monkeypatch) -> None:
    closed = _children(monkeypatch, [5555], [5555])
    assert T.account(_t(True), mt5=FakeMt5(acc=ACC))["state"] == "OK"
    assert closed == []


@pytest.mark.parametrize(("names", "pair", "want"), [
    (["EURUSD"], "EURUSD", "EURUSD"),
    (["EURUSDm", "EURUSDm_raw"], "EURUSD", "EURUSDm"),
    (["EURUSD.pro"], "EURUSD", "EURUSD.pro"),
    (["XAUUSDm"], "XAUUSD", "XAUUSDm"),
    (["GOLD", "GOLDEUR"], "XAUUSD", "GOLD"),
    (["USTECm"], "US100", "USTECm"),
    (["DE30"], "GER40", "DE30"),
    (["EURGBP"], "EURUSD", None),
])
def test_main_pairs_take_the_broker_name(names, pair, want) -> None:
    rows = {r["pair"]: r["symbol"] for r in T.main_pairs(names)}
    assert rows[pair] == want


def test_market_watch_wins_over_a_shorter_hidden_name() -> None:
    rows = T.main_pairs([{"name": "EURUSD", "visible": False}, {"name": "EURUSDm", "visible": True}])
    assert rows[0] == {"pair": "EURUSD", "symbol": "EURUSDm"}


@pytest.mark.skipif(os.name != "nt", reason="process parents are read with the Windows toolhelp API")
def test_process_parents_and_liveness_are_read_from_windows() -> None:
    assert T._parent_pids()[os.getpid()] == os.getppid()
    assert T._alive(os.getpid()) is True
    assert T.started_by_us("C:/no/such/terminal64.exe") == []
