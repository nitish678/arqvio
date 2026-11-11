"""Deploy v3: a bot names its terminal, the terminal starts outside the engine's process tree, symbols map across
brokers."""
from __future__ import annotations

from pathlib import Path

from engine.api.routes_app_deploy import broker_symbol
from engine.research.deploy import full_deploy as fd


def test_the_terminal_is_started_through_cmd_start_so_closing_arqvio_leaves_it_running(monkeypatch, tmp_path) -> None:
    seen: dict = {}

    class FakePopen:
        def __init__(self, argv, **kw):
            seen["argv"], seen["kw"] = argv, kw

    monkeypatch.setattr(fd.subprocess, "Popen", FakePopen)
    exe = tmp_path / "mt5" / "terminal64.exe"
    fd.launch_detached(exe, tmp_path / "start.ini")
    argv = seen["argv"]
    assert argv[:4] == ["cmd", "/c", "start", ""]  # cmd exits at once: MT5's parent is gone, not the engine
    assert argv[4:6] == ["/D", str(exe.parent)] and argv[6] == str(exe)
    assert argv[7] == f"/config:{tmp_path / 'start.ini'}"


def test_same_terminal_scopes_staging_to_one_terminal() -> None:
    assert fd.same_terminal(r"C:\mt5\2\terminal64.exe", r"c:\MT5\2\terminal64.exe")
    assert not fd.same_terminal(r"C:\mt5\2\terminal64.exe", r"C:\mt5\3\terminal64.exe")
    assert fd.same_terminal(None, r"C:\mt5\3\terminal64.exe")  # a bot from before v3 belongs to every terminal
    assert fd.same_terminal(r"C:\mt5\2\terminal64.exe", None)  # no scope: every bot


def test_a_strategy_bred_on_one_broker_maps_to_another_brokers_name() -> None:
    exness = [{"name": "EURUSDm", "visible": True}, {"name": "XAUUSDm", "visible": True}]
    raw = [{"name": "EURUSD", "visible": True}, {"name": "XAUUSD", "visible": True}]
    assert broker_symbol("EURUSD", exness) == "EURUSDm"
    assert broker_symbol("XAUUSDm", raw) == "XAUUSD"
    assert broker_symbol("EURUSD", raw) == "EURUSD"
    assert broker_symbol("BTCUSD", raw) is None


def test_start_config_runs_the_attach_script_with_algo_trading_on(tmp_path: Path) -> None:
    ini = fd.start_config(tmp_path, {"symbol": "EURUSD", "timeframe": "H1"})
    text = ini.read_text(encoding="ascii")
    assert "AllowLiveTrading=1" in text and "Symbol=EURUSD" in text and "Period=H1" in text
