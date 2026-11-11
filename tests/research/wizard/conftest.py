"""Shared fixtures for the wizard tests: an isolated home with an accounts config, and a fake MetaTrader5."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

LOGIN = 900001


class FakeMT5:
    """Just enough of MetaTrader5 for the wizard's read-only steps."""

    __version__ = "fake"

    def __init__(self, symbols: dict[str, dict] | None = None, *, login: int = LOGIN, mode: int = 0):
        self.symbols = symbols or {}
        self.login, self.mode = login, mode
        self.shut = False

    def initialize(self, **kw):
        self.kw = kw
        return True

    def account_info(self):
        return SimpleNamespace(login=self.login, trade_mode=self.mode)

    def symbol_select(self, symbol, enable):
        return symbol in self.symbols

    def symbol_info(self, symbol):
        return SimpleNamespace(**self.symbols[symbol]) if symbol in self.symbols else None

    def symbol_info_tick(self, symbol):
        return None  # a closed market: probe_basis raises, the audit records "unresolved"

    def terminal_info(self):
        return SimpleNamespace(build=1)

    def last_error(self):
        return (-1, "fake")

    def shutdown(self):
        self.shut = True


@pytest.fixture
def wizard_home(isolated_arqvio_home: Path) -> Path:
    """isolated_arqvio_home with an accounts config naming acct1 (no literal password)."""
    cfg = isolated_arqvio_home / "config"
    cfg.mkdir(parents=True, exist_ok=True)
    acct = {"id": "acct1", "login": LOGIN, "server": "Fake-Demo", "terminal": "C:/fake/terminal64.exe",
            "password_env": "ARQVIO_FAKE_PW_UNSET", "probe_symbol": "AAAm", "pinned_symbol": "AAAm"}
    (cfg / "accounts.json").write_text(json.dumps({"accounts": [acct]}), encoding="utf-8")
    return isolated_arqvio_home
