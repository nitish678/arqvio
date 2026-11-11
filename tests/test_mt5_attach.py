"""The tests-only MT5 attach policy (U4 E1, ruling U3-Q1), on a fake MetaTrader5: an absent environment SKIPS; a
wrong account, a non-DEMO account or a missing symbol on the attached terminal FAILS (skip-by-ruling never)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tests import mt5_attach


class FakeMT5:
    def __init__(self, *, ok: bool = True, login: int = 111, mode: int = 0, symbols: tuple[str, ...] = ("USDCADm",)):
        self.ok, self.login, self.mode, self.symbols = ok, login, mode, symbols
        self.kw: dict = {}
        self.shut = False

    def initialize(self, **kw):
        self.kw = kw
        return self.ok

    def account_info(self):
        return SimpleNamespace(login=self.login, trade_mode=self.mode)

    def symbol_select(self, symbol, enable):
        return symbol in self.symbols

    def last_error(self):
        return (-1, "fake")

    def shutdown(self):
        self.shut = True


@pytest.fixture
def row(tmp_path: Path, monkeypatch) -> dict:
    exe = tmp_path / "terminal64.exe"
    exe.write_bytes(b"")
    r = {"id": "acct1", "login": 111, "server": "Demo-Server", "terminal": str(exe), "password_env": "ARQVIO_TEST_PW_UNSET"}
    monkeypatch.setattr(mt5_attach, "acct1", lambda: r)
    monkeypatch.delenv("ARQVIO_TEST_PW_UNSET", raising=False)
    return r


def test_attach_is_by_explicit_login_and_verified(row) -> None:
    fake = FakeMT5()
    assert mt5_attach.attach(fake) == row
    assert fake.kw == {"path": row["terminal"], "login": 111, "server": "Demo-Server"}


def test_an_absent_environment_skips(row, monkeypatch) -> None:
    with pytest.raises(pytest.skip.Exception):
        mt5_attach.attach(FakeMT5(ok=False))
    monkeypatch.setattr(mt5_attach, "acct1", lambda: None)
    with pytest.raises(pytest.skip.Exception):
        mt5_attach.attach(FakeMT5())


@pytest.mark.parametrize("fake", [FakeMT5(login=222), FakeMT5(mode=2)], ids=["other-login", "non-demo"])
def test_the_wrong_account_fails_and_detaches(row, fake) -> None:
    with pytest.raises(pytest.fail.Exception):
        mt5_attach.attach(fake)
    assert fake.shut


def test_a_missing_symbol_fails_never_skips() -> None:
    mt5_attach.require_symbol(FakeMT5(), "USDCADm")
    with pytest.raises(pytest.fail.Exception):
        mt5_attach.require_symbol(FakeMT5(), "NOSUCHm")
