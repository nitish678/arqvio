"""Tests-only MT5 attach by EXPLICIT account login (U4 E1; owner ruling U3-Q1).

Live-MT5 tests attach to install 1's account ``acct1`` from <ARQVIO_HOME>/config/accounts.json (resolved at call time
through engine.research.health.probe.accounts_path). They never attach to whichever terminal a bare initialize()
reaches or a hard-coded path names, and they verify account_info().login.

- An ABSENT environment SKIPS: no MetaTrader5 package, no accounts.json on this host, no terminal executable, or the
  terminal not attachable. An unavailable terminal is not a failing check.
- Anything that means the RIGHT terminal is wrong FAILS: a malformed config, no acct1, a login mismatch, a non-DEMO
  account, a missing symbol. Skip-by-ruling never.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import pytest

ACCOUNT_ID = "acct1"
DEMO = 0  # MetaTrader5.ACCOUNT_TRADE_MODE_DEMO

_ATTACH_TIMEOUT_S = 30.0


def acct1() -> dict[str, Any] | None:
    """acct1's config row, or None when this host has no accounts config. Reads the config only; no MT5 call."""
    from engine.research.health.probe import accounts_path, load_accounts

    if not accounts_path().is_file():
        return None
    rows = [a for a in load_accounts() if a["id"] == ACCOUNT_ID]  # a malformed config raises HealthError: a FAIL
    if len(rows) != 1:
        raise RuntimeError(f"accounts config has {len(rows)} rows with id {ACCOUNT_ID!r}, expected exactly 1")
    return rows[0]


def acct1_terminal() -> str:
    """acct1's terminal path, or '' when this host has no accounts config (the attach fixture then skips)."""
    row = acct1()
    return row["terminal"] if row else ""


def _initialize_with_timeout(mt5: Any, **kw: Any) -> tuple[bool, tuple[int, str] | None]:
    """Run mt5.initialize() in a worker thread with a hard ceiling.

    The MetaTrader5 module's initialize() can block indefinitely if the
    terminal cannot start or reach a login. This wrapper enforces a timeout
    so the attach fixture skips rather than hangs the entire suite.
    """
    box: list[Any] = [None]

    def worker() -> None:
        try:
            box[0] = mt5.initialize(**kw)
        except Exception as exc:
            box[0] = exc

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout=_ATTACH_TIMEOUT_S)

    if t.is_alive():
        return False, None
    if isinstance(box[0], Exception):
        return False, None
    return bool(box[0]), None


def attach(mt5: Any) -> dict[str, Any]:
    """initialize() acct1 by path + login + server and verify the account; skip or fail as the module docstring says."""
    row = acct1()
    if row is None:
        pytest.skip("no accounts config on this host")
    if not Path(row["terminal"]).is_file():
        pytest.skip(f"{ACCOUNT_ID} terminal not installed at {row['terminal']}")
    kw = {"path": row["terminal"], "login": int(row["login"]), "server": row["server"]}
    if os.environ.get(row["password_env"]):
        kw["password"] = os.environ[row["password_env"]]
    ok, _ = _initialize_with_timeout(mt5, **kw)
    if not ok:
        pytest.skip(f"MT5 attach to {ACCOUNT_ID} failed (timed out or error): {mt5.last_error()}")
    info = mt5.account_info()
    if info is None or int(info.login) != int(row["login"]) or int(info.trade_mode) != DEMO:
        mt5.shutdown()
        pytest.fail(f"attached to the wrong account: expected {ACCOUNT_ID} (DEMO), got "
                    f"{'no account' if info is None else 'another login or a non-DEMO account'}")
    return row


def require_symbol(mt5: Any, symbol: str) -> None:
    """On the verified acct1 terminal, a symbol that cannot be selected is a defect: FAIL, never skip."""
    if not mt5.symbol_select(symbol, True):
        pytest.fail(f"symbol_select({symbol}) failed on {ACCOUNT_ID}: {mt5.last_error()}")
