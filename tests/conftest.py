"""Shared pytest fixtures.

WHAT: Provide a fresh Settings instance and a clean test environment.
WHY: Tests must not depend on the developer's real ARQVIO_HOME; we override env.
HOW: monkeypatch ARQVIO_HOME to a tmp_path; reload settings cache.
TRADE-OFF: Slight fixture overhead per test in exchange for hermetic tests.
TEST: tests/config/test_settings.py asserts the override.
FAILURE: If ARQVIO_HOME is unset, the test fails loudly, not silently.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture
def isolated_arqvio_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point ARQVIO_HOME at a tmp_path; do not touch the real user dir."""
    home = tmp_path / "ARQVIO"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("ARQVIO_HOME", str(home))
    # Clear cached settings so they re-read env on next access.
    from engine.config import settings as settings_module

    settings_module.get_settings.cache_clear()
    yield home
    settings_module.get_settings.cache_clear()


@pytest.fixture
def mt5_acct1() -> Iterator[dict]:
    """A live MT5 attach to acct1 by explicit login, verified (tests/mt5_attach.py); shut down afterwards."""
    mt5 = pytest.importorskip("MetaTrader5")
    from tests.mt5_attach import attach

    acct = attach(mt5)
    yield acct
    mt5.shutdown()


@pytest.fixture
def fresh_event_bus() -> object:  # type: ignore[name-defined]
    """Return a brand-new EventBus (not the default module-level one)."""
    from engine.events.bus import EventBus

    return EventBus()
