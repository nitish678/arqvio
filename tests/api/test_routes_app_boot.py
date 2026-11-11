"""The boot screen's checks (GET /api/v2/app/boot): one line per real check, grouped by start-up phase."""
from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from engine.api.server import create_app

GROUPS = ("engine", "environment", "library", "session")


def test_boot_lines_are_grouped_and_named(isolated_arqvio_home: Path, monkeypatch) -> None:
    from engine.research.data import terminals

    monkeypatch.setattr(terminals, "discover", lambda: [])  # no MetaTrader 5 on a test machine
    body = TestClient(create_app()).get("/api/v2/app/boot").json()
    checks = body["checks"]
    names = [c["check"] for c in checks]
    assert names == ["engine", "runtime", "backtest kernel", "data folder", "disk", "MetaTrader 5", "vault",
                     "gene registry", "launch queue", "deployed bots", "campaign"]
    assert all(c["group"] in GROUPS for c in checks)
    # phases run in order: a group never comes back after the next one started
    seen = [c["group"] for c in checks]
    assert seen == sorted(seen, key=GROUPS.index)
    mt5 = next(c for c in checks if c["check"] == "MetaTrader 5")
    assert mt5["ok"] is False and "no MetaTrader 5" in mt5["detail"]
    # no MetaTrader 5 is a warning, not a failed start: the vault and settings still work
    assert body["ok"] is True
    assert str(isolated_arqvio_home) in next(c for c in checks if c["check"] == "data folder")["detail"]
