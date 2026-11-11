"""Add market over HTTP: one background run at a time, its answer recorded verbatim, the new cell served as QUICK."""
from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from engine.api.server import create_app
from tests.research.cells.test_add_market import NOW, FakeMt5, seed_symbol_files

REQ = {"symbol": "EURUSDm", "timeframe": "H4", "start": "2024-01-01", "end": "2026-01-01", "cross_cell": "usdcadm_h1"}


@pytest.fixture
def c(isolated_arqvio_home: Path, monkeypatch) -> TestClient:
    for sub in ("specs", "spread"):
        (isolated_arqvio_home / "artifacts" / sub).mkdir(parents=True)
    seed_symbol_files(isolated_arqvio_home)
    monkeypatch.setattr("engine.research.data.mt5_terminal.resolve_terminal", lambda: SimpleNamespace(path=None))
    return TestClient(create_app())


def settle(c: TestClient, timeout: float = 60.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = c.get("/api/v2/app/markets").json()["job"]
        if job and job["state"] != "RUNNING":
            return job
        time.sleep(0.2)
    raise AssertionError("the Add market run did not finish")


def test_an_add_runs_in_the_background_and_its_cell_is_served_as_quick(c: TestClient, monkeypatch) -> None:
    from engine.research.cells import add_market

    monkeypatch.setitem(sys.modules, "MetaTrader5", FakeMt5())
    real_run = add_market.run
    monkeypatch.setattr(add_market, "run", lambda *a, **k: real_run(*a, **k, now=NOW))  # the fake's clock
    first = c.get("/api/v2/app/markets").json()
    assert first["quick_cells"] == [] and first["job"] is None and "EURUSDm" in first["known_symbols"]
    assert first["timeframes"][0] == "M1" and first["timeframes"][-1] == "D1"
    r = c.post("/api/v2/app/markets/add", json=REQ)
    assert r.json() == {"state": "RUNNING", "cell_id": "eurusdm_h4"}
    job = settle(c)
    assert (job["state"], job["cell_id"], job["bars"], job["specs"], job["spread"]) == (
        "DONE", "eurusdm_h4", 3000, "kept", "kept")
    over = c.get("/api/v2/app/markets").json()
    assert [q["cell_id"] for q in over["quick_cells"]] == ["eurusdm_h4"]
    cells = c.get("/api/v2/app/cells").json()["cells"]
    quick = next(x for x in cells if x["cell_id"] == "eurusdm_h4")
    assert quick["status"] == "QUICK" and quick["render_state"] == "QUICK"
    assert quick["dataset_window"]["artifact_path"].endswith("user_cells.json")
    assert c.delete("/api/v2/app/markets/eurusdm_h4").json()["removed"] == "eurusdm_h4"
    assert c.delete("/api/v2/app/markets/eurusdm_h4").status_code == 404


def test_a_refusal_is_recorded_with_its_step(c: TestClient, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "MetaTrader5", FakeMt5(offered=()))
    c.post("/api/v2/app/markets/add", json=REQ)
    job = settle(c)
    assert (job["state"], job["step"]) == ("REFUSED", "symbol")
    assert "not offered by this terminal" in job["reason"]
    assert c.get("/api/v2/app/markets").json()["quick_cells"] == []
