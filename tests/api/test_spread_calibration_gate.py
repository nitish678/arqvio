"""F2: launch is refused for a symbol with no calibrated spread matrix.

A campaign costed from a missing matrix is costed from nothing: the
backtester falls back to the symbol spec's single spread_points value,
which has no hour-of-day shape, so the rollover hours come out as cheap as
London noon and every candidate looks better than it is. That is a wrong
result, not a degraded one, and preflight/launch refuse it in the same
class as a missing audit row.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from engine.api.routes_s5 import SPREAD_NOT_CALIBRATED, spread_blocked_reason
from engine.api.server import create_app


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


def test_spread_blocked_reason_missing_artifact(tmp_path, monkeypatch) -> None:
    import engine.api.routes_s5 as routes

    monkeypatch.setattr(
        "engine.data.spread_calibration.spread_dir", lambda: tmp_path
    )
    reason = routes.spread_blocked_reason("NOSUCHm")
    assert reason is not None
    assert reason.startswith(SPREAD_NOT_CALIBRATED)
    assert "NOSUCHm.json" in reason


def test_spread_blocked_reason_unparseable(tmp_path, monkeypatch) -> None:
    (tmp_path / "JUNKm.json").write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(
        "engine.data.spread_calibration.spread_dir", lambda: tmp_path
    )
    reason = spread_blocked_reason("JUNKm")
    assert reason is not None
    assert SPREAD_NOT_CALIBRATED in reason


def test_spread_blocked_reason_wrong_bucket_count(tmp_path, monkeypatch) -> None:
    import json

    (tmp_path / "SHORTm.json").write_text(
        json.dumps(
            {
                "symbol": "SHORTm",
                "buckets": [10.0] * 12,
                "mean_buckets": [10.0] * 12,
                "sample_count": 100,
                "source": "MT5",
                "calibrator_version": 2,
                "calibrated_utc": "2026-09-24T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "engine.data.spread_calibration.spread_dir", lambda: tmp_path
    )
    reason = spread_blocked_reason("SHORTm")
    assert reason is not None
    assert "12 buckets" in reason


def test_spread_blocked_reason_none_when_calibrated(tmp_path, monkeypatch) -> None:
    import json

    (tmp_path / "GOODm.json").write_text(
        json.dumps(
            {
                "symbol": "GOODm",
                "buckets": [float(9 + (h % 4)) for h in range(24)],
                "mean_buckets": [float(9 + (h % 4)) for h in range(24)],
                "sample_count": 1000,
                "source": "MT5",
                "calibrator_version": 2,
                "calibrated_utc": "2026-09-24T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "engine.data.spread_calibration.spread_dir", lambda: tmp_path
    )
    assert spread_blocked_reason("GOODm") is None


def test_launch_refuses_uncalibrated_spread(client, tmp_path, monkeypatch) -> None:
    """The headline case: an uncalibrated symbol cannot start a campaign."""
    # An empty spread dir makes every symbol uncalibrated.
    monkeypatch.setattr(
        "engine.data.spread_calibration.spread_dir", lambda: tmp_path
    )

    res = client.post(
        "/api/v1/s5/launch",
        json={"population": 100, "selected": ["USDCADm:D1"]},
    )
    assert res.status_code == 422, res.text
    # The app wraps HTTPException detail as {"error": ..., "status_code": ...}
    # rather than FastAPI's default {"detail": ...}; assert the real contract.
    detail = res.json()["error"]
    assert detail["error"] == SPREAD_NOT_CALIBRATED
    blocked = detail["blocked"]
    assert len(blocked) == 1
    assert blocked[0]["cell"] == "USDCADm:D1"
    assert SPREAD_NOT_CALIBRATED in blocked[0]["reason"]


def test_preflight_marks_uncalibrated_cell_blocked(
    client, tmp_path, monkeypatch
) -> None:
    """Same fact, reported by preflight: the cell shows as unavailable."""
    monkeypatch.setattr(
        "engine.data.spread_calibration.spread_dir", lambda: tmp_path
    )

    res = client.get("/api/v1/s5/preflight", params={"selected": "USDCADm:D1"})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["all_satisfied"] is False
    (cell,) = body["cells"]
    assert cell["symbol"] == "USDCADm"
    assert cell["available"] is False
    assert SPREAD_NOT_CALIBRATED in cell["reason"]


def test_calibrated_symbol_is_not_blocked_by_the_spread_gate(client) -> None:
    """The gate must not refuse a symbol that IS calibrated.

    Uses the real artifact directory. This asserts only that the refusal
    is not the spread one -- MinBTL may still fail the cell, which is a
    different gate.
    """
    res = client.get("/api/v1/s5/preflight", params={"selected": "USDCADm:D1"})
    assert res.status_code == 200, res.text
    (cell,) = res.json()["cells"]
    reason = cell.get("reason") or ""
    assert SPREAD_NOT_CALIBRATED not in reason
