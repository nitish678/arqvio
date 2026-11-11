"""Stage 1 smoke test.

WHAT: Verifies the engine is importable, the FastAPI app builds, and /healthz returns 'ok'.
WHY: This is the one test that proves the round-trip the spec asks for.
HOW: FastAPI TestClient; we do not start uvicorn.
TRADE-OFF: In-process TestClient is slightly different from a live server, but is
        deterministic and exercises the same handlers. Live-server check is in scripts/.
TEST: Self.
FAILURE: If /healthz does not return 200, this test fails the stage.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from engine import __version__
from engine.api.server import create_app


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


def test_app_builds() -> None:
    app = create_app()
    assert app.title == "ARQVIO Engine"
    assert app.version == __version__


def test_healthz_returns_ok(client: TestClient) -> None:
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["version"] == __version__
    assert isinstance(body["api_port"], int)
    assert isinstance(body["ts"], str)
    assert body["ts"].endswith("Z")  # ISO-8601 with Z suffix


def test_ping_returns_pong(client: TestClient) -> None:
    resp = client.get("/api/v1/ping")
    assert resp.status_code == 200
    body = resp.json()
    assert body["pong"] is True
    assert body["version"] == __version__
    assert body["ts"].endswith("Z")


def test_unknown_route_returns_404(client: TestClient) -> None:
    resp = client.get("/nope")
    assert resp.status_code == 404
