"""CORS: only the desktop app's own origins may call the engine from a browser context."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from engine.api.server import APP_ORIGINS, create_app


@pytest.mark.parametrize("origin", APP_ORIGINS)
def test_app_origin_is_allowed(origin: str) -> None:
    resp = TestClient(create_app()).get("/api/v1/ping", headers={"Origin": origin})
    assert resp.headers.get("access-control-allow-origin") == origin


@pytest.mark.parametrize("origin", ["https://example.com", "http://localhost:3000", "null"])
def test_foreign_origin_gets_no_cors_grant(origin: str) -> None:
    resp = TestClient(create_app()).get("/api/v1/ping", headers={"Origin": origin})
    assert "access-control-allow-origin" not in resp.headers


def test_foreign_preflight_is_refused() -> None:
    resp = TestClient(create_app()).options(
        "/api/v1/ping",
        headers={"Origin": "https://example.com", "Access-Control-Request-Method": "GET"},
    )
    assert resp.status_code == 400
    assert "access-control-allow-origin" not in resp.headers
