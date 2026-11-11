"""Dedicated /healthz round-trip tests (more thorough than test_stage_1)."""
from __future__ import annotations

from fastapi.testclient import TestClient

from engine.api.server import create_app
from engine.events.schemas import SystemHealthEvent


def test_healthz_publishes_event() -> None:
    """Every /healthz hit should emit a SystemHealthEvent for any subscriber."""
    from engine.events.bus import get_default_bus

    bus = get_default_bus()
    seen: list[SystemHealthEvent] = []
    bus.subscribe(SystemHealthEvent, seen.append)

    client = TestClient(create_app())
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert len(seen) >= 1
    last = seen[-1]
    assert last.status == "ok"
    assert isinstance(last.api_port, int)
