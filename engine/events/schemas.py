"""Event schemas for the in-process bus.

WHAT: Pydantic models for every event type the bus carries.
WHY: A typed contract; producers and consumers agree on the shape.
HOW: One BaseModel per event; fields are explicit; no Any.
TRADE-OFF: Stage 1 defines only system events. New events land in their own modules.
TEST: tests/events/test_bus.py publishes a SystemStartupEvent and asserts it round-trips.
FAILURE: A model that cannot validate a payload is a hard error; the publisher must fix it.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from engine.core.time import iso_utc


class SystemEvent(BaseModel):
    """Base for system-level events (startup, shutdown, health)."""

    ts: str = Field(default_factory=iso_utc)


class SystemStartupEvent(SystemEvent):
    """Published once when the engine starts."""

    version: str


class SystemHealthEvent(SystemEvent):
    """Published periodically or on /healthz request."""

    status: str  # "ok" only at Stage 1
    version: str
    api_port: int


__all__ = ["SystemEvent", "SystemStartupEvent", "SystemHealthEvent"]
