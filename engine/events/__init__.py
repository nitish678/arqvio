"""Event bus + schemas.

Stage 1: synchronous in-memory bus, three system event types.
"""
from engine.events.bus import EventBus, get_default_bus, publish_startup_event
from engine.events.schemas import SystemEvent, SystemHealthEvent, SystemStartupEvent

__all__ = [
    "EventBus",
    "get_default_bus",
    "publish_startup_event",
    "SystemEvent",
    "SystemHealthEvent",
    "SystemStartupEvent",
]
