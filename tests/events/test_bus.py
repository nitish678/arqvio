"""Event bus: pub/sub isolation, handler error tolerance, type discrimination."""
from __future__ import annotations

from engine.events.bus import EventBus
from engine.events.schemas import SystemEvent, SystemHealthEvent, SystemStartupEvent


def test_publish_invokes_subscriber() -> None:
    bus = EventBus()
    seen: list[SystemStartupEvent] = []
    bus.subscribe(SystemStartupEvent, seen.append)
    bus.publish(SystemStartupEvent(version="0.1.0"))
    assert len(seen) == 1
    assert seen[0].version == "0.1.0"


def test_handlers_isolated_by_type() -> None:
    bus = EventBus()
    a: list[SystemStartupEvent] = []
    b: list[SystemHealthEvent] = []
    bus.subscribe(SystemStartupEvent, a.append)
    bus.subscribe(SystemHealthEvent, b.append)
    bus.publish(SystemStartupEvent(version="0.1.0"))
    assert len(a) == 1
    assert len(b) == 0


def test_failing_handler_does_not_stop_others() -> None:
    bus = EventBus()
    seen: list[SystemEvent] = []

    def boom(_e: SystemEvent) -> None:
        raise RuntimeError("kaboom")

    bus.subscribe(SystemStartupEvent, boom)
    bus.subscribe(SystemStartupEvent, seen.append)
    bus.publish(SystemStartupEvent(version="0.1.0"))
    assert len(seen) == 1


def test_clear_removes_all_handlers() -> None:
    bus = EventBus()
    seen: list[SystemEvent] = []
    bus.subscribe(SystemStartupEvent, seen.append)
    bus.clear()
    bus.publish(SystemStartupEvent(version="0.1.0"))
    assert seen == []


def test_handlers_for_unknown_type_is_empty() -> None:
    bus = EventBus()
    assert list(bus.handlers_for(SystemHealthEvent)) == []
