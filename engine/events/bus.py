"""In-process typed event bus.

WHAT: A simple pub/sub bus carrying Pydantic event models.
WHY: Decouples producers (data hub, AI, news) from consumers (UI SSE bridge, loggers).
HOW: A registry of subscriber callables; events dispatched synchronously by default.
TRADE-OFF: Synchronous dispatch is simple and predictable. Async dispatch arrives in Stage 13.
TEST: tests/events/test_bus.py asserts publish/subscribe and isolation between topics.
FAILURE: A subscriber raising logs the error but does not stop other subscribers.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import TypeVar

from pydantic import BaseModel

from engine.core.logging import get_logger
from engine.core.time import iso_utc

E = TypeVar("E", bound=BaseModel)
Handler = Callable[[BaseModel], None]


class EventBus:
    """A typed in-process pub/sub bus.

    Usage:
        bus = EventBus()
        bus.subscribe(SystemEvent, my_handler)
        bus.publish(SystemEvent(name="startup"))
    """

    def __init__(self) -> None:
        self._handlers: dict[type[BaseModel], list[Handler]] = defaultdict(list)
        self._log = get_logger()

    def subscribe(self, event_type: type[BaseModel], handler: Handler) -> None:
        """Register a handler for a specific event type."""
        self._handlers[event_type].append(handler)

    def handlers_for(self, event_type: type[BaseModel]) -> Iterable[Handler]:
        return tuple(self._handlers.get(event_type, ()))

    def publish(self, event: BaseModel) -> int:
        """Publish an event. Returns the number of handlers invoked.

        Handler exceptions are logged and do not stop other handlers.
        """
        handlers = self._handlers.get(type(event), ())
        for handler in handlers:
            try:
                handler(event)
            except Exception as exc:
                self._log.opt(exception=True).error(
                    "Event handler {handler} failed for {event_type}: {err}",
                    handler=getattr(handler, "__name__", repr(handler)),
                    event_type=type(event).__name__,
                    err=exc,
                )
        return len(handlers)

    def clear(self) -> None:
        """Remove all subscribers. Useful for tests."""
        self._handlers.clear()


# Module-level default bus. Tests should construct their own bus, not rely on this.
_default_bus = EventBus()


def get_default_bus() -> EventBus:
    """Return the process-wide default EventBus."""
    return _default_bus


def publish_startup_event(version: str) -> None:
    """Convenience: publish a SystemStartupEvent on the default bus."""
    from engine.events.schemas import SystemStartupEvent  # local import to avoid cycle

    _default_bus.publish(SystemStartupEvent(ts=iso_utc(), version=version))
