"""Time helpers.

WHAT: UTC normalization and ISO-8601 formatting.
WHY: All internal timestamps are UTC. Broker server time is recorded alongside, never mixed in.
HOW: datetime.now(timezone.utc); serialize with trailing 'Z' to match ISO-8601.
TRADE-OFF: We use stdlib datetime; pendulum/pytz are heavier and not needed yet.
TEST: tests/core/test_time.py asserts UTC tzinfo and format.
FAILURE: A missing import is a hard error.
"""
from __future__ import annotations

from datetime import UTC, datetime


def utc_now() -> datetime:
    """Return current UTC time as an aware datetime."""
    return datetime.now(UTC)


def iso_utc(ts: datetime | None = None) -> str:
    """Return ISO-8601 string with 'Z' suffix. If ts is None, uses utc_now()."""
    when = ts if ts is not None else utc_now()
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    # datetime.isoformat() produces '+00:00'; we prefer 'Z' for compactness and convention.
    return when.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
