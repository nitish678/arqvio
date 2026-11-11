"""Time helpers."""
from __future__ import annotations

from datetime import UTC, datetime

from engine.core.time import iso_utc, utc_now


def test_utc_now_is_aware() -> None:
    ts = utc_now()
    assert ts.tzinfo is not None
    assert ts.utcoffset() == UTC.utcoffset(ts)


def test_iso_utc_format() -> None:
    s = iso_utc()
    assert s.endswith("Z")
    # Round-trip: parse as ISO and compare.
    parsed = datetime.fromisoformat(s.replace("Z", "+00:00"))
    assert parsed.tzinfo is not None


def test_iso_utc_accepts_naive() -> None:
    naive = datetime(2026, 1, 1, 12, 0, 0)
    s = iso_utc(naive)
    assert s.startswith("2026-01-01T12:00:00")
    assert s.endswith("Z")
