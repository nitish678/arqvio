"""Provenance for every backtest result.

A backtest result is never just a number. It carries:

  source          BUNDLED | CACHED | MT5_LIVE
  dataset_hash    sha256 of the underlying bar bytes
  as_of           the moment the data was current at
  mixed_source    True iff the result was computed across two or more
                  sources (e.g. MT5_LIVE tail stitched to BUNDLED head)
                  - such results are blocked from campaign launch by
                  default.
  staleness_seconds  ``(now - as_of)`` rounded to the second.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta


class DataSource(str, enum.Enum):
    BUNDLED = "BUNDLED"
    CACHED = "CACHED"
    MT5_LIVE = "MT5_LIVE"

    @classmethod
    def parse(cls, value: str) -> DataSource:
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"unknown data source: {value!r}")


def parse_source(value: str) -> DataSource:
    return DataSource.parse(value)


@dataclass(frozen=True)
class Provenance:
    source: DataSource
    dataset_hash: str
    as_of: datetime
    mixed_source: bool = False

    def staleness(self, now: datetime | None = None) -> timedelta:
        now = now or datetime.now(tz=UTC)
        return now - self.as_of

    def staleness_seconds(self, now: datetime | None = None) -> int:
        return int(self.staleness(now).total_seconds())


def _format_human(delta: timedelta) -> str:
    seconds = int(delta.total_seconds())
    if seconds < 0:
        return "in the future"
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    if seconds < 86400:
        return f"{seconds // 3600}h"
    days = seconds // 86400
    return f"{days} day{'s' if days != 1 else ''}"


def report_staleness_line(p: Provenance, *, now: datetime | None = None) -> str:
    """The literal banner the spec requires.

    Examples:
        DATA: MT5_LIVE · synced 12s ago
        DATA: BUNDLED
        DATA: CACHED, 3 days old
        DATA: MT5_LIVE · unknown freshness
    """
    now = now or datetime.now(tz=UTC)
    if p.source is DataSource.MT5_LIVE:
        delta = p.staleness(now)
        if delta.total_seconds() < 0:
            return "DATA: MT5_LIVE · unknown freshness"
        return f"DATA: MT5_LIVE · synced {_format_human(delta)} ago"
    if p.source is DataSource.BUNDLED:
        return "DATA: BUNDLED"
    # CACHED
    return f"DATA: CACHED, {_format_human(p.staleness(now))} old"


def format_staleness(p: Provenance, *, now: datetime | None = None) -> str:
    return report_staleness_line(p, now=now)


__all__ = [
    "DataSource",
    "Provenance",
    "format_staleness",
    "parse_source",
    "report_staleness_line",
]
