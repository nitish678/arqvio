"""Per-date broker UTC offset.

Different brokers sit in different timezones, and most change offset
twice a year for DST. The S2 spec requires that the data layer convert
broker-server time to UTC using a PER-DATE lookup, not a single static
offset. This is so that, e.g., a bar stamped "2024-03-10 22:00 EET" is
correctly identified as the rollover into DST (EET -> EEST).

For MT5/Exness the broker is in EET/EEST (UTC+2 winter, UTC+3 summer).
The DST transition dates are the last Sunday of March (spring forward
at 01:00 UTC -> 03:00 local) and the last Sunday of October (fall back
at 01:00 UTC -> 02:00 local). We encode these directly.
"""
from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

# Broker = Exness, server in EET/EEST.
DEFAULT_BROKER_TZ_NAME = "Europe/Helsinki"  # proxy for EET/EEST (same rules)

_WINTER_OFFSET_HOURS = 2   # EET
_SUMMER_OFFSET_HOURS = 3   # EEST


def _last_sunday(year: int, month: int) -> date:
    """Return the last Sunday of the given month/year.

    Mar 31, 2024 = Sunday, so this should return 2024-03-31 (not 30).
    """
    if month == 12:
        last_day_of_month = date(year, 12, 31)
    else:
        first_of_next = date(year, month + 1, 1)
        last_day_of_month = date.fromordinal(first_of_next.toordinal() - 1)
    # weekday(): Mon=0, Sun=6. Subtract that many days to land on Sunday.
    days_back = (last_day_of_month.weekday() + 1) % 7
    return date.fromordinal(last_day_of_month.toordinal() - days_back)


def _is_summer(dst_zone: str, d: date) -> bool:
    """True if date d is in the summer half of the year for the given zone.

    Currently only EET/EEST (last Sunday of March .. last Sunday of October)
    is supported. Other zones raise NotImplementedError.
    """
    if dst_zone == DEFAULT_BROKER_TZ_NAME:
        dst_start = _last_sunday(d.year, 3)
        dst_end = _last_sunday(d.year, 10)
        return dst_start <= d < dst_end
    raise NotImplementedError(f"DST rules for {dst_zone!r} not implemented")


def offset_hours_for_date(d: date, *, broker_tz: str = DEFAULT_BROKER_TZ_NAME) -> int:
    """Return the broker's UTC offset in hours for the given calendar date.

    Example: 2024-07-15 is summer -> +3. 2024-01-15 is winter -> +2.
    """
    return _SUMMER_OFFSET_HOURS if _is_summer(broker_tz, d) else _WINTER_OFFSET_HOURS


def broker_to_utc(
    broker_dt: datetime, *, broker_tz: str = DEFAULT_BROKER_TZ_NAME
) -> datetime:
    """Convert a naive broker-server datetime to a UTC-aware datetime.

    The naive datetime is interpreted as broker-local; the offset is
    looked up by the calendar date in the broker's local timezone.
    """
    if broker_dt.tzinfo is not None:
        raise ValueError("broker_dt must be naive (no tzinfo)")
    offset = offset_hours_for_date(broker_dt.date(), broker_tz=broker_tz)
    return broker_dt.replace(tzinfo=UTC) - timedelta(hours=offset)
