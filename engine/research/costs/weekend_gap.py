"""Weekend gap fill model.

FX is closed Friday ~22:00 broker-local to Sunday ~22:00 broker-local.
A strategy that was holding a position over the weekend cannot be filled
"during" Saturday or Sunday - it must be filled at the Monday open.

Per spec: positions held over the weekend are valued at the Monday open,
not at the last Friday close. This is the gap: the difference between
the Friday close and the Monday open is the weekend gap; the backtester
realises the P&L of that gap on Monday's open bar.

This module exposes the rule and a helper to detect whether a date is in
the weekend gap and what the opening bar should be.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

# Broker (Exness) trading week:
#   Friday close:  22:00 broker-local (EET/EEST)
#   Sunday open:   22:00 broker-local
# Sat is fully closed; Sun morning is closed.
FRIDAY_CLOSE_HOUR = 22
SUNDAY_OPEN_HOUR = 22


@dataclass(frozen=True)
class WeekendGap:
    """A realised gap between Friday close and Monday open.

    For a LONG position: gap = monday_open - friday_close
    For a SHORT: gap = -(monday_open - friday_close) i.e. profit when price falls.
    """

    friday_close_date: date
    monday_open_date: date
    friday_close: float
    monday_open: float
    direction: str        # "long" or "short"

    @property
    def gap_price(self) -> float:
        return self.monday_open - self.friday_close

    @property
    def gap_signed(self) -> float:
        return self.gap_price if self.direction == "long" else -self.gap_price

    def to_dict(self) -> dict[str, object]:
        return {
            "friday_close_date": self.friday_close_date.isoformat(),
            "monday_open_date": self.monday_open_date.isoformat(),
            "friday_close": self.friday_close,
            "monday_open": self.monday_open,
            "direction": self.direction,
            "gap_price": self.gap_price,
            "gap_signed": self.gap_signed,
        }


def is_weekend(d: date) -> bool:
    """True if d is Saturday or Sunday (the broker is closed)."""
    return d.weekday() in (5, 6)


def is_friday_close_bar(bar_open_utc: datetime) -> bool:
    """True if a bar's open time is the Friday close (22:00 UTC)."""
    # The Friday close at the broker is 22:00 EET/EEST, which is 19:00 UTC
    # winter or 18:00 UTC summer. We do not yet account for DST here -
    # the backtester converts bar times to UTC via the per-date offset
    # table; this function checks the UTC hour directly.
    return bar_open_utc.weekday() == 4 and bar_open_utc.hour == 22


def next_monday_open(d: date) -> date:
    """Given any date, return the date of the next Monday market open.

    If d is Mon-Fri: returns d if d is Monday, else d + days_to_monday.
    If d is Sat: returns the following Monday.
    If d is Sun: returns the following day (Monday).
    """
    if d.weekday() == 0:  # already Monday
        return d
    if d.weekday() <= 4:  # Tue-Fri
        return d + timedelta(days=(7 - d.weekday()))
    if d.weekday() == 5:  # Sat
        return d + timedelta(days=2)
    return d + timedelta(days=1)  # Sun
