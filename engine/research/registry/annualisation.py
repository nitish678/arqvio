"""The annualisation law: bars per year for a timeframe (DR-2026-10-07-u1 §4).

bars/year = 252 trading days x 24 h x 60 min / tf_minutes -- the 252-day convention every annualised Sharpe in
ARQVIO is on (engine/validation/stats.py, engine/data/timeframes.py). A timeframe outside the table is refused,
never defaulted: a cell on an unknown scale cannot be annualised honestly.
"""
from __future__ import annotations

from engine.core.errors import ConfigError

TRADING_DAYS_PER_YEAR = 252
MINUTES_PER_DAY = 24 * 60

TF_MINUTES: dict[str, int] = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H2": 120, "H4": 240, "D1": 1440}


class UnknownTimeframe(ConfigError):
    """A timeframe the annualisation law has no row for."""


def bars_per_year(timeframe: str) -> int:
    """The law value for ``timeframe``: 252 x 24 x 60 / tf_minutes (exact for every row of TF_MINUTES)."""
    if timeframe not in TF_MINUTES:
        raise UnknownTimeframe(f"no annualisation law for timeframe {timeframe!r}; known {sorted(TF_MINUTES)}")
    return TRADING_DAYS_PER_YEAR * MINUTES_PER_DAY // TF_MINUTES[timeframe]
