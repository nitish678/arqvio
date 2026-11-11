"""Timeframes supported by ARQVIO.

The seven timeframes in the spec (M5 M15 M30 H1 H2 H4 D1) are the only ones
the campaign builder will offer by default. Each entry records its
periods-per-year (used for annualisation) and a per-cell enable flag.

Per-symbol/per-timeframe enable flags default to ``False``. The user opts
in explicitly. New timeframes (e.g. W1, MN) are not added in this pass.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

PERIODS_PER_YEAR: dict[str, int] = {
    "M5": 60 * 24 * 252 // 5,        # intraday session approximation
    "M15": 60 * 24 * 252 // 15,
    "M30": 60 * 24 * 252 // 30,
    "H1": 24 * 252,
    "H2": 12 * 252,
    "H4": 6 * 252,
    "D1": 252,
}


class Timeframe(str, Enum):
    M5 = "M5"
    M15 = "M15"
    M30 = "M30"
    H1 = "H1"
    H2 = "H2"
    H4 = "H4"
    D1 = "D1"

    @property
    def periods_per_year(self) -> int:
        return PERIODS_PER_YEAR[self.value]


DEFAULT_TIMEFRAMES: tuple[str, ...] = ("M5", "M15", "M30", "H1", "H2", "H4", "D1")


@dataclass
class TimeframeState:
    """User-facing enable flags for a single timeframe on a single symbol."""

    timeframe: str
    enabled: bool = False
    long_enabled: bool = False
    short_enabled: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "timeframe": self.timeframe,
            "enabled": self.enabled,
            "long_enabled": self.long_enabled,
            "short_enabled": self.short_enabled,
        }


@dataclass
class SymbolState:
    """User-facing state for one symbol: per-direction enable, per-timeframe
    enable, and a per-timeframe years_available cache."""

    symbol: str
    category: str = "OTHER"
    enabled: bool = False
    long_enabled: bool = False
    short_enabled: bool = False
    spread_calibrated: bool = False
    median_spread_points: float | None = None
    last_ingest_utc: str | None = None
    timeframes: dict[str, TimeframeState] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "category": self.category,
            "enabled": self.enabled,
            "long_enabled": self.long_enabled,
            "short_enabled": self.short_enabled,
            "spread_calibrated": self.spread_calibrated,
            "median_spread_points": self.median_spread_points,
            "last_ingest_utc": self.last_ingest_utc,
            "timeframes": {tf: s.to_dict() for tf, s in self.timeframes.items()},
        }
