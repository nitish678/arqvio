"""History depth audit.

For every (symbol, timeframe) cell, report years_available, bar_count,
first_bar_time, last_bar_time, source, as_of. If MT5 is unavailable, the
cell is reported as UNAVAILABLE - we never synthesise years.

This is the single most decision-changing artefact in the build: it
decides which timeframes are worth searching at all.
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from engine.data.tiers import TIER_1_TIMEFRAMES, TIER_2_TIMEFRAMES, TIER_3_TIMEFRAMES


@dataclass(frozen=True)
class HistoryCell:
    symbol: str
    timeframe: str
    years_available: float | None   # None == UNAVAILABLE
    bar_count: int
    first_bar_utc: str | None
    last_bar_utc: str | None
    source: str                      # BUNDLED | CACHED | MT5_LIVE | UNAVAILABLE
    as_of: str                       # ISO-8601 UTC

    @property
    def is_unavailable(self) -> bool:
        return self.years_available is None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _utc(seconds: int) -> str:
    return datetime.fromtimestamp(seconds, tz=UTC).isoformat()


def _tier_for(timeframe: str) -> str:
    if timeframe in TIER_1_TIMEFRAMES:
        return "TIER_1"
    if timeframe in TIER_2_TIMEFRAMES:
        return "TIER_2"
    if timeframe in TIER_3_TIMEFRAMES:
        return "TIER_3"
    return "UNSUPPORTED"


def audit_live(
    terminal_path: str,
    symbols: list[str],
    timeframes: list[str],
    out_csv: Path,
    *,
    bar_cap: int = 100_000,
    mt5_module: object = None,  # injected for tests
) -> list[HistoryCell]:
    """Probe MT5 for every (symbol, timeframe) and write a CSV.

    No synthetic fallback. An unavailable cell is reported as years=None.
    """
    if mt5_module is None:
        import MetaTrader5 as _mt5_module  # noqa: N813  # upstream convention
        mt5_module = _mt5_module
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows: list[HistoryCell] = []
    as_of = datetime.now(tz=UTC).isoformat()
    mt5: Any = mt5_module
    mt5.initialize(path=terminal_path)
    try:
        for symbol in symbols:
            for timeframe in timeframes:
                tf_const = getattr(mt5, f"TIMEFRAME_{timeframe}")
                rates = mt5.copy_rates_from_pos(
                    symbol, tf_const, 0, int(bar_cap)
                )
                if rates is None or len(rates) == 0:
                    rows.append(
                        HistoryCell(
                            symbol=symbol, timeframe=timeframe,
                            years_available=None, bar_count=0,
                            first_bar_utc=None, last_bar_utc=None,
                            source="UNAVAILABLE", as_of=as_of,
                        )
                    )
                    continue
                first = int(rates["time"][0])
                last = int(rates["time"][-1])
                years = (last - first) / (365.2425 * 86400.0)
                rows.append(
                    HistoryCell(
                        symbol=symbol, timeframe=timeframe,
                        years_available=float(years), bar_count=int(len(rates)),
                        first_bar_utc=_utc(first), last_bar_utc=_utc(last),
                        source="MT5_LIVE", as_of=as_of,
                    )
                )
    finally:
        mt5.shutdown()
    _write_csv(rows, out_csv)
    return rows


def audit_offline(cells: list[HistoryCell], out_csv: Path) -> None:
    """Write the audit CSV without contacting MT5 (used by tests)."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(cells, out_csv)


def _write_csv(rows: list[HistoryCell], out_csv: Path) -> None:
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "symbol", "timeframe", "years_available",
                "bar_count", "first_bar_utc", "last_bar_utc",
                "source", "as_of",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "symbol": row.symbol,
                    "timeframe": row.timeframe,
                    "years_available": (
                        f"{row.years_available:.4f}"
                        if row.years_available is not None else "UNAVAILABLE"
                    ),
                    "bar_count": row.bar_count,
                    "first_bar_utc": row.first_bar_utc or "",
                    "last_bar_utc": row.last_bar_utc or "",
                    "source": row.source,
                    "as_of": row.as_of,
                }
            )


def best_cells_by_history(
    rows: list[HistoryCell], top: int = 5
) -> list[HistoryCell]:
    """Return the cells with the longest reliable history.

    UNAVAILABLE cells sort to the end. Used by the campaign scope planner.
    """
    reliable = [r for r in rows if r.years_available is not None]
    reliable.sort(key=lambda r: r.years_available or 0.0, reverse=True)
    return reliable[:top]


__all__ = [
    "HistoryCell",
    "audit_live",
    "audit_offline",
    "best_cells_by_history",
]
