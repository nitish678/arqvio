"""Incremental MT5 sync.

Fetches only the gap between the broker's most recent bar and the last
stored bar on disk. Never a full re-pull. Reports "37 bars added" per cell.
"""
from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from engine.core.errors import DataError
from engine.data.parquet_store import BarBatch, load_bars, store_bars
from engine.data.tiers import tier3_path


@dataclass(frozen=True)
class SyncReport:
    symbol: str
    timeframe: str
    bars_added: int
    last_stored_utc: str
    now_utc: str
    elapsed_seconds: float

    def to_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "bars_added": self.bars_added,
            "last_stored_utc": self.last_stored_utc,
            "now_utc": self.now_utc,
            "elapsed_seconds": self.elapsed_seconds,
        }

    def line(self) -> str:
        return f"{self.symbol} {self.timeframe}: {self.bars_added} bars added"


def _latest_bar_time(path: Path) -> int | None:
    if not path.exists():
        return None
    table = load_bars(path)
    if table.n_bars() == 0:
        return None
    return int(table.time[-1])


def incremental_sync(
    symbol: str,
    timeframe: str,
    *,
    pull_fn: Callable[[int, int], BarBatch | None],
    max_bars: int = 5000,
) -> SyncReport:
    """Pull from ``last_stored_bar_time`` forward; never a full re-pull.

    ``pull_fn(from_utc_seconds, to_utc_seconds)`` is injected so the runner
    can use the live MT5 puller in production and a fake puller in tests.
    """
    start = time.monotonic()
    now = datetime.now(tz=UTC)
    last_year = now.year
    target_path = tier3_path(symbol, timeframe, last_year)
    last = _latest_bar_time(target_path)
    from_utc = last + 1 if last is not None else 0
    to_utc = int(now.timestamp())
    bars = pull_fn(from_utc, to_utc)
    bars_added = 0
    if bars is not None and bars.n_bars() > 0:
        bars_added = min(int(bars.n_bars()), int(max_bars))
        if last is None:
            store_bars(target_path, bars, symbol=symbol, timeframe=timeframe)
        else:
            existing = load_bars(target_path)
            # G12: refuse to concatenate bars of different timestamp bases.
            # Mixing a broker-local history with broker-local new bars of a
            # different DST regime would silently mis-bucket hours. Only "utc"
            # is a known-good basis.
            if (
                existing.time_basis != "utc"
                or bars.time_basis != "utc"
            ):
                raise DataError(
                    f"refusing mixed-basis sync: existing.time_basis="
                    f"{existing.time_basis!r}, new.time_basis={bars.time_basis!r}"
                )
            combined = BarBatch(
                time=np.concatenate([existing.time, bars.time]),
                open=np.concatenate([existing.open, bars.open]),
                high=np.concatenate([existing.high, bars.high]),
                low=np.concatenate([existing.low, bars.low]),
                close=np.concatenate([existing.close, bars.close]),
                tick_volume=np.concatenate([existing.tick_volume, bars.tick_volume]),
                spread=np.concatenate([existing.spread, bars.spread]),
                real_volume=np.concatenate([existing.real_volume, bars.real_volume]),
                time_basis="utc",
            )
            store_bars(target_path, combined, symbol=symbol, timeframe=timeframe)
    elapsed = time.monotonic() - start
    last_stored = datetime.fromtimestamp(
        to_utc if last is None else last, tz=UTC
    ).isoformat()
    return SyncReport(
        symbol=symbol, timeframe=timeframe, bars_added=bars_added,
        last_stored_utc=last_stored, now_utc=now.isoformat(),
        elapsed_seconds=float(elapsed),
    )


def read_sync_manifest(directory: Path) -> list[dict[str, object]]:
    """List all sync reports that have been written to ``directory``."""
    if not directory.exists():
        return []
    out: list[dict[str, object]] = []
    for path in directory.rglob("sync.json"):
        try:
            out.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return out


__all__ = [
    "SyncReport",
    "incremental_sync",
    "read_sync_manifest",
]
