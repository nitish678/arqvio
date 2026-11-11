"""Parquet + zstd storage for OHLCV bars.

~45 bytes per bar is the target (per the S5 spec). We store the
canonical bar columns as ``float64`` and rely on Parquet's zstd codec
plus the fact that all values are bounded to compress well.

Partitioning: ``{symbol}/{timeframe}/{year}.parquet``. The directory
layout mirrors the data-tier model: tier1 is read-only, tier2 sits
under the optional pack root, tier3 is the live MT5 sync target.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class BarBatch:
    time: np.ndarray  # int64 unix seconds, UTC
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    tick_volume: np.ndarray
    spread: np.ndarray
    real_volume: np.ndarray
    # "utc" is the only supported basis; recorded so a future broker-local
    # basis can be refused at concatenation (G12) instead of silently
    # mixing timestamps of different bases.
    time_basis: str = "utc"

    def n_bars(self) -> int:
        return int(self.time.size)


def store_bars(
    path: Path,
    bars: BarBatch,
    *,
    symbol: str,
    timeframe: str,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pyarrow is required for Parquet storage") from exc
    arrays = [
        pa.array(bars.time.astype("<i8")),
        pa.array(bars.open.astype("<f8")),
        pa.array(bars.high.astype("<f8")),
        pa.array(bars.low.astype("<f8")),
        pa.array(bars.close.astype("<f8")),
        pa.array(bars.tick_volume.astype("<i8")),
        pa.array(bars.spread.astype("<f8")),
        pa.array(bars.real_volume.astype("<i8")),
    ]
    names = [
        "time", "open", "high", "low", "close",
        "tick_volume", "spread", "real_volume",
    ]
    table = pa.table(arrays, names=names)
    pq.write_table(
        table, path,
        compression="zstd",
        compression_level=9,
        use_dictionary=False,
        write_statistics=False,
    )
    return path


def load_bars(path: Path) -> BarBatch:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pyarrow is required for Parquet loading") from exc
    table = pq.read_table(path)
    cols = {name: table.column(name).to_numpy() for name in table.column_names}
    return BarBatch(
        time=cols["time"].astype("<i8"),
        open=cols["open"].astype("<f8"),
        high=cols["high"].astype("<f8"),
        low=cols["low"].astype("<f8"),
        close=cols["close"].astype("<f8"),
        tick_volume=cols["tick_volume"].astype("<i8"),
        spread=cols["spread"].astype("<f8"),
        real_volume=cols["real_volume"].astype("<i8"),
    )


def approx_bytes_per_bar(path: Path) -> float:
    """Best-effort: file size / row count. Useful for verifying the ~45 B/bar target."""
    table_columns = load_bars(path)
    size = path.stat().st_size
    n = max(1, table_columns.n_bars())
    return float(size) / float(n)


def filter_years(
    bars: Iterable[BarBatch], years: list[int]
) -> BarBatch:
    """Concatenate batches and keep only bars in the given UTC years."""
    all_time: list[np.ndarray] = []
    all_open: list[np.ndarray] = []
    all_high: list[np.ndarray] = []
    all_low: list[np.ndarray] = []
    all_close: list[np.ndarray] = []
    all_tick_volume: list[np.ndarray] = []
    all_spread: list[np.ndarray] = []
    all_real_volume: list[np.ndarray] = []
    for batch in bars:
        years_arr = (
            np.array([(int(t) // 31556952) + 1970 for t in batch.time])
        )
        mask = np.isin(years_arr, np.asarray(years, dtype=np.int64))
        if not mask.any():
            continue
        all_time.append(batch.time[mask])
        all_open.append(batch.open[mask])
        all_high.append(batch.high[mask])
        all_low.append(batch.low[mask])
        all_close.append(batch.close[mask])
        all_tick_volume.append(batch.tick_volume[mask])
        all_spread.append(batch.spread[mask])
        all_real_volume.append(batch.real_volume[mask])
    if not all_time:
        return BarBatch(
            time=np.zeros(0, dtype="<i8"),
            open=np.zeros(0, dtype="<f8"),
            high=np.zeros(0, dtype="<f8"),
            low=np.zeros(0, dtype="<f8"),
            close=np.zeros(0, dtype="<f8"),
            tick_volume=np.zeros(0, dtype="<i8"),
            spread=np.zeros(0, dtype="<f8"),
            real_volume=np.zeros(0, dtype="<i8"),
        )
    return BarBatch(
        time=np.concatenate(all_time),
        open=np.concatenate(all_open),
        high=np.concatenate(all_high),
        low=np.concatenate(all_low),
        close=np.concatenate(all_close),
        tick_volume=np.concatenate(all_tick_volume),
        spread=np.concatenate(all_spread),
        real_volume=np.concatenate(all_real_volume),
    )


__all__ = [
    "BarBatch",
    "approx_bytes_per_bar",
    "filter_years",
    "load_bars",
    "store_bars",
]
