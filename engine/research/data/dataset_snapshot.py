"""Pinned bar snapshots, and a hash that can actually tell two apart.

WHY THIS EXISTS (G14). ``load_cell_bars`` calls MT5's
``copy_rates_from_pos(symbol, tf, 0, count)``, which returns the MOST
RECENT ``count`` bars. The window therefore slides forward with
wall-clock time: same length, different bars. Between the S1 XAUUSDm
acceptance run and a re-run one hour later, one bar left the front and
one joined the back, and seven positions that had been open at
end-of-data gained a bar in which to touch a level -- EOD 104 -> 97,
redistributing into TP and SL, with no code change.

The provenance field that exists to catch exactly this could not see
it. ``dataset_hash`` was ``f"real-mt5-{symbol}-{timeframe}-{n}"``
(real_backtest.py:319) -- symbol, timeframe and BAR COUNT. Two
different datasets of equal length hash identically, and both of those
runs recorded ``real-mt5-XAUUSDm-H1-57381``.

So no result computed on a live window was ever reproducible, and no
fast-vs-slow parity claim can be made against one. A snapshot fixes the
bars; the content hash makes a substitution detectable.

THE CANONICAL BYTE LAYOUT
-------------------------

``dataset_hash`` is the first 16 hex characters of a SHA-256 over the
concatenation, in this exact order, of:

    1. b"ASTRA-DATASET-v1"                  format tag (frozen: the dataset ids are hashes under it)
    2. symbol,  UTF-8, NUL-terminated
    3. timeframe, UTF-8, NUL-terminated
    4. n_bars as int64, little-endian
    5. time   as int64   little-endian, n_bars values
    6. open   as float64 little-endian, n_bars values
    7. high   as float64 little-endian, n_bars values
    8. low    as float64 little-endian, n_bars values
    9. close  as float64 little-endian, n_bars values
   10. spread as int32   little-endian, n_bars values

Byte order is forced little-endian so the id does not depend on the
machine that computed it. Volume is deliberately NOT hashed: the
backtest never reads it, and including it would make the id change for
a reason that cannot affect a result. The format tag is version 1 so a
future layout change cannot silently collide with an existing id.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from engine.core.errors import DataError
from engine.core.paths import get_artifacts_dir

DATASETS_DIRNAME = "datasets"
FORMAT_TAG = b"ASTRA-DATASET-v1"  # frozen: every pinned dataset id is a hash under this tag (the pre-ARQVIO name)
HASH_CHARS = 16


def datasets_dir() -> Path:
    return get_artifacts_dir() / DATASETS_DIRNAME


def dataset_hash(
    *,
    symbol: str,
    timeframe: str,
    time: np.ndarray,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    spread: np.ndarray,
) -> str:
    """Content digest of a bar set. See the module docstring for the layout."""
    n = int(np.asarray(time).size)
    for name, arr in (
        ("open", open_), ("high", high), ("low", low),
        ("close", close), ("spread", spread),
    ):
        if int(np.asarray(arr).size) != n:
            raise DataError(f"{name} has {np.asarray(arr).size} values, expected {n}")

    digest = hashlib.sha256()
    digest.update(FORMAT_TAG)
    digest.update(symbol.encode("utf-8") + b"\x00")
    digest.update(str(timeframe).encode("utf-8") + b"\x00")
    digest.update(np.int64(n).astype("<i8").tobytes())
    digest.update(np.asarray(time, dtype=np.int64).astype("<i8").tobytes())
    for arr in (open_, high, low, close):
        digest.update(np.asarray(arr, dtype=np.float64).astype("<f8").tobytes())
    digest.update(np.asarray(spread, dtype=np.int32).astype("<i4").tobytes())
    return digest.hexdigest()[:HASH_CHARS]


@dataclass(frozen=True)
class SnapshotMeta:
    """The sidecar: what was fetched, when, and from where."""

    dataset_id: str
    symbol: str
    timeframe: str
    n_bars: int
    start_utc: str
    end_utc: str
    server: str | None
    snapshot_utc: str
    time_basis: str = "utc"

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "n_bars": self.n_bars,
            "start_utc": self.start_utc,
            "end_utc": self.end_utc,
            "server": self.server,
            "snapshot_utc": self.snapshot_utc,
            "time_basis": self.time_basis,
        }


def _paths(symbol: str, timeframe: str, digest: str, directory: Path):
    stem = f"{symbol}_{timeframe}_{digest}"
    return directory / f"{stem}.npz", directory / f"{stem}.json"


def save_snapshot(
    *,
    symbol: str,
    timeframe: str,
    time: np.ndarray,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    spread: np.ndarray,
    tick_volume: np.ndarray | None = None,
    real_volume: np.ndarray | None = None,
    server: str | None = None,
    time_basis: str = "utc",
    target_dir: Path | None = None,
) -> SnapshotMeta:
    """Persist a bar set under its own content hash."""
    directory = datasets_dir() if target_dir is None else target_dir
    directory.mkdir(parents=True, exist_ok=True)

    time = np.asarray(time, dtype=np.int64)
    n = int(time.size)
    if n == 0:
        raise DataError("refusing to snapshot an empty bar set")

    digest = dataset_hash(
        symbol=symbol, timeframe=timeframe, time=time, open_=open_,
        high=high, low=low, close=close, spread=spread,
    )
    npz_path, json_path = _paths(symbol, timeframe, digest, directory)

    zeros_i = np.zeros(n, dtype=np.int64)
    np.savez(
        npz_path,
        time=time,
        open=np.asarray(open_, dtype=np.float64),
        high=np.asarray(high, dtype=np.float64),
        low=np.asarray(low, dtype=np.float64),
        close=np.asarray(close, dtype=np.float64),
        spread=np.asarray(spread, dtype=np.int32),
        tick_volume=(
            zeros_i if tick_volume is None
            else np.asarray(tick_volume, dtype=np.int64)
        ),
        real_volume=(
            zeros_i if real_volume is None
            else np.asarray(real_volume, dtype=np.int64)
        ),
    )

    if time_basis not in ("utc", "server_time"):
        raise DataError(
            f"time_basis must be 'utc' or 'server_time'; got {time_basis!r}. Mixed-basis "
            f"timestamps are refused rather than mis-bucket."
        )
    meta = SnapshotMeta(
        dataset_id=digest,
        symbol=symbol,
        timeframe=str(timeframe),
        n_bars=n,
        start_utc=datetime.fromtimestamp(int(time[0]), tz=UTC).isoformat(),
        end_utc=datetime.fromtimestamp(int(time[-1]), tz=UTC).isoformat(),
        server=server,
        snapshot_utc=datetime.now(tz=UTC).isoformat(),
        time_basis=time_basis,
    )
    json_path.write_text(
        json.dumps(meta.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    return meta


def find_snapshot(dataset_id: str, target_dir: Path | None = None) -> Path | None:
    """The .npz for an id, or None. The id alone identifies the file."""
    directory = datasets_dir() if target_dir is None else target_dir
    if not directory.is_dir():
        return None
    matches = sorted(directory.glob(f"*_{dataset_id}.npz"))
    return matches[0] if matches else None


def load_meta(dataset_id: str, target_dir: Path | None = None) -> SnapshotMeta | None:
    npz = find_snapshot(dataset_id, target_dir)
    if npz is None:
        return None
    sidecar = npz.with_suffix(".json")
    if not sidecar.is_file():
        return None
    return SnapshotMeta(**json.loads(sidecar.read_text(encoding="utf-8")))


def load_snapshot(dataset_id: str, target_dir: Path | None = None):
    """Rebuild a BarData from a pinned snapshot.

    Re-derives the hash from the loaded arrays and refuses a mismatch:
    a snapshot whose contents no longer match its own id is exactly the
    substitution this store exists to detect.
    """
    from engine.research.data.mt5_source import BarData, Timeframe

    npz = find_snapshot(dataset_id, target_dir)
    if npz is None:
        raise DataError(
            f"no dataset snapshot {dataset_id!r} in "
            f"{datasets_dir() if target_dir is None else target_dir}"
        )
    meta = load_meta(dataset_id, target_dir)
    if meta is None:
        raise DataError(f"snapshot {dataset_id!r} has no sidecar metadata")

    with np.load(npz) as data:
        arrays = {k: data[k] for k in data.files}

    recomputed = dataset_hash(
        symbol=meta.symbol,
        timeframe=meta.timeframe,
        time=arrays["time"],
        open_=arrays["open"],
        high=arrays["high"],
        low=arrays["low"],
        close=arrays["close"],
        spread=arrays["spread"],
    )
    if recomputed != dataset_id:
        raise DataError(
            f"snapshot {dataset_id!r} does not hash to its own id "
            f"(got {recomputed!r}); the file has been modified or replaced"
        )

    return BarData(
        time=arrays["time"],
        open=arrays["open"],
        high=arrays["high"],
        low=arrays["low"],
        close=arrays["close"],
        tick_volume=arrays["tick_volume"],
        spread=arrays["spread"],
        real_volume=arrays["real_volume"],
        symbol=meta.symbol,
        timeframe=Timeframe[meta.timeframe] if hasattr(Timeframe, meta.timeframe)
        else getattr(Timeframe, str(meta.timeframe), None),
        time_basis=meta.time_basis,
    )


def available_snapshots(target_dir: Path | None = None) -> tuple[SnapshotMeta, ...]:
    directory = datasets_dir() if target_dir is None else target_dir
    if not directory.is_dir():
        return ()
    out = []
    for sidecar in sorted(directory.glob("*.json")):
        try:
            out.append(SnapshotMeta(**json.loads(sidecar.read_text(encoding="utf-8"))))
        except Exception:
            continue
    return tuple(out)


__all__ = [
    "DATASETS_DIRNAME",
    "FORMAT_TAG",
    "SnapshotMeta",
    "available_snapshots",
    "dataset_hash",
    "datasets_dir",
    "find_snapshot",
    "load_meta",
    "load_snapshot",
    "save_snapshot",
]
