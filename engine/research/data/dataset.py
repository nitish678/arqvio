"""Dataset identity.

A Dataset is a frozen, content-addressed record of:

    symbol + timeframe + date range + source broker + the byte-hash of the
    actual bar data + config hash + git SHA + seed.

The byte-hash is what makes a backtest reproducible: any change to a bar
file, a config value, or the engine version produces a new dataset_id
and breaks the link to a previous result. This is the keystone of the
no-fake-data invariant: every result carries a dataset_hash, and
mismatched hashes between campaign and backtest mean the backtest is
fabricated.

Per the S2 gate: reloading the same source data with the same parameters
must produce an IDENTICAL hash.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from typing import Any

import numpy as np

from engine.core.errors import DataError
from engine.research.data.mt5_source import BarData, Timeframe


@dataclass(frozen=True)
class DatasetId:
    """Content-addressed dataset identity."""

    dataset_id: str        # sha256 of (symbol, tf, range, byte-hash, config-hash, git-sha, seed)
    symbol: str
    timeframe: Timeframe
    date_from_utc: str     # ISO-8601
    date_to_utc: str       # ISO-8601
    source: str            # e.g. "MT5:Exness:C:\\mt5\\1"
    byte_hash: str         # sha256 of the bar bytes
    config_hash: str       # sha256 of any config that produced the data
    git_sha: str           # HEAD of the repo when the dataset was made
    seed: int
    n_bars: int
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "symbol": self.symbol,
            "timeframe": self.timeframe.value,
            "date_from_utc": self.date_from_utc,
            "date_to_utc": self.date_to_utc,
            "source": self.source,
            "byte_hash": self.byte_hash,
            "config_hash": self.config_hash,
            "git_sha": self.git_sha,
            "seed": self.seed,
            "n_bars": self.n_bars,
            "notes": self.notes,
        }


def _hash_bars(bars: BarData) -> str:
    """sha256 over a canonical byte representation of the bars."""
    h = hashlib.sha256()
    # Time first (deterministic ordering guaranteed because fetch_bars
    # returns ascending time).
    h.update(bars.time.astype("<i8").tobytes())
    for arr in (bars.open, bars.high, bars.low, bars.close,
                bars.tick_volume, bars.spread, bars.real_volume):
        h.update(arr.astype("<f8").tobytes())
    return h.hexdigest()


def _git_sha() -> str:
    """Current HEAD sha, or 'unknown' if not in a git repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()[:40]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def make_dataset_id(
    bars: BarData,
    *,
    config_hash: str = "no-config",
    seed: int = 0,
    notes: str = "",
) -> DatasetId:
    """Compute the dataset identity for a given bar pull.

    Per S2 gate: calling make_dataset_id twice on the SAME bar object
    (or on bars that hash to the same bytes) must produce the same
    dataset_id. This is the reproducibility test.
    """
    if bars.n_bars == 0:
        raise DataError("cannot fingerprint an empty bar pull")
    byte_hash = _hash_bars(bars)
    git_sha = _git_sha()
    payload = {
        "symbol": bars.symbol,
        "timeframe": bars.timeframe.value,
        "source": bars.source,
        "byte_hash": byte_hash,
        "config_hash": config_hash,
        "git_sha": git_sha,
        "seed": int(seed),
        "n_bars": bars.n_bars,
    }
    canon = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    dataset_id = hashlib.sha256(canon.encode("utf-8")).hexdigest()
    return DatasetId(
        dataset_id=dataset_id,
        symbol=bars.symbol,
        timeframe=bars.timeframe,
        date_from_utc=(
            np.datetime64(int(bars.time[0]), "s").astype("M8[s]")
            .astype("datetime64[s]")
            .item()
            .isoformat() + "Z"
        ),
        date_to_utc=(
            np.datetime64(int(bars.time[-1]), "s").astype("M8[s]")
            .astype("datetime64[s]")
            .item()
            .isoformat() + "Z"
        ),
        source=bars.source,
        byte_hash=byte_hash,
        config_hash=config_hash,
        git_sha=git_sha,
        seed=seed,
        n_bars=bars.n_bars,
        notes=notes,
    )
