"""C0: pinned-dataset hash invariant — survives working-set edits.

Asserts both S2-E acceptance snapshots still hash to their recorded SHA-256
identifies through every C0 change. If the dataset content were touched, the
hash would change — this test catches that directly rather than trusting a
skip message.

Uses the same loaders the existing snapshot tests use (find_snapshot / load_meta)
so it exercises the production code path, not a re-implementation.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from engine.research.data.dataset_snapshot import (
    FORMAT_TAG,
    dataset_hash,
    find_snapshot,
    load_meta,
)

PINNED = [
    ("USDCADm", "aed3bf314539ec5a", "aed3bf314539ec5abcb0bc480219189ea71e42b52eec106392c8cc0f095656a4"),
    ("XAUUSDm", "447b843f87438a9a", "447b843f87438a9a5c12ba04dfbcf2a6dc9b57380c90d936dd61acaca31329e0"),
]


@pytest.mark.parametrize("symbol, dataset_id, expected_full_hash", PINNED)
def test_pinned_snapshot_hash_survives_working_set_edits(
    symbol: str, dataset_id: str, expected_full_hash: str,
) -> None:
    """The pinned snapshot content must hash to the exact recorded value.

    Regardless of current working-set edits, if the snapshot file on disk
    matches its original content, full SHA-256 must equal the pinned value.
    """
    path = find_snapshot(dataset_id)
    if path is None:
        pytest.skip(f"{symbol} snapshot {dataset_id} not present on this machine")

    meta = load_meta(dataset_id)
    assert meta is not None
    assert meta.symbol == symbol
    assert meta.timeframe == "H1"

    # Reconstruct the content hash via the canonical dataset_hash function,
    # which is the same path real_backtest.load_cell_bars_from_snapshot uses.
    data = np.load(path)
    recomputed_id = dataset_hash(
        symbol=meta.symbol,
        timeframe=meta.timeframe,
        time=np.asarray(data["time"], dtype=np.int64),
        open_=np.asarray(data["open"], dtype=np.float64),
        high=np.asarray(data["high"], dtype=np.float64),
        low=np.asarray(data["low"], dtype=np.float64),
        close=np.asarray(data["close"], dtype=np.float64),
        spread=np.asarray(data["spread"], dtype=np.int32),
    )
    assert recomputed_id == dataset_id[:16], (
        f"dataset_hash (id) diverged: expected {dataset_id[:16]}, got {recomputed_id}"
    )

    # And the full 64-char digest must match the pinned value exactly.
    digest = hashlib.sha256()
    digest.update(FORMAT_TAG)
    digest.update(meta.symbol.encode("utf-8") + b"\x00")
    digest.update(meta.timeframe.encode("utf-8") + b"\x00")
    digest.update(np.int64(data["time"].size).astype("<i8").tobytes())
    digest.update(np.asarray(data["time"], dtype=np.int64).astype("<i8").tobytes())
    for key in ("open", "high", "low", "close"):
        digest.update(np.asarray(data[key], dtype=np.float64).astype("<f8").tobytes())
    digest.update(np.asarray(data["spread"], dtype=np.int32).astype("<i4").tobytes())
    full_hash = digest.hexdigest()
    assert full_hash == expected_full_hash, (
        f"{symbol} ({dataset_id}) full SHA-256 diverged from the pinned value. "
        f"Data was touched: expected {expected_full_hash}, got {full_hash}."
    )
