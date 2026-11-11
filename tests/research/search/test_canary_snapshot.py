"""The pre-commit canary reads a pinned snapshot that ships in the repository: present, the right bars, its own hash."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from engine.core.errors import DataError
from engine.research.search.real_backtest import load_cell_bars_from_snapshot
from scripts.canary_fast import CANARY_DATA, DATASET_ID, EXPECTED_BARS, SYMBOL, TIMEFRAME


def test_the_canary_snapshot_ships_and_hashes_to_its_id() -> None:
    cell = load_cell_bars_from_snapshot(SYMBOL, TIMEFRAME, DATASET_ID, target_dir=CANARY_DATA)
    assert (cell.n_bars, cell.source, cell.dataset_hash) == (EXPECTED_BARS, "MT5_SNAPSHOT", DATASET_ID)
    meta = json.loads((CANARY_DATA / f"{SYMBOL}_{TIMEFRAME}_{DATASET_ID}.json").read_text(encoding="utf-8"))
    assert meta["time_basis"] == "utc" and meta["n_bars"] == EXPECTED_BARS
    assert np.all(np.diff(np.asarray(cell.bars.time)) > 0)


def test_a_modified_canary_snapshot_is_refused(tmp_path: Path) -> None:
    for p in CANARY_DATA.glob(f"*_{DATASET_ID}.*"):
        shutil.copy2(p, tmp_path / p.name)
    npz = tmp_path / f"{SYMBOL}_{TIMEFRAME}_{DATASET_ID}.npz"
    with np.load(npz) as data:
        arrays = {k: data[k].copy() for k in data.files}
    arrays["close"][100] += 1e-5
    np.savez(npz, **arrays)
    with pytest.raises(DataError, match="does not hash to its own id"):
        load_cell_bars_from_snapshot(SYMBOL, TIMEFRAME, DATASET_ID, target_dir=tmp_path)
