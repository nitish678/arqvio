"""v4 held-out split: breeding gets the first 70% of a pulled window as its own pinned snapshot; the full window stays."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from engine.research.cells import add_market
from engine.research.data.dataset_snapshot import dataset_hash, find_snapshot, load_snapshot, save_snapshot


def _pull(n: int) -> dict:
    t = np.arange(n, dtype=np.int64) * 3600 + 1_700_000_000
    c = 1.1 + np.cumsum(np.full(n, 1e-5))
    arrays = {"time": t, "open_": c, "high": c + 1e-4, "low": c - 1e-4, "close": c, "spread": np.full(n, 12, dtype=np.int32)}
    h16 = dataset_hash(symbol="EURUSD", timeframe="H1", **arrays)
    save_snapshot(symbol="EURUSD", timeframe="H1", **arrays, server="test", time_basis="utc")
    return {"cell_id": "eurusd_h1", "symbol": "EURUSD", "timeframe": "H1", "dataset_hash16": h16,
            "window": {"first_bar": "x", "last_bar": "y", "bars": n}}


def test_the_cell_breeds_on_the_first_70_percent_and_the_full_window_is_kept(isolated_arqvio_home: Path) -> None:
    rec = _pull(4000)
    out = add_market.hold_out(rec)
    ho = out["held_out"]
    assert ho["state"] == "KEPT" and ho["split_bar"] == 2800 and ho["bars"] == 1200
    assert ho["dataset_hash16"] == rec["dataset_hash16"] and out["dataset_hash16"] != rec["dataset_hash16"]
    train = load_snapshot(out["dataset_hash16"])
    full = load_snapshot(rec["dataset_hash16"])
    assert len(train.time) == 2800 and np.array_equal(train.close, full.close[:2800])
    assert out["window"]["bars"] == 2800
    assert (add_market.cell_dir("eurusd_h1") / "add_market.json").is_file()


def test_a_short_window_is_not_split_and_says_why(isolated_arqvio_home: Path) -> None:
    rec = _pull(2500)
    out = add_market.hold_out(rec)
    assert out["held_out"]["state"] == "NONE" and "too few" in out["held_out"]["reason"]
    assert out["dataset_hash16"] == rec["dataset_hash16"] and find_snapshot(rec["dataset_hash16"]) is not None
