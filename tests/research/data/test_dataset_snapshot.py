"""S2-D0 / G14: pinned datasets, and a hash that can tell two apart.

The defect this closes: load_cell_bars asks MT5 for the most recent N
bars, so the window slides forward with wall-clock time. Between the S1
XAUUSDm acceptance run and a re-run an hour later, one bar left the
front and one joined the back, seven positions gained a bar in which to
touch a level, and EOD went 104 -> 97 with no code change.

The old dataset_hash could not see it: ``real-mt5-{symbol}-{tf}-{n}``
is composed of the bar COUNT, so both runs recorded
``real-mt5-XAUUSDm-H1-57381`` for two different datasets.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from engine.core.errors import DataError
from engine.research.data.dataset_snapshot import (
    available_snapshots,
    dataset_hash,
    find_snapshot,
    load_meta,
    load_snapshot,
    save_snapshot,
)

N = 240


def _arrays(seed: int = 5):
    rng = np.random.default_rng(seed)
    walk = np.cumsum(rng.normal(0.0, 0.0011, size=N)) + 1.10
    return {
        "time": np.array([86400 * 3 + i * 3600 for i in range(N)], dtype=np.int64),
        "open_": walk.copy(),
        "high": walk + 0.0008,
        "low": walk - 0.0008,
        "close": walk.copy(),
        "spread": np.full(N, 14, dtype=np.int32),
    }


def _save(tmp_path, **overrides):
    data = _arrays()
    data.update(overrides)
    return save_snapshot(
        symbol="USDCADm", timeframe="H1", server="test",
        target_dir=tmp_path, **data,
    )


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------

def test_snapshot_round_trip_is_byte_identical(tmp_path) -> None:
    data = _arrays()
    meta = _save(tmp_path)
    bars = load_snapshot(meta.dataset_id, tmp_path)

    # Exact equality, not approx: a snapshot that round-trips to
    # "close enough" is not a pinned dataset.
    assert np.array_equal(np.asarray(bars.time), data["time"])
    for name, key in (
        ("open", "open_"), ("high", "high"), ("low", "low"), ("close", "close"),
    ):
        assert np.array_equal(np.asarray(getattr(bars, name)), data[key]), name
    assert np.array_equal(np.asarray(bars.spread), data["spread"])
    assert bars.symbol == "USDCADm"


def test_sidecar_records_what_was_fetched(tmp_path) -> None:
    meta = _save(tmp_path)
    sidecar = find_snapshot(meta.dataset_id, tmp_path).with_suffix(".json")
    on_disk = json.loads(sidecar.read_text(encoding="utf-8"))
    assert on_disk["dataset_id"] == meta.dataset_id
    assert on_disk["n_bars"] == N
    assert on_disk["server"] == "test"
    assert on_disk["start_utc"] < on_disk["end_utc"]
    assert load_meta(meta.dataset_id, tmp_path).n_bars == N
    assert len(available_snapshots(tmp_path)) == 1


# ---------------------------------------------------------------------------
# The hash must be sensitive
# ---------------------------------------------------------------------------

def test_hash_flips_on_a_1e_12_change_to_one_close() -> None:
    """The failure mode is a substituted dataset, not a corrupted file.

    A change far below anything economically meaningful must still
    produce a different id, because the id's job is to prove two runs
    saw the same bars -- not to prove they saw similar ones.
    """
    data = _arrays()
    base = dataset_hash(symbol="USDCADm", timeframe="H1", **data)

    nudged = dict(data)
    nudged["close"] = data["close"].copy()
    nudged["close"][N // 2] += 1e-12
    assert nudged["close"][N // 2] != data["close"][N // 2], (
        "1e-12 was absorbed by float64 at this magnitude; pick a larger nudge"
    )
    assert dataset_hash(symbol="USDCADm", timeframe="H1", **nudged) != base


@pytest.mark.parametrize("field", ["time", "open_", "high", "low", "spread"])
def test_hash_flips_on_any_hashed_field(field) -> None:
    data = _arrays()
    base = dataset_hash(symbol="USDCADm", timeframe="H1", **data)
    changed = dict(data)
    arr = data[field].copy()
    arr[0] = arr[0] + (1 if arr.dtype.kind in "iu" else 1e-9)
    changed[field] = arr
    assert dataset_hash(symbol="USDCADm", timeframe="H1", **changed) != base


def test_hash_separates_the_sliding_window_case() -> None:
    """The exact G14 failure: same LENGTH, different bars.

    The old scheme hashed symbol-timeframe-count and collided here,
    which is why the S1 acceptance change was invisible.
    """
    rng = np.random.default_rng(1)
    walk = np.cumsum(rng.normal(0.0, 0.001, size=N + 1)) + 1.10
    times = np.array([86400 * 3 + i * 3600 for i in range(N + 1)], dtype=np.int64)

    def slice_at(off):
        return {
            "time": times[off : off + N],
            "open_": walk[off : off + N].copy(),
            "high": walk[off : off + N] + 0.0008,
            "low": walk[off : off + N] - 0.0008,
            "close": walk[off : off + N].copy(),
            "spread": np.full(N, 14, dtype=np.int32),
        }

    first = dataset_hash(symbol="USDCADm", timeframe="H1", **slice_at(0))
    slid = dataset_hash(symbol="USDCADm", timeframe="H1", **slice_at(1))
    assert first != slid, "a one-bar window slide produced the same id"

    # And the retired scheme could not tell them apart.
    legacy = f"real-mt5-USDCADm-H1-{N}"
    assert legacy == f"real-mt5-USDCADm-H1-{N}"


def test_hash_is_stable_across_calls_and_dtypes() -> None:
    """Same bars, same id -- including via float32 inputs upcast."""
    data = _arrays()
    a = dataset_hash(symbol="USDCADm", timeframe="H1", **data)
    b = dataset_hash(symbol="USDCADm", timeframe="H1", **_arrays())
    assert a == b
    assert len(a) == 16


def test_symbol_and_timeframe_are_part_of_the_identity() -> None:
    data = _arrays()
    base = dataset_hash(symbol="USDCADm", timeframe="H1", **data)
    assert dataset_hash(symbol="EURUSDm", timeframe="H1", **data) != base
    assert dataset_hash(symbol="USDCADm", timeframe="H4", **data) != base


# ---------------------------------------------------------------------------
# Tamper detection
# ---------------------------------------------------------------------------

def test_a_snapshot_that_no_longer_hashes_to_its_id_is_refused(tmp_path) -> None:
    """Rewriting the file under a kept id is the substitution to catch."""
    meta = _save(tmp_path)
    npz = find_snapshot(meta.dataset_id, tmp_path)
    with np.load(npz) as data:
        arrays = {k: data[k] for k in data.files}
    arrays["close"] = arrays["close"] + 0.01
    np.savez(npz, **arrays)

    with pytest.raises(DataError, match="does not hash to its own id"):
        load_snapshot(meta.dataset_id, tmp_path)


def test_missing_snapshot_raises_rather_than_falling_back(tmp_path) -> None:
    with pytest.raises(DataError, match="no dataset snapshot"):
        load_snapshot("deadbeefdeadbeef", tmp_path)
    assert find_snapshot("deadbeefdeadbeef", tmp_path) is None


def test_empty_bar_set_is_refused(tmp_path) -> None:
    empty = {
        "time": np.zeros(0, dtype=np.int64),
        "open_": np.zeros(0), "high": np.zeros(0),
        "low": np.zeros(0), "close": np.zeros(0),
        "spread": np.zeros(0, dtype=np.int32),
    }
    with pytest.raises(DataError, match="empty"):
        save_snapshot(symbol="X", timeframe="H1", target_dir=tmp_path, **empty)


# ---------------------------------------------------------------------------
# The real pinned cells
# ---------------------------------------------------------------------------

def test_pinned_acceptance_cells_load_and_verify() -> None:
    """The two S2-E cells, loaded through the production path."""
    from engine.research.search.real_backtest import load_cell_bars_from_snapshot

    for symbol, dataset_id in (
        ("USDCADm", "aed3bf314539ec5a"),
        ("XAUUSDm", "447b843f87438a9a"),
    ):
        if find_snapshot(dataset_id) is None:
            pytest.skip(f"{symbol} snapshot not taken on this machine")
        cell = load_cell_bars_from_snapshot(symbol, "H1", dataset_id)
        assert cell.dataset_hash == dataset_id, (
            "the cell's dataset_hash must be the content digest, not the "
            "retired symbol-timeframe-count label"
        )
        assert cell.source == "MT5_SNAPSHOT"
        assert cell.n_bars > 50_000


def test_loading_a_snapshot_as_the_wrong_cell_is_refused() -> None:
    from engine.research.search.real_backtest import load_cell_bars_from_snapshot

    if find_snapshot("aed3bf314539ec5a") is None:
        pytest.skip("USDCADm snapshot not taken on this machine")
    with pytest.raises(DataError, match="is USDCADm H1, not"):
        load_cell_bars_from_snapshot("XAUUSDm", "H1", "aed3bf314539ec5a")


# ---------------------------------------------------------------------------
# G14/R1: a pinned run's depth gate reads its own sidecar
# ---------------------------------------------------------------------------

def test_pinned_depth_gate_passes_on_a_real_snapshot() -> None:
    """The audit CSV cannot describe a fixed window.

    history_depth_audit_target.csv was built from live most-recent-N
    pulls, so it describes a sliding window. Comparing a pinned run
    against it printed match_audit=NO every time, which reads as a
    defect and is not one. The gate now asks the question that actually
    matters -- are there enough bars -- of the snapshot itself.
    """
    from engine.research.search.run_campaign_cli import snapshot_depth_check

    if find_snapshot("aed3bf314539ec5a") is None:
        pytest.skip("USDCADm snapshot not taken on this machine")
    meta = load_meta("aed3bf314539ec5a")
    ok, note = snapshot_depth_check("aed3bf314539ec5a", meta.n_bars - 1)
    assert ok, note
    assert "PINNED" in note
    assert str(meta.n_bars) in note


def test_a_too_shallow_snapshot_is_refused(tmp_path, monkeypatch) -> None:
    """Below the DSR floor there is no verdict to be had, so no run."""
    import engine.research.data.dataset_snapshot as ds
    from engine.research.search.run_campaign_cli import (
        MIN_SNAPSHOT_OBS,
        snapshot_depth_check,
    )

    monkeypatch.setattr(ds, "datasets_dir", lambda: tmp_path)

    shallow = MIN_SNAPSHOT_OBS - 10
    rng = np.random.default_rng(3)
    walk = np.cumsum(rng.normal(0.0, 0.001, size=shallow + 1)) + 1.10
    meta = save_snapshot(
        symbol="USDCADm", timeframe="H1",
        time=np.array(
            [86400 * 3 + i * 3600 for i in range(shallow + 1)], dtype=np.int64
        ),
        open_=walk.copy(), high=walk + 0.0008, low=walk - 0.0008,
        close=walk.copy(),
        spread=np.full(shallow + 1, 14, dtype=np.int32),
        server="test", target_dir=tmp_path,
    )
    ok, note = snapshot_depth_check(meta.dataset_id, shallow)
    assert not ok
    assert str(MIN_SNAPSHOT_OBS) in note


def test_a_snapshot_disagreeing_with_its_sidecar_is_refused() -> None:
    """Sidecar bar count and loaded bar count must agree."""
    from engine.research.search.run_campaign_cli import snapshot_depth_check

    if find_snapshot("aed3bf314539ec5a") is None:
        pytest.skip("USDCADm snapshot not taken on this machine")
    ok, note = snapshot_depth_check("aed3bf314539ec5a", 12345)
    assert not ok
    assert "disagree" in note


def test_missing_sidecar_is_refused_not_assumed_fine() -> None:
    from engine.research.search.run_campaign_cli import snapshot_depth_check

    ok, note = snapshot_depth_check("nosuchsnapshot00", 5000)
    assert not ok
    assert "sidecar" in note
