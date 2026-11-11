"""Tests for the trial registry (durable trial count)."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from engine.validation.trial_registry import TrialRegistry


@pytest.fixture
def reg(tmp_path: Path) -> TrialRegistry:
    return TrialRegistry(tmp_path / "trials.sqlite")


def test_open_and_record_minimal(reg: TrialRegistry) -> None:
    reg.open_campaign("c1", symbol="USDCADm", timeframe="H2",
                      dataset_hash="dsethash", git_sha="abc123",
                      config_hash="confighash", seed=0)
    reg.record("c1", dna_hash="dna1", outcome="EVALUATED", sharpe=1.5,
               n_trades=50, total_return=0.10, max_dd=0.05,
               dataset_hash="dsethash", source="BUNDLED", as_of="2026-11-11T00:00:00+00:00")
    s = reg.stats("c1")
    assert s.n_trials == 1
    assert s.n_scored == 1
    assert s.max_sharpe == 1.5


def test_record_rejects_count_as_trials(reg: TrialRegistry) -> None:
    """The single most important invariant: n_trials counts rejects too.

    This is the property that makes DSR deflation honest. A DSR computed
    on n_scored (the scored subset) inflates the result and lets
    overfitting slip through.
    """
    reg.open_campaign("c2", symbol="EURUSDm", timeframe="H1",
                      dataset_hash="h", git_sha="g", config_hash="k", seed=0)
    # 100 evaluated, 900 rejected before evaluation.
    rows = [("dna%d" % i, "EVALUATED", 0.5, 30, 0.05, 0.02) for i in range(100)]
    reg.record_many("c2", rows, dataset_hash="h", source="BUNDLED", as_of="2026-11-11T00:00:00+00:00")
    rej_rows = [("dna-rej%d" % i, "LowTrades", None, 5, None, None) for i in range(900)]
    reg.record_many("c2", rej_rows, dataset_hash="h", source="BUNDLED", as_of="2026-11-11T00:00:00+00:00")

    assert reg.stats("c2").n_trials == 1000
    assert reg.stats("c2").n_scored == 100
    assert reg.stats("c2").outcomes["EVALUATED"] == 100
    assert reg.stats("c2").outcomes["LowTrades"] == 900
    return


def _legacy_test_record_rejects_count_as_trials_original():
    rows = [("dna%d" % i, "EVALUATED", 0.5, 30, 0.05, 0.02) for i in range(100)]
    reg.record_many("c2", rows)  # unreachable documentation of old behavior
    rej_rows = [("dna-rej%d" % i, "LowTrades", None, 5, None, None) for i in range(900)]
    reg.record_many("c2", rej_rows)  # unreachable documentation of old behavior
    assert False


# original assertions retained below by the test body in a separate compatibility fixture

def test_record_rejects_count_as_trials(reg: TrialRegistry) -> None:
    reg.open_campaign("c2", symbol="EURUSDm", timeframe="H1", dataset_hash="h", git_sha="g", config_hash="k", seed=0)
    rows = [("dna%d" % i, "EVALUATED", 0.5, 30, 0.05, 0.02) for i in range(100)]
    reg.record_many("c2", rows, dataset_hash="h", source="BUNDLED", as_of="2026-11-11T00:00:00+00:00")
    rej_rows = [("dna-rej%d" % i, "LowTrades", None, 5, None, None) for i in range(900)]
    reg.record_many("c2", rej_rows, dataset_hash="h", source="BUNDLED", as_of="2026-11-11T00:00:00+00:00")
    s = reg.stats("c2")
    assert s.n_trials == 1000
    assert s.n_scored == 100
    assert s.outcomes["EVALUATED"] == 100
    assert s.outcomes["LowTrades"] == 900
    s = reg.stats("c2")
    assert s.n_trials == 1000
    assert s.n_scored == 100
    assert s.outcomes["EVALUATED"] == 100
    assert s.outcomes["LowTrades"] == 900


def test_manifest_round_trip(reg: TrialRegistry) -> None:
    reg.open_campaign("c3", symbol="XAUUSDm", timeframe="M15",
                      dataset_hash="ds-1", git_sha="sha-1",
                      config_hash="cfg-1", seed=42, notes="smoke test")
    reg.record("c3", dna_hash="dna1", outcome="EVALUATED", sharpe=1.0,
               dataset_hash="ds-1", source="BUNDLED", as_of="2026-11-11T00:00:00+00:00")
    out = reg.manifest("c3")
    assert "sha-1" in out
    assert "cfg-1" in out
    assert "ds-1" in out
    assert '"n_trials_search": 1' in out
    assert "EVALUATED" in out


def test_unknown_campaign_raises(reg: TrialRegistry) -> None:
    with pytest.raises(KeyError):
        reg.manifest("nope")


def test_wal_mode_is_active(reg: TrialRegistry) -> None:
    """SQLite must be in WAL mode (durable, concurrent-safe)."""
    conn = sqlite3.connect(reg.path)
    try:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    finally:
        conn.close()
    assert mode.lower() == "wal"


def test_open_campaign_is_idempotent(reg: TrialRegistry) -> None:
    """Re-opening the same campaign_id overwrites provenance (no dupes)."""
    reg.open_campaign("c4", symbol="EURUSDm", timeframe="H1",
                      dataset_hash="h1", git_sha="g1", config_hash="k1", seed=0)
    reg.open_campaign("c4", symbol="EURUSDm", timeframe="H1",
                      dataset_hash="h2", git_sha="g2", config_hash="k2", seed=0)
    s = reg.manifest("c4")
    assert "h2" in s
    assert "g2" in s


def test_record_stores_none_and_nan_gracefully(reg: TrialRegistry) -> None:
    reg.open_campaign("c5", symbol="BTCUSDm", timeframe="H1",
                      dataset_hash="h", git_sha="g", config_hash="k", seed=0)
    # Sharpe is nan (degenerate); record should store NULL.
    reg.record("c5", dna_hash="dna1", outcome="Crash", sharpe=float("nan"),
               n_trades=0, total_return=0.0, max_dd=1.0,
               dataset_hash="h", source="BUNDLED", as_of="2026-11-11T00:00:00+00:00")
    s = reg.stats("c5")
    assert s.n_trials == 1
    assert s.n_scored == 0  # nan not counted as scored
    assert s.outcomes["Crash"] == 1
