"""Tests for the data-tier / symbol / spread-calibration / history-audit layer."""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from engine.data.history_audit import (
    HistoryCell,
    audit_live,
    audit_offline,
    best_cells_by_history,
)
from engine.data.spread_calibration import (
    SpreadMatrix,
    calibration_failure_message,
    is_calibration_failure,
    load,
    persist,
)
from engine.data.symbols import (
    build_record,
    by_category,
    categorise,
    detect_suffix,
)
from engine.data.tiers import (
    TIER_1_TIMEFRAMES,
    TIER_2_TIMEFRAMES,
    TIER_3_TIMEFRAMES,
    tier3_path,
)
from engine.data.timeframes import (
    DEFAULT_TIMEFRAMES,
    PERIODS_PER_YEAR,
)

# --- tiers -----------------------------------------------------------------

def test_tier_classification_is_stable() -> None:
    assert "D1" in TIER_1_TIMEFRAMES
    assert "H4" in TIER_1_TIMEFRAMES
    assert "H1" in TIER_2_TIMEFRAMES
    assert "H2" in TIER_2_TIMEFRAMES
    assert "M5" in TIER_3_TIMEFRAMES
    assert "M15" in TIER_3_TIMEFRAMES
    assert "D1" in TIER_3_TIMEFRAMES  # tier3 can also host D1 via incremental


def test_tier3_path_layout(tmp_path: Path) -> None:
    p = tier3_path("XAUUSDm", "H2", 2026)
    assert p.as_posix().endswith("/tier3/XAUUSDm/H2/2026.parquet")
    # The directory must not be created on import - it is created on write.
    assert not p.exists()


# --- timeframes ------------------------------------------------------------

def test_periods_per_year_table_has_documented_keys() -> None:
    for tf in DEFAULT_TIMEFRAMES:
        assert tf in PERIODS_PER_YEAR
        assert PERIODS_PER_YEAR[tf] > 0


# --- symbols ---------------------------------------------------------------

def test_detect_suffix_exness_micro() -> None:
    assert detect_suffix("XAUUSDm") == ("XAUUSD", "m")
    assert detect_suffix("EURUSDm") == ("EURUSD", "m")
    assert detect_suffix("BTCUSDm") == ("BTCUSD", "m")


def test_detect_suffix_no_suffix() -> None:
    assert detect_suffix("EURUSD") == ("EURUSD", "")
    assert detect_suffix("XAUUSD") == ("XAUUSD", "")


def test_categorise_major_minor_exotic() -> None:
    assert categorise("EURUSD", "Forex") == "FOREX_MAJOR"
    assert categorise("EURJPY", "Forex") == "FOREX_MINOR"
    assert categorise("USDSGD", "Forex") == "FOREX_EXOTIC"
    assert categorise("XAUUSD", "Metals") == "METALS"
    assert categorise("BTCUSD", "Crypto") == "CRYPTO"
    assert categorise("USOILm", "Commodities") == "ENERGY"
    assert categorise("US500m", "Indices") == "INDICES"
    assert categorise("FOO", "Other") == "OTHER"


def test_build_record_round_trip() -> None:
    rec = build_record(
        "XAUUSDm", path="Demo\\Metals\\XAUUSDm",
        visible=True, digits=2, point=0.01, spread_points=20,
    )
    assert rec.name == "XAUUSDm"
    assert rec.base == "XAUUSD"
    assert rec.suffix == "m"
    assert rec.category == "METALS"
    d = rec.to_dict()
    assert d["name"] == "XAUUSDm"
    assert d["base"] == "XAUUSD"
    assert d["category"] == "METALS"


def test_by_category_groups_records() -> None:
    recs = [
        build_record("EURUSDm", "Demo\\Forex\\EURUSDm", True, 5, 0.00001, 10),
        build_record("XAUUSDm", "Demo\\Metals\\XAUUSDm", True, 2, 0.01, 20),
    ]
    grouped = by_category(recs)
    assert "FOREX_MAJOR" in grouped
    assert "METALS" in grouped
    assert len(grouped["FOREX_MAJOR"]) == 1
    assert len(grouped["METALS"]) == 1


# --- spread calibration ----------------------------------------------------

def test_flat_24h_row_is_calibration_failure() -> None:
    flat = SpreadMatrix(symbol="XAUUSDm", buckets=tuple([1.5] * 24), mean_buckets=tuple([1.5] * 24), sample_count=100, source="MT5")
    assert flat.is_flat
    assert is_calibration_failure(flat)
    assert "SPREAD_UNCALIBRATED" in calibration_failure_message("XAUUSDm")


def test_real_24h_row_is_not_a_failure() -> None:
    real = SpreadMatrix(
        symbol="XAUUSDm",
        buckets=tuple(np.linspace(1.2, 3.5, 24)),
        mean_buckets=tuple(np.linspace(1.2, 3.5, 24)),
        sample_count=10_000,
        source="MT5",
    )
    assert not real.is_flat
    assert not is_calibration_failure(real)


def test_persist_and_load_round_trip(tmp_path: Path) -> None:
    real = SpreadMatrix(
        symbol="XAUUSDm",
        buckets=tuple(np.linspace(1.2, 3.5, 24)),
        mean_buckets=tuple(np.linspace(1.2, 3.5, 24)),
        sample_count=10_000,
        source="MT5",
    )
    persist(real, tmp_path)
    loaded = load("XAUUSDm", tmp_path)
    assert loaded is not None
    assert loaded.buckets == real.buckets
    assert loaded.mean_buckets == real.mean_buckets
    assert loaded.symbol == real.symbol


# --- history audit ---------------------------------------------------------

class _FakeMT5:
    """Records the most-recent request and returns canned bars."""

    def __init__(self, bars: int = 100) -> None:
        self.bars = bars
        self.connected = True
        self.init_calls = 0
        self.shutdown_calls = 0

    TIMEFRAME_D1 = 16408
    TIMEFRAME_H4 = 16388
    TIMEFRAME_H1 = 16377
    TIMEFRAME_H2 = 16378
    TIMEFRAME_M15 = 16375

    def initialize(self, path: str) -> bool:
        self.init_calls += 1
        return True

    def shutdown(self) -> None:
        self.shutdown_calls += 1

    def last_error(self) -> tuple[int, str]:
        return (1, "Success")

    def copy_rates_from_pos(self, symbol: str, tf: int, start: int, count: int) -> np.ndarray:
        n = min(int(self.bars), int(count))
        if n <= 0:
            return np.zeros(0, dtype=self._dtype())
        # Spread bars across the last 5 years (synthetic).
        now = int(datetime.now(tz=UTC).timestamp())
        step = 86400  # daily step; we don't need per-TF fidelity for the test
        first = now - n * step
        times = np.arange(first, first + n * step, step, dtype="<i8")
        opens = 100.0 + np.linspace(0, 5, n)
        highs = opens + 0.5
        lows = opens - 0.5
        closes = opens + 0.1
        tv = np.full(n, 1000, dtype="<i8")
        spread = np.full(n, 1, dtype="<f8")
        rv = np.zeros(n, dtype="<i8")
        return np.zeros(
            n,
            dtype=[("time", "<i8"), ("open", "<f8"), ("high", "<f8"),
                   ("low", "<f8"), ("close", "<f8"), ("tick_volume", "<i8"),
                   ("spread", "<f8"), ("real_volume", "<i8")],
        )

    @staticmethod
    def _dtype():
        return np.dtype([
            ("time", "<i8"), ("open", "<f8"), ("high", "<f8"),
            ("low", "<f8"), ("close", "<f8"), ("tick_volume", "<i8"),
            ("spread", "<f8"), ("real_volume", "<i8"),
        ])


def test_history_audit_records_all_cells(tmp_path: Path) -> None:
    fake = _FakeMT5(bars=200)
    out_csv = tmp_path / "audit.csv"
    rows = audit_live(
        terminal_path="C:/fake/terminal64.exe",
        symbols=["XAUUSDm", "EURUSDm"],
        timeframes=["H2", "D1"],
        out_csv=out_csv,
        mt5_module=fake,
    )
    assert len(rows) == 4
    assert all(r.years_available is not None for r in rows)
    assert out_csv.exists()
    text = out_csv.read_text(encoding="utf-8")
    assert "XAUUSDm,H2" in text
    assert "EURUSDm,D1" in text


def test_history_audit_unavailable_is_not_synthesised(tmp_path: Path) -> None:
    fake = _FakeMT5(bars=0)
    out_csv = tmp_path / "audit.csv"
    rows = audit_live(
        terminal_path="C:/fake/terminal64.exe",
        symbols=["FOO"],
        timeframes=["H2"],
        out_csv=out_csv,
        mt5_module=fake,
    )
    assert len(rows) == 1
    assert rows[0].is_unavailable
    assert rows[0].years_available is None
    assert "UNAVAILABLE" in out_csv.read_text(encoding="utf-8")


def test_best_cells_by_history_orders_by_years() -> None:
    rows = [
        HistoryCell("A", "D1", years_available=2.0, bar_count=100, first_bar_utc="", last_bar_utc="", source="MT5_LIVE", as_of="2026"),
        HistoryCell("B", "H4", years_available=10.0, bar_count=200, first_bar_utc="", last_bar_utc="", source="MT5_LIVE", as_of="2026"),
        HistoryCell("C", "M5", years_available=None, bar_count=0, first_bar_utc=None, last_bar_utc=None, source="UNAVAILABLE", as_of="2026"),
    ]
    best = best_cells_by_history(rows, top=2)
    assert [b.symbol for b in best] == ["B", "A"]
    # UNAVAILABLE never appears in best cells.
    assert all(b.years_available is not None for b in best)


def test_audit_offline_does_not_contact_mt5(tmp_path: Path) -> None:
    rows = [
        HistoryCell("X", "D1", years_available=5.0, bar_count=100,
                    first_bar_utc="2021-01-01T00:00:00+00:00",
                    last_bar_utc="2026-01-01T00:00:00+00:00",
                    source="BUNDLED", as_of="2026-09-16T00:00:00+00:00"),
    ]
    out = tmp_path / "audit.csv"
    audit_offline(rows, out)
    assert out.exists()
