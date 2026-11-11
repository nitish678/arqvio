"""Lifecycle + incremental sync + provenance header formatting."""
from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from engine.data.parquet_store import BarBatch
from engine.mt5.lifecycle import (
    PROBE_TIMEOUT_SECONDS,
    RETRY_INTERVAL_SECONDS,
    ConnectionController,
    ConnectionHeader,
    ConnectionState,
    default_probe_loop,
)
from engine.mt5.sync import SyncReport, incremental_sync
from engine.research.provenance import DataSource, Provenance, report_staleness_line

# --- lifecycle ------------------------------------------------------------

def test_state_machine_transitions() -> None:
    c = ConnectionController()
    c.transition(ConnectionState.CONNECTING)
    c.transition(ConnectionState.CONNECTED)
    c.transition(ConnectionState.SYNCING)
    c.transition(ConnectionState.LIVE)
    assert c.state is ConnectionState.LIVE
    c.transition(ConnectionState.DEGRADED)
    assert c.state is ConnectionState.DEGRADED
    assert len(c.transition_log) == 5


def test_header_render_live() -> None:
    h = ConnectionHeader(
        state=ConnectionState.LIVE,
        last_synced_utc=datetime.now(tz=UTC).isoformat(),
    )
    text = h.render()
    assert text.startswith("MT5 LIVE")
    assert "synced" in text


def test_header_render_degraded() -> None:
    h = ConnectionHeader(
        state=ConnectionState.DEGRADED,
        data_as_of_utc=datetime(2026, 9, 14, tzinfo=UTC).isoformat(),
    )
    text = h.render()
    assert "DEGRADED" in text
    assert "cache" in text


def test_header_render_offline() -> None:
    h = ConnectionHeader(
        state=ConnectionState.DISCONNECTED,
        data_as_of_utc=datetime(2026, 9, 14, tzinfo=UTC).isoformat(),
    )
    text = h.render()
    assert "OFFLINE" in text


def test_probe_loop_retries_and_succeeds() -> None:
    c = ConnectionController()
    attempts: list[bool] = [False, False, True]
    def probe() -> bool:
        return attempts.pop(0)
    def stop() -> bool:
        return not attempts
    default_probe_loop(c, probe, stop_condition=stop)
    assert c.state is ConnectionState.CONNECTED


def test_probe_constants_match_spec() -> None:
    assert PROBE_TIMEOUT_SECONDS == 3.0
    assert RETRY_INTERVAL_SECONDS == 30.0


# --- sync -----------------------------------------------------------------

def test_incremental_sync_fetches_gap_only(tmp_path) -> None:
    # Pre-populate 100 bars as the "last stored" state.
    from engine.data.parquet_store import store_bars
    from engine.data.tiers import tier3_path

    path = tier3_path("USDCADm", "H2", 2026)
    n_initial = 100
    t0 = int(datetime(2026, 9, 16, 0, 0, 0, tzinfo=UTC).timestamp())
    bar_dt = 2 * 3600  # 2 hours
    initial = BarBatch(
        time=np.arange(t0, t0 + n_initial * bar_dt, bar_dt, dtype="<i8"),
        open=np.linspace(1.30, 1.31, n_initial, dtype="<f8"),
        high=np.linspace(1.31, 1.32, n_initial, dtype="<f8"),
        low=np.linspace(1.29, 1.30, n_initial, dtype="<f8"),
        close=np.linspace(1.30, 1.31, n_initial, dtype="<f8"),
        tick_volume=np.full(n_initial, 1000, dtype="<i8"),
        spread=np.full(n_initial, 1, dtype="<f8"),
        real_volume=np.zeros(n_initial, dtype="<i8"),
    )
    store_bars(path, initial, symbol="USDCADm", timeframe="H2")
    # The last stored bar time is the LAST bar in the file, i.e. t0 + (n-1)*bar_dt.
    last_stored = int(initial.time[-1])
    captured: dict[str, int] = {}
    def puller(from_utc: int, to_utc: int) -> BarBatch:
        captured["from"] = from_utc
        captured["to"] = to_utc
        # Add 10 new bars AFTER the last stored one.
        new_t0 = last_stored + bar_dt
        new_time = np.arange(new_t0, new_t0 + 10 * bar_dt, bar_dt, dtype="<i8")
        return BarBatch(
            time=new_time,
            open=np.full(10, 1.31, dtype="<f8"),
            high=np.full(10, 1.32, dtype="<f8"),
            low=np.full(10, 1.30, dtype="<f8"),
            close=np.full(10, 1.31, dtype="<f8"),
            tick_volume=np.full(10, 1000, dtype="<i8"),
            spread=np.full(10, 1, dtype="<f8"),
            real_volume=np.zeros(10, dtype="<i8"),
        )
    report: SyncReport = incremental_sync("USDCADm", "H2", pull_fn=puller)
    assert captured["from"] == last_stored + 1
    assert report.bars_added == 10
    assert "USDCADm H2: 10 bars added" in report.line()


# --- provenance -----------------------------------------------------------

def test_provenance_staleness_strings() -> None:
    p_cached = Provenance(
        source=DataSource.CACHED,
        dataset_hash="h",
        as_of=datetime(2026, 9, 13, tzinfo=UTC),
    )
    line = report_staleness_line(p_cached, now=datetime(2026, 9, 16, 12, 0, 0, tzinfo=UTC))
    assert line == "DATA: CACHED, 3 days old"

    p_bundled = Provenance(
        source=DataSource.BUNDLED,
        dataset_hash="h",
        as_of=datetime(2026, 9, 16, 0, 0, 0, tzinfo=UTC),
    )
    line = report_staleness_line(p_bundled, now=datetime(2026, 9, 16, 12, 0, 0, tzinfo=UTC))
    assert line == "DATA: BUNDLED"

    p_live = Provenance(
        source=DataSource.MT5_LIVE,
        dataset_hash="h",
        as_of=datetime(2026, 9, 16, 11, 59, 48, tzinfo=UTC),
    )
    line = report_staleness_line(p_live, now=datetime(2026, 9, 16, 12, 0, 0, tzinfo=UTC))
    assert line == "DATA: MT5_LIVE · synced 12s ago"
