"""Deploy refuses a broker on another clock than the one a strategy learned its hours on."""
from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from engine.research.data import server_clock
from engine.research.data.dataset_snapshot import save_snapshot
from engine.research.data.time_basis import BasisProbe
from engine.research.deploy import clock_guard

NOW = datetime(2026, 11, 1, 12, 0, tzinfo=UTC)


def _snap(server: str, basis: str) -> str:
    n = 50
    t = np.arange(n, dtype=np.int64) * 3600 + 1_700_000_000
    p = np.linspace(1.0, 1.1, n) + (0.5 if basis == "server_time" else 0.0)  # distinct bars, distinct snapshots
    meta = save_snapshot(symbol="EURUSD", timeframe="H1", time=t, open_=p, high=p + 0.001, low=p - 0.001, close=p,
                         spread=np.full(n, 10, dtype=np.int64), server=server, time_basis=basis)
    return getattr(meta, "dataset_id", meta)


def test_the_guard(isolated_arqvio_home) -> None:
    server_clock.remember("Exness-UTC", BasisProbe("server_equals_utc", 0, 1.0, 1), "EURUSD", NOW)
    server_clock.remember("Aiwa-GMT3", BasisProbe("broker_offset", 10800, 10800.0, 1), "EURUSD", NOW)
    utc_bars, gmt3_bars = _snap("Exness-UTC", "utc"), _snap("Aiwa-GMT3", "server_time")
    assert clock_guard.check(utc_bars, "Exness-UTC")["ok"]
    assert clock_guard.check(gmt3_bars, "Aiwa-GMT3")["ok"]
    bad = clock_guard.check(utc_bars, "Aiwa-GMT3")
    assert not bad["ok"] and "UTC clock" in bad["reason"] and "GMT+3" in bad["reason"] and "3 hours" in bad["reason"]
    assert not clock_guard.check(gmt3_bars, "Exness-UTC")["ok"]
    unknown = clock_guard.check(utc_bars, "Never-Measured")
    assert unknown["ok"] and "not been measured" in unknown["warn"]
    assert clock_guard.check(None, "Aiwa-GMT3")["ok"]  # an old record without bars: nothing to compare
