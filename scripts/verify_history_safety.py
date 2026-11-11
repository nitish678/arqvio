"""Explicit read-only tester probe; saves raw evidence only under isolated ARQVIO_HOME."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    import MetaTrader5 as mt5
    import numpy as np

    from engine.core.paths import get_artifacts_dir
    from engine.research.cells.add_market import pull_bars
    from engine.research.data.terminals import MT5_LOCK, running_terminals

    terminal = Path("C:/mt5/tester/terminal64.exe")
    if str(terminal).lower() not in {str(Path(p)).lower() for p in running_terminals()}:
        raise RuntimeError("Approved tester is not running; no initialize attempted")
    end = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    start = end - timedelta(days=90)
    directory = get_artifacts_dir() / "safety_history" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    directory.mkdir(parents=True, exist_ok=False)
    with MT5_LOCK:
        if not mt5.initialize(path=str(terminal), timeout=10000):
            raise RuntimeError("Approved tester attach failed")
        try:
            info = mt5.terminal_info()
            if info is None or Path(info.path).resolve() != terminal.parent.resolve():
                raise RuntimeError("Attached terminal identity mismatch")
            if mt5.account_info() is None:
                raise RuntimeError("Tester account is unavailable")
            bars = pull_bars(mt5, "GBPUSDm", "M1", start, end)
        finally:
            mt5.shutdown()  # disconnect Python only; leave terminal running
    times = bars["time"]
    assert len(times) >= 2000
    assert np.all(np.diff(times) > 0), "history is not strictly chronological"
    assert np.all((times >= start.timestamp()) & (times < end.timestamp()))
    raw = directory / "GBPUSDm_M1.npz"
    np.savez_compressed(raw, **bars)
    with np.load(raw, allow_pickle=False) as loaded:
        for name, values in bars.items():
            assert np.array_equal(values, loaded[name]), name
    report = {"schema": "arqvio.history_safety", "version": 1, "symbol": "GBPUSDm", "timeframe": "M1",
              "bars": len(times), "first_utc": datetime.fromtimestamp(int(times[0]), UTC).isoformat(),
              "last_utc": datetime.fromtimestamp(int(times[-1]), UTC).isoformat(),
              "requested_start_utc": start.isoformat(), "requested_end_utc": end.isoformat(),
              "strictly_chronological": True, "unique_timestamps": True, "within_window": True,
              "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(), "saved_bytes_reloaded": True,
              "raw_data_public": False, "trading_calls": False,
              "limits": "Terminal-capped window, not proof of full requested history or execution parity"}
    (directory / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
