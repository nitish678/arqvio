"""The CAMPAIGN tab's sidecar: equity curves from per-period returns, the kernel trade log as a tape, the file."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from engine.research.backtest.kernel import TR_DIR, TR_EXIT_BAR, TR_EXIT_REASON, TR_NET, TR_SIZE
from engine.research.search import live_view


def test_curve_is_cumulative_percent_and_downsampled() -> None:
    r = np.full(1000, 0.001)
    c = live_view.curve_of(r)
    assert c is not None and len(c) == live_view.CURVE_POINTS
    assert c[0] == 0.1  # the first bar's return
    assert abs(c[-1] - ((1.001 ** 1000) - 1) * 100) < 1e-3
    assert live_view.curve_of(None) is None and live_view.curve_of([]) is None
    assert len(live_view.curve_of([0.01, -0.01, 0.02])) == 3  # never more points than bars


def _fake_bt(rows: np.ndarray):
    calls = []

    def bt(dna):
        calls.append(dict(bt.capture))
        if bt.capture.get("on"):
            bt.capture.update(trades=rows)

    bt.capture = {}
    bt.calls = calls
    return bt


def _row(direction: int, exit_bar: int, net: float, reason: int) -> np.ndarray:
    r = np.zeros(TR_SIZE)
    r[TR_DIR], r[TR_EXIT_BAR], r[TR_NET], r[TR_EXIT_REASON] = direction, exit_bar, net, reason
    return r


def test_tape_reads_closed_trades_and_turns_the_log_off_again() -> None:
    rows = np.array([_row(1, 10, 100.0, 3), _row(-1, 20, -50.0, 2), _row(1, -1, 0.0, 0)])
    bt = _fake_bt(rows)
    tape = live_view.tape_of(bt, object(), "ABC123")
    assert [(t["dir"], t["bar"], t["pct"], t["reason"]) for t in tape] == [(1, 10, 1.0, "TP"), (-1, 20, -0.5, "SL")]
    assert bt.capture == {}  # the next evaluation runs without the trade log
    assert live_view.tape_of(object(), object(), "X") == []  # no kernel capture: no tape


def test_merged_tape_is_newest_first_across_bots() -> None:
    a = _fake_bt(np.array([_row(1, 5, 10.0, 3), _row(1, 30, 10.0, 3)]))
    tape = live_view.merged_tape(a, [("A", object()), ("B", object())])
    assert [t["bar"] for t in tape] == [30, 30, 5, 5]


def test_write_then_read(tmp_path: Path) -> None:
    live_view.write(tmp_path, generation=3, generations=20, phase="TRIBES", survivors=7,
                    bots=[{"id": "A", "curve": [0.0, 1.0]}] * 50, tape=[])
    doc = live_view.read(tmp_path)
    assert doc is not None and doc["generation"] == 3 and len(doc["bots"]) == live_view.MAX_BOTS
    assert live_view.read(tmp_path / "missing") is None
