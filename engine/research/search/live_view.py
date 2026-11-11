"""What the CAMPAIGN tab draws while a campaign breeds: ``<run dir>/live.json``, rewritten after every generation.

Read-only sidecar (nothing in the search reads it back, so results and checkpoints are unchanged):
- ``bots``: the generation's best survivors, each with its equity curve on the training bars (cumulative return %,
  downsampled to ``CURVE_POINTS``) -- the curves the tab replays behind its scan line;
- ``tape``: the backtest trades of the top few survivors (direction, P&L as % of the 10,000 test account, exit
  reason), from the fast kernel's trade log. These are simulated trades on history, never live ones.
idea: research §7.3 (CAMPAIGN generations fan + live trade tape); format and code are ARQVIO's.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "arqvio.live_view"
VERSION = 1
FILE = "live.json"
CURVE_POINTS = 96
MAX_BOTS = 40
TAPE_BOTS = 5
TAPE_TRADES = 40
INITIAL = 10_000.0  # the inspector's test account (engine/research/vault/inspect.py)
REASONS = {0: "OPEN", 1: "SIGNAL", 2: "SL", 3: "TP", 4: "SL/TP", 5: "FRIDAY"}


def curve_of(per_period_returns: Any, points: int = CURVE_POINTS) -> list[float] | None:
    """Cumulative return in % at ``points`` evenly spaced bars (first and last included)."""
    if per_period_returns is None:
        return None
    r = np.asarray(per_period_returns, dtype=np.float64)
    if r.size == 0:
        return None
    eq = np.cumprod(1.0 + np.nan_to_num(r)) - 1.0
    idx = np.unique(np.linspace(0, eq.size - 1, min(points, eq.size)).round().astype(int))
    return [round(float(x) * 100.0, 3) for x in eq[idx]]


def tape_of(backtest_fn: Any, dna: Any, bot: str, limit: int = TAPE_TRADES) -> list[dict[str, Any]]:
    """The last ``limit`` closed trades of one DNA from the kernel's trade log; [] when the DNA is not
    kernel-expressible (the slow engine is too slow to re-run during a campaign)."""
    cap = getattr(backtest_fn, "capture", None)
    if cap is None:
        return []
    from engine.research.backtest.kernel import TR_DIR, TR_EXIT_BAR, TR_EXIT_REASON, TR_NET

    cap.clear()
    cap["on"] = True
    try:
        backtest_fn(dna)
        rows = cap.get("trades")
    finally:
        cap.clear()  # the next evaluation runs without the trade log
    if rows is None or len(rows) == 0:
        return []
    closed = rows[rows[:, TR_EXIT_BAR] >= 0][-limit:]
    return [{"bot": bot, "dir": int(r[TR_DIR]), "bar": int(r[TR_EXIT_BAR]),
             "pct": round(float(r[TR_NET]) / INITIAL * 100.0, 3), "reason": REASONS.get(int(r[TR_EXIT_REASON]), "?")}
            for r in closed]


def merged_tape(backtest_fn: Any, top: Iterable[tuple[str, Any]], limit: int = TAPE_TRADES) -> list[dict[str, Any]]:
    """The top bots' trades interleaved by exit bar, newest first."""
    rows: list[dict[str, Any]] = []
    for bot, dna in top:
        try:
            rows += tape_of(backtest_fn, dna, bot)
        except Exception:  # a sidecar never stops a campaign
            continue
    rows.sort(key=lambda x: (-x["bar"], x["bot"]))
    return rows[:limit]


def write(run_dir: Path, *, generation: int, generations: int, phase: str, survivors: int,
          bots: list[dict[str, Any]], tape: list[dict[str, Any]]) -> None:
    doc = {"schema": SCHEMA, "version": VERSION, "generation": generation, "generations": generations,
           "phase": phase, "survivors": survivors, "bots": bots[:MAX_BOTS], "tape": tape,
           "utc": datetime.now(tz=UTC).isoformat(timespec="seconds")}
    p = Path(run_dir) / FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, separators=(",", ":")), encoding="utf-8")
    tmp.replace(p)


def read(run_dir: Path) -> dict[str, Any] | None:
    try:
        doc = json.loads((Path(run_dir) / FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return doc if doc.get("schema") == SCHEMA and doc.get("version") == VERSION else None


def safe(fn: Callable[[], None]) -> None:
    """Run a sidecar write; a failure is ignored (the campaign's own files are what count)."""
    try:
        fn()
    except Exception:
        pass
