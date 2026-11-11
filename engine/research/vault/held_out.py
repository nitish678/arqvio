"""The held-out test: a vault strategy run on bars its breeding never saw (ARQVIO_V4.md 10.3).

A launch breeds each market on the first 70% of its window (cells/add_market.py ``hold_out``); the full window stays
pinned. After breeding, every new vault strategy is run ONCE over the full window by the campaign's fast kernel and
judged on the part after the split only: the return, Sharpe, drawdown and trade count of the held-out bars. The
position state at the split carries over (the strategy runs continuously, as it would live). One run per strategy,
after the search -- the result never feeds back into breeding, so it is not one more trial.
"""
from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

from engine.research.registry.annualisation import bars_per_year

MIN_HELD_OUT_TRADES = 15


def record_for(entry: dict[str, Any]) -> dict[str, Any] | None:
    from engine.research.cells.add_market import cell_dir

    p = cell_dir(entry["cell"]) / "add_market.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def metrics(equity: np.ndarray, fills: np.ndarray, split: int, timeframe: str) -> dict[str, Any]:
    """Held-out numbers from a full-window equity curve (one value per bar boundary) and the trades' fill bars."""
    eq = np.asarray(equity, dtype=np.float64)[split:]
    r = np.diff(eq) / eq[:-1] if eq.size >= 2 and np.all(eq[:-1] > 0) else np.zeros(0)
    sharpe = float(np.mean(r) / np.std(r, ddof=1) * math.sqrt(bars_per_year(timeframe))) if r.size >= 2 and np.std(r) > 0 else 0.0
    peak = np.maximum.accumulate(eq) if eq.size else eq
    dd = float(-((eq - peak) / np.maximum(peak, 1e-9)).min()) if eq.size else 0.0
    return {"return": float(eq[-1] / eq[0] - 1.0) if eq.size >= 2 else 0.0, "sharpe": sharpe, "max_dd": dd,
            "trades": int(np.sum(np.asarray(fills) >= split)), "bars": int(eq.size - 1)}


def evaluate(entry: dict[str, Any]) -> dict[str, Any]:
    from engine.research.backtest.kernel import TR_FILL_BAR
    from engine.research.search.dna import DNA
    from engine.research.search.real_backtest import fast_backtest_factory, load_cell_bars_from_snapshot

    rec = record_for(entry)
    ho = (rec or {}).get("held_out") or {}
    if ho.get("state") != "KEPT":
        return {"state": "NONE", "reason": ho.get("reason") or "this market was bred on its whole window (before v4)"}
    cell = load_cell_bars_from_snapshot(entry["symbol"], entry["timeframe"], ho["dataset_hash16"])
    bt = fast_backtest_factory(cell)
    bt.capture["on"] = True
    bt(DNA(**entry["dna"]))
    if "trades" not in bt.capture:
        return {"state": "FAILED", "reason": "the strategy is not kernel-expressible; the held-out run needs the kernel"}
    rows, equity = bt.capture["trades"], bt.capture["equity"]
    m = metrics(equity, rows[:, TR_FILL_BAR] if len(rows) else np.zeros(0), int(ho["split_bar"]), entry["timeframe"])
    return {"state": "DONE", "split_time": ho["split_time"], "frac": ho["frac"], **m}


__all__ = ["MIN_HELD_OUT_TRADES", "evaluate", "metrics", "record_for"]
