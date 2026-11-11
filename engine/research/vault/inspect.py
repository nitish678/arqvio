"""The strategy inspector: a vault strategy re-run on its own pinned data, read trade by trade.

The vault keeps what the campaign recorded (Sharpe, return, drawdown, trades). The inspector panel also shows what
only the individual trades tell -- win rate, profit factor, average win and loss, streaks, long/short split, holding
time -- and the equity curve. The run is the campaign's own fast kernel with its trade log kept (``trades_out``, the
rows the parity battery checks against the slow oracle), so its return must equal the vault's recorded one;
``matches_vault`` says whether it does.

The result is cached per strategy (``<home>/artifacts/vault/inspect/<dna_hash>_<cell>.json``): the same DNA on the
same pinned dataset always gives the same trades.
"""
from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from engine.research.registry.annualisation import TF_MINUTES

CURVE_POINTS = 400
SCHEMA = "arqvio.strategy_inspect"
VERSION = 2  # v4: monthly P&L and the spread gate per month


def cache_path(entry: dict[str, Any]) -> Path:
    from engine.research.vault.library import vault_dir

    return vault_dir() / "inspect" / f"{entry['dna_hash']}_{entry['cell']}.json"


def _streaks(wins: list[bool]) -> tuple[int, int]:
    best_w = best_l = run = 0
    last: bool | None = None
    for w in wins:
        run = run + 1 if w == last else 1
        last = w
        if w:
            best_w = max(best_w, run)
        else:
            best_l = max(best_l, run)
    return best_w, best_l


def _curve(times: np.ndarray, equity: np.ndarray) -> list[list[float]]:
    n = len(equity)
    if n == 0:
        return []
    idx = np.unique(np.linspace(0, n - 1, min(CURVE_POINTS, n)).astype(int))
    return [[int(times[min(i, len(times) - 1)]), round(float(equity[i]), 2)] for i in idx]


def summarise(pnl: list[float], dirs: list[int], holds: list[int], equity: np.ndarray, times: np.ndarray,
              timeframe: str, initial: float) -> dict[str, Any]:
    """Trade statistics from closed trades (realized P&L in account currency) and the equity curve."""
    wins = [p for p in pnl if p > 0]
    losses = [p for p in pnl if p <= 0]
    gross_win, gross_loss = sum(wins), -sum(losses)
    best_w, best_l = _streaks([p > 0 for p in pnl])
    eq = np.asarray(equity, dtype=np.float64)
    peak = np.maximum.accumulate(eq) if eq.size else eq
    dd = float(-((eq - peak) / np.maximum(peak, 1e-9)).min()) if eq.size else 0.0
    final = float(eq[-1]) if eq.size else initial
    n = len(pnl)
    longs = sum(1 for d in dirs if d > 0)
    return {
        "trades": n,
        "win_rate": (len(wins) / n) if n else None,
        "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else None,
        "avg_win": (gross_win / len(wins)) if wins else None,
        "avg_loss": (-gross_loss / len(losses)) if losses else None,
        "avg_trade": (sum(pnl) / n) if n else None,
        "largest_win": max(pnl) if pnl else None,
        "largest_loss": min(pnl) if pnl else None,
        "max_win_streak": best_w,
        "max_loss_streak": best_l,
        "long_trades": longs,
        "short_trades": n - longs,
        "avg_hold_hours": (sum(holds) / len(holds) * TF_MINUTES[timeframe] / 60.0) if holds else None,
        "total_return": final / initial - 1.0,
        "max_drawdown": dd,
        "net_profit": final - initial,
        "initial_equity": initial,
        "curve": _curve(times, eq),
    }


def _kernel_run(cell: Any, dna: Any) -> tuple[np.ndarray, np.ndarray] | None:
    """(trade rows, equity) from the campaign's fast kernel; None when this DNA falls back to the slow engine."""
    from engine.research.search.real_backtest import fast_backtest_factory

    bt = fast_backtest_factory(cell)
    bt.capture["on"] = True
    bt(dna)
    if "trades" not in bt.capture:
        return None
    return bt.capture["trades"], bt.capture["equity"]


def inspect(entry: dict[str, Any], *, refresh: bool = False) -> dict[str, Any]:
    """The inspector record for one vault entry (cached)."""
    p = cache_path(entry)
    if p.is_file() and not refresh:
        cached = json.loads(p.read_text(encoding="utf-8"))
        if cached.get("version") == VERSION:
            return cached
    from engine.research.backtest.kernel import TR_DIR, TR_ENTRY_BAR, TR_EXIT_BAR, TR_NET
    from engine.research.search.dna import DNA
    from engine.research.search.real_backtest import load_cell_bars_from_snapshot, spread_cost_mode
    from engine.research.vault.library import recorded_cost_model

    dna = DNA(**entry["dna"])
    costs = recorded_cost_model(entry)  # the record's own cost model: "matches" compares like with like
    cell = load_cell_bars_from_snapshot(entry["symbol"], entry["timeframe"], entry["dataset_id"])
    times = np.asarray(cell.bars.time)
    initial = 10_000.0
    with spread_cost_mode(costs):
        run = _kernel_run(cell, dna)
    if run is not None:
        rows, equity = run
        closed = rows[rows[:, TR_EXIT_BAR] >= 0]
        pnl = [float(x) for x in closed[:, TR_NET]]
        dirs = [int(x) for x in closed[:, TR_DIR]]
        holds = [int(r[TR_EXIT_BAR] - r[TR_ENTRY_BAR]) for r in closed]
        exits = [int(r[TR_EXIT_BAR]) for r in closed]
        engine = "fast kernel (trade log)"
    else:  # not kernel-expressible: the slow engine's own trades
        from engine.research.search.real_backtest import (
            dna_to_strategy,
            run_slow_backtest,
            slow_cell_setup,
        )

        cfg, strategy = dna_to_strategy(dna)
        with spread_cost_mode(costs):
            setup = slow_cell_setup(cell)
        res = run_slow_backtest(cell, setup, dna, cfg, strategy, notes=f"inspect {dna.dna_hash}")
        trades = list(res.closed_trades)
        pnl = [float(t.realized_pnl) for t in trades]
        dirs = [int(getattr(t.direction, "value", t.direction)) for t in trades]
        holds = [int(t.exit_bar_index) - int(t.entry_bar_index) for t in trades]
        exits = [int(t.exit_bar_index) for t in trades]
        equity, engine = np.asarray(res.equity_curve), "slow engine"
    stats = summarise(pnl, dirs, holds, equity, times, entry["timeframe"], initial)
    monthly, gate = _monthly(cell, times, pnl, exits)
    recorded = (entry.get("metrics") or {}).get("total_return")
    matches = (recorded is not None and math.isfinite(recorded)
               and abs(stats["total_return"] - recorded) <= 1e-9 * max(1.0, abs(recorded)))
    doc = {"schema": SCHEMA, "version": VERSION, "id": entry["id"], "dna_hash": entry["dna_hash"],
           "cell": entry["cell"], "symbol": entry["symbol"], "timeframe": entry["timeframe"],
           "dataset_id": entry["dataset_id"], "bars": int(cell.n_bars),
           "first_bar": datetime.fromtimestamp(int(times[0]), tz=UTC).isoformat(),
           "last_bar": datetime.fromtimestamp(int(times[-1]), tz=UTC).isoformat(),
           "engine": engine, "stats": stats, "monthly": monthly, "spread_gate": gate,
           "recorded_total_return": recorded,
           "matches_vault": bool(matches), "computed_utc": datetime.now(tz=UTC).isoformat(timespec="seconds")}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc) + "\n", encoding="utf-8")
    return doc


def _monthly(cell: Any, times: np.ndarray, pnl: list[float], exits: list[int]) -> tuple[list[dict[str, Any]], dict]:
    """Per UTC month: closed trades and their P&L (by exit bar), and the bars on which the cell's spread gate would
    refuse an entry -- the months a flat equity curve sits in show WHY (ARQVIO_V4.md 10.1)."""
    from engine.research.costs.spread_regime import blocked_by_month
    from engine.research.search.real_backtest import slow_cell_setup

    setup = slow_cell_setup(cell)
    if setup.spread_cap_by_bar is not None:
        cap, gate = setup.spread_cap_by_bar, {"mode": "regime", "detail": "3x the median spread of the previous 20 days"}
    elif setup.max_spread_points is not None:
        cap, gate = setup.max_spread_points, {"mode": "fixed", "detail": f"{setup.max_spread_points:.1f} points"}
    else:
        cap, gate = None, {"mode": "off", "detail": "no spread calibration for this symbol"}
    blocked = blocked_by_month(times, np.asarray(cell.bars.spread), cap)
    months = times.astype("datetime64[s]").astype("datetime64[M]").astype(str).tolist()
    out: dict[str, dict[str, Any]] = {}
    for m in dict.fromkeys(months):
        out[m] = {"month": m, "trades": 0, "pnl": 0.0, "bars": 0, "gate_bars": blocked.get(m, 0)}
    for m in months:
        out[m]["bars"] += 1
    for v, i in zip(pnl, exits, strict=True):
        m = months[min(max(i, 0), len(months) - 1)]
        out[m]["trades"] += 1
        out[m]["pnl"] += v
    return [{**r, "pnl": round(r["pnl"], 2)} for r in out.values()], gate


__all__ = ["CURVE_POINTS", "cache_path", "inspect", "summarise"]
