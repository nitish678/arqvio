"""Minimal common gate input: real fills/closed P&L/equity, never invented kernel rows.

Legacy kernel tuple callers stay supported by the V1 gate. Oracle fallback uses
its Trade objects; this module does not change fills, costs or engine interfaces.
"""
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GateResult:
    fills: np.ndarray
    exits: np.ndarray
    net: np.ndarray
    equity: np.ndarray
    execution_path: str
    fallback_reason: str = ""


def kernel_view(rows: np.ndarray, equity: np.ndarray) -> GateResult:
    from engine.research.backtest.kernel import TR_EXIT_BAR, TR_FILL_BAR, TR_NET

    return GateResult(rows[:, TR_FILL_BAR], rows[:, TR_EXIT_BAR], rows[:, TR_NET],
                      np.asarray(equity, dtype=np.float64), "kernel")


def oracle_view(result: Any, *, reason: str) -> GateResult:
    trades = result.trades
    return GateResult(
        np.asarray([t.entry_fill.bar_index for t in trades], dtype=np.int64),
        np.asarray([-1 if t.exit_bar_index is None else t.exit_bar_index for t in trades], dtype=np.int64),
        np.asarray([t.realized_pnl for t in trades], dtype=np.float64),
        np.asarray(result.equity_curve, dtype=np.float64), "slow_fallback", reason,
    )


def full_run(dna, cell, fast_fn) -> GateResult:
    capture = fast_fn.capture
    capture.clear()
    capture["on"] = True
    state = getattr(fast_fn, "fallback_state", {})
    before = int(state.get("fallbacks", 0))
    fast_fn(dna)
    if "trades" in capture and "equity" in capture:
        return kernel_view(capture["trades"], capture["equity"])
    # No kernel log: rerun the exact oracle rather than approximate or silently reject.
    from engine.research.search.real_backtest import dna_to_strategy, run_slow_backtest, slow_cell_setup

    cfg, strategy = dna_to_strategy(dna)
    setup = slow_cell_setup(cell)
    reasons = state.get("reasons") or []
    reason = str(reasons[-1]) if int(state.get("fallbacks", 0)) > before and reasons else "kernel trade capture unavailable"
    result = run_slow_backtest(cell, setup, dna, cfg, strategy, notes=f"finalist gate {dna.dna_hash}")
    return oracle_view(result, reason=reason)
