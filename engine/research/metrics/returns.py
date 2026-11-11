"""Convert a :class:`BacktestResult` into a 1-D returns array.

WHAT: Two pure helpers. ``bar_returns`` gives per-bar arithmetic returns
      from the equity curve (the series S1's DSR / PBO / vs_random
      consume). ``trade_returns`` gives per-trade net P&L (in account
      currency) from the closed-trade ledger.

WHY: The backtester has TWO timelines (bars and trades). The metrics
     layer must expose both so downstream consumers (validation core,
     multi-strategy runner) can pick the right granularity without
     re-parsing the result.

HOW: No I/O, no RNG, no engine calls. ``bar_returns`` requires the
     equity curve to have at least 2 finite values; ``trade_returns``
     drops still-open trades (their P&L is unrealised, not realised).

TRADE-OFF: We use arithmetic returns (not log) because the S1 DSR/PSR
           formulas and Bailey & Lopez de Prado are written in terms
           of arithmetic returns. Geometric returns would disagree by
           ~ 0.5 * sigma^2 per period for daily data; the literature
           is unanimous on arithmetic here.
"""
from __future__ import annotations

import numpy as np

from engine.research.backtest.engine import BacktestResult
from engine.research.backtest.order import Trade


def bar_returns(result: BacktestResult) -> np.ndarray:
    """Per-bar arithmetic returns from the equity curve.

    Returns ``r_t = equity[t] / equity[t-1] - 1`` for ``t = 1..N``.
    Drops any non-finite pair so the array is safe to feed to
    ``sharpe`` / ``moments`` in ``engine.validation.stats``.

    Returns an empty array when the equity curve has fewer than 2
    finite points.
    """
    eq = np.asarray(result.equity_curve, dtype=np.float64).ravel()
    if eq.size < 2:
        return np.empty(0, dtype=np.float64)
    finite = np.isfinite(eq)
    if finite.sum() < 2:
        return np.empty(0, dtype=np.float64)
    eq = eq[finite]
    prev = eq[:-1]
    cur = eq[1:]
    # Guard against a zero / negative equity (would blow up division).
    safe = prev > 0
    r = np.where(safe, cur / np.where(safe, prev, 1.0) - 1.0, 0.0)
    return r.astype(np.float64)


def trade_returns(result: BacktestResult) -> np.ndarray:
    """Per-closed-trade net P&L (account ccy), in chronological order.

    Drops trades that are still open (``exit_fill is None``) because
    their P&L is mark-to-market, not realised. The S4 distribution
    metrics (win rate, expectancy, profit factor) operate on this
    series; S1 stats do NOT consume it (they want bar returns).
    """
    closed = [t for t in result.trades if t.exit_fill is not None]
    if not closed:
        return np.empty(0, dtype=np.float64)
    pnls = np.asarray([t.realized_pnl for t in closed], dtype=np.float64)
    return pnls[np.isfinite(pnls)]


def closed_trade_pnl_split(result: BacktestResult) -> tuple[np.ndarray, np.ndarray]:
    """Per-closed-trade (gross_price_pnl, realized_pnl) split.

    Used by :mod:`engine.research.metrics.cost_impact` to compute
    ``(gross - net) / gross`` honestly. Returns two equal-length
    arrays; empty when no closed trades.
    """
    closed: list[Trade] = [t for t in result.trades if t.exit_fill is not None]
    if not closed:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    gross = np.asarray([t.gross_price_pnl for t in closed], dtype=np.float64)
    net = np.asarray([t.realized_pnl for t in closed], dtype=np.float64)
    m = np.isfinite(gross) & np.isfinite(net)
    return gross[m], net[m]
