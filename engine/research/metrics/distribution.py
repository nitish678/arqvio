"""Trade-distribution metrics (win / loss / R / expectancy / profit factor).

WHAT: Per-trade statistics from the closed-trade ledger. All values
      are in account currency unless explicitly noted.

WHY: Ratios (Sharpe / Sortino / Calmar) describe the equity curve as
     a time series. Distribution metrics describe the ATOMIC P&L
     events. A strategy can have a great Sharpe with only 3 trades
     (lucky) and a mediocre Sharpe with 300 trades (real edge) - the
     distribution tells you which.

HOW: Pure NumPy on the per-trade net-P&L array. ``win_rate`` is the
     fraction of closed trades with positive net P&L. ``expectancy``
     is the mean of that same array. ``profit_factor`` is the sum of
     gains divided by the absolute sum of losses (None when there
     are no losses, since the ratio is undefined). ``avg_r_multiple``
     is the ratio of mean win to mean |loss| (None when there are no
     losses).

TRADE-OFF: We count "winners" as ``net_pnl > 0`` strictly. A trade
           that breaks even after costs is a LOSER from a P&L
           perspective - this matches what an accountant reports.
"""
from __future__ import annotations

import math

import numpy as np


def win_rate(net_pnls: np.ndarray) -> float:
    """Fraction of closed trades with net P&L > 0, in [0, 1].

    Returns 0.0 for an empty array.
    """
    p = np.asarray(net_pnls, dtype=np.float64).ravel()
    p = p[np.isfinite(p)]
    if p.size == 0:
        return 0.0
    return float(np.mean(p > 0.0))


def avg_win(net_pnls: np.ndarray) -> float:
    """Mean net P&L of winning trades; 0.0 if no winners."""
    p = np.asarray(net_pnls, dtype=np.float64).ravel()
    p = p[np.isfinite(p)]
    wins = p[p > 0.0]
    if wins.size == 0:
        return 0.0
    return float(np.mean(wins))


def avg_loss(net_pnls: np.ndarray) -> float:
    """Mean net P&L of losing trades (negative or zero); 0.0 if no losers."""
    p = np.asarray(net_pnls, dtype=np.float64).ravel()
    p = p[np.isfinite(p)]
    losses = p[p < 0.0]
    if losses.size == 0:
        return 0.0
    return float(np.mean(losses))


def expectancy(net_pnls: np.ndarray) -> float:
    """Mean of per-trade net P&L (account ccy). 0.0 for an empty array."""
    p = np.asarray(net_pnls, dtype=np.float64).ravel()
    p = p[np.isfinite(p)]
    if p.size == 0:
        return 0.0
    return float(np.mean(p))


def profit_factor(net_pnls: np.ndarray) -> float | None:
    """Gross gains / |gross losses|.

    Returns ``None`` when there are no losing trades (the ratio is
    mathematically undefined - a "profit factor of infinity" is
    marketing, not math). When there are gains and no losses, also
    returns ``None`` for the same reason. When there are losses and
    no gains, returns 0.0.
    """
    p = np.asarray(net_pnls, dtype=np.float64).ravel()
    p = p[np.isfinite(p)]
    gains = float(np.sum(p[p > 0.0]))
    losses = float(np.sum(p[p < 0.0]))
    if losses == 0.0:
        return None
    if gains == 0.0:
        return 0.0
    return gains / abs(losses)


def avg_r_multiple(net_pnls: np.ndarray) -> float | None:
    """Mean win / |mean loss| (the R-multiple).

    Returns ``None`` when there are no losing trades (R is undefined).
    When there are no winning trades but there are losers, returns
    0.0 (the strategy never wins).
    """
    p = np.asarray(net_pnls, dtype=np.float64).ravel()
    p = p[np.isfinite(p)]
    losses = p[p < 0.0]
    if losses.size == 0:
        return None
    mean_loss = float(np.mean(losses))
    if mean_loss == 0.0:
        return None
    wins = p[p > 0.0]
    if wins.size == 0:
        return 0.0
    mean_win = float(np.mean(wins))
    return mean_win / abs(mean_loss)


def assert_finite(value: float, name: str) -> float:
    """Helper for tests / callers: raise on non-finite.

    Used at the orchestrator boundary so a NaN from a deeper layer
    surfaces as an explicit failure rather than silently shipping
    NaN to the dashboard.
    """
    if not math.isfinite(value):
        raise ValueError(f"{name} is not finite: {value!r}")
    return value
