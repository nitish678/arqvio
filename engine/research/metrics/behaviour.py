"""Trade-behaviour metrics (B2).

The existing metrics describe the RETURN distribution. These describe
how the strategy behaved to get it: how long it held, which side it
took, how its wins and losses clustered, and whether its equity grew
steadily or in one lucky stretch.

Two strategies with the same Sharpe and the same win rate are not the
same strategy if one made its money in a single month and the other
made it every month. Linearity and persistence are what tell them
apart.

Definitions are standard usage and each is stated where it is
computed. Everything here returns ``None`` when it is undefined rather
than a stand-in value: a recovery factor with no drawdown is not 0.0,
it is undefined, and reporting 0.0 would rank it below a strategy that
actually recovered.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from engine.research.backtest.order import Trade

# A rolling window of 20 trades, of which at least half must be
# profitable. Both are the spec's stated values for persistence.
PERSISTENCE_WINDOW_TRADES = 20
PERSISTENCE_MIN_PROFITABLE_FRACTION = 0.5

# Chronological in-sample / out-of-sample split for the IS-vs-OOS
# counts. 70/30 is the common convention; it is a PARAMETER so a caller
# with a reason can change it, and it is reported alongside the numbers
# so nobody has to guess which split produced them.
DEFAULT_IS_FRACTION = 0.70


@dataclass(frozen=True)
class StreakStats:
    max_wins: int
    avg_wins: float | None
    max_losses: int
    avg_losses: float | None


def _streak_lengths(flags: list[bool], want: bool) -> list[int]:
    runs: list[int] = []
    current = 0
    for flag in flags:
        if flag is want:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return runs


def streaks(net_pnls: np.ndarray) -> StreakStats:
    """Longest and average consecutive win / loss runs.

    A trade at exactly zero is neither: counting it as a win would
    extend a winning streak on a trade that made nothing.
    """
    arr = np.asarray(net_pnls, dtype=np.float64)
    wins = [bool(x > 0) for x in arr]
    losses = [bool(x < 0) for x in arr]
    win_runs = _streak_lengths(wins, True)
    loss_runs = _streak_lengths(losses, True)
    return StreakStats(
        max_wins=max(win_runs) if win_runs else 0,
        avg_wins=float(np.mean(win_runs)) if win_runs else None,
        max_losses=max(loss_runs) if loss_runs else 0,
        avg_losses=float(np.mean(loss_runs)) if loss_runs else None,
    )


def avg_hold_bars(trades: tuple[Trade, ...]) -> float | None:
    """Mean bars held, over CLOSED trades only.

    An open trade has no holding period yet; including it as
    "held until the data ended" would report the dataset's length
    rather than the strategy's behaviour.
    """
    spans = [
        int(t.exit_bar_index) - int(t.entry_bar_index)
        for t in trades
        if t.exit_bar_index is not None and t.entry_bar_index is not None
    ]
    if not spans:
        return None
    return float(np.mean(spans))


def long_short_split(trades: tuple[Trade, ...]) -> tuple[int, int, float | None]:
    """(n_long, n_short, long_fraction). Fraction is None with no trades."""
    n_long = sum(1 for t in trades if t.direction == "long")
    n_short = sum(1 for t in trades if t.direction == "short")
    total = n_long + n_short
    return n_long, n_short, (n_long / total if total else None)


def recovery_factor(total_return: float, max_drawdown: float) -> float | None:
    """Net return divided by the worst drawdown it survived.

    max_drawdown arrives negative. None when there was no drawdown:
    dividing by zero is undefined, and 0.0 would rank a strategy that
    never drew down BELOW one that recovered from a small dip.
    """
    dd = abs(float(max_drawdown))
    if dd <= 0.0:
        return None
    return float(total_return) / dd


def linearity_r2(equity: np.ndarray) -> float | None:
    """R^2 of the equity curve against a straight line through time.

    1.0 is a perfectly steady climb; near 0 means the shape is
    anything but. This is what separates a strategy that earned its
    Sharpe every month from one that earned it in a single week and
    then sat still.

    None when there are fewer than three points or the curve is
    perfectly flat, in which case a line explains nothing because
    there is nothing to explain.
    """
    y = np.asarray(equity, dtype=np.float64)
    if y.size < 3:
        return None
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot <= 0.0:
        return None
    x = np.arange(y.size, dtype=np.float64)
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope * x + intercept)
    ss_res = float(np.sum(residual**2))
    return float(1.0 - ss_res / ss_tot)


def persistence(
    net_pnls: np.ndarray,
    *,
    window: int = PERSISTENCE_WINDOW_TRADES,
    min_fraction: float = PERSISTENCE_MIN_PROFITABLE_FRACTION,
) -> float | None:
    """Fraction of rolling trade-windows that were profitable.

    A window is profitable when its net P&L sums above zero. The result
    is the share of windows clearing that bar; ``min_fraction`` is the
    threshold a caller compares against, carried here so the rule and
    its parameter travel together.

    None when there are fewer trades than one window: a single partial
    window is not evidence of persistence either way.
    """
    arr = np.asarray(net_pnls, dtype=np.float64)
    if arr.size < window:
        return None
    sums = np.convolve(arr, np.ones(window, dtype=np.float64), mode="valid")
    del min_fraction
    return float(np.mean(sums > 0.0))


@dataclass(frozen=True)
class InOutSample:
    is_fraction: float
    n_trades_is: int
    n_trades_oos: int
    return_is: float
    return_oos: float


def in_out_sample(
    net_pnls: np.ndarray, *, is_fraction: float = DEFAULT_IS_FRACTION,
) -> InOutSample | None:
    """Trade counts and summed P&L either side of a chronological split.

    The split is on TRADE ORDER, not calendar time, so both halves
    contain trades. ``is_fraction`` is reported back so the numbers
    cannot be read without knowing which split produced them.
    """
    arr = np.asarray(net_pnls, dtype=np.float64)
    if arr.size < 2:
        return None
    cut = int(round(arr.size * float(is_fraction)))
    cut = max(1, min(arr.size - 1, cut))
    return InOutSample(
        is_fraction=float(is_fraction),
        n_trades_is=int(cut),
        n_trades_oos=int(arr.size - cut),
        return_is=float(np.sum(arr[:cut])),
        return_oos=float(np.sum(arr[cut:])),
    )


__all__ = [
    "DEFAULT_IS_FRACTION",
    "PERSISTENCE_MIN_PROFITABLE_FRACTION",
    "PERSISTENCE_WINDOW_TRADES",
    "InOutSample",
    "StreakStats",
    "avg_hold_bars",
    "in_out_sample",
    "linearity_r2",
    "long_short_split",
    "persistence",
    "recovery_factor",
    "streaks",
]
