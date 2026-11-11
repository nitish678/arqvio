"""Exposure: how much of the time the strategy is in the market.

WHAT: ``exposure_time`` = bars-in-position / total-bars. ``trades_per_month``
      = closed round-trip count / months covered by the run.

WHY: Two strategies with the same Sharpe can have wildly different
     exposure: an always-in MA-crossover vs a sparse mean-reversion.
     The latter pays fewer spreads; the former compounds capital more
     often. Exposure + cost_impact together distinguish them.

HOW: ``exposure_time`` walks the trade ledger and counts the bars
     each round-trip covered, divided by the bar count of the run.
     Open (still-running) trades count from their entry_bar_index to
     the last bar of the run. ``trades_per_month`` uses the actual
     calendar span from the first to the last bar in the run (in
     case the engine is later wired with bar timestamps); for now
     we fall back to ``bars / periods_per_year * 12`` when no
     timestamps are available, which is the right answer for a
     uniformly-spaced bar series.

TRADE-OFF: We measure exposure in BARS, not time, because the engine
           doesn't yet attach a timestamp to the equity curve. The
           conversion to "fraction of the year in market" is exact
           when bars are uniformly spaced (the common case in S3).
"""
from __future__ import annotations

import numpy as np

from engine.research.backtest.engine import BacktestResult


def exposure_time(result: BacktestResult) -> float:
    """Fraction of the run during which a position was open.

    A round-trip covers bars ``entry_bar_index`` through
    ``exit_bar_index`` (inclusive on both ends - 1 bar = 1 exposure
    unit). Still-open trades count from their entry bar to the last
    bar of the run. Returns 0.0 for an empty run.
    """
    n_bars = int(result.bars_n)
    if n_bars <= 0:
        return 0.0
    covered = 0
    last_bar = n_bars - 1
    for t in result.trades:
        if t.entry_bar_index < 0 or t.entry_bar_index > last_bar:
            continue
        end = t.exit_bar_index if t.exit_bar_index is not None else last_bar
        if end < t.entry_bar_index:
            continue
        covered += end - t.entry_bar_index + 1
    # An in-progress trade that was opened on the very last bar adds
    # one exposure unit; clamp to n_bars so the fraction is in [0, 1].
    covered = min(covered, n_bars)
    return float(covered) / float(n_bars)


def trades_per_month(result: BacktestResult, periods_per_year: int) -> float:
    """Closed round-trip count per calendar month.

    Months are derived from ``bars_n / periods_per_year * 12``. If
    ``bars_n == 0`` or ``periods_per_year <= 0`` returns 0.0.
    """
    if result.n_round_trips <= 0:
        return 0.0
    if result.bars_n <= 0 or periods_per_year <= 0:
        return 0.0
    months = float(result.bars_n) / float(periods_per_year) * 12.0
    if months <= 0.0:
        return 0.0
    return float(result.n_round_trips) / months


def months_in_run(bar_count: int, periods_per_year: int) -> float:
    """Calendar months represented by ``bar_count`` bars.

    Pure helper, exported for tests. ``0.0`` if any input is non-positive.
    """
    if bar_count <= 0 or periods_per_year <= 0:
        return 0.0
    return float(bar_count) / float(periods_per_year) * 12.0


def _coerce_returns_array(value: object) -> np.ndarray:
    """Defensive coercion for callers passing list/array."""
    return np.asarray(value, dtype=np.float64).ravel()
