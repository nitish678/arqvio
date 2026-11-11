"""Orchestrator: turn a :class:`BacktestResult` into a :class:`ResearchMetrics`.

WHAT: The single function every consumer calls. It dispatches to the
      per-metric modules, derives ``periods_per_year`` from the
      backtest config when possible, and assembles the frozen
      ``ResearchMetrics`` dataclass.

WHY: One entry point, one signature, one output shape. The validation
     core (S1) and the multi-strategy runner (S5) both call this.

HOW: Pulls per-bar returns and per-trade P&L from
     :mod:`engine.research.metrics.returns`, then composes the rest.
     Raises :class:`DataError` only for malformed input shapes (the
     S3 engine already validates that; this is a safety net).

TRADE-OFF: ``periods_per_year`` is derived from
           ``BacktestConfig.timeframe`` via S1's
           ``PERIODS_PER_YEAR`` table. When the timeframe is
           unknown (e.g. ``"custom"``) we fall back to 252 (the
           daily default). Callers can override via the
           ``periods_per_year`` kwarg.
"""
from __future__ import annotations

import math
from typing import Final

import numpy as np

from engine.core.errors import DataError
from engine.research.backtest.engine import BacktestResult
from engine.research.metrics.cost_impact import cost_impact
from engine.research.metrics.distribution import (
    avg_loss,
    avg_r_multiple,
    avg_win,
    expectancy,
    profit_factor,
    win_rate,
)
from engine.research.metrics.drawdown import max_drawdown as _max_dd
from engine.research.metrics.drawdown import time_to_recovery_bars
from engine.research.metrics.exposure import exposure_time, trades_per_month
from engine.research.metrics.ratios import (
    annualised_sharpe,
    annualised_sortino,
    calmar,
)
from engine.research.metrics.returns import (
    bar_returns,
    closed_trade_pnl_split,
    trade_returns,
)
from engine.research.metrics.schema import ResearchMetrics
from engine.research.metrics.stability import rolling_sharpe_stability
from engine.validation.stats import PERIODS_PER_YEAR

_DEFAULT_PERIODS_PER_YEAR: Final[int] = 252


def _derive_periods_per_year(result: BacktestResult) -> int:
    """Map ``config.timeframe`` to a periods-per-year integer.

    Falls back to 252 for unknown timeframes. The map mirrors S1's
    ``PERIODS_PER_YEAR`` so S4 annualised metrics and S1 DSR/PSR
    use the same factor when given the same result.
    """
    tf = result.config.timeframe
    mapped = PERIODS_PER_YEAR.get(tf)
    if mapped is None or mapped <= 0:
        return _DEFAULT_PERIODS_PER_YEAR
    return int(mapped)


def _cagr(initial: float, final: float, months: float) -> float | None:
    """Compound annual growth rate; ``None`` if undefined.

    ``months`` is the calendar span of the run. ``None`` when the
    span is < 12 months, when initial equity is non-positive, or when
    the resulting CAGR is not finite.
    """
    if months < 12.0 or initial <= 0.0 or final <= 0.0:
        return None
    years = months / 12.0
    if years <= 0.0:
        return None
    ratio = final / initial
    if ratio <= 0.0:
        return None
    try:
        val = float(ratio ** (1.0 / years) - 1.0)
    except (OverflowError, ValueError):
        return None
    if not math.isfinite(val):
        return None
    return val


def compute_metrics(
    result: BacktestResult,
    *,
    periods_per_year: int | None = None,
    dataset_version_id: str = "",
    data_hash: str = "",
) -> ResearchMetrics:
    """Compute all S4 metrics from a backtest result.

    Args:
        result: the engine's output. Must have a non-empty equity curve.
        periods_per_year: override the auto-derived annualisation factor.
            ``None`` means "derive from ``config.timeframe``".
        dataset_version_id: carried on the output for traceability.
        data_hash: hash of the cost model + data slice; carried verbatim.

    Returns:
        A frozen :class:`ResearchMetrics` with every field populated.

    Raises:
        DataError: when ``equity_curve`` is missing or has fewer than 2 points.
    """
    eq = np.asarray(result.equity_curve, dtype=np.float64).ravel()
    if eq.size < 2:
        raise DataError(
            f"equity_curve must have >= 2 points; got {eq.size}"
        )

    ppy = periods_per_year if periods_per_year is not None else _derive_periods_per_year(result)
    if ppy <= 0:
        raise DataError(f"periods_per_year must be > 0; got {ppy}")

    initial_equity = float(result.config.initial_equity)
    final_equity = float(result.final_equity)
    total_return = final_equity / initial_equity - 1.0 if initial_equity > 0 else 0.0

    # Bar-based series.
    rets = bar_returns(result)

    # Equity-curve metrics.
    mdd = _max_dd(eq)
    recovery = time_to_recovery_bars(eq)

    # Trade ledger.
    trade_pnls = trade_returns(result)
    gross_arr, net_arr = closed_trade_pnl_split(result)
    months = float(result.bars_n) / float(ppy) * 12.0
    cagr_val = _cagr(initial_equity, final_equity, months)

    return ResearchMetrics(
        # bookkeeping
        n_trades=result.n_trades,
        n_round_trips=result.n_round_trips,
        n_bars=result.bars_n,
        final_equity=final_equity,
        initial_equity=initial_equity,
        total_return=total_return,
        # annualised
        cagr=cagr_val,
        sharpe=annualised_sharpe(rets, ppy),
        sortino=annualised_sortino(rets, ppy),
        calmar=calmar(cagr_val, mdd),
        # drawdown
        max_drawdown=mdd,
        time_to_recovery_bars=recovery,
        # distribution
        profit_factor=profit_factor(trade_pnls),
        win_rate=win_rate(trade_pnls),
        expectancy=expectancy(trade_pnls),
        avg_win=avg_win(trade_pnls),
        avg_loss=avg_loss(trade_pnls),
        avg_r_multiple=avg_r_multiple(trade_pnls),
        # cost / exposure / stability
        cost_impact=cost_impact(gross_arr, net_arr),
        exposure_time=exposure_time(result),
        trades_per_month=trades_per_month(result, ppy),
        stability=rolling_sharpe_stability(eq),
        # traceability
        dataset_version_id=dataset_version_id,
        data_hash=data_hash,
        periods_per_year=ppy,
    )
