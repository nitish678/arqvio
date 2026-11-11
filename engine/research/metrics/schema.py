"""Research metrics output schema.

WHAT: A single frozen dataclass that holds every metric the S3 backtester
      can report. S5 (multi-strategy runner) and S1's validation core
      (DSR / PBO / vs_random) read this object.

WHY: Metrics live in ONE shape; consumers do not need to remember which
     function returned which tuple. ``None`` marks "undefined" (e.g. no
     trades, zero variance, insufficient data); a malformed input is a
     ``DataError`` raised by the orchestrator.

HOW: Pure data, no methods, no I/O. Carries dataset_version_id and
     data_hash so every metric is traceable to the exact cost-model
     snapshot it was computed on.

TRADE-OFF: ``None`` everywhere "undefined" can happen keeps the dataclass
           honest but forces downstream code to handle the None case
           instead of getting 0.0 / nan by accident.

NOT IMPLEMENTED: live-track-record attribution (S7+), risk-adjusted by
                 leverage (S5), per-symbol portfolio aggregation (S6+).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResearchMetrics:
    """All metrics computed from a single :class:`BacktestResult`.

    A field is ``None`` when it is mathematically undefined for this
    result (e.g. Sharpe on zero-variance returns, Calmar when max
    drawdown is zero). It is ``0.0`` when the metric is well-defined
    but happens to be zero (e.g. no round-trip trades => win_rate = 0).
    """

    # ---- counts / bookkeeping (always defined) -----------------------------
    n_trades: int                # total trade records (open + closed)
    n_round_trips: int           # closed trade count
    n_bars: int                  # bars in the run
    final_equity: float          # account ccy
    initial_equity: float        # account ccy
    total_return: float          # final_equity / initial_equity - 1

    # ---- time-based annualised metrics (None when undefined) ---------------
    cagr: float | None           # None if span < 1y or negative equity
    sharpe: float | None         # None if n_returns < 2 or zero variance
    sortino: float | None        # None if zero downside variance
    calmar: float | None         # None if max_drawdown is 0

    # ---- drawdown (always defined; max_dd is negative) ---------------------
    max_drawdown: float          # negative; e.g. -0.18 means -18%
    time_to_recovery_bars: int | None  # None if never recovered

    # ---- trade distribution ------------------------------------------------
    profit_factor: float | None  # None if no losing trades
    win_rate: float              # in [0, 1]
    expectancy: float            # mean net P&L per closed trade (account ccy)
    avg_win: float               # mean net P&L of winners (account ccy)
    avg_loss: float              # mean net P&L of losers (account ccy, <= 0)
    avg_r_multiple: float | None # mean win / |mean loss|; None if no losses

    # ---- cost / exposure / stability ---------------------------------------
    cost_impact: float | None    # None if no closed trades; else in [0, 1]
    exposure_time: float         # in [0, 1]: bars-in-position / total bars
    trades_per_month: float      # n_round_trips / months_covered
    stability: float | None      # std of rolling Sharpe; None if too few bars

    # ---- traceability ------------------------------------------------------
    dataset_version_id: str      # e.g. "usdcadm_h2_2024_v3"
    data_hash: str               # hash of the cost model + raw data slice
    periods_per_year: int        # annualisation factor used for sharpe/sortino
