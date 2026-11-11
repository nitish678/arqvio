"""The research score and the two tribes' fitness.

Score 0-100, 8 weighted parts (research PART 4.3, homepage verbatim): Persistence 20, Recovery 15, Sharpe 15,
Trades 15, Held-out linearity 12, Max drawdown 10, Profit factor 10, In-sample linearity 3; < 55 WEAK, >= 55 GOOD.
The research gives the weights, not how each part maps a number to [0, 1]; the mappings below are ARQVIO's
(research PART 6: behaviour-equivalent, not byte-equivalent) and each is stated next to its part.

Fitness during breeding is the research's log-scaled composite (TOP FITNESS 4.27 style): ln(1 + 100 q) with q in
[0, 1], so it runs 0 .. 4.62. TRIBE A · CONSISTENCY weighs smoothness and drawdown, TRIBE B · GROWTH weighs return
and profit factor.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from engine.research.registry.annualisation import bars_per_year

WEIGHTS: dict[str, float] = {
    "persistence": 20.0, "recovery": 15.0, "sharpe": 15.0, "trades": 15.0,
    "held_out_linearity": 12.0, "max_drawdown": 10.0, "profit_factor": 10.0, "in_sample_linearity": 3.0,
}
GOOD_AT = 55.0
TRIBE_A, TRIBE_B, FINAL_WAR, WILD = "TRIBE A", "TRIBE B", "FINAL WAR", "Wild"

# Cull reasons and engine states (research PART 4.8).
STAGNATION, DRAWDOWN_BREACH, LOW_FITNESS = "STAGNATION", "DRAWDOWN_BREACH", "LOW_FITNESS"


def _clamp01(x: float) -> float:
    return 0.0 if not math.isfinite(x) else max(0.0, min(1.0, x))


def equity(returns: np.ndarray) -> np.ndarray:
    return np.cumprod(1.0 + np.asarray(returns, dtype=np.float64))


def linearity(returns: np.ndarray) -> float:
    """R^2 of the equity curve against a straight line, 0 when the line does not rise."""
    eq = equity(returns)
    if eq.size < 3 or not np.all(np.isfinite(eq)):
        return 0.0
    x = np.arange(eq.size, dtype=np.float64)
    slope, intercept = np.polyfit(x, eq, 1)
    if slope <= 0.0:
        return 0.0
    ss_tot = float(np.sum((eq - eq.mean()) ** 2))
    if ss_tot <= 0.0:
        return 0.0
    return _clamp01(1.0 - float(np.sum((eq - (slope * x + intercept)) ** 2)) / ss_tot)


def max_drawdown(returns: np.ndarray) -> float:
    eq = equity(returns)
    if eq.size == 0:
        return 0.0
    peak = np.maximum.accumulate(np.concatenate(([1.0], eq)))[1:]
    return float(-np.min((eq - peak) / peak))


def sharpe(returns: np.ndarray, timeframe: str) -> float:
    r = np.asarray(returns, dtype=np.float64)
    if r.size < 2:
        return 0.0
    sd = float(np.std(r, ddof=1))
    return float(np.mean(r) / sd * math.sqrt(bars_per_year(timeframe))) if sd > 0 else 0.0


def profit_factor(trade_pnl: np.ndarray | None, returns: np.ndarray) -> float:
    """Gross profit / gross loss of the closed trades; from the bar returns when no trade log came back."""
    x = np.asarray(trade_pnl if trade_pnl is not None and len(trade_pnl) else returns, dtype=np.float64)
    gain, loss = float(x[x > 0].sum()), float(-x[x < 0].sum())
    if loss <= 0.0:
        return 10.0 if gain > 0.0 else 0.0
    return min(gain / loss, 10.0)


@dataclass(frozen=True)
class Metrics:
    """One strategy on one stretch of bars."""

    sharpe: float
    total_return: float
    max_dd: float
    n_trades: int
    profit_factor: float
    linearity: float

    def to_dict(self) -> dict[str, Any]:
        return {"sharpe": self.sharpe, "total_return": self.total_return, "max_dd": self.max_dd,
                "n_trades": self.n_trades, "profit_factor": self.profit_factor, "linearity": self.linearity}


def metrics(returns: np.ndarray, timeframe: str, n_trades: int, trade_pnl: np.ndarray | None = None) -> Metrics:
    r = np.asarray(returns, dtype=np.float64)
    eq = equity(r)
    return Metrics(sharpe=sharpe(r, timeframe), total_return=float(eq[-1] - 1.0) if eq.size else 0.0,
                   max_dd=max_drawdown(r), n_trades=int(n_trades), profit_factor=profit_factor(trade_pnl, r),
                   linearity=linearity(r))


def _q_common(m: Metrics) -> dict[str, float]:
    return {
        "sharpe": _clamp01(m.sharpe / 2.0),                       # 1 at an annual Sharpe of 2
        "drawdown": 1.0 - _clamp01(m.max_dd / 0.30),              # 1 at no drawdown, 0 at 30 %
        "trades": _clamp01(m.n_trades / 100.0),                   # 1 at 100 trades
        "pf": _clamp01((m.profit_factor - 1.0) / 1.0),            # 1 at a profit factor of 2
        "recovery": _clamp01((m.total_return / max(m.max_dd, 1e-6)) / 3.0) if m.total_return > 0 else 0.0,
    }


def fitness(m: Metrics, tribe: str) -> float:
    """ln(1 + 100 q): TRIBE A weighs linearity and drawdown, TRIBE B (and the Final War's growth side) return."""
    q = _q_common(m)
    if tribe == TRIBE_A:
        mix = 0.30 * m.linearity + 0.25 * q["drawdown"] + 0.25 * q["sharpe"] + 0.20 * q["trades"]
    else:
        mix = 0.35 * q["recovery"] + 0.25 * q["sharpe"] + 0.20 * q["pf"] + 0.20 * q["trades"]
    return math.log1p(100.0 * _clamp01(mix))


def war_fitness(m: Metrics) -> float:
    """The Final War: every survivor judged on both tribes' measures at once."""
    return 0.5 * (fitness(m, TRIBE_A) + fitness(m, TRIBE_B))


def score(in_sample: Metrics, held_out: Metrics, full: Metrics) -> dict[str, Any]:
    """The 0-100 research score from the in-sample, held-out and whole-window numbers."""
    q = _q_common(full)
    is_rpt = in_sample.total_return / in_sample.n_trades if in_sample.n_trades else 0.0
    ho_rpt = held_out.total_return / held_out.n_trades if held_out.n_trades else 0.0
    unit = {
        # research: "PERSISTENCE = unseen earn/trade vs training" -- 1 when held-out earns per trade what training did
        "persistence": _clamp01(ho_rpt / is_rpt) if is_rpt > 0 else 0.0,
        "recovery": q["recovery"],
        "sharpe": q["sharpe"],
        "trades": q["trades"],
        "held_out_linearity": held_out.linearity,
        "max_drawdown": q["drawdown"],
        "profit_factor": q["pf"],
        "in_sample_linearity": in_sample.linearity,
    }
    parts = {k: round(WEIGHTS[k] * unit[k], 2) for k in WEIGHTS}
    total = round(sum(parts.values()), 1)
    return {"score": total, "verdict": "GOOD" if total >= GOOD_AT else "WEAK", "parts": parts}


__all__ = [
    "DRAWDOWN_BREACH", "FINAL_WAR", "GOOD_AT", "LOW_FITNESS", "Metrics", "STAGNATION", "TRIBE_A", "TRIBE_B",
    "WEIGHTS", "WILD", "equity", "fitness", "linearity", "max_drawdown", "metrics", "profit_factor", "score",
    "sharpe", "war_fitness",
]
