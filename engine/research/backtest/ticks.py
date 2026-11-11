"""Deterministic bid/ask tick execution model for research-only experiments.

This module does not place orders. It consumes an ordered tick stream and
simulates market fills. Latency is an explicit tick-index delay; no queue or
market-impact claim is made.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Side = Literal["long", "short"]


@dataclass(frozen=True)
class TickData:
    time_msc: np.ndarray
    bid: np.ndarray
    ask: np.ndarray
    symbol: str
    source: str
    time_basis: str = "utc"

    def __post_init__(self) -> None:
        n = len(self.time_msc)
        if n == 0 or len(self.bid) != n or len(self.ask) != n:
            raise ValueError("tick arrays must be non-empty and equal length")
        if not np.all(np.isfinite(self.bid)) or not np.all(np.isfinite(self.ask)):
            raise ValueError("tick prices must be finite")
        if np.any(self.bid <= 0) or np.any(self.ask <= 0) or np.any(self.ask < self.bid):
            raise ValueError("ticks require positive ask >= bid")
        if np.any(np.diff(self.time_msc) < 0):
            raise ValueError("ticks must be ordered by time_msc")
        if self.time_basis != "utc":
            raise ValueError("tick execution requires UTC timestamps")

    @property
    def spread(self) -> np.ndarray:
        return self.ask - self.bid


@dataclass(frozen=True)
class TickExecutionConfig:
    latency_ticks: int = 0
    slippage_points: float = 0.0
    point: float = 1e-5

    def __post_init__(self) -> None:
        if self.latency_ticks < 0 or self.slippage_points < 0 or self.point <= 0:
            raise ValueError("latency_ticks >= 0, slippage_points >= 0 and point > 0 required")


@dataclass(frozen=True)
class TickFill:
    tick_index: int
    time_msc: int
    side: Side
    requested_price: float
    fill_price: float
    spread_price: float
    slippage_price: float


def fill_market(ticks: TickData, signal_index: int, side: Side, config: TickExecutionConfig) -> TickFill:
    """Fill at ask for long/buy and bid for short/sell after explicit tick latency."""
    if signal_index < 0 or signal_index >= len(ticks.time_msc):
        raise IndexError("signal_index outside tick data")
    i = signal_index + config.latency_ticks
    if i >= len(ticks.time_msc):
        raise ValueError("latency pushes fill beyond available tick data")
    requested = float(ticks.ask[i] if side == "long" else ticks.bid[i])
    slip = config.slippage_points * config.point
    fill = requested + slip if side == "long" else requested - slip
    mid = (float(ticks.ask[i]) + float(ticks.bid[i])) / 2.0
    return TickFill(i, int(ticks.time_msc[i]), side, requested, fill,
                    float(ticks.ask[i] - ticks.bid[i]), slip)
