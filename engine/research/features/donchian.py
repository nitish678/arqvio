"""Donchian channel features (structural look-ahead safety).

Each function reads a window ending at bar t (inclusive) and never reaches
into the future. The registry enforces the lookback guard at the indexing layer.
"""
from __future__ import annotations

import numpy as np


def donchian_high_series(high: np.ndarray, period: int = 20) -> np.ndarray:
    """Upper Donchian channel: max(high) over the last `period` bars ending at t."""
    high = np.asarray(high, dtype=np.float64)
    n = high.size
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return out
    for i in range(period - 1, n):
        out[i] = float(np.max(high[i - period + 1 : i + 1]))
    return out


def donchian_low_series(low: np.ndarray, period: int = 20) -> np.ndarray:
    """Lower Donchian channel: min(low) over the last `period` bars ending at t."""
    low = np.asarray(low, dtype=np.float64)
    n = low.size
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return out
    for i in range(period - 1, n):
        out[i] = float(np.min(low[i - period + 1 : i + 1]))
    return out


__all__ = ["donchian_high_series", "donchian_low_series"]
