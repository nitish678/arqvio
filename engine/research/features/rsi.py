"""Relative Strength Index.

RSI(feature) = 100 - 100 / (1 + RS)

where RS is the ratio of the average gain to the average loss over the
lookback window. The average is the Wilder smoothing:

    avg_gain[t] = (prev_avg_gain * (period - 1) + gain[t]) / period
    avg_loss[t] = (prev_avg_loss * (period - 1) + loss[t]) / period

This matches MT5's ``iRSI`` semantics on default settings. The feature is
registered as ``rsi_14`` with a lookback of 15 -- the first 14 bars
seed the Wilder average, and bar t is the 15th.
"""
from __future__ import annotations

import numpy as np

from engine.core.errors import DataError

DEFAULT_RSI_PERIOD = 14


def rsi_series(
    close: np.ndarray,
    period: int = DEFAULT_RSI_PERIOD,
) -> np.ndarray:
    """RSI for every bar, NaN for the first ``period`` bars (warmup).

    Returns a float64 array the same length as ``close``. Indices
    ``< period`` are NaN: Wilder smoothing needs a full window before it
    produces a value, and callers treat NaN as "do not trade yet".
    """
    if period < 1:
        raise DataError(f"rsi period must be >= 1; got {period}")
    close = np.asarray(close, dtype=np.float64)
    if close.size == 0:
        return np.zeros(0, dtype=np.float64)
    up = np.diff(close)
    gains = np.where(up > 0, up, 0.0)
    losses = np.where(up < 0, -up, 0.0)
    # Seed: SMA of the first `period` gains/losses.
    if close.size <= period:
        return np.full(close.size, np.nan, dtype=np.float64)
    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))
    out = np.full(close.size, np.nan, dtype=np.float64)
    out[period] = _rsi_value(avg_gain, avg_loss)
    for i in range(period + 1, close.size):
        g = gains[i - 1]
        l = losses[i - 1]
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + l) / period
        out[i] = _rsi_value(avg_gain, avg_loss)
    return out


def _rsi_value(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0.0 and avg_gain == 0.0:
        return 50.0
    if avg_loss == 0.0:
        return 100.0
    if avg_gain == 0.0:
        return 0.0
    rs = avg_gain / avg_loss
    return float(100.0 - 100.0 / (1.0 + rs))


__all__ = [
    "DEFAULT_RSI_PERIOD",
    "rsi_series",
]
