"""Average True Range.

ATR is the input to the SL/TP distance genes: a stop placed a fixed
number of price units away is a different stop on gold than on EURUSD,
and a different stop in a quiet week than a volatile one. Expressing it
as a multiple of ATR makes ``sl_atr`` mean the same thing everywhere.

True range is the registry's definition (registry.py ``_true_range``):

    TR(t) = max(high - low, |high - close[t-1]|, |low - close[t-1]|)

SMA-of-TR == MT5 iATR semantics; Wilder's intentionally NOT used. See G9.

That is the whole reason this file is not the textbook ATR. MT5's
``iATR()`` averages true range over a fixed window, and ATR sets every
stop distance in the system, so an S6 MQL5 port that used Wilder's
smoothing would drift from the Python backtest on every single trade --
a parity failure that compounds rather than cancels. A fixed window is
also honest about causality: the value at bar t depends on exactly 14
bars, not on an unbounded recursive tail seeded somewhere off-screen.

WARMUP. ATR is undefined for the first 13 bars: the 14-bar window is not
full. Undefined is returned as ``None``, never as a partial average --
a stop distance derived from 6 bars is not a 14-bar ATR, and callers
must not run ATR-derived logic there at all.

LOOK-AHEAD. Every function reads bars at or before ``end_index`` and
never past it. The slice is taken explicitly rather than trusted to the
caller.
"""
from __future__ import annotations

import numpy as np

from engine.core.errors import DataError

DEFAULT_ATR_PERIOD = 14


def true_range_series(
    high: np.ndarray, low: np.ndarray, close: np.ndarray,
) -> np.ndarray:
    """TR for every bar. TR[0] falls back to high-low (no previous close)."""
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    if high.shape != low.shape or high.shape != close.shape:
        raise DataError("high/low/close must be the same length")
    if high.size == 0:
        return np.zeros(0, dtype=np.float64)

    tr = np.empty(high.size, dtype=np.float64)
    tr[0] = high[0] - low[0]
    if high.size > 1:
        prev_close = close[:-1]
        tr[1:] = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - prev_close),
                np.abs(low[1:] - prev_close),
            ),
        )
    return tr


def sma_atr_series(tr: np.ndarray, period: int = DEFAULT_ATR_PERIOD) -> np.ndarray:
    """Simple moving average of true range -- MT5 ``iATR`` semantics.

    Index ``i`` holds ``mean(TR[i - period + 1 .. i])``. Indices before
    ``period - 1`` are NaN: the window is not full there.
    """
    if period < 1:
        raise DataError(f"atr period must be >= 1; got {period}")
    tr = np.asarray(tr, dtype=np.float64)
    out = np.full(tr.size, np.nan, dtype=np.float64)
    if tr.size < period:
        return out
    # Cumulative-sum window: out[i] = (csum[i] - csum[i - period]) / period.
    csum = np.cumsum(tr, dtype=np.float64)
    out[period - 1] = csum[period - 1] / period
    if tr.size > period:
        out[period:] = (csum[period:] - csum[:-period]) / period
    return out


def atr_at(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    end_index: int,
    period: int = DEFAULT_ATR_PERIOD,
) -> float | None:
    """ATR as of ``end_index`` inclusive, or None if it is undefined there.

    None means "undefined", and the caller must treat it as such. A
    fallback (a default distance, some multiple of point) would put a
    fabricated number into every warmup trade. None is also returned for
    a degenerate zero ATR: a flat series gives a zero-distance stop,
    which is not a stop.
    """
    if end_index < 0:
        return None
    stop = end_index + 1
    if stop < period:
        return None
    # Sliced to end_index INCLUSIVE: bar end_index + 1 is not in the array.
    tr = true_range_series(high[:stop], low[:stop], close[:stop])
    series = sma_atr_series(tr, period=period)
    value = float(series[-1])
    if not np.isfinite(value) or value <= 0.0:
        return None
    return value


__all__ = [
    "DEFAULT_ATR_PERIOD",
    "atr_at",
    "sma_atr_series",
    "true_range_series",
]
