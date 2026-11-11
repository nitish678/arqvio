"""The spread-regime entry gate: a bar's spread against the spread of its own recent past.

The fixed gate (A5) refused entries on bars whose spread exceeded the p95 of the symbol's calibration window --
the last ~28 days of ticks. Applied to a year of history it judged every bar by today's spread: XAGUSD M5's
Feb-Sep 2026 bars sat at 21-35 points against an 18-point cap, 99-100% of them were above it, 881 entries were
refused and the equity curve went flat for eight months (owner's vault, 2026-10-26; ARQVIO_V4.md section 10.1).

The regime gate asks the question the fixed gate meant to ask -- "has the spread blown out?" -- relative to the
bar's own era: an entry is refused when ``spread[i] > multiple * median(spread[i-window .. i-1])``. Only past
bars enter the median (no look-ahead). Until half a window of past bars exists the cap is NaN (no gate). The
median is floored at ``FLOOR_POINTS`` so a zero-spread stretch cannot turn every positive spread into a refusal.

The slow oracle compares ``spread[i] > cap[i]``; the kernel receives ``spread - cap`` against a threshold of 0.0,
which is the same comparison exactly (IEEE subtraction is correctly rounded, so a - b > 0 iff a > b for finite
values; NaN compares False either way). The live EA computes the same median from its own closed bars.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

DEFAULT_DAYS = 20
DEFAULT_MULTIPLE = 3.0
FLOOR_POINTS = 1.0
MIN_FILL = 0.5

TF_MINUTES = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H2": 120, "H4": 240, "D1": 1440}


def window_bars(timeframe: str, days: int = DEFAULT_DAYS) -> int:
    """Bars in ``days`` calendar days of a 24-hour market (at least 20)."""
    tf = str(timeframe).upper().replace("TIMEFRAME.", "")
    minutes = TF_MINUTES.get(tf)
    if minutes is None:
        raise ValueError(f"spread regime: unknown timeframe {timeframe!r}")
    return max(20, int(days) * 1440 // minutes)


@njit(cache=True)
def _rolling_past_median(v: np.ndarray, window: int, min_count: int) -> np.ndarray:
    """out[i] = median(v[i-window .. i-1]) over integer values >= 0; NaN with fewer than min_count past values.

    A Fenwick tree over the value range gives each order statistic in O(log K)."""
    n = v.size
    out = np.full(n, np.nan)
    if n == 0:
        return out
    k_max = 0
    for i in range(n):
        if v[i] > k_max:
            k_max = v[i]
    size = k_max + 1
    tree = np.zeros(size + 1, dtype=np.int64)
    log = 1
    while (1 << log) <= size:
        log += 1
    count = 0
    for i in range(n):
        if count >= min_count:
            # k-th smallest (1-based) via Fenwick descent
            a = 0.0
            for which in range(2):
                k = (count + 1) // 2 if which == 0 else count // 2 + 1
                pos = 0
                rem = k
                step = 1 << log
                while step > 0:
                    nxt = pos + step
                    if nxt <= size and tree[nxt] < rem:
                        pos = nxt
                        rem -= tree[nxt]
                    step >>= 1
                a += pos  # value index = pos (0-based value)
            out[i] = a / 2.0
        # add v[i]; drop v[i-window]
        j = v[i] + 1
        while j <= size:
            tree[j] += 1
            j += j & (-j)
        count += 1
        if i - window >= 0:
            j = v[i - window] + 1
            while j <= size:
                tree[j] -= 1
                j += j & (-j)
            count -= 1
    return out


def regime_cap(spread: np.ndarray, *, window: int, multiple: float = DEFAULT_MULTIPLE) -> np.ndarray:
    """Per-bar cap in points: ``multiple * max(median of the previous window bars, FLOOR_POINTS)``; NaN = no gate."""
    if window < 1:
        raise ValueError("spread regime: window must be >= 1")
    if not (math.isfinite(multiple) and multiple > 0):
        raise ValueError("spread regime: multiple must be > 0")
    s = np.asarray(spread, dtype=np.float64)
    v = np.clip(np.nan_to_num(np.rint(s), nan=0.0), 0, None).astype(np.int64)
    med = _rolling_past_median(v, int(window), max(1, int(math.ceil(window * MIN_FILL))))
    return float(multiple) * np.maximum(med, FLOOR_POINTS)  # NaN stays NaN


def gate_excess(spread: np.ndarray, cap: np.ndarray) -> np.ndarray:
    """The kernel's gate input: ``spread - cap`` (refuse when > 0.0)."""
    return np.asarray(spread, dtype=np.float64) - np.asarray(cap, dtype=np.float64)


def blocked_by_month(times: np.ndarray, spread: np.ndarray, cap: np.ndarray | float | None) -> dict[str, int]:
    """Bars per UTC month on which the gate would refuse an entry (the inspector's "blocked entries" context)."""
    t = np.asarray(times, dtype=np.int64)
    s = np.asarray(spread, dtype=np.float64)
    if cap is None:
        return {}
    c = np.broadcast_to(np.asarray(cap, dtype=np.float64), s.shape)
    over = s > c
    out: dict[str, int] = {}
    months = t.astype("datetime64[s]").astype("datetime64[M]").astype(str)
    for m, o in zip(months, over, strict=True):
        if o:
            out[m] = out.get(m, 0) + 1
    return out


def fill_spread_points(spread: np.ndarray) -> np.ndarray:
    """The spread in points charged on a fill at each bar: the signal candle's (bar i-1's) own historical spread
    (ARQVIO_V4.md 10.1 cost; the MT5 tester charges the spread of the bar the order was decided on). NaN where the
    bar's spread is unknown -- the first bar, or a recorded spread of 0 (a zero-spread history carries no cost
    information) -- and the engine falls back to the calibrated hourly bucket there."""
    v = np.asarray(spread, dtype=np.float64)
    out = np.full(v.size, np.nan, dtype=np.float64)
    if v.size > 1:
        prev = v[:-1]
        out[1:] = np.where(prev > 0.0, prev, np.nan)
    return out


__all__ = ["DEFAULT_DAYS", "DEFAULT_MULTIPLE", "FLOOR_POINTS", "blocked_by_month", "fill_spread_points", "gate_excess", "regime_cap",
           "window_bars"]
