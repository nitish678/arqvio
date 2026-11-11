"""ARQVIO entry gates: whether an entry may open on a signal bar, per direction.

Two genes (engine/research/search/dna.py), both off by default:

    session    the signal bar's UTC hour lies inside a named window (end exclusive):
               ASIA 00-08, LONDON 07-16, NEWYORK 12-21, LONDON_NY 07-21
    trend_ema  an EMA of the close (seeded with the first close; alpha = 2 / (period + 1)) sets the side: a long
               only when the close is above it, a short only below; no entry before the EMA has `period` bars

Semantics, the same in the oracle (strategies/gated.py), the kernel (kernel.run_candidate) and the live EA: a gate
acts on an OPENING intent only. When it refuses one, the intent is dropped and the strategy is flat -- exactly as
if the signal had not fired. Closing intents are never gated. Every value is causal: bar i's gate reads bars 0..i.

The EMA is one recursion, ``e = e + alpha * (close - e)``, evaluated in the same order everywhere (``TrendEma`` for
the bar-by-bar oracle, ``gate_arrays`` for the kernel's precomputed arrays), so the two agree to the last bit.
"""
from __future__ import annotations

import numpy as np

SESSIONS: dict[str, tuple[int, int]] = {"ASIA": (0, 8), "LONDON": (7, 16), "NEWYORK": (12, 21), "LONDON_NY": (7, 21)}


def session_open(hour: int, session: str) -> bool:
    if not session:
        return True
    lo, hi = SESSIONS[session]
    return lo <= hour < hi


class TrendEma:
    """The trend EMA one bar at a time (the oracle's view: it never reads a bar ahead)."""

    def __init__(self, period: int) -> None:
        self.period = int(period)
        self.alpha = 2.0 / (self.period + 1.0)
        self.value = 0.0
        self.n = 0

    def update(self, close: float) -> None:
        self.value = float(close) if self.n == 0 else self.value + self.alpha * (float(close) - self.value)
        self.n += 1

    def allows(self, direction: int, close: float) -> bool:
        if self.n < self.period:
            return False
        return close > self.value if direction > 0 else close < self.value


def gate_arrays(hours: np.ndarray, close: np.ndarray, session: str, trend_ema: int) -> tuple[np.ndarray, np.ndarray]:
    """(long_ok, short_ok) for every bar: the gates the oracle applies bar by bar, precomputed for the kernel.
    ``hours`` is each bar's UTC hour, (time // 3600) % 24 -- prep's bar_hour_index."""
    t = np.asarray(hours, dtype=np.int64)
    c = np.asarray(close, dtype=np.float64)
    long_ok = np.ones(t.size, dtype=np.bool_)
    short_ok = np.ones(t.size, dtype=np.bool_)
    if session:
        lo, hi = SESSIONS[session]
        inside = (t >= lo) & (t < hi)
        long_ok &= inside
        short_ok &= inside
    if trend_ema:
        ema = TrendEma(trend_ema)
        up = np.zeros(t.size, dtype=np.bool_)
        down = np.zeros(t.size, dtype=np.bool_)
        for i in range(t.size):
            ema.update(c[i])
            up[i] = ema.allows(1, float(c[i]))
            down[i] = ema.allows(-1, float(c[i]))
        long_ok &= up
        short_ok &= down
    return long_ok, short_ok


__all__ = ["SESSIONS", "TrendEma", "gate_arrays", "session_open"]
