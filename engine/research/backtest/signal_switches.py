"""The research entry signals as switches (v4 gene batch 2.6; research PART 4.1 ``use_sig_*``), the SIGNALS family.

A DNA of family SIGNALS carries ``signal_switches``: the research signals it trades. On every bar each switch says
+1 (a long entry), -1 (a short entry) or 0; the family enters long when some switch says +1 and none says -1, short
the other way round (research: "bias AND signals AND filters" -- signals are alternatives). Each switch is a small
causal state machine stepping bar i and nothing later; the oracle (strategies/signals.py) steps them bar by bar and
the kernel gets ``signal_arrays`` -- the SAME classes run over the whole series -- so the two agree to the last bit
by construction (as backtest/switch_gates.py). The live EA runs the same rules (deploy/templates/ea_live.mq5.j2).

Research names each signal and its parameters (engine/research/genes/registry.py); the fixed values here are
ARQVIO's (typical values of the research seeds, research APPENDIX-D), documented per signal:

    cci            CCI(14) of the typical price crosses up through -100 (long) / down through +100 (short)
    mom_break      close - previous close > 0.5 ATR(14) (long) / < -0.5 ATR(14) (short)
    engulfing      the bar's body engulfs the previous opposite-coloured body (long: bullish engulfing)
    inside_break   the previous bar was inside the one before it; the close breaks its high (long) / low (short)
    wick_rejection a lower wick >= 2 x the body with the close in the bar's upper third (long); mirrored (short)
    liqsweep       the low takes out the lowest low of the prior 20 bars and the close is back above it (long);
                   mirrored on the highs (short)
    fade           close < SMA(10) - 2 ATR(14) (long, fading the stretch) / close > SMA(10) + 2 ATR(14) (short)
    squeeze        the previous bar's Bollinger(20, 2) sat inside its Keltner(20, 1.5); the close breaks the upper
                   band (long) / lower band (short)
    three_soldiers three rising bullish bars, each body >= 0.4 ATR(14) (long); three falling bearish bars (short)
    rsi            RSI(14) crosses up through 30 (long) / down through 70 (short)
    bb             the close comes back inside Bollinger(20, 2) from below (long) / above (short)
    pin_bar        body <= 33 % of the range with the lower wick >= 2/3 of it (long); upper wick (short)
    williams       Williams %R(14) crosses up through -80 (long) / down through -20 (short)
    -- 2026-11-01, the last five --
    fib            the swing of the 50 bars before this one (lowest low to highest high): in an up leg (the high came
                   after the low) the bar dips to the 61.8 % retracement and closes back above it (long); in a down
                   leg the bar reaches up to it and closes back below it (short)
    atr_run        three rising closes in a row whose move (close - close 3 bars ago) exceeds 1.5 ATR(14) (long);
                   three falling closes, mirrored (short)
    vwap           the close crosses up (long) / down (short) through the day's VWAP: typical price weighted by the
                   bar's tick volume, anchored at 00:00 of the bar-time day (no signal while the day has no volume)
    orb            opening range: the high/low of the first 4 bars of each bar-time day; afterwards the first close
                   above it (long) or below it (short) that day, once per day
    kalman         a local-level Kalman filter of the close (process variance 0.01, measurement variance 1, so the
                   gain is scale-free); its slope turns up (long) / down (short); no signal before 20 bars

ATR(14) here is the simple mean of the true range (first bar: high - low), as switch_gates. No signal fires before
its full lookback. ``v`` is the bar's tick volume (only vwap reads it); "day" is ``time // 86400`` of the bar label.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

# Order matters: the sampler draws one coin per READY signal, in this order (new signals go last).
READY_SIGNALS: tuple[str, ...] = ("cci", "mom_break", "engulfing", "inside_break", "wick_rejection", "liqsweep",
                                  "fade", "squeeze", "three_soldiers", "rsi", "bb", "pin_bar", "williams",
                                  "fib", "atr_run", "vwap", "orb", "kalman")


class _Mean:
    def __init__(self, n: int) -> None:
        self.n = n
        self.q: deque[float] = deque(maxlen=n)

    def push(self, v: float) -> None:
        self.q.append(float(v))

    @property
    def full(self) -> bool:
        return len(self.q) == self.n

    def value(self) -> float:
        s = 0.0
        for v in self.q:
            s += v
        return s / self.n


class _Atr:
    def __init__(self, n: int = 14) -> None:
        self.m = _Mean(n)
        self.prev_close: float | None = None

    def push(self, h: float, lo: float, c: float) -> None:
        p = self.prev_close
        self.m.push(h - lo if p is None else max(h - lo, abs(h - p), abs(lo - p)))
        self.prev_close = c

    @property
    def full(self) -> bool:
        return self.m.full

    def value(self) -> float:
        return self.m.value()


def _bands(q: deque[float], k: float) -> tuple[float, float, float]:
    """(mid, lower, upper): SMA and population standard deviation of the window."""
    n = len(q)
    s = 0.0
    for v in q:
        s += v
    m = s / n
    ss = 0.0
    for v in q:
        ss += (v - m) * (v - m)
    sd = math.sqrt(ss / n)  # correctly rounded, as MQL5 MathSqrt (x ** 0.5 is pow)
    return m, m - k * sd, m + k * sd


class SigCci:
    def __init__(self) -> None:
        self.q: deque[float] = deque(maxlen=14)
        self.prev: float | None = None
        self.now: float | None = None

    def update(self, t: int, o: float, h: float, lo: float, c: float, v: float) -> None:
        self.q.append((h + lo + c) / 3.0)
        self.prev = self.now
        if len(self.q) < 14:
            self.now = None
            return
        s = 0.0
        for v in self.q:
            s += v
        m = s / 14.0
        md = 0.0
        for v in self.q:
            md += abs(v - m)
        md /= 14.0
        self.now = 0.0 if md == 0 else (self.q[-1] - m) / (0.015 * md)

    def signal(self) -> int:
        if self.prev is None or self.now is None:
            return 0
        if self.prev <= -100.0 < self.now:
            return 1
        if self.prev >= 100.0 > self.now:
            return -1
        return 0


class SigMomBreak:
    def __init__(self) -> None:
        self.atr = _Atr()
        self.prev_c: float | None = None
        self.move = 0.0

    def update(self, t: int, o: float, h: float, lo: float, c: float, v: float) -> None:
        self.move = 0.0 if self.prev_c is None else c - self.prev_c
        self.prev_c = c
        self.atr.push(h, lo, c)

    def signal(self) -> int:
        if not self.atr.full:
            return 0
        a = self.atr.value()
        if self.move > 0.5 * a:
            return 1
        if self.move < -0.5 * a:
            return -1
        return 0


class SigEngulfing:
    def __init__(self) -> None:
        self.bars: deque[tuple[float, float]] = deque(maxlen=2)  # (open, close)

    def update(self, t: int, o: float, h: float, lo: float, c: float, v: float) -> None:
        self.bars.append((o, c))

    def signal(self) -> int:
        if len(self.bars) < 2:
            return 0
        (o1, c1), (o, c) = self.bars
        if c1 < o1 and c > o and o <= c1 and c >= o1:
            return 1
        if c1 > o1 and c < o and o >= c1 and c <= o1:
            return -1
        return 0


class SigInsideBreak:
    def __init__(self) -> None:
        self.bars: deque[tuple[float, float, float]] = deque(maxlen=3)  # (high, low, close)

    def update(self, t: int, o: float, h: float, lo: float, c: float, v: float) -> None:
        self.bars.append((h, lo, c))

    def signal(self) -> int:
        if len(self.bars) < 3:
            return 0
        (h2, l2, _), (h1, l1, _), (_, _, c) = self.bars
        if not (h1 <= h2 and l1 >= l2):
            return 0
        if c > h1:
            return 1
        if c < l1:
            return -1
        return 0


class SigWickRejection:
    def __init__(self) -> None:
        self.bar: tuple[float, float, float, float] | None = None

    def update(self, t: int, o: float, h: float, lo: float, c: float, v: float) -> None:
        self.bar = (o, h, lo, c)

    def signal(self) -> int:
        if self.bar is None:
            return 0
        o, h, lo, c = self.bar
        rng = h - lo
        if rng <= 0:
            return 0
        body = abs(c - o)
        lower, upper = min(o, c) - lo, h - max(o, c)
        if lower >= 2.0 * body and c >= lo + 2.0 * rng / 3.0:
            return 1
        if upper >= 2.0 * body and c <= lo + rng / 3.0:
            return -1
        return 0


class SigLiqSweep:
    def __init__(self) -> None:
        self.hi: deque[float] = deque(maxlen=21)
        self.lo: deque[float] = deque(maxlen=21)
        self.close = 0.0

    def update(self, t: int, o: float, h: float, lo: float, c: float, v: float) -> None:
        self.hi.append(float(h))
        self.lo.append(float(lo))
        self.close = c

    def signal(self) -> int:
        if len(self.lo) < 21:
            return 0
        prior_lo = min(list(self.lo)[:-1])
        prior_hi = max(list(self.hi)[:-1])
        if self.lo[-1] < prior_lo and self.close > prior_lo:
            return 1
        if self.hi[-1] > prior_hi and self.close < prior_hi:
            return -1
        return 0


class SigFade:
    def __init__(self) -> None:
        self.m = _Mean(10)
        self.atr = _Atr()
        self.close = 0.0

    def update(self, t: int, o: float, h: float, lo: float, c: float, v: float) -> None:
        self.m.push(c)
        self.atr.push(h, lo, c)
        self.close = c

    def signal(self) -> int:
        if not self.atr.full:
            return 0
        mid, a = self.m.value(), self.atr.value()
        if self.close < mid - 2.0 * a:
            return 1
        if self.close > mid + 2.0 * a:
            return -1
        return 0


class SigSqueeze:
    def __init__(self) -> None:
        self.q: deque[float] = deque(maxlen=20)
        self.atr = _Atr(20)
        self.squeezed_prev = False
        self.squeezed = False
        self.close = 0.0
        self.lower = self.upper = 0.0

    def update(self, t: int, o: float, h: float, lo: float, c: float, v: float) -> None:
        self.q.append(float(c))
        self.atr.push(h, lo, c)
        self.close = c
        self.squeezed_prev = self.squeezed
        if len(self.q) < 20 or not self.atr.full:
            self.squeezed = False
            return
        mid, self.lower, self.upper = _bands(self.q, 2.0)
        a = self.atr.value()
        self.squeezed = self.lower > mid - 1.5 * a and self.upper < mid + 1.5 * a

    def signal(self) -> int:
        if not self.squeezed_prev or len(self.q) < 20:
            return 0
        if self.close > self.upper:
            return 1
        if self.close < self.lower:
            return -1
        return 0


class SigThreeSoldiers:
    def __init__(self) -> None:
        self.bars: deque[tuple[float, float]] = deque(maxlen=3)
        self.atr = _Atr()

    def update(self, t: int, o: float, h: float, lo: float, c: float, v: float) -> None:
        self.bars.append((o, c))
        self.atr.push(h, lo, c)

    def signal(self) -> int:
        if len(self.bars) < 3 or not self.atr.full:
            return 0
        need = 0.4 * self.atr.value()
        b = list(self.bars)
        if all(c - o >= need for o, c in b) and b[1][1] > b[0][1] and b[2][1] > b[1][1]:
            return 1
        if all(o - c >= need for o, c in b) and b[1][1] < b[0][1] and b[2][1] < b[1][1]:
            return -1
        return 0


class SigRsi:
    def __init__(self) -> None:
        self.gain, self.loss = _Mean(14), _Mean(14)
        self.prev_c: float | None = None
        self.prev: float | None = None
        self.now: float | None = None

    def update(self, t: int, o: float, h: float, lo: float, c: float, v: float) -> None:
        if self.prev_c is not None:
            ch = c - self.prev_c
            self.gain.push(ch if ch > 0 else 0.0)
            self.loss.push(-ch if ch < 0 else 0.0)
        self.prev_c = c
        self.prev = self.now
        if not self.gain.full:
            self.now = None
            return
        g, lw = self.gain.value(), self.loss.value()
        self.now = 100.0 if lw == 0 else 100.0 - 100.0 / (1.0 + g / lw)

    def signal(self) -> int:
        if self.prev is None or self.now is None:
            return 0
        if self.prev <= 30.0 < self.now:
            return 1
        if self.prev >= 70.0 > self.now:
            return -1
        return 0


class SigBb:
    def __init__(self) -> None:
        self.q: deque[float] = deque(maxlen=20)
        self.prev: tuple[float, float, float] | None = None  # (close, lower, upper) of the previous bar
        self.now: tuple[float, float, float] | None = None

    def update(self, t: int, o: float, h: float, lo: float, c: float, v: float) -> None:
        self.q.append(float(c))
        self.prev = self.now
        if len(self.q) < 20:
            self.now = None
            return
        _, lower, upper = _bands(self.q, 2.0)
        self.now = (c, lower, upper)

    def signal(self) -> int:
        if self.prev is None or self.now is None:
            return 0
        (pc, pl, pu), (c, lw, up) = self.prev, self.now
        if pc < pl and c >= lw:
            return 1
        if pc > pu and c <= up:
            return -1
        return 0


class SigPinBar:
    def __init__(self) -> None:
        self.bar: tuple[float, float, float, float] | None = None

    def update(self, t: int, o: float, h: float, lo: float, c: float, v: float) -> None:
        self.bar = (o, h, lo, c)

    def signal(self) -> int:
        if self.bar is None:
            return 0
        o, h, lo, c = self.bar
        rng = h - lo
        if rng <= 0 or abs(c - o) > 0.33 * rng:
            return 0
        if min(o, c) - lo >= 2.0 * rng / 3.0:
            return 1
        if h - max(o, c) >= 2.0 * rng / 3.0:
            return -1
        return 0


class SigWilliams:
    def __init__(self) -> None:
        self.hi: deque[float] = deque(maxlen=14)
        self.lo: deque[float] = deque(maxlen=14)
        self.prev: float | None = None
        self.now: float | None = None

    def update(self, t: int, o: float, h: float, lo: float, c: float, v: float) -> None:
        self.hi.append(float(h))
        self.lo.append(float(lo))
        self.prev = self.now
        if len(self.hi) < 14:
            self.now = None
            return
        hh, ll = max(self.hi), min(self.lo)
        self.now = -50.0 if hh == ll else -100.0 * (hh - c) / (hh - ll)

    def signal(self) -> int:
        if self.prev is None or self.now is None:
            return 0
        if self.prev <= -80.0 < self.now:
            return 1
        if self.prev >= -20.0 > self.now:
            return -1
        return 0


class SigFib:
    LOOKBACK = 50
    LEVEL = 0.618

    def __init__(self) -> None:
        self.hi: deque[float] = deque(maxlen=self.LOOKBACK + 1)
        self.lo: deque[float] = deque(maxlen=self.LOOKBACK + 1)
        self.close = 0.0

    def update(self, t: int, o: float, h: float, lo: float, c: float, v: float) -> None:
        self.hi.append(float(h))
        self.lo.append(float(lo))
        self.close = c

    def signal(self) -> int:
        if len(self.hi) <= self.LOOKBACK:
            return 0
        his, los = list(self.hi), list(self.lo)
        ih, hh, il, ll = 0, his[0], 0, los[0]
        for k in range(1, self.LOOKBACK):  # the oldest extreme wins a tie
            if his[k] > hh:
                ih, hh = k, his[k]
            if los[k] < ll:
                il, ll = k, los[k]
        if hh <= ll:
            return 0
        h, lo, c = his[-1], los[-1], self.close
        if ih > il:
            level = hh - self.LEVEL * (hh - ll)
            return 1 if lo <= level < c else 0
        if il > ih:
            level = ll + self.LEVEL * (hh - ll)
            return -1 if h >= level > c else 0
        return 0


class SigAtrRun:
    def __init__(self) -> None:
        self.q: deque[float] = deque(maxlen=4)
        self.atr = _Atr()

    def update(self, t: int, o: float, h: float, lo: float, c: float, v: float) -> None:
        self.q.append(float(c))
        self.atr.push(h, lo, c)

    def signal(self) -> int:
        if len(self.q) < 4 or not self.atr.full:
            return 0
        q, need = self.q, 1.5 * self.atr.value()
        if q[1] > q[0] and q[2] > q[1] and q[3] > q[2] and q[3] - q[0] > need:
            return 1
        if q[1] < q[0] and q[2] < q[1] and q[3] < q[2] and q[0] - q[3] > need:
            return -1
        return 0


class DayVwap:
    """The day's VWAP so far: sum(typical price x tick volume) / sum(tick volume), oldest bar first; None while
    the day has no volume. Shared with the vwap bias (switch_gates.BiasVwap)."""

    def __init__(self) -> None:
        self.day: int | None = None
        self.pv = self.vv = 0.0

    def push(self, t: int, h: float, lo: float, c: float, v: float) -> float | None:
        day = int(t) // 86400
        if day != self.day:
            self.day, self.pv, self.vv = day, 0.0, 0.0
        self.pv += (h + lo + c) / 3.0 * v
        self.vv += v
        return self.pv / self.vv if self.vv > 0.0 else None


class SigVwap:
    def __init__(self) -> None:
        self.vw = DayVwap()
        self.prev: tuple[float, float] | None = None
        self.now: tuple[float, float] | None = None

    def update(self, t: int, o: float, h: float, lo: float, c: float, v: float) -> None:
        w = self.vw.push(t, h, lo, c, v)
        self.prev = self.now
        self.now = None if w is None else (c, w)

    def signal(self) -> int:
        if self.prev is None or self.now is None:
            return 0
        (pc, pw), (c, w) = self.prev, self.now
        if pc <= pw and c > w:
            return 1
        if pc >= pw and c < w:
            return -1
        return 0


class SigOrb:
    RANGE_BARS = 4

    def __init__(self) -> None:
        self.day: int | None = None
        self.k = 0
        self.hi = self.lo = 0.0
        self.fired = False
        self.vote = 0

    def update(self, t: int, o: float, h: float, lo: float, c: float, v: float) -> None:
        day = int(t) // 86400
        if day != self.day:
            self.day, self.k, self.fired = day, 0, False
        self.vote = 0
        if self.k < self.RANGE_BARS:
            self.hi = h if self.k == 0 else max(self.hi, h)
            self.lo = lo if self.k == 0 else min(self.lo, lo)
        elif not self.fired:
            if c > self.hi:
                self.vote, self.fired = 1, True
            elif c < self.lo:
                self.vote, self.fired = -1, True
        self.k += 1

    def signal(self) -> int:
        return self.vote


class SigKalman:
    Q, R, WARMUP = 0.01, 1.0, 20

    def __init__(self) -> None:
        self.x: float | None = None
        self.p = 1.0
        self.n = 0
        self.d_prev: float | None = None
        self.d_now: float | None = None

    def update(self, t: int, o: float, h: float, lo: float, c: float, v: float) -> None:
        self.n += 1
        if self.x is None:
            self.x = c
            return
        pp = self.p + self.Q
        k = pp / (pp + self.R)
        x = self.x + k * (c - self.x)
        self.p = (1.0 - k) * pp
        self.d_prev, self.d_now = self.d_now, x - self.x
        self.x = x

    def signal(self) -> int:
        if self.n < self.WARMUP or self.d_prev is None or self.d_now is None:
            return 0
        if self.d_now > 0.0 >= self.d_prev:
            return 1
        if self.d_now < 0.0 <= self.d_prev:
            return -1
        return 0


SIGNAL_CLASSES = {"cci": SigCci, "mom_break": SigMomBreak, "engulfing": SigEngulfing, "inside_break": SigInsideBreak,
                  "wick_rejection": SigWickRejection, "liqsweep": SigLiqSweep, "fade": SigFade,
                  "squeeze": SigSqueeze, "three_soldiers": SigThreeSoldiers, "rsi": SigRsi, "bb": SigBb,
                  "pin_bar": SigPinBar, "williams": SigWilliams, "fib": SigFib, "atr_run": SigAtrRun,
                  "vwap": SigVwap, "orb": SigOrb, "kalman": SigKalman}
assert tuple(SIGNAL_CLASSES) == READY_SIGNALS


def make(names: tuple[str, ...]) -> list:
    out = []
    for name in names:
        if name not in SIGNAL_CLASSES:
            raise ValueError(f"signal switch {name!r} is not READY (READY: {READY_SIGNALS})")
        out.append(SIGNAL_CLASSES[name]())
    return out


def combine(votes: list[int]) -> int:
    """+1 when some switch says long and none says short; -1 the other way round; else 0."""
    up, down = any(v > 0 for v in votes), any(v < 0 for v in votes)
    return 1 if up and not down else -1 if down and not up else 0


def signal_arrays(time: np.ndarray, open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                  names: tuple[str, ...], volume: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(long_signal, short_signal) as float64 1.0 / 0.0 for every bar, from the classes the oracle steps.
    ``volume`` (the bars' tick volume) is required when a switch reads it (vwap)."""
    n = len(close)
    long_s = np.zeros(n, dtype=np.float64)
    short_s = np.zeros(n, dtype=np.float64)
    sigs = make(tuple(names))
    if volume is None and "vwap" in names:
        raise ValueError("the vwap signal needs the bars' tick volume")
    t = np.asarray(time, dtype=np.int64)
    o, h, lo, c = (np.asarray(x, dtype=np.float64) for x in (open_, high, low, close))
    vol = np.zeros(n, dtype=np.float64) if volume is None else np.asarray(volume, dtype=np.float64)
    for i in range(n):
        for s in sigs:
            s.update(int(t[i]), float(o[i]), float(h[i]), float(lo[i]), float(c[i]), float(vol[i]))
        v = combine([s.signal() for s in sigs])
        if v > 0:
            long_s[i] = 1.0
        elif v < 0:
            short_s[i] = 1.0
    return long_s, short_s


__all__ = ["READY_SIGNALS", "SIGNAL_CLASSES", "DayVwap", "combine", "make", "signal_arrays"]
