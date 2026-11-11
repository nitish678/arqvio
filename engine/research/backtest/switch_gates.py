"""Bias and filter switches as entry gates (v4 gene batch B; worker4/02_PLAN.md section 4).

A BIAS switch sets the side an opening may take; a FILTER switch vetoes openings on a bar. Each is a small causal
state machine: ``update`` takes bar i (and nothing later), ``allows(direction)`` answers for bar i. The oracle
(strategies/gated.py) steps them one bar at a time; the kernel gets ``switch_arrays`` -- the SAME classes run over
the whole series -- so the two agree to the last bit by construction, exactly like the trend EMA (gates.py).
Semantics are the session/trend gates': only an OPENING is gated; a refused opening is dropped and the strategy is
flat. Parameters are fixed per switch (below) and documented; the live EA runs the same rules.

Bias (long when, short when):
    sma            close > SMA(100) of the close            | close < SMA(100)
    momentum       close > close 20 bars ago                | close < close 20 bars ago
    donchian_mid   close > midpoint of the 20-bar high/low  | close < that midpoint
    -- batch C (2026-11-01) --
    rsi            RSI(14) > 50                             | RSI(14) < 50
    adx            +DI(14) > -DI(14)                        | -DI(14) > +DI(14)   (simple means of DM and TR)
    chandelier     close > HH(22) - 3 x ATR(22)             | close < LL(22) + 3 x ATR(22)
    daily_mid      close > midpoint of the previous UTC day's high/low | close < it
    htf            close > SMA(200) of the close            | close < SMA(200)  (the slow trend on the bred bars)
    -- 2026-11-01, the last four --
    trailing       an ATR trailing stop (3 x ATR(14)) says up | says down: the stop trails the close by 3 ATR and
                   flips side when a close crosses it (it starts on the up side at its first full ATR)
    market_struct  the last 10 bars made a higher high AND a higher low than the 10 before | lower high AND lower low
    vwap           close > the day's VWAP (tick-volume weighted typical price, from 00:00 of the bar day) | close <
    psar           Parabolic SAR (step 0.02, max 0.2; Wilder) in an up trend | in a down trend
Filters (an opening is allowed when):
    no_open_friday the bar is not on a Friday (UTC)
    volatility     ATR(14) / ATR(100) lies in [0.5, 2.0] (simple averages of the true range)
    rsi            RSI(14) (simple averages of gains/losses) lies in [30, 70]
    -- batch D (2026-11-01) --
    bb             close inside SMA(20) +- 2 population standard deviations of the 20 closes
    cci            CCI(20) of the typical price lies in [-100, 100]
    consec         not the 4th rising close in a row for a long (4th falling for a short): no chasing a run
    adr_exhaust    today's (UTC) range so far < 80 % of the mean range of the last 14 complete days
    adx            ADX(14) >= 20 (a trend to trade)
    keltner        close inside SMA(20) +- 1.5 x ATR(20)
    -- 2026-11-01 --
    sma            SMA(10) and SMA(30) of the close are at least 0.25 x ATR(14) apart (a trend, not a tangle)
    receding       volatility is not receding: the mean true range of the last 5 bars >= that of the 5 before
    doji           the bar is not a doji: its body is more than 10 % of its range (a zero range is a doji)
    regime         Choppiness(14) < 61.8 (trending): sum of 14 true ranges < (HH14 - LL14) x 14^0.618
    event          no price shock: the bar's range <= 3 x ATR of the 14 bars before it and its open gaps from the
                   previous close by <= 1 x that ATR (research event_atr_mult / gap_atr_mult; no news calendar)
No gate allows an opening before it has its full lookback. The research names these genes and their parameters
(engine/research/genes/registry.py); the fixed values above are ARQVIO's, as for batch B.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from engine.research.backtest.signal_switches import DayVwap

# Order matters: the sampler draws one coin per READY switch, in this order.
READY_BIAS: tuple[str, ...] = ("sma", "momentum", "donchian_mid", "rsi", "adx", "chandelier", "daily_mid", "htf",
                               "trailing", "market_struct", "vwap", "psar")
READY_FILTERS: tuple[str, ...] = ("no_open_friday", "volatility", "rsi", "bb", "cci", "consec", "adr_exhaust",
                                  "adx", "keltner", "sma", "receding", "doji", "regime", "event")


class _Mean:
    """A rolling mean of the last n values: the sum is recomputed over the window (no drifting running sum)."""

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


class BiasSma:
    def __init__(self) -> None:
        self.m = _Mean(100)
        self.close = 0.0

    def update(self, t: int, o: float, h: float, lo: float, c: float, v: float) -> None:
        self.m.push(c)
        self.close = c

    def allows(self, d: int) -> bool:
        if not self.m.full:
            return False
        v = self.m.value()
        return self.close > v if d > 0 else self.close < v


class BiasMomentum:
    def __init__(self) -> None:
        self.q: deque[float] = deque(maxlen=21)

    def update(self, t: int, o: float, h: float, lo: float, c: float, v: float) -> None:
        self.q.append(float(c))

    def allows(self, d: int) -> bool:
        if len(self.q) < 21:
            return False
        return self.q[-1] > self.q[0] if d > 0 else self.q[-1] < self.q[0]


class BiasDonchianMid:
    def __init__(self) -> None:
        self.hi: deque[float] = deque(maxlen=20)
        self.lo: deque[float] = deque(maxlen=20)
        self.close = 0.0

    def update(self, t: int, o: float, h: float, lo: float, c: float, v: float) -> None:
        self.hi.append(float(h))
        self.lo.append(float(lo))
        self.close = c

    def allows(self, d: int) -> bool:
        if len(self.hi) < 20:
            return False
        mid = (max(self.hi) + min(self.lo)) / 2.0
        return self.close > mid if d > 0 else self.close < mid


class FilterNoOpenFriday:
    def __init__(self) -> None:
        self.friday = False

    def update(self, t: int, o: float, h: float, lo: float, c: float, v: float) -> None:
        self.friday = ((int(t) // 86400) + 4) % 7 == 5  # 1970-01-01 was a Thursday: day 0 -> weekday 4

    def allows(self, d: int) -> bool:
        return not self.friday


class FilterVolatility:
    def __init__(self) -> None:
        self.fast, self.slow = _Mean(14), _Mean(100)
        self.prev_close: float | None = None

    def update(self, t: int, o: float, h: float, lo: float, c: float, v: float) -> None:
        tr = h - lo if self.prev_close is None else max(h - lo, abs(h - self.prev_close), abs(lo - self.prev_close))
        self.fast.push(tr)
        self.slow.push(tr)
        self.prev_close = c

    def allows(self, d: int) -> bool:
        if not self.slow.full:
            return False
        s = self.slow.value()
        if s <= 0:
            return False
        r = self.fast.value() / s
        return 0.5 <= r <= 2.0


class FilterRsi:
    def __init__(self) -> None:
        self.gain, self.loss = _Mean(14), _Mean(14)
        self.prev: float | None = None

    def update(self, t: int, o: float, h: float, lo: float, c: float, v: float) -> None:
        if self.prev is not None:
            ch = c - self.prev
            self.gain.push(ch if ch > 0 else 0.0)
            self.loss.push(-ch if ch < 0 else 0.0)
        self.prev = c

    def allows(self, d: int) -> bool:
        if not self.gain.full:
            return False
        g, lo = self.gain.value(), self.loss.value()
        rsi = 100.0 if lo == 0 else 100.0 - 100.0 / (1.0 + g / lo)
        return 30.0 <= rsi <= 70.0


class _TrueRange:
    """The true range of each bar (the first bar: high - low), as FilterVolatility."""

    def __init__(self) -> None:
        self.prev_close: float | None = None

    def push(self, h: float, lo: float, c: float) -> float:
        p = self.prev_close
        tr = h - lo if p is None else max(h - lo, abs(h - p), abs(lo - p))
        self.prev_close = c
        return tr


def _rsi(gain: _Mean, loss: _Mean) -> float:
    g, lo = gain.value(), loss.value()
    return 100.0 if lo == 0 else 100.0 - 100.0 / (1.0 + g / lo)


class BiasRsi(FilterRsi):
    def allows(self, d: int) -> bool:
        if not self.gain.full:
            return False
        r = _rsi(self.gain, self.loss)
        return r > 50.0 if d > 0 else r < 50.0


class _Dmi:
    """+DI, -DI and ADX(14) from simple means (DM and TR over 14, DX over 14)."""

    def __init__(self, n: int = 14) -> None:
        self.n = n
        self.pdm, self.mdm, self.tr, self.dx = _Mean(n), _Mean(n), _Mean(n), _Mean(n)
        self.prev: tuple[float, float, float] | None = None
        self.pdi = self.mdi = 0.0

    def update(self, t: int, o: float, h: float, lo: float, c: float, v: float) -> None:
        if self.prev is not None:
            ph, pl, pc = self.prev
            up, down = h - ph, pl - lo
            self.pdm.push(up if up > down and up > 0 else 0.0)
            self.mdm.push(down if down > up and down > 0 else 0.0)
            self.tr.push(max(h - lo, abs(h - pc), abs(lo - pc)))
            if self.tr.full:
                atr = self.tr.value()
                self.pdi = 100.0 * self.pdm.value() / atr if atr > 0 else 0.0
                self.mdi = 100.0 * self.mdm.value() / atr if atr > 0 else 0.0
                s = self.pdi + self.mdi
                self.dx.push(100.0 * abs(self.pdi - self.mdi) / s if s > 0 else 0.0)
        self.prev = (h, lo, c)


class BiasAdx(_Dmi):
    def allows(self, d: int) -> bool:
        if not self.tr.full:
            return False
        return self.pdi > self.mdi if d > 0 else self.mdi > self.pdi


class FilterAdx(_Dmi):
    def allows(self, d: int) -> bool:
        return self.dx.full and self.dx.value() >= 20.0


class BiasChandelier:
    def __init__(self) -> None:
        self.hi: deque[float] = deque(maxlen=22)
        self.lo: deque[float] = deque(maxlen=22)
        self.atr, self.trs = _Mean(22), _TrueRange()
        self.close = 0.0

    def update(self, t: int, o: float, h: float, lo: float, c: float, v: float) -> None:
        self.hi.append(float(h))
        self.lo.append(float(lo))
        self.atr.push(self.trs.push(h, lo, c))
        self.close = c

    def allows(self, d: int) -> bool:
        if not self.atr.full:
            return False
        a = self.atr.value()
        return self.close > max(self.hi) - 3.0 * a if d > 0 else self.close < min(self.lo) + 3.0 * a


class BiasDailyMid:
    def __init__(self) -> None:
        self.day = None
        self.day_hi = self.day_lo = 0.0
        self.mid: float | None = None
        self.close = 0.0

    def update(self, t: int, o: float, h: float, lo: float, c: float, v: float) -> None:
        day = int(t) // 86400
        if self.day is None or day != self.day:
            if self.day is not None:
                self.mid = (self.day_hi + self.day_lo) / 2.0
            self.day, self.day_hi, self.day_lo = day, h, lo
        else:
            self.day_hi, self.day_lo = max(self.day_hi, h), min(self.day_lo, lo)
        self.close = c

    def allows(self, d: int) -> bool:
        if self.mid is None:
            return False
        return self.close > self.mid if d > 0 else self.close < self.mid


class BiasHtf(BiasSma):
    def __init__(self) -> None:
        self.m = _Mean(200)
        self.close = 0.0


class FilterBb:
    def __init__(self) -> None:
        self.q: deque[float] = deque(maxlen=20)

    def update(self, t: int, o: float, h: float, lo: float, c: float, v: float) -> None:
        self.q.append(float(c))

    def allows(self, d: int) -> bool:
        if len(self.q) < 20:
            return False
        s = 0.0
        for v in self.q:
            s += v
        m = s / 20.0
        ss = 0.0
        for v in self.q:
            ss += (v - m) * (v - m)
        sd = (ss / 20.0) ** 0.5
        c = self.q[-1]
        return m - 2.0 * sd <= c <= m + 2.0 * sd


class FilterCci:
    def __init__(self) -> None:
        self.q: deque[float] = deque(maxlen=20)

    def update(self, t: int, o: float, h: float, lo: float, c: float, v: float) -> None:
        self.q.append((h + lo + c) / 3.0)

    def allows(self, d: int) -> bool:
        if len(self.q) < 20:
            return False
        s = 0.0
        for v in self.q:
            s += v
        m = s / 20.0
        md = 0.0
        for v in self.q:
            md += abs(v - m)
        md /= 20.0
        cci = 0.0 if md == 0 else (self.q[-1] - m) / (0.015 * md)
        return -100.0 <= cci <= 100.0


class FilterConsec:
    def __init__(self) -> None:
        self.q: deque[float] = deque(maxlen=5)

    def update(self, t: int, o: float, h: float, lo: float, c: float, v: float) -> None:
        self.q.append(float(c))

    def allows(self, d: int) -> bool:
        if len(self.q) < 5:
            return False
        q = self.q
        rising = all(q[k] > q[k - 1] for k in range(1, 5))
        falling = all(q[k] < q[k - 1] for k in range(1, 5))
        return not rising if d > 0 else not falling


class FilterAdrExhaust:
    def __init__(self) -> None:
        self.ranges: deque[float] = deque(maxlen=14)
        self.day = None
        self.day_hi = self.day_lo = 0.0

    def update(self, t: int, o: float, h: float, lo: float, c: float, v: float) -> None:
        day = int(t) // 86400
        if self.day is None or day != self.day:
            if self.day is not None:
                self.ranges.append(self.day_hi - self.day_lo)
            self.day, self.day_hi, self.day_lo = day, h, lo
        else:
            self.day_hi, self.day_lo = max(self.day_hi, h), min(self.day_lo, lo)

    def allows(self, d: int) -> bool:
        if len(self.ranges) < 14:
            return False
        s = 0.0
        for v in self.ranges:
            s += v
        return (self.day_hi - self.day_lo) < 0.8 * (s / 14.0)


class FilterKeltner:
    def __init__(self) -> None:
        self.m, self.atr, self.trs = _Mean(20), _Mean(20), _TrueRange()
        self.close = 0.0

    def update(self, t: int, o: float, h: float, lo: float, c: float, v: float) -> None:
        self.m.push(c)
        self.atr.push(self.trs.push(h, lo, c))
        self.close = c

    def allows(self, d: int) -> bool:
        if not self.m.full:
            return False
        mid, a = self.m.value(), self.atr.value()
        return mid - 1.5 * a <= self.close <= mid + 1.5 * a


class BiasTrailing:
    def __init__(self) -> None:
        self.atr, self.trs = _Mean(14), _TrueRange()
        self.dir = 0
        self.stop = 0.0

    def update(self, t: int, o: float, h: float, lo: float, c: float, v: float) -> None:
        self.atr.push(self.trs.push(h, lo, c))
        if not self.atr.full:
            return
        a = self.atr.value()
        if self.dir == 0:
            self.dir, self.stop = 1, c - 3.0 * a
        elif self.dir > 0:
            if c < self.stop:
                self.dir, self.stop = -1, c + 3.0 * a
            else:
                self.stop = max(self.stop, c - 3.0 * a)
        elif c > self.stop:
            self.dir, self.stop = 1, c - 3.0 * a
        else:
            self.stop = min(self.stop, c + 3.0 * a)

    def allows(self, d: int) -> bool:
        return self.dir > 0 if d > 0 else self.dir < 0


class BiasMarketStruct:
    def __init__(self) -> None:
        self.hi: deque[float] = deque(maxlen=20)
        self.lo: deque[float] = deque(maxlen=20)

    def update(self, t: int, o: float, h: float, lo: float, c: float, v: float) -> None:
        self.hi.append(float(h))
        self.lo.append(float(lo))

    def allows(self, d: int) -> bool:
        if len(self.hi) < 20:
            return False
        hi, lo = list(self.hi), list(self.lo)
        h1, h2, l1, l2 = max(hi[:10]), max(hi[10:]), min(lo[:10]), min(lo[10:])
        return (h2 > h1 and l2 > l1) if d > 0 else (h2 < h1 and l2 < l1)


class BiasVwap:
    def __init__(self) -> None:
        self.vw = DayVwap()
        self.w: float | None = None
        self.close = 0.0

    def update(self, t: int, o: float, h: float, lo: float, c: float, v: float) -> None:
        self.w = self.vw.push(t, h, lo, c, v)
        self.close = c

    def allows(self, d: int) -> bool:
        if self.w is None:
            return False
        return self.close > self.w if d > 0 else self.close < self.w


class BiasPsar:
    STEP, MAX = 0.02, 0.2

    def __init__(self) -> None:
        self.n = 0
        self.up = True
        self.sar = self.ep = 0.0
        self.af = self.STEP
        self.h1 = self.l1 = self.h2 = self.l2 = 0.0  # the previous bar's and the one before's high/low
        self.c1 = 0.0

    def update(self, t: int, o: float, h: float, lo: float, c: float, v: float) -> None:
        if self.n == 1:  # the first two closes set the first trend; the SAR starts at the first bar's extreme
            self.up = c >= self.c1
            self.sar, self.ep = (self.l1, max(self.h1, h)) if self.up else (self.h1, min(self.l1, lo))
        elif self.n >= 2:
            sar = self.sar + self.af * (self.ep - self.sar)
            if self.up:
                sar = min(sar, self.l1, self.l2)
                if lo < sar:
                    self.up, sar, self.ep, self.af = False, self.ep, lo, self.STEP
                elif h > self.ep:
                    self.ep, self.af = h, min(self.af + self.STEP, self.MAX)
            else:
                sar = max(sar, self.h1, self.h2)
                if h > sar:
                    self.up, sar, self.ep, self.af = True, self.ep, h, self.STEP
                elif lo < self.ep:
                    self.ep, self.af = lo, min(self.af + self.STEP, self.MAX)
            self.sar = sar
        self.h2, self.l2, self.h1, self.l1, self.c1 = self.h1, self.l1, h, lo, c
        self.n += 1

    def allows(self, d: int) -> bool:
        if self.n < 3:
            return False
        return self.up if d > 0 else not self.up


class FilterSma:
    def __init__(self) -> None:
        self.fast, self.slow = _Mean(10), _Mean(30)
        self.atr, self.trs = _Mean(14), _TrueRange()

    def update(self, t: int, o: float, h: float, lo: float, c: float, v: float) -> None:
        self.fast.push(c)
        self.slow.push(c)
        self.atr.push(self.trs.push(h, lo, c))

    def allows(self, d: int) -> bool:
        if not self.slow.full:
            return False
        return abs(self.fast.value() - self.slow.value()) >= 0.25 * self.atr.value()


class FilterReceding:
    def __init__(self) -> None:
        self.trs = _TrueRange()
        self.q: deque[float] = deque(maxlen=10)

    def update(self, t: int, o: float, h: float, lo: float, c: float, v: float) -> None:
        self.q.append(self.trs.push(h, lo, c))

    def allows(self, d: int) -> bool:
        if len(self.q) < 10:
            return False
        q = list(self.q)
        old = new = 0.0
        for k in range(5):
            old += q[k]
            new += q[k + 5]
        return new / 5.0 >= old / 5.0


class FilterDoji:
    def __init__(self) -> None:
        self.body = self.rng = 0.0

    def update(self, t: int, o: float, h: float, lo: float, c: float, v: float) -> None:
        self.body, self.rng = abs(c - o), h - lo

    def allows(self, d: int) -> bool:
        return self.rng > 0.0 and self.body > 0.1 * self.rng


CHOP_RATIO = 14.0 ** 0.618  # Choppiness(14) < 61.8  <=>  sum TR < (HH - LL) x 14^0.618 (no logarithm per bar)


class FilterRegime:
    def __init__(self) -> None:
        self.trs = _TrueRange()
        self.tr: deque[float] = deque(maxlen=14)
        self.hi: deque[float] = deque(maxlen=14)
        self.lo: deque[float] = deque(maxlen=14)

    def update(self, t: int, o: float, h: float, lo: float, c: float, v: float) -> None:
        self.tr.append(self.trs.push(h, lo, c))
        self.hi.append(float(h))
        self.lo.append(float(lo))

    def allows(self, d: int) -> bool:
        if len(self.tr) < 14:
            return False
        s = 0.0
        for x in self.tr:
            s += x
        rng = max(self.hi) - min(self.lo)
        return rng > 0.0 and s < rng * CHOP_RATIO


class FilterEvent:
    def __init__(self) -> None:
        self.trs = _TrueRange()
        self.tr: deque[float] = deque(maxlen=15)
        self.rng = self.gap = 0.0

    def update(self, t: int, o: float, h: float, lo: float, c: float, v: float) -> None:
        prev = self.trs.prev_close
        self.gap = 0.0 if prev is None else abs(o - prev)
        self.rng = h - lo
        self.tr.append(self.trs.push(h, lo, c))

    def allows(self, d: int) -> bool:
        if len(self.tr) < 15:
            return False
        s = 0.0
        for k in range(14):  # the 14 bars before this one
            s += self.tr[k]
        a = s / 14.0
        return self.rng <= 3.0 * a and self.gap <= 1.0 * a


BIAS_CLASSES = {"sma": BiasSma, "momentum": BiasMomentum, "donchian_mid": BiasDonchianMid, "rsi": BiasRsi,
                "adx": BiasAdx, "chandelier": BiasChandelier, "daily_mid": BiasDailyMid, "htf": BiasHtf,
                "trailing": BiasTrailing, "market_struct": BiasMarketStruct, "vwap": BiasVwap, "psar": BiasPsar}
FILTER_CLASSES = {"no_open_friday": FilterNoOpenFriday, "volatility": FilterVolatility, "rsi": FilterRsi,
                  "bb": FilterBb, "cci": FilterCci, "consec": FilterConsec, "adr_exhaust": FilterAdrExhaust,
                  "adx": FilterAdx, "keltner": FilterKeltner, "sma": FilterSma, "receding": FilterReceding,
                  "doji": FilterDoji, "regime": FilterRegime, "event": FilterEvent}
assert tuple(BIAS_CLASSES) == READY_BIAS and tuple(FILTER_CLASSES) == READY_FILTERS


def make(bias: tuple[str, ...], filters: tuple[str, ...]) -> list:
    """The gate objects for a DNA's switches; a switch without a gate here is refused (it is not READY)."""
    out = []
    for name in bias:
        if name not in BIAS_CLASSES:
            raise ValueError(f"bias switch {name!r} is not READY (READY: {READY_BIAS})")
        out.append(BIAS_CLASSES[name]())
    for name in filters:
        if name not in FILTER_CLASSES:
            raise ValueError(f"filter switch {name!r} is not READY (READY: {READY_FILTERS})")
        out.append(FILTER_CLASSES[name]())
    return out


# The gates that read the bar's open (doji, event) or its tick volume (vwap): switch_arrays needs those arrays.
NEEDS_OPEN = frozenset({("filter", "doji"), ("filter", "event")})
NEEDS_VOLUME = frozenset({("bias", "vwap")})


def switch_arrays(time: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                  bias: tuple[str, ...], filters: tuple[str, ...], *, open_: np.ndarray | None = None,
                  volume: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(long_ok, short_ok) for every bar from the same gate objects the oracle steps."""
    n = len(close)
    long_ok = np.ones(n, dtype=np.bool_)
    short_ok = np.ones(n, dtype=np.bool_)
    gates = make(tuple(bias), tuple(filters))
    if not gates:
        return long_ok, short_ok
    named = {("bias", b) for b in bias} | {("filter", f) for f in filters}
    if open_ is None and named & NEEDS_OPEN:
        raise ValueError(f"{sorted(named & NEEDS_OPEN)} need the bars' open")
    if volume is None and named & NEEDS_VOLUME:
        raise ValueError(f"{sorted(named & NEEDS_VOLUME)} need the bars' tick volume")
    t = np.asarray(time, dtype=np.int64)
    h, lo, c = (np.asarray(x, dtype=np.float64) for x in (high, low, close))
    o = np.zeros(n, dtype=np.float64) if open_ is None else np.asarray(open_, dtype=np.float64)
    vol = np.zeros(n, dtype=np.float64) if volume is None else np.asarray(volume, dtype=np.float64)
    for i in range(n):
        for g in gates:
            g.update(int(t[i]), float(o[i]), float(h[i]), float(lo[i]), float(c[i]), float(vol[i]))
        long_ok[i] = all(g.allows(1) for g in gates)
        short_ok[i] = all(g.allows(-1) for g in gates)
    return long_ok, short_ok


__all__ = ["BIAS_CLASSES", "FILTER_CLASSES", "READY_BIAS", "READY_FILTERS", "make", "switch_arrays"]
