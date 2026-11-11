"""S3 PART 2.3: late-index tripwires, one per non-MA family.

Each test drives the REAL strategy's ``on_bar`` through engine-style
``bars_up_to`` views on the pinned USDCADm H1 snapshot to bar >= 3000, then
compares the indicator value the strategy's cache holds at that bar with a
from-scratch value computed here in plain Python (no engine indicator code),
at 1e-12.

This guards every cache key: a key that does not track the visible window
(the old ``id()`` keys, or a key missing a period) returns a series from a
shorter window -- IndexError or a stale value at a late index. The walk is
sparse early and dense around the target so the cache sees many distinct
visible lengths, including consecutive ones.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from engine.research.backtest.strategies.donchian_breakout import (
    DonchianBreakoutConfig, DonchianBreakoutStrategy,
)
from engine.research.backtest.strategies.macd_cross import (
    MACDCrossConfig, MACDCrossStrategy,
)
from engine.research.backtest.strategies.rsi import RSIConfig, RSIStrategy
from engine.research.backtest.strategies.stoch_cross import (
    StochCrossConfig, StochCrossStrategy,
)
from engine.research.data.dataset_snapshot import find_snapshot, load_snapshot
from engine.research.data.mt5_source import BarData
from engine.research.features.forward_block import bars_up_to

DATASET_ID = "aed3bf314539ec5a"  # USDCADm H1
SLICE_BARS = 3_200
TARGET = 3_000
TOL = 1e-12
_WALK = list(range(0, TARGET - 50, 50)) + list(range(TARGET - 50, TARGET + 2))


@pytest.fixture(scope="module")
def bars() -> BarData:
    if find_snapshot(DATASET_ID) is None:
        pytest.skip(f"snapshot {DATASET_ID} not taken on this machine")
    b = load_snapshot(DATASET_ID)
    s = slice(-SLICE_BARS, None)
    return BarData(
        time=np.asarray(b.time)[s], open=np.asarray(b.open)[s],
        high=np.asarray(b.high)[s], low=np.asarray(b.low)[s],
        close=np.asarray(b.close)[s], tick_volume=np.asarray(b.tick_volume)[s],
        spread=np.asarray(b.spread)[s], real_volume=np.asarray(b.real_volume)[s],
        symbol=b.symbol, timeframe=b.timeframe,
    )


def _walk(strategy, bars: BarData) -> None:
    state: dict[str, object] = {}
    for i in _WALK:
        strategy.on_bar(i, bars_up_to(bars, i), state)
    # The last call saw visible length TARGET + 2; call AT the target last so
    # the cache holds exactly the window [0, TARGET].
    strategy.on_bar(TARGET, bars_up_to(bars, TARGET), state)


def _close_to(a: float, b: float) -> None:
    assert math.isfinite(a) and math.isfinite(b), (a, b)
    assert abs(a - b) <= TOL * max(1.0, abs(b)), (a, b, a - b)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _ema_from_scratch(x: list[float], n: int) -> list[float]:
    """EMA alpha=2/(n+1), SMA-seeded on the first n FINITE values."""
    out = [math.nan] * len(x)
    finite = [i for i, v in enumerate(x) if math.isfinite(v)]
    if len(finite) < n:
        return out
    seed_end = finite[n - 1]
    out[seed_end] = _mean(x[finite[0] : seed_end + 1])
    alpha = 2.0 / (n + 1)
    for i in range(seed_end + 1, len(x)):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
    return out


def test_rsi_late_index_tripwire(bars: BarData) -> None:
    p = 9
    strat = RSIStrategy(RSIConfig(period=p, oversold=25.0, overbought=75.0))
    _walk(strat, bars)
    got = float(strat._rsi_cache[TARGET])

    c = [float(v) for v in np.asarray(bars.close)[: TARGET + 1]]
    d = [c[j + 1] - c[j] for j in range(len(c) - 1)]
    g = [max(v, 0.0) for v in d]
    lo = [max(-v, 0.0) for v in d]
    ag, al = _mean(g[:p]), _mean(lo[:p])
    for i in range(p + 1, TARGET + 1):
        ag = (ag * (p - 1) + g[i - 1]) / p
        al = (al * (p - 1) + lo[i - 1]) / p
    if ag == 0.0 and al == 0.0:
        want = 50.0
    elif al == 0.0:
        want = 100.0
    else:
        want = 100.0 - 100.0 / (1.0 + ag / al)
    _close_to(got, want)


def test_stoch_late_index_tripwire(bars: BarData) -> None:
    k, d, s = 11, 3, 2
    strat = StochCrossStrategy(StochCrossConfig(
        k_period=k, d_period=d, smoothing=s, k_oversold=20.0, k_overbought=80.0,
    ))
    _walk(strat, bars)
    got_k, got_d = float(strat._k_cache[TARGET]), float(strat._d_cache[TARGET])

    h = [float(v) for v in np.asarray(bars.high)[: TARGET + 1]]
    lw = [float(v) for v in np.asarray(bars.low)[: TARGET + 1]]
    c = [float(v) for v in np.asarray(bars.close)[: TARGET + 1]]

    def raw(i: int) -> float:
        hh, ll = max(h[i - k + 1 : i + 1]), min(lw[i - k + 1 : i + 1])
        return math.nan if hh == ll else (c[i] - ll) / (hh - ll) * 100.0

    def pct_k(i: int) -> float:
        return _mean([raw(j) for j in range(i - s + 1, i + 1)])

    want_k = pct_k(TARGET)
    want_d = _mean([pct_k(j) for j in range(TARGET - d + 1, TARGET + 1)])
    _close_to(got_k, want_k)
    _close_to(got_d, want_d)
    assert 0.0 <= got_k <= 100.0


def test_macd_late_index_tripwire(bars: BarData) -> None:
    fast, slow, sig = 7, 29, 6
    strat = MACDCrossStrategy(MACDCrossConfig(
        fast_period=fast, slow_period=slow, signal_period=sig,
    ))
    _walk(strat, bars)
    got_macd = float(strat._macd_cache[0][TARGET])
    got_sig = float(strat._macd_cache[1][TARGET])

    c = [float(v) for v in np.asarray(bars.close)[: TARGET + 1]]
    ef, es = _ema_from_scratch(c, fast), _ema_from_scratch(c, slow)
    macd = [a - b for a, b in zip(ef, es)]
    signal = _ema_from_scratch(macd, sig)
    _close_to(got_macd, macd[TARGET])
    _close_to(got_sig, signal[TARGET])


def test_donchian_late_index_tripwire(bars: BarData) -> None:
    p = 23
    strat = DonchianBreakoutStrategy(DonchianBreakoutConfig(period=p))
    _walk(strat, bars)
    upper, lower = strat._rolling_cache
    # on_bar at TARGET reads the PRIOR channel [TARGET-p, TARGET-1].
    got_u, got_l = float(upper[TARGET - 1]), float(lower[TARGET - 1])

    h = [float(v) for v in np.asarray(bars.high)[:TARGET]]
    lw = [float(v) for v in np.asarray(bars.low)[:TARGET]]
    _close_to(got_u, max(h[TARGET - p : TARGET]))
    _close_to(got_l, min(lw[TARGET - p : TARGET]))
    assert upper.size == TARGET + 1
