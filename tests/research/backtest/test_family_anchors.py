"""S3 PART 2.4: strategy-level, hand-derived semantics anchors (F1-F4).

These are NOT the feature-level unit tests in tests/research/features/.
Each anchor drives the REAL strategy's ``on_bar`` through engine-style
``bars_up_to`` views on a tiny synthetic series whose indicator values and
signals were derived by hand (exact rational arithmetic; the tables below
are the derivation). The implementation defers to these numbers: if a
refactor changes them, the refactor is wrong.

SL/TP genes are 0 in every anchor, so ATR is never consulted and each
entry intent is emitted on the signal bar itself.
"""
from __future__ import annotations

import numpy as np
import pytest

from engine.research.backtest.strategies.donchian_breakout import (
    DonchianBreakoutConfig, DonchianBreakoutStrategy,
)
from engine.research.backtest.strategies.macd_cross import (
    MACDCrossConfig, MACDCrossStrategy, macd_series,
)
from engine.research.backtest.strategies.rsi import RSIConfig, RSIStrategy
from engine.research.backtest.strategies.stoch_cross import (
    StochCrossConfig, StochCrossStrategy,
)
from engine.research.data.mt5_source import BarData
from engine.research.features.forward_block import bars_up_to

TOL = 1e-12
VOL = 0.1


def _bars(close, high=None, low=None) -> BarData:
    c = np.asarray(close, dtype=np.float64)
    h = c.copy() if high is None else np.asarray(high, dtype=np.float64)
    lo = c.copy() if low is None else np.asarray(low, dtype=np.float64)
    n = c.size
    return BarData(
        time=np.arange(n, dtype=np.int64) * 3600, open=c.copy(), high=h, low=lo,
        close=c, tick_volume=np.zeros(n, dtype=np.int64),
        spread=np.zeros(n, dtype=np.int64), real_volume=np.zeros(n, dtype=np.int64),
        symbol="USDCADm", timeframe="H1",
    )


def _drive(strategy, bars: BarData) -> dict[int, tuple[str, float]]:
    """Run on_bar over every bar; return {bar: (direction, volume)}."""
    state: dict[str, object] = {}
    out: dict[int, tuple[str, float]] = {}
    for i in range(bars.close.size):
        intent = strategy.on_bar(i, bars_up_to(bars, i), state)
        if intent is not None:
            out[i] = (intent.direction, float(intent.volume))
    return out


# ---------------------------------------------------------------- F1: RSI

RSI_CLOSES = [10, 11, 12, 13, 14, 13, 12, 11, 12, 13, 14, 15, 14, 13, 12, 13]

# Wilder RSI, p=5. d_i = c_i - c_{i-1}:
#   d1..d5 = +1,+1,+1,+1,-1   d6..d8 = -1,-1,+1   d9..d15 = +1,+1,+1,-1,-1,-1,+1
# Seed (index p=5): avg_gain = (1+1+1+1+0)/5 = 0.8, avg_loss = (0+0+0+0+1)/5 = 0.2
# Recursion: avg = (prev*4 + x)/5. Every |d| = 1, so avg_gain + avg_loss = 1
# at every index and RSI = 100*avg_gain/(avg_gain+avg_loss) = 100*avg_gain.
#
#   idx  d   avg_gain            avg_loss            RSI
#    5   -1  0.8                 0.2                 80.0
#    6   -1  0.64                0.36                64.0
#    7   -1  0.512               0.488               51.2
#    8   +1  0.6096              0.3904              60.96
#    9   +1  0.68768             0.31232             68.768
#   10   +1  0.750144            0.249856            75.0144
#   11   +1  0.8001152           0.1998848           80.01152
#   12   -1  0.64009216          0.35990784          64.009216
#   13   -1  0.512073728         0.487926272         51.2073728
#   14   -1  0.4096589824        0.5903410176        40.96589824
#   15   +1  0.52772718592       0.47227281408       52.772718592
RSI_TABLE = {
    5: 80.0, 6: 64.0, 7: 51.2, 8: 60.96, 9: 68.768, 10: 75.0144,
    11: 80.01152, 12: 64.009216, 13: 51.2073728, 14: 40.96589824,
    15: 52.772718592,
}


def test_rsi_anchor_wilder_values_through_idx15() -> None:
    strat = RSIStrategy(RSIConfig(period=5, oversold=45.0, overbought=78.0))
    bars = _bars(RSI_CLOSES)
    _drive(strat, bars)
    rsi = strat._rsi_cache
    assert np.all(np.isnan(rsi[:5])), "warmup (< p) must be NaN"
    for idx, want in RSI_TABLE.items():
        assert abs(float(rsi[idx]) - want) <= TOL, (idx, float(rsi[idx]), want)


def test_rsi_anchor_level_entries_and_reversal() -> None:
    """os=45, ob=78 (oracle entry rule is level-triggered while flat):
    idx5 80.0 > 78 -> SHORT; idx11 80.01 > 78 but already short -> nothing;
    idx14 40.97 < 45 while short -> close (pending long); idx15 -> LONG."""
    strat = RSIStrategy(RSIConfig(period=5, oversold=45.0, overbought=78.0, volume=VOL))
    got = _drive(strat, _bars(RSI_CLOSES))
    assert got == {5: ("short", VOL), 14: ("long", 0.0), 15: ("long", VOL)}


def test_rsi_anchor_midline_exit() -> None:
    """os=35, ob=78, exit_midline: idx5 SHORT; RSI never < 35; idx15 crosses
    50 upward (prev 40.97 <= 50, now 52.77 > 50) -> flat close of the short."""
    strat = RSIStrategy(RSIConfig(
        period=5, oversold=35.0, overbought=78.0, exit_midline=True, volume=VOL,
    ))
    got = _drive(strat, _bars(RSI_CLOSES))
    assert got == {5: ("short", VOL), 15: ("short", 0.0)}


# -------------------------------------------------------------- F2: STOCH

# k=5, slowing=3, d=3; high = close+1, low = close-1.
# raw = 100*(C - LL5)/(HH5 - LL5) over [t-4, t]; %K = SMA3(raw); %D = SMA3(%K).
#
#   i  close  HH5  LL5  raw       %K                %D
#   4   24    31   23   12.5      -                 -
#   5   22    29   21   12.5      -                 -
#   6   21    28   20   12.5      12.5              -
#   7   19    26   18   12.5      12.5              -
#   8   18    25   17   12.5      12.5              12.5
#   9   17    23   16   100/7     275/21=13.0952..  800/63=12.6984..
#  10   16    22   15   100/7     575/42=13.6905..  1100/84=13.0952..
#  11   17    20   15   40        160/7=22.8571..   695/42=16.5476..
#
# Warmup: on_bar is silent for bar < k+d+s-2 = 9. At bar 9: K8 == D8 == 12.5
# (exact) and K9 > D9 -> bull cross; both 13.10 and 12.70 <= 25 -> LONG.
STOCH_CLOSES = [30, 28, 27, 25, 24, 22, 21, 19, 18, 17, 16, 17, 19, 22, 24]
STOCH_K = {6: 12.5, 7: 12.5, 8: 12.5, 9: 275 / 21, 10: 575 / 42, 11: 160 / 7}
STOCH_D = {8: 12.5, 9: 800 / 63, 10: 1100 / 84, 11: 695 / 42}


def _stoch() -> StochCrossStrategy:
    return StochCrossStrategy(StochCrossConfig(
        k_period=5, d_period=3, smoothing=3,
        k_oversold=25.0, k_overbought=80.0, volume=VOL,
    ))


def test_stoch_anchor_hand_k_d_and_first_cross() -> None:
    c = np.asarray(STOCH_CLOSES, dtype=np.float64)
    strat = _stoch()
    got = _drive(strat, _bars(c, c + 1.0, c - 1.0))
    k, d = strat._k_cache, strat._d_cache
    assert np.all(np.isnan(k[:6])) and np.all(np.isnan(d[:8]))
    for i, want in STOCH_K.items():
        assert abs(float(k[i]) - want) <= TOL, (i, float(k[i]), want)
    for i, want in STOCH_D.items():
        assert abs(float(d[i]) - want) <= TOL, (i, float(d[i]), want)
    assert got == {9: ("long", VOL)}


def test_stoch_anchor_hh_equals_ll_emits_no_signal() -> None:
    """Same series with bars 5..9 flat (H == L == C == 22): at bar 9 the whole
    5-bar window is flat, HH == LL, raw %K is undefined (NaN) -> %K NaN for
    bars 9..11 and %D NaN for 9..13 -> NO signal anywhere (the bar-9 LONG of
    the un-flattened series is gone)."""
    c = np.asarray(STOCH_CLOSES, dtype=np.float64)
    h, lo = c + 1.0, c - 1.0
    c[5:10] = h[5:10] = lo[5:10] = 22.0
    strat = _stoch()
    got = _drive(strat, _bars(c, h, lo))
    assert np.isnan(strat._k_cache[9])
    assert got == {}


# --------------------------------------------------------------- F3: MACD

# EMA alpha = 2/(n+1), SMA seed at index n-1 from the first n finite values.
# fast=3, slow=6, signal=3 (series-level: the committed gap>=5 ban rejects a
# 3/6 CONFIG, so the strategy-level cross anchor below uses 3/8/3).
#
#   i  close  EMA3     EMA6            MACD            SIGNAL
#   2   12    11 (seed = (10+11+12)/3)
#   3   13    12 (= .5*13 + .5*11)
#   4   12    12 (= .5*12 + .5*12)
#   5   11    23/2     23/2 (seed)     0
#   6   10    43/4     155/14          -9/28
#   7   11    87/8     1083/98         -69/392         -65/392 (seed=mean m5..m7)
#   8   13    191/16   7963/686        1809/5488       899/10976
#   9   14    415/32   59023/4802      52047/76832     14585/38416
# EMA6 recursion alpha = 2/7: 155/14 = (2/7)*10 + (5/7)*(23/2); signal alpha = 1/2.
MACD_CLOSES = [10, 11, 12, 13, 12, 11, 10, 11, 13, 14, 13, 12, 11, 10, 9, 10, 12]
MACD_LINE = {5: 0.0, 6: -9 / 28, 7: -69 / 392, 8: 1809 / 5488, 9: 52047 / 76832}
MACD_SIG = {7: -65 / 392, 8: 899 / 10976, 9: 14585 / 38416}


def test_macd_anchor_sma_seeds_and_first_two_recursions() -> None:
    line, sig, _ = macd_series(
        np.asarray(MACD_CLOSES, dtype=np.float64),
        fast_period=3, slow_period=6, signal_period=3,
    )
    assert np.all(np.isnan(line[:5])) and np.all(np.isnan(sig[:7]))
    for i, want in MACD_LINE.items():
        assert abs(float(line[i]) - want) <= TOL, (i, float(line[i]), want)
    for i, want in MACD_SIG.items():
        assert abs(float(sig[i]) - want) <= TOL, (i, float(sig[i]), want)


# Strategy-level, fast=3 slow=8 signal=3 (warmup: silent for bar < 8+3-1 = 10):
#   i   MACD-SIGNAL (exact rational -> float)
#   9   +0.5622   10  +0.1960   11  -0.1093  <- bear cross: SHORT
#  12..15 negative              16  +0.3418  <- bull cross while short: close
#  17   +0.5957 (pending -> LONG)  18 +0.5578  19 +0.1680
#  20   -0.2977  <- bear cross while long: close; 21 pending -> SHORT
MACD_CLOSES_LONG = MACD_CLOSES + [14, 15, 14, 12, 10]


def test_macd_anchor_strategy_crosses_and_reversals() -> None:
    strat = MACDCrossStrategy(MACDCrossConfig(
        fast_period=3, slow_period=8, signal_period=3, volume=VOL,
    ))
    got = _drive(strat, _bars(MACD_CLOSES_LONG))
    assert got == {
        11: ("short", VOL), 16: ("long", 0.0), 17: ("long", VOL),
        20: ("short", 0.0), 21: ("short", VOL),
    }


# ----------------------------------------------------------- F4: DONCHIAN

# period=4 (the committed gene space has ONE channel; mission entry_ch=4 /
# exit_ch=3 collapses to period=4). high = close+1, low = close-1.
# Prior channel for bar t is [t-4, t-1]:
#
#   t  close  upper_prev  lower_prev  event
#   4   12    14          9
#   5   11    14          10
#   6   12    14          10
#   7   15    14          10          15 > 14 -> first break: LONG
#   8   13    16          10
#   9   10    16          10          10 == lower: strict, no break
#  10    8    16          9           8 < 9 while long -> close (pending short)
#  11    9    -           -           pending -> SHORT
DONCHIAN_CLOSES = [10, 12, 11, 13, 12, 11, 12, 15, 13, 10, 8, 9]
DONCHIAN_UPPER = {4: 14.0, 5: 14.0, 6: 14.0, 7: 14.0, 8: 16.0, 9: 16.0, 10: 16.0}
DONCHIAN_LOWER = {4: 9.0, 5: 10.0, 6: 10.0, 7: 10.0, 8: 10.0, 9: 10.0, 10: 9.0}


def test_donchian_anchor_prior_channel_and_first_break() -> None:
    c = np.asarray(DONCHIAN_CLOSES, dtype=np.float64)
    strat = DonchianBreakoutStrategy(DonchianBreakoutConfig(period=4, volume=VOL))
    got = _drive(strat, _bars(c, c + 1.0, c - 1.0))
    upper, lower = strat._rolling_cache
    for t, want in DONCHIAN_UPPER.items():
        assert float(upper[t - 1]) == pytest.approx(want, abs=TOL), t
    for t, want in DONCHIAN_LOWER.items():
        assert float(lower[t - 1]) == pytest.approx(want, abs=TOL), t
    assert got == {7: ("long", VOL), 10: ("short", 0.0), 11: ("short", VOL)}
