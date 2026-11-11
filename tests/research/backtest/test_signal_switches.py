"""The research entry signals (backtest/signal_switches.py): each fires on its textbook shape and not before its
lookback; combine() keeps opposing votes out; the oracle and the kernel take the same trades (SIGNALS family)."""
from __future__ import annotations

import numpy as np
import pytest

from engine.research.backtest.signal_switches import READY_SIGNALS, combine, make, signal_arrays

H = 3600


def _arrays(o, h, lo, c, name, t=None, v=None):
    n = len(c)
    t = np.arange(n, dtype=np.int64) * H if t is None else np.asarray(t, dtype=np.int64)
    return signal_arrays(t, np.asarray(o, float), np.asarray(h, float), np.asarray(lo, float),
                         np.asarray(c, float), (name,), volume=np.ones(n) if v is None else np.asarray(v, float))


def test_combine_needs_one_side_only():
    assert combine([1, 0]) == 1 and combine([0, -1]) == -1 and combine([1, -1]) == 0 and combine([]) == 0


def test_unknown_signal_is_refused():
    with pytest.raises(ValueError, match="not READY"):
        make(("macd",))  # a DNA family, never a switch


@pytest.mark.parametrize("name", READY_SIGNALS)
def test_no_signal_on_the_first_bar(name):
    lng, sht = _arrays([1.0], [1.1], [0.9], [1.0], name)
    assert lng[0] == 0.0 and sht[0] == 0.0


def test_engulfing_bullish_and_bearish():
    lng, sht = _arrays([1.0, 0.95], [1.02, 1.06], [0.94, 0.94], [0.96, 1.05], "engulfing")
    assert lng[1] == 1.0 and sht[1] == 0.0
    lng, sht = _arrays([1.0, 1.05], [1.06, 1.06], [0.99, 0.94], [1.04, 0.98], "engulfing")
    assert sht[1] == 1.0 and lng[1] == 0.0


def test_inside_break():
    o = [1.0, 1.0, 1.0]
    h = [1.10, 1.05, 1.08]
    lo = [0.90, 0.95, 0.97]
    c = [1.0, 1.0, 1.07]
    lng, _ = _arrays(o, h, lo, c, "inside_break")
    assert lng[2] == 1.0


def test_pin_bar_and_wick_rejection_on_a_hammer():
    # open 1.00, close 1.01, low 0.90, high 1.012: long lower wick, close near the top
    for name in ("pin_bar", "wick_rejection"):
        lng, sht = _arrays([1.0], [1.012], [0.90], [1.01], name)
        assert lng[0] == 1.0 and sht[0] == 0.0, name


def test_liqsweep_takes_the_low_and_reclaims_it():
    n = 22
    lo = np.full(n, 1.0)
    h = np.full(n, 1.1)
    c = np.full(n, 1.05)
    o = np.full(n, 1.05)
    lo[-1] = 0.95  # below the prior 20-bar low 1.0
    c[-1] = 1.02   # closes back above it
    lng, _ = _arrays(o, h, lo, c, "liqsweep")
    assert lng[-1] == 1.0 and lng[:-1].sum() == 0.0


def test_three_soldiers_after_atr_warmup():
    n = 20
    o = 1.0 + 0.01 * np.arange(n)
    c = o + 0.008
    h, lo = c + 0.001, o - 0.001
    lng, _ = _arrays(o, h, lo, c, "three_soldiers")
    assert lng[13] == 1.0 and lng[:13].sum() == 0.0


def test_rsi_and_williams_cross_up_out_of_the_low_band():
    down = list(1.0 - 0.01 * np.arange(20))
    c = np.array(down + [down[-1] + 0.05, down[-1] + 0.10])
    o = c.copy()
    h, lo = c + 0.002, c - 0.002
    for name in ("rsi", "williams"):
        lng, sht = _arrays(o, h, lo, c, name)
        assert lng[-2:].sum() >= 1.0 and sht[-2:].sum() == 0.0, name


def test_vwap_needs_the_tick_volume():
    with pytest.raises(ValueError, match="tick volume"):
        signal_arrays(np.zeros(2, dtype=np.int64), np.ones(2), np.ones(2), np.ones(2), np.ones(2), ("vwap",))


def test_fib_buys_the_dip_to_61_8_in_an_up_leg():
    n = 52
    c = np.linspace(1.0, 2.0, n)  # a 50-bar leg up: low 1.0 first, high last
    h, lo, o = c + 0.001, c - 0.001, c.copy()
    lv = h[50] - 0.618 * (h[50] - lo[1])  # the swing of the 50 bars before bar 51 (bars 1..50)
    h[51], lo[51], c[51], o[51] = 2.0, 1.3, 1.6, 1.9
    lng, sht = _arrays(o, h, lo, c, "fib")
    assert lng[51] == 1.0 and sht.sum() == 0.0 and lng[:51].sum() == 0.0 and lo[51] <= lv


def test_atr_run_three_closes_beyond_one_and_a_half_atr():
    n = 20
    c = np.ones(n)
    c[-3:] = [1.01, 1.02, 1.03]
    o = np.r_[1.0, c[:-1]]
    h, lo = np.maximum(o, c) + 0.001, np.minimum(o, c) - 0.001
    lng, sht = _arrays(o, h, lo, c, "atr_run")
    assert lng[-1] == 1.0 and lng[:-1].sum() == 0.0 and sht.sum() == 0.0


def test_vwap_cross_and_day_reset():
    t = np.array([0, 3600, 7200, 86400, 90000], dtype=np.int64)
    c = np.array([1.0, 0.9, 1.2, 1.0, 1.0])
    h, lo = c + 0.01, c - 0.01
    h[3] = 1.05  # day 2 opens with a typical price above its close
    lng, sht = _arrays(c, h, lo, c, "vwap", t=t, v=[10, 10, 1, 5, 5])
    assert sht[1] == 1.0 and lng[2] == 1.0
    assert sht[3] == 1.0  # a new day: the VWAP restarts at bar 3's own typical price (~1.013), above its close


def test_orb_trades_the_first_break_of_the_day_only():
    t = np.arange(8, dtype=np.int64) * H
    c = np.array([1.0, 1.01, 0.99, 1.0, 1.05, 1.06, 0.9, 1.0])
    h, lo = c + 0.005, c - 0.005
    lng, sht = _arrays(c, h, lo, c, "orb", t=t)
    assert lng[4] == 1.0 and lng.sum() == 1.0 and sht.sum() == 0.0  # 0.9 later that day is not a second break


def test_kalman_turns_with_the_trend():
    c = np.r_[np.linspace(1.0, 1.2, 40), np.linspace(1.2, 1.0, 40)]
    lng, sht = _arrays(c, c + 0.001, c - 0.001, c, "kalman")
    assert sht[40:].sum() == 1.0 and lng[:40].sum() == 0.0 and lng[20:].sum() == 0.0
