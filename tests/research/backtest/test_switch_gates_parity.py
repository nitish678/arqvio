"""v4 gene batch B: bias/filter switches as entry gates -- the kernel (gate arrays) against the oracle (GatedStrategy)."""
from __future__ import annotations

import numpy as np
import pytest

from engine.research.backtest.engine import BacktestConfig, BacktestEngine
from engine.research.backtest.fast_path import record_to_dict, run_fast
from engine.research.backtest.gates import gate_arrays
from engine.research.backtest.kernel import TR_SIZE
from engine.research.backtest.prep import prep_cell
from engine.research.backtest.strategies.gated import GatedStrategy
from engine.research.backtest.strategies.ma_crossover import MACrossConfig, MACrossStrategy
from engine.research.backtest.switch_gates import READY_BIAS, READY_FILTERS, switch_arrays
from engine.research.data.mt5_source import BarData, Timeframe
from engine.research.search.real_backtest import spec_for
from tests.research.backtest.test_kernel_parity_steps import slow_trade_rows, trade_list_diffs

SYMBOL = "USDCADm"


def _bars(n: int = 2400, seed: int = 5) -> BarData:
    rng = np.random.default_rng(seed)
    walk = np.cumsum(rng.normal(0.0, 0.0018, size=n)) + 1.10
    # the open is the previous close plus a small gap (the doji/event filters read it), tick volume varies (vwap)
    open_ = np.r_[walk[0], walk[:-1]] + rng.normal(0.0, 0.0004, size=n)
    return BarData(time=np.array([86400 * 3 + i * 3600 for i in range(n)], dtype=np.int64), open=open_,
                   high=np.maximum(open_, walk) + rng.uniform(0.0004, 0.002, n),
                   low=np.minimum(open_, walk) - rng.uniform(0.0004, 0.002, n), close=walk.copy(),
                   tick_volume=rng.integers(1, 900, n).astype(np.float64), spread=np.full(n, 12, dtype=np.int32),
                   real_volume=np.zeros(n), symbol=SYMBOL, timeframe=Timeframe.H1)


CASES = [((b,), ()) for b in READY_BIAS] + [((), (f,)) for f in READY_FILTERS] + [
    (("sma",), ("rsi",)), (("momentum", "donchian_mid"), ("volatility", "no_open_friday")),
    (("adx", "daily_mid"), ("bb", "consec")), (("chandelier",), ("adr_exhaust", "keltner", "cci")),
    (("trailing", "vwap"), ("doji", "event")), (("psar",), ("sma", "receding")), (("market_struct",), ("doji",))]


@pytest.mark.parametrize(("bias", "filters"), CASES)
def test_the_kernel_takes_exactly_the_openings_the_oracle_takes(bias, filters) -> None:
    bars = _bars()
    spec = spec_for(SYMBOL)
    cfg = BacktestConfig(strategy_id="sw", symbol=SYMBOL, timeframe="H1", initial_equity=10_000.0, charge_swap=True,
                         charge_commission=True, commission_per_lot_per_side=3.5, slippage_mean_points=0.3,
                         slippage_std_points=0.2)
    ma = MACrossConfig(fast=5, slow=21, volume=0.1, sl_atr=1.5, tp_atr=3.0, strategy_id="sw")
    strat = GatedStrategy(MACrossStrategy(ma), session="", trend_ema=0, bias_switches=bias, filter_switches=filters)
    slow = BacktestEngine(spec=spec, config=cfg, rng_seed=0).run(bars, strat)
    prepped = prep_cell(bars, spec)
    gl, gs = gate_arrays(prepped.bar_hour_index, prepped.close, "", 0)
    sl, ss = switch_arrays(bars.time, prepped.high, prepped.low, prepped.close, bias, filters, open_=prepped.open,
                           volume=prepped.tick_volume)
    trades = np.zeros((prepped.n_bars, TR_SIZE), dtype=np.float64)
    record, _, status = run_fast(prepped, trades_out=trades, fast=5, slow=21, volume=0.1, sl_atr=1.5, tp_atr=3.0,
                                 gates=(gl & sl, gs & ss), initial_equity=10_000.0, charge_swap=True,
                                 charge_commission=True, commission_per_lot_per_side=3.5, slippage_mean_points=0.3,
                                 slippage_std_points=0.2, rng_seed=0)
    assert status == 0
    got = record_to_dict(record)
    assert slow.n_trades > 3, "fixture: the gated strategy must still trade"
    assert got["n_trades"] == slow.n_trades
    assert not trade_list_diffs([tuple(r) for r in trades[: got["n_trades"]]], slow_trade_rows(slow))
    assert got["final_equity"] == slow.final_equity


def test_a_switch_that_is_not_ready_is_refused() -> None:
    with pytest.raises(ValueError, match="not READY"):
        switch_arrays(np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), ("ema",), ())


def test_a_gate_that_reads_the_open_or_the_volume_needs_it() -> None:
    z = np.zeros(3)
    with pytest.raises(ValueError, match="open"):
        switch_arrays(z, z, z, z, (), ("doji",))
    with pytest.raises(ValueError, match="tick volume"):
        switch_arrays(z, z, z, z, ("vwap",), (), open_=z)


def test_no_open_friday_closes_exactly_fridays() -> None:
    t = np.array([86400 * d for d in range(14)], dtype=np.int64)  # 1970-01-01 (Thu) .. 01-14
    lo, so = switch_arrays(t, np.ones(14), np.ones(14), np.ones(14), (), ("no_open_friday",))
    assert [d for d in range(14) if not lo[d]] == [1, 8] and (lo == so).all()


H = 3600


def _trend(n: int, step: float = 0.001) -> tuple[np.ndarray, ...]:
    c = 1.0 + step * np.arange(n, dtype=np.float64)
    return np.arange(n, dtype=np.int64) * H, c + 0.0005, c - 0.0005, c


@pytest.mark.parametrize("name", ["rsi", "adx", "chandelier", "htf", "daily_mid", "trailing", "market_struct",
                                  "vwap", "psar"])
def test_batch_c_bias_follows_a_steady_rise(name) -> None:
    t, h, lo, c = _trend(400)
    lng, sht = switch_arrays(t, h, lo, c, (name,), (), open_=c - 0.0002, volume=np.ones(len(c)))
    assert lng[-1] and not sht[-1]
    assert not lng[0] and not sht[0]  # no gate allows an opening before its lookback


def test_consec_refuses_the_fourth_close_in_the_trade_direction() -> None:
    t, h, lo, c = _trend(10)
    lng, sht = switch_arrays(t, h, lo, c, (), ("consec",))
    assert not lng[4] and sht[4] and not lng[3]


def test_adr_exhaust_compares_today_with_the_14_day_mean() -> None:
    days = 16
    t = np.repeat(np.arange(days, dtype=np.int64) * 86400, 2) + np.tile([0, 3600], days)
    rng = np.ones(days)
    rng[-1] = 0.5
    h = np.repeat(1.0 + rng / 2, 2)
    lo = np.repeat(1.0 - rng / 2, 2)
    c = np.ones(2 * days)
    lng, _ = switch_arrays(t, h, lo, c, (), ("adr_exhaust",))
    assert lng[-1]                  # 0.5 < 0.8 x mean(1.0)
    h2 = h.copy()
    h2[-2:] = 1.7
    lng2, _ = switch_arrays(t, h2, lo, c, (), ("adr_exhaust",))
    assert not lng2[-1]             # today's range 0.95 >= 0.8


def test_bb_and_keltner_refuse_a_spike() -> None:
    n = 60
    t = np.arange(n, dtype=np.int64) * H
    c = np.ones(n)
    c[-1] = 1.5
    h, lo = c + 0.001, c - 0.001
    for f in ("bb", "keltner", "cci"):
        lng, sht = switch_arrays(t, h, lo, c, (), (f,))
        assert not lng[-1] and not sht[-1], f


def test_doji_refuses_a_bar_without_a_body() -> None:
    t, h, lo, c = _trend(3)
    o = c.copy()
    o[1] -= 0.0004  # a body of 0.0004 in a range of 0.001: not a doji
    lng, sht = switch_arrays(t, h, lo, c, (), ("doji",), open_=o)
    assert list(lng) == [False, True, False] and (lng == sht).all()


def test_event_refuses_a_shock_bar_and_a_gap() -> None:
    t, h, lo, c = _trend(40)
    o = c - 0.0002
    lng, _ = switch_arrays(t, h, lo, c, (), ("event",), open_=o)
    assert not lng[13] and lng[14:].all()  # 14 bars before the first judged one
    h2, o2 = h.copy(), o.copy()
    h2[30] += 0.01   # a range of 11 x the usual
    o2[35] += 0.005  # a gap of 5 ATR
    lng2, _ = switch_arrays(t, h2, lo, c, (), ("event",), open_=o2)
    assert not lng2[30] and not lng2[35] and lng2[29] and lng2[34]


def test_regime_trends_pass_and_chop_is_refused() -> None:
    t, h, lo, c = _trend(40)
    assert switch_arrays(t, h, lo, c, (), ("regime",))[0][-1]
    chop = 1.0 + 0.004 * (np.arange(40) % 2)  # up, down, up, down: every true range spans the whole box
    assert not switch_arrays(t, chop + 0.0005, chop - 0.0005, chop, (), ("regime",))[0][-1]


def test_sma_and_receding_filters() -> None:
    t, h, lo, c = _trend(60)
    assert switch_arrays(t, h, lo, c, (), ("sma",))[0][-1]           # SMA(10) - SMA(30) = 0.01 >> 0.25 ATR
    flat = np.ones(60)
    assert not switch_arrays(t, flat + 0.001, flat - 0.001, flat, (), ("sma",))[0][-1]
    wide = np.where(np.arange(60) < 55, 0.01, 0.001)                 # the last 5 ranges shrink: receding
    assert not switch_arrays(t, flat + wide, flat - wide, flat, (), ("receding",))[0][-1]
    assert switch_arrays(t, flat + wide[::-1], flat - wide[::-1], flat, (), ("receding",))[0][-1]
