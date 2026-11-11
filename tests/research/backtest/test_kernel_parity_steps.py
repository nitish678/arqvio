"""S2-C steps 1-10: the kernel against the oracle, feature by feature.

Every test here runs the SAME synthetic bars through engine.py and
through the kernel and asserts they agree. Nothing is compared against
a hand-written expectation, because the slow engine is the
specification -- a hand-written number could only encode my reading of
it, and my reading is the thing under test.

The features are switched on one at a time, in the order the brief
lists them, so a divergence points at the feature that introduced it
instead of at the whole engine.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from engine.research.backtest.engine import BacktestConfig, BacktestEngine
from engine.research.backtest.fast_path import (
    record_exit_reasons,
    record_to_dict,
    run_fast,
)
from engine.research.backtest.kernel import (
    TR_REASON_FRIDAY_CLOSE,
    TR_REASON_OPEN,
    TR_REASON_SIGNAL,
    TR_REASON_SL,
    TR_REASON_SLTP_DEFERRED,
    TR_REASON_TP,
    TR_SIZE,
)
from engine.research.backtest.prep import prep_cell
from engine.research.backtest.strategies.ma_crossover import (
    MACrossConfig,
    MACrossStrategy,
)
from engine.research.data.mt5_source import BarData, Timeframe
from engine.research.search.real_backtest import spec_for

SYMBOL = "USDCADm"

_REASON = {
    "SIGNAL": TR_REASON_SIGNAL, "SL": TR_REASON_SL, "TP": TR_REASON_TP,
    "SLTP_DEFERRED_ZERO_SPREAD": TR_REASON_SLTP_DEFERRED,
    "FRIDAY_CLOSE": TR_REASON_FRIDAY_CLOSE,
}


def slow_trade_rows(result) -> list[tuple]:
    """engine.py Trades in the kernel's trade-log layout."""
    rows = []
    for t in result.trades:
        closed = t.exit_fill is not None
        rows.append((
            t.entry_bar_index, t.entry_fill.bar_index,
            1 if t.direction == "long" else -1,
            t.entry_fill.filled_price, t.volume,
            math.nan if t.sl_price is None else t.sl_price,
            math.nan if t.tp_price is None else t.tp_price,
            t.exit_bar_index if closed else -1,
            t.exit_fill.filled_price if closed else math.nan,
            _REASON[t.exit_reason] if closed else TR_REASON_OPEN,
            t.realized_pnl,
        ))
    return rows


def trade_list_diffs(fast_rows, slow_rows) -> list[str]:
    """Every field of every trade, compared exactly (NaN == NaN)."""
    out = []
    if len(fast_rows) != len(slow_rows):
        return [f"len fast={len(fast_rows)} slow={len(slow_rows)}"]
    for k, (f, s) in enumerate(zip(fast_rows, slow_rows)):
        for j, (x, y) in enumerate(zip(f, s)):
            same = (math.isnan(x) and math.isnan(y)) or x == y
            if not same:
                out.append(f"trade {k} field {j}: fast={x!r} slow={y!r}")
    return out


def _bars(
    n: int = 400,
    seed: int = 7,
    start_hour: int = 0,
    spread_points: int = 14,
    vol: float = 0.0016,
) -> BarData:
    rng = np.random.default_rng(seed)
    walk = np.cumsum(rng.normal(0.0, vol, size=n)) + 1.10
    return BarData(
        time=np.array(
            [86400 * 3 + start_hour * 3600 + i * 3600 for i in range(n)],
            dtype=np.int64,
        ),
        open=walk.copy(),
        high=walk + 0.0012,
        low=walk - 0.0012,
        close=walk.copy(),
        tick_volume=np.ones(n),
        spread=np.full(n, spread_points, dtype=np.int32),
        real_volume=np.zeros(n),
        symbol=SYMBOL,
        timeframe=Timeframe.H1,
    )


def _compare(
    bars: BarData,
    *,
    fast: int = 9,
    slow: int = 37,
    volume: float = 0.1,
    sl_atr: float = 0.0,
    tp_atr: float = 0.0,
    no_trade_hours: tuple[int, ...] = (),
    max_spread_points: float | None = None,
    hourly_spread_points: tuple[float, ...] | None = None,
    daily_loss_limit_pct: float | None = None,
    friday_close_hour: int | None = None,
    charge_swap: bool = False,
    charge_commission: bool = False,
    commission_per_lot_per_side: float = 0.0,
    slippage_mean_points: float = 0.0,
    slippage_std_points: float = 0.0,
    label: str = "",
):
    """Run both engines on the same bars and assert they agree."""
    spec = spec_for(SYMBOL)
    cfg = BacktestConfig(
        strategy_id="parity",
        symbol=SYMBOL,
        timeframe="H1",
        initial_equity=10_000.0,
        charge_swap=charge_swap,
        charge_commission=charge_commission,
        commission_per_lot_per_side=commission_per_lot_per_side,
        slippage_mean_points=slippage_mean_points,
        slippage_std_points=slippage_std_points,
        no_trade_hours=no_trade_hours,
        max_spread_points=max_spread_points,
        hourly_spread_points=hourly_spread_points,
        daily_loss_limit_pct=daily_loss_limit_pct,
        friday_close_hour=friday_close_hour,
    )
    ma = MACrossConfig(
        fast=fast, slow=slow, volume=volume, sl_atr=sl_atr, tp_atr=tp_atr,
        strategy_id="parity",
    )
    slow_result = BacktestEngine(spec=spec, config=cfg, rng_seed=0).run(
        bars, MACrossStrategy(ma)
    )

    prepped = prep_cell(
        bars, spec,
        no_trade_hours=no_trade_hours,
        max_spread_points=max_spread_points,
        hourly_spread_points=hourly_spread_points,
    )
    trades = np.zeros((prepped.n_bars, TR_SIZE), dtype=np.float64)
    record, equity, status = run_fast(
        prepped, trades_out=trades,
        fast=fast, slow=slow, volume=volume, sl_atr=sl_atr, tp_atr=tp_atr,
        initial_equity=10_000.0,
        max_spread_points=max_spread_points,
        daily_loss_limit_pct=daily_loss_limit_pct,
        friday_close_hour=friday_close_hour,
        charge_swap=charge_swap,
        charge_commission=charge_commission,
        commission_per_lot_per_side=commission_per_lot_per_side,
        slippage_mean_points=slippage_mean_points,
        slippage_std_points=slippage_std_points,
        rng_seed=0,
    )
    assert status == 0, f"{label}: kernel returned status {status}"

    got = record_to_dict(record)
    tag = f"{label}: " if label else ""

    assert got["n_trades"] == slow_result.n_trades, (
        f"{tag}n_trades fast={got['n_trades']} slow={slow_result.n_trades}"
    )
    assert got["n_round_trips"] == slow_result.n_round_trips, (
        f"{tag}n_round_trips fast={got['n_round_trips']} "
        f"slow={slow_result.n_round_trips}"
    )
    assert got["n_open"] == slow_result.n_open, (
        f"{tag}n_open fast={got['n_open']} slow={slow_result.n_open}"
    )
    assert record_exit_reasons(record) == dict(slow_result.exit_reasons), (
        f"{tag}exit_reasons fast={record_exit_reasons(record)} "
        f"slow={dict(slow_result.exit_reasons)}"
    )
    for name, slow_value in (
        ("blocked_zero_spread_events", slow_result.blocked_zero_spread_events),
        ("deferred_zero_spread_fills", slow_result.deferred_zero_spread_fills),
        ("sltp_deferred_zero_spread", slow_result.sltp_deferred_zero_spread),
        ("blocked_max_spread", slow_result.blocked_max_spread),
        ("blocked_daily_loss", slow_result.blocked_daily_loss),
        ("friday_flattens", slow_result.friday_flattens),
    ):
        assert got[name] == slow_value, (
            f"{tag}{name} fast={got[name]} slow={slow_value}"
        )
    diffs = trade_list_diffs(
        [tuple(r) for r in trades[: got["n_trades"]]], slow_trade_rows(slow_result),
    )
    assert not diffs, f"{tag}trade list: {diffs[:5]}"
    # Money is compared EXACTLY: lot size, spread, slippage, commission and
    # swap are all computed in the oracle's operation order.
    assert got["final_equity"] == slow_result.final_equity, (
        f"{tag}final_equity fast={got['final_equity']!r} "
        f"slow={slow_result.final_equity!r}"
    )
    for name, slow_value in (
        ("cost_commission", slow_result.cost_commission),
        ("cost_spread", slow_result.cost_spread),
        ("cost_slippage", slow_result.cost_slippage),
        ("cost_swap", slow_result.cost_swap),
    ):
        assert got[name] == slow_value, (
            f"{tag}{name} fast={got[name]!r} slow={slow_value!r}"
        )
    return slow_result, record, equity


# --- step 1: signals only, no stops, no costs ---------------------------

def test_step1_signals_only() -> None:
    result, _, _ = _compare(_bars(), label="signals-only")
    assert result.n_trades > 5, "fixture produced too few trades to prove anything"


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_step1_signals_across_several_series(seed: int) -> None:
    _compare(_bars(seed=seed), label=f"signals seed={seed}")


@pytest.mark.parametrize(("fast", "slow"), [(5, 10), (9, 37), (20, 50), (12, 199)])
def test_step2_entries_market2_across_windows(fast: int, slow: int) -> None:
    """MARKET2: signal at t, fill at open t+1, across MA window shapes."""
    _compare(_bars(n=500), fast=fast, slow=slow, label=f"ma {fast}/{slow}")


# --- step 3: SL/TP, including the pessimistic cases ---------------------

@pytest.mark.parametrize(
    ("sl_atr", "tp_atr"),
    [(1.0, 0.0), (0.0, 1.0), (1.0, 2.0), (0.5, 0.5), (3.0, 6.0)],
)
def test_step3_sl_tp(sl_atr: float, tp_atr: float) -> None:
    result, record, _ = _compare(
        _bars(n=500), sl_atr=sl_atr, tp_atr=tp_atr,
        label=f"sl={sl_atr} tp={tp_atr}",
    )
    reasons = record_exit_reasons(record)
    assert reasons.get("SL", 0) + reasons.get("TP", 0) > 0, (
        "fixture never hit a stop, so SL/TP parity is untested here"
    )
    del result


def test_step3_both_touched_is_sl_first_in_both_engines() -> None:
    """A wide-range fixture where both levels sit inside many bars."""
    _compare(
        _bars(n=500, vol=0.004), sl_atr=0.5, tp_atr=0.5,
        label="both-touched",
    )


# --- step 4: zero-spread deferral ---------------------------------------

@pytest.mark.parametrize("hours", [(21,), (3,), (0, 12), (21, 22)])
def test_step4_zero_spread_deferral(hours: tuple[int, ...]) -> None:
    # Short MA windows on a choppy series, so intents are emitted often
    # enough to actually land inside a one-hour-per-day window. With
    # 9/37 the fixture crossed too rarely to block anything, and a
    # passing parity check on a feature that never fired proves nothing.
    result, record, _ = _compare(
        _bars(n=900, vol=0.0030), fast=4, slow=9,
        sl_atr=1.0, tp_atr=2.0, no_trade_hours=hours,
        label=f"no_trade={hours}",
    )
    assert result.blocked_zero_spread_events > 0, (
        f"no_trade_hours={hours} never blocked anything in this fixture"
    )
    del record


# --- step 5: gap fills --------------------------------------------------

def test_step5_gap_fills() -> None:
    """A series with jumps, so levels are gapped past rather than touched."""
    n = 500
    rng = np.random.default_rng(3)
    walk = np.cumsum(rng.normal(0.0, 0.0015, size=n)) + 1.10
    walk[200:] += 0.02          # one large session gap
    walk[350:] -= 0.03          # and one the other way
    bars = BarData(
        time=np.array([86400 * 3 + i * 3600 for i in range(n)], dtype=np.int64),
        open=walk.copy(), high=walk + 0.0010, low=walk - 0.0010,
        close=walk.copy(), tick_volume=np.ones(n),
        spread=np.full(n, 14, dtype=np.int32), real_volume=np.zeros(n),
        symbol=SYMBOL, timeframe=Timeframe.H1,
    )
    _compare(bars, sl_atr=1.0, tp_atr=1.5, label="gaps")


# --- step 6: per-leg tick values, slippage cap, max-spread gate ----------

def test_step6_slippage_is_drawn_from_the_same_stream() -> None:
    """The kernel consumes numpy's draws; numba's RNG could not match."""
    _compare(
        _bars(n=500), sl_atr=1.0,
        slippage_mean_points=0.3, slippage_std_points=0.2,
        label="slippage",
    )


def test_step6_zero_std_consumes_no_draw() -> None:
    """std==0 returns the mean without touching the generator.

    Consuming a draw here would desync every later fill.
    """
    _compare(
        _bars(n=400), slippage_mean_points=0.5, slippage_std_points=0.0,
        label="slippage std=0",
    )


@pytest.mark.parametrize("threshold", [10.0, 13.0, 14.0, 20.0])
def test_step6_max_spread_gate(threshold: float) -> None:
    _compare(
        _bars(n=400, spread_points=14), max_spread_points=threshold,
        label=f"gate={threshold}",
    )


# --- W0: hourly spread cost ------------------------------------------------

@pytest.mark.parametrize("hourly_spread_points", [
    tuple(float(h % 10 + 1) for h in range(24)),
    (1.0,) * 24,
    (0.5,) * 12 + (2.5,) * 12,
])
def test_w0_hourly_spread_cost_matches_engine(hourly_spread_points: tuple[float, ...]) -> None:
    """W0: kernel and slow engine both charge
    hourly_spread_points[hour] * point * volume * per_unit, in that order,
    so the spread cost line agrees bit for bit."""
    _compare(
        _bars(n=500),
        hourly_spread_points=hourly_spread_points,
        label=f"w0-hourly",
    )


def test_w0_hourly_spread_cost_affects_fee_line() -> None:
    """When hourly_spread_points differs from flat, the spread cost must differ."""
    result_flat, record_flat, _ = _compare(
        _bars(n=500),
        label="flat-spread",
    )
    hourly = tuple(float(h % 10 + 1) for h in range(24))
    result_hr, record_hr, _ = _compare(
        _bars(n=500),
        hourly_spread_points=hourly,
        label="hourly-spread",
    )
    from engine.research.backtest.fast_path import record_to_dict

    flat_cost = record_to_dict(record_flat)["cost_spread"]
    hr_cost = record_to_dict(record_hr)["cost_spread"]
    assert flat_cost != pytest.approx(hr_cost, rel=1e-12), (
        "hourly_spread_points had no effect on the spread cost line; "
        "either the wiring or the test is wrong"
    )


# --- step 7: swap --------------------------------------------------------

def test_step7_swap_sign_and_triple_day() -> None:
    result, record, _ = _compare(
        _bars(n=800), charge_swap=True, label="swap",
    )
    assert record_to_dict(record)["cost_swap"] != 0.0, (
        "fixture held nothing across a day roll, so swap parity is untested"
    )
    del result


def test_step7_swap_with_stops_held_over_rolls() -> None:
    _compare(_bars(n=800), sl_atr=2.0, tp_atr=4.0, charge_swap=True,
             label="swap+stops")


# --- step 8: commission --------------------------------------------------

@pytest.mark.parametrize("per_side", [0.0, 3.5, 7.0])
def test_step8_commission(per_side: float) -> None:
    _compare(
        _bars(n=400), charge_commission=True,
        commission_per_lot_per_side=per_side,
        label=f"commission={per_side}",
    )


# --- step 9: daily loss guardrail ---------------------------------------

@pytest.mark.parametrize("pct", [0.05, 0.5, 4.0])
def test_step9_daily_loss_limit(pct: float) -> None:
    _compare(
        _bars(n=600), sl_atr=1.0, daily_loss_limit_pct=pct,
        label=f"daily_loss={pct}",
    )


# --- step 10: Friday close ----------------------------------------------

@pytest.mark.parametrize("hour", [0, 12, 17, 22])
def test_step10_friday_close(hour: int) -> None:
    _compare(
        _bars(n=800), sl_atr=1.0, friday_close_hour=hour,
        label=f"friday={hour}",
    )


def test_step10_friday_close_respects_no_trade_hours() -> None:
    _compare(
        _bars(n=800), sl_atr=1.0, friday_close_hour=17,
        no_trade_hours=(17,), label="friday+blocked",
    )


# --- everything at once --------------------------------------------------

def test_all_features_together() -> None:
    """The combination is where ordering mistakes surface."""
    _compare(
        _bars(n=900),
        fast=9, slow=37, sl_atr=1.5, tp_atr=3.0,
        no_trade_hours=(21,),
        max_spread_points=20.0,
        daily_loss_limit_pct=2.0,
        friday_close_hour=20,
        charge_swap=True,
        charge_commission=True,
        commission_per_lot_per_side=3.5,
        slippage_mean_points=0.3,
        slippage_std_points=0.2,
        label="all-features",
    )
