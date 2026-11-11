"""W4 (A5/A6): the cost-model gaps the verify sweep found.

A2 (per-date DST), A3 (swap + Wednesday triple) and A4 (commission)
already existed and are covered by tests/research/test_s2_gate.py. What
was missing: a cap on the slippage draw, a max-spread entry gate, and
weekend-gap fills wired to the exit engine.
"""
from __future__ import annotations

import numpy as np
import pytest

from engine.research.backtest.engine import BacktestConfig, BacktestEngine
from engine.research.backtest.order import EXIT_SL, EXIT_TP, OrderIntent
from engine.research.costs.slippage import (
    SLIPPAGE_CAP_POINTS_FX,
    SLIPPAGE_CAP_POINTS_METALS,
    realised_slippage_points,
    slippage_cap_for_symbol,
)
from engine.research.data.mt5_source import BarData, Timeframe
from engine.research.search.real_backtest import spec_for

SYMBOL = "USDCADm"


def _bars(
    n: int,
    price: float = 1.10000,
    spread_points: int = 0,
    opens: list[float] | None = None,
    highs: list[float] | None = None,
    lows: list[float] | None = None,
) -> BarData:
    close = np.full(n, price, dtype=np.float64)
    return BarData(
        time=np.array([86400 + i * 3600 for i in range(n)], dtype=np.int64),
        open=np.array(opens, dtype=np.float64) if opens else close.copy(),
        high=np.array(highs, dtype=np.float64) if highs else close + 0.0005,
        low=np.array(lows, dtype=np.float64) if lows else close - 0.0005,
        close=close.copy(),
        tick_volume=np.ones(n),
        spread=np.full(n, spread_points, dtype=np.int32),
        real_volume=np.zeros(n),
        symbol=SYMBOL,
        timeframe=Timeframe.H1,
    )


class _EnterOnBar:
    strategy_id = "enter"

    def __init__(self, bar: int, sl_dist=None, tp_dist=None, direction="long"):
        self.bar, self.sl_dist, self.tp_dist = bar, sl_dist, tp_dist
        self.direction = direction

    def on_bar(self, bar_index, bars, state):
        if bar_index != self.bar:
            return None
        return OrderIntent(
            direction=self.direction, volume=0.1, bar_index=bar_index,
            sl_dist=self.sl_dist, tp_dist=self.tp_dist,
        )


def _engine(**overrides) -> BacktestEngine:
    config = BacktestConfig(
        strategy_id="w4",
        symbol=SYMBOL,
        timeframe="H1",
        initial_equity=10_000.0,
        charge_swap=False,
        charge_commission=False,
        slippage_mean_points=0.0,
        slippage_std_points=0.0,
        **overrides,
    )
    return BacktestEngine(spec=spec_for(SYMBOL), config=config, rng_seed=0)


# ---------------------------------------------------------------------------
# A5: slippage cap
# ---------------------------------------------------------------------------

def test_slippage_draw_is_capped() -> None:
    """The half-normal is unbounded above; one tail sample must not
    charge an entry hundreds of points."""
    rng = np.random.default_rng(0)
    uncapped = [
        realised_slippage_points(mean_points=5.0, std_points=50.0, rng=rng)
        for _ in range(2000)
    ]
    assert max(uncapped) > SLIPPAGE_CAP_POINTS_FX, (
        "fixture must actually produce draws above the cap"
    )

    rng = np.random.default_rng(0)
    capped = [
        realised_slippage_points(
            mean_points=5.0, std_points=50.0, rng=rng,
            cap_points=SLIPPAGE_CAP_POINTS_FX,
        )
        for _ in range(2000)
    ]
    assert max(capped) <= SLIPPAGE_CAP_POINTS_FX
    assert min(capped) >= 0.0, "slippage is adverse-only"


def test_slippage_cap_is_per_asset_class() -> None:
    """The two sanctioned values, and nothing invented for either."""
    assert slippage_cap_for_symbol("USDCADm") == SLIPPAGE_CAP_POINTS_FX
    assert slippage_cap_for_symbol("EURUSDm") == SLIPPAGE_CAP_POINTS_FX
    assert slippage_cap_for_symbol("XAUUSDm") == SLIPPAGE_CAP_POINTS_METALS
    assert slippage_cap_for_symbol("XAGUSDm") == SLIPPAGE_CAP_POINTS_METALS
    # An unknown class takes the TIGHTER cap rather than the looser one.
    assert slippage_cap_for_symbol("US500m") == SLIPPAGE_CAP_POINTS_FX


# ---------------------------------------------------------------------------
# A5: max-spread entry gate
# ---------------------------------------------------------------------------

def test_max_spread_gate_rejects_an_extreme_spread_bar() -> None:
    """An entry due on a blown-out bar is refused and counted."""
    bars = _bars(10, spread_points=500)
    result = _engine(max_spread_points=50.0).run(bars, _EnterOnBar(bar=2))
    assert result.n_trades == 0, "the entry filled on a 500-point spread bar"
    assert result.blocked_max_spread == 1

    # The same bars with the gate off DO trade, so the assertion above
    # is about the gate and not about the fixture being untradeable.
    allowed = _engine().run(bars, _EnterOnBar(bar=2))
    assert allowed.n_trades == 1
    assert allowed.blocked_max_spread == 0


def test_max_spread_gate_leaves_normal_bars_alone() -> None:
    bars = _bars(10, spread_points=12)
    result = _engine(max_spread_points=50.0).run(bars, _EnterOnBar(bar=2))
    assert result.n_trades == 1
    assert result.blocked_max_spread == 0


def test_max_spread_gate_does_not_trap_an_open_position() -> None:
    """Exits are NOT gated.

    Refusing to close a position because spreads widened traps it in
    exactly the conditions it should be leaving.
    """
    n = 12
    spread = np.full(n, 10, dtype=np.int32)
    spread[5:] = 900                      # blows out while holding
    price = np.full(n, 1.10000)
    lows = price - 0.0005
    lows[6] = 1.09000                     # takes out the stop
    bars = BarData(
        time=np.array([86400 + i * 3600 for i in range(n)], dtype=np.int64),
        open=price.copy(), high=price + 0.0005, low=lows, close=price.copy(),
        tick_volume=np.ones(n), spread=spread, real_volume=np.zeros(n),
        symbol=SYMBOL, timeframe=Timeframe.H1,
    )
    result = _engine(max_spread_points=50.0).run(
        bars, _EnterOnBar(bar=2, sl_dist=0.00100)
    )
    assert result.n_trades == 1
    assert result.trades[0].exit_reason == EXIT_SL, (
        "the stop did not execute because the gate blocked the exit"
    )


def test_gate_threshold_comes_from_the_symbols_own_spreads() -> None:
    """p95 of the calibrated row, not a hardcoded pip count.

    Gold and EURUSD are two orders of magnitude apart; one number
    cannot serve both.
    """
    from engine.data.spread_calibration import load, spread_dir

    usdcad = load("USDCADm", spread_dir())
    xau = load("XAUUSDm", spread_dir())
    if usdcad is None or xau is None:
        pytest.skip("spread artifacts not calibrated on this machine")

    fx_p95 = usdcad.spread_percentile_points(95.0)
    metal_p95 = xau.spread_percentile_points(95.0)
    assert fx_p95 is not None
    assert metal_p95 is not None
    assert metal_p95 > fx_p95 * 5, (
        f"gold p95 {metal_p95} is not far above FX p95 {fx_p95}; a single "
        "hardcoded threshold would be wrong for one of them"
    )


def test_zero_spread_hours_do_not_drag_the_percentile_down() -> None:
    """The F1 no-trade hours are excluded from the gate threshold."""
    from engine.data.spread_calibration import SpreadMatrix

    buckets = [100.0] * 24
    with_zero = list(buckets)
    with_zero[21] = 0.0
    a = SpreadMatrix(symbol="Xm", buckets=tuple(buckets),
                     mean_buckets=tuple(buckets),
                     sample_count=0, source="MT5")
    b = SpreadMatrix(symbol="Xm", buckets=tuple(with_zero),
                     mean_buckets=tuple(with_zero),
                     sample_count=0, source="MT5")
    assert a.spread_percentile_points(95.0) == b.spread_percentile_points(95.0)


def test_no_usable_spread_disables_the_gate_rather_than_guessing() -> None:
    from engine.data.spread_calibration import SpreadMatrix

    m = SpreadMatrix(symbol="Xm", buckets=tuple([0.0] * 24),
                     mean_buckets=tuple([0.0] * 24),
                     sample_count=0, source="MT5")
    assert m.spread_percentile_points(95.0) is None


# ---------------------------------------------------------------------------
# A6: weekend gap
# ---------------------------------------------------------------------------

def test_weekend_gap_fill_at_open() -> None:
    """LONG SL 1.1000, Friday close 1.1005, Monday open 1.0990 -> 1.0990.

    The spec's own case. The level was never available, so the fill is
    the first price that existed, which is worse than the stop.
    """
    n = 10
    price = np.full(n, 1.10050)
    opens = list(price)
    highs = list(price + 0.0002)
    lows = list(price - 0.0002)
    # Bar 5 gaps well below the stop and never trades back up to it.
    opens[5], highs[5], lows[5] = 1.09900, 1.09920, 1.09850
    bars = BarData(
        time=np.array([86400 + i * 3600 for i in range(n)], dtype=np.int64),
        open=np.array(opens), high=np.array(highs), low=np.array(lows),
        close=price.copy(), tick_volume=np.ones(n),
        spread=np.zeros(n, dtype=np.int32), real_volume=np.zeros(n),
        symbol=SYMBOL, timeframe=Timeframe.H1,
    )
    # Entry fills at bar 2's open = 1.10050; SL 0.00050 below = 1.10000.
    result = _engine().run(bars, _EnterOnBar(bar=1, sl_dist=0.00050))
    trade = result.trades[0]
    assert trade.sl_price == pytest.approx(1.10000, abs=1e-10)
    assert trade.exit_reason == EXIT_SL
    assert trade.exit_fill.filled_price == pytest.approx(1.09900, abs=1e-10), (
        "the gapped stop filled at its level instead of the open"
    )
    assert trade.exit_fill.filled_price < trade.sl_price


def test_weekend_gap_in_favour_fills_at_open_too() -> None:
    """A target gapped THROUGH fills better, not at the level.

    First-available-price cuts both ways. Clamping the good side to the
    level while taking the open on the bad side would be a bias.
    """
    n = 10
    price = np.full(n, 1.10000)
    opens = list(price)
    highs = list(price + 0.0002)
    lows = list(price - 0.0002)
    opens[5], highs[5], lows[5] = 1.10300, 1.10350, 1.10280
    bars = BarData(
        time=np.array([86400 + i * 3600 for i in range(n)], dtype=np.int64),
        open=np.array(opens), high=np.array(highs), low=np.array(lows),
        close=price.copy(), tick_volume=np.ones(n),
        spread=np.zeros(n, dtype=np.int32), real_volume=np.zeros(n),
        symbol=SYMBOL, timeframe=Timeframe.H1,
    )
    result = _engine().run(bars, _EnterOnBar(bar=1, tp_dist=0.00050))
    trade = result.trades[0]
    assert trade.tp_price == pytest.approx(1.10050, abs=1e-10)
    assert trade.exit_reason == EXIT_TP
    assert trade.exit_fill.filled_price == pytest.approx(1.10300, abs=1e-10)
    assert trade.exit_fill.filled_price > trade.tp_price


# ---------------------------------------------------------------------------
# A3: two swap bugs that only became visible once W3 ingested real rates
# ---------------------------------------------------------------------------

def _daily_bars(n: int) -> BarData:
    price = np.full(n, 1.10000, dtype=np.float64)
    return BarData(
        time=np.array([86400 * (10 + i) for i in range(n)], dtype=np.int64),
        open=price.copy(), high=price + 0.0005, low=price - 0.0005,
        close=price.copy(), tick_volume=np.ones(n),
        spread=np.zeros(n, dtype=np.int32), real_volume=np.zeros(n),
        symbol=SYMBOL, timeframe=Timeframe.D1,
    )


def _held_position_swap(n_days: int) -> float:
    """Swap charged on one position held across n_days day-rolls."""
    config = BacktestConfig(
        strategy_id="swap", symbol=SYMBOL, timeframe="D1",
        initial_equity=10_000.0, charge_swap=True, charge_commission=False,
        slippage_mean_points=0.0, slippage_std_points=0.0,
    )
    engine = BacktestEngine(spec=spec_for(SYMBOL), config=config, rng_seed=0)
    result = engine.run(
        _daily_bars(n_days), _EnterOnBar(bar=0, direction="short")
    )
    return float(result.trades[0].swap_charged)


def test_swap_is_charged_once_per_night_not_cumulatively() -> None:
    """Each roll charges its OWN nights, not the whole history again.

    _charge_swap used to compute swap from the ENTRY date to today and
    add the whole thing at every roll, so night 1 was charged N times.
    The tell is quadratic growth: doubling the holding period more than
    doubles the charge.
    """
    short_hold = _held_position_swap(6)
    long_hold = _held_position_swap(11)
    assert short_hold != 0.0, "fixture must actually accrue swap"

    # Linear accrual: ~2x the nights is ~2x the charge. Quadratic
    # accumulation would be ~4x or worse.
    ratio = long_hold / short_hold
    assert ratio < 3.0, (
        f"swap grew {ratio:.2f}x when the holding period doubled; that is "
        "super-linear, i.e. earlier nights are being re-charged"
    )


def test_swap_on_a_negative_rate_is_a_cost_not_a_credit() -> None:
    """MT5 signs a rate the trader PAYS as negative.

    swap_cost returns that signed adjustment; Trade.swap_charged is a
    COST, because _close_trade_pnl_split subtracts it. Without the
    negation a -8.4pt short swap paid the strategy to hold.
    """
    spec = spec_for(SYMBOL)
    if spec.swap_short >= 0:
        pytest.skip(f"{SYMBOL} swap_short is {spec.swap_short}, not a cost")

    charged = _held_position_swap(8)
    assert charged > 0.0, (
        f"swap_charged={charged} on a negative (pay) rate; a cost was "
        "recorded as a credit"
    )

    # And it reaches P&L in the right direction.
    config = BacktestConfig(
        strategy_id="swap-pnl", symbol=SYMBOL, timeframe="D1",
        initial_equity=10_000.0, charge_commission=False,
        slippage_mean_points=0.0, slippage_std_points=0.0,
    )
    bars = _daily_bars(8)
    with_swap = BacktestEngine(
        spec=spec, config=BacktestConfig(**{**config.__dict__, "charge_swap": True}),
        rng_seed=0,
    ).run(bars, _EnterOnBar(bar=0, direction="short"))
    without_swap = BacktestEngine(
        spec=spec, config=BacktestConfig(**{**config.__dict__, "charge_swap": False}),
        rng_seed=0,
    ).run(bars, _EnterOnBar(bar=0, direction="short"))
    assert with_swap.final_equity < without_swap.final_equity, (
        "charging swap on a pay rate INCREASED equity"
    )


def test_cost_decomposition_splits_four_ways_and_sums() -> None:
    """fee_total is only meaningful if the parts are reported."""
    config = BacktestConfig(
        strategy_id="decomp", symbol=SYMBOL, timeframe="D1",
        initial_equity=10_000.0, charge_swap=True, charge_commission=True,
        commission_per_lot_per_side=3.5,
        slippage_mean_points=0.5, slippage_std_points=0.2,
    )
    result = BacktestEngine(
        spec=spec_for(SYMBOL), config=config, rng_seed=0
    ).run(_daily_bars(12), _EnterOnBar(bar=0, direction="short"))

    d = result.cost_decomposition()
    assert d["fee_total"] == pytest.approx(
        d["commission"] + d["spread"] + d["slippage"] + d["swap"], abs=1e-9
    )
    assert d["commission"] > 0, "commission config did not reach the fill"
    assert d["spread"] > 0
    assert d["slippage"] > 0
    shares = sum(
        d[f"{k}_pct_of_fees"] for k in ("commission", "spread", "slippage", "swap")
    )
    assert shares == pytest.approx(100.0, abs=1e-6)
