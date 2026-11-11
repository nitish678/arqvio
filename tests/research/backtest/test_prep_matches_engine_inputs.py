"""S2-B: prep arrays must agree with the slow engine's per-bar view.

The slow engine is the specification. These assertions are written
against what engine.py ACTUALLY does, re-deriving each quantity the way
the engine's own loop derives it, so the two cannot drift apart without
this failing.

Two of them encode oracle behaviour that differs from the S2 brief, and
they say so in place: the engine works in UTC with no DST anywhere, and
its spread COST is a flat per-symbol number rather than an hourly
lookup.
"""
from __future__ import annotations

import numpy as np
import pytest

from engine.research.backtest.prep import prep_cell
from engine.research.costs.spread import spread_cost
from engine.research.data.mt5_source import BarData, Timeframe
from engine.research.features.atr import atr_at
from engine.research.search.real_backtest import spec_for

SYMBOL = "USDCADm"
N = 400
SPOT_CHECK_BARS = 20


def _bars() -> BarData:
    rng = np.random.default_rng(11)
    walk = np.cumsum(rng.normal(0.0, 0.0012, size=N)) + 1.10
    # Hourly bars so every UTC hour and several weekdays are covered.
    return BarData(
        time=np.array([86400 * 3 + i * 3600 for i in range(N)], dtype=np.int64),
        open=walk.copy(),
        high=walk + 0.0009,
        low=walk - 0.0009,
        close=walk.copy(),
        tick_volume=np.ones(N),
        spread=np.full(N, 14, dtype=np.int32),
        real_volume=np.zeros(N),
        symbol=SYMBOL,
        timeframe=Timeframe.H1,
    )


def _prep(**kwargs):
    return prep_cell(_bars(), spec_for(SYMBOL), **kwargs)


def _spot_indices(n: int) -> list[int]:
    """Evenly spread, deterministic, and always including the edges."""
    idx = list(np.linspace(0, n - 1, SPOT_CHECK_BARS, dtype=int))
    return sorted({int(i) for i in idx})


# ---------------------------------------------------------------------------
# Time: UTC, no DST (engine.py:341, :389, :217)
# ---------------------------------------------------------------------------

def test_hour_matches_the_engines_utc_derivation() -> None:
    """engine.py:341 -- hour = (bar_time // 3600) % 24.

    The data layer (mt5_source.fetch_bars, real_backtest.load_cell_bars)
    now converts broker-server time to UTC via time_offset.broker_to_utc
    (G12), so bar_time here is genuinely UTC -- not accidentally-UTC. The
    slow engine still buckets on the raw UTC stamp (no offset inside
    engine.py), so prep must reproduce that derivation, not improve on it.
    """
    bars = _bars()
    p = _prep()
    for i in _spot_indices(p.n_bars):
        expected = (int(bars.time[i]) // 3600) % 24
        assert int(p.bar_hour_index[i]) == expected, f"bar {i}"


def test_day_index_and_friday_match_the_engine() -> None:
    """engine.py:389/:491 day index; engine.py:217 Friday test."""
    bars = _bars()
    p = _prep()
    for i in _spot_indices(p.n_bars):
        t = int(bars.time[i])
        assert int(p.day_index[i]) == t // 86400, f"bar {i}"
        assert bool(p.is_friday[i]) == ((t // 86400) % 7 == 1), f"bar {i}"


def test_no_dst_offset_is_applied_anywhere() -> None:
    """Guards oracle parity, not correctness.

    If the engine ever starts applying a broker offset, prep must too,
    and this failing is how that gets noticed rather than showing up as
    a silent parity divergence in the kernel.
    """
    import inspect

    from engine.research.backtest import engine as slow_engine

    source = inspect.getsource(slow_engine)
    assert "time_offset" not in source, (
        "engine.py now imports time_offset; prep.py still derives UTC "
        "hours and days and must be updated to match"
    )


# ---------------------------------------------------------------------------
# The zero-spread window (F1)
# ---------------------------------------------------------------------------

def test_no_trade_mask_matches_the_engines_hour_test() -> None:
    """engine.py:342 -- blocked = hour in no_trade_hours."""
    hours = (21, 3)
    p = _prep(no_trade_hours=hours)
    for i in _spot_indices(p.n_bars):
        assert bool(p.no_trade_mask[i]) == (int(p.bar_hour_index[i]) in hours), f"bar {i}"
    assert p.no_trade_mask.sum() > 0, "fixture never hits a blocked hour"


def test_no_trade_mask_is_empty_without_configured_hours() -> None:
    assert _prep().no_trade_mask.sum() == 0


# ---------------------------------------------------------------------------
# ATR (G9)
# ---------------------------------------------------------------------------

def test_atr_matches_atr_at_bar_for_bar() -> None:
    """The kernel must see the same ATR the strategy computed.

    atr_at is what ma_crossover._stop_distances calls, so this compares
    the precomputed column against the exact function the slow path
    used -- including the 13-bar warmup, where both must be undefined.
    """
    bars = _bars()
    p = _prep()
    for i in _spot_indices(p.n_bars):
        expected = atr_at(bars.high, bars.low, bars.close, i)
        got = p.atr14[i]
        if expected is None:
            assert np.isnan(got), f"bar {i}: prep has {got}, atr_at undefined"
        else:
            assert got == pytest.approx(expected, rel=0, abs=1e-15), f"bar {i}"


def test_atr_warmup_is_exactly_thirteen_bars() -> None:
    p = _prep()
    assert np.all(np.isnan(p.atr14[:13]))
    assert not np.isnan(p.atr14[13])


# ---------------------------------------------------------------------------
# Spread: flat cost, per-bar gate (engine.py:459, costs/spread.py:58)
# ---------------------------------------------------------------------------

def test_spread_cost_flat_when_no_hourly() -> None:
    """Without hourly_spread_points, the flat spec spread is used everywhere."""
    spec = spec_for(SYMBOL)
    p = _prep()
    assert p.spread_cost_one_way == pytest.approx(
        spread_cost(spec, 1.0).one_way, rel=0, abs=1e-15
    )
    # hourly_spread_cost_per_lot is the flat cost repeated across all 24 hours.
    assert len(p.hourly_spread_cost_per_lot) == 24
    assert p.hourly_spread_cost_per_lot[0] == pytest.approx(
        p.spread_cost_one_way, rel=0, abs=1e-15
    )
    assert p.hourly_spread_cost_per_lot[12] == pytest.approx(
        p.spread_cost_one_way, rel=0, abs=1e-15
    )


def test_hourly_spread_cost_uses_bucket_per_hour() -> None:
    """When hourly_spread_points is set, each hour's cost is
    spread_points[hour] * point * per_unit_per_lot at 1.0 lot.
    """
    spec = spec_for(SYMBOL)
    # Synthetic 24-point row, varied so the test is meaningful.
    hourly_pts = tuple(float(h % 10 + 1) for h in range(24))
    p = _prep(hourly_spread_points=hourly_pts)
    per_unit_per_lot = max(
        spec.trade_tick_value_profit, spec.trade_tick_value_loss
    ) / spec.trade_tick_size
    for h in range(24):
        expected = hourly_pts[h] * spec.point * per_unit_per_lot
        assert p.hourly_spread_cost_per_lot[h] == pytest.approx(
            expected, rel=0, abs=1e-15
        ), f"hour {h}"


def test_bar_spread_column_is_carried_for_the_gate_only() -> None:
    """engine.py:459 compares float(bars.spread[i]) against the gate."""
    bars = _bars()
    p = _prep(max_spread_points=50.0)
    for i in _spot_indices(p.n_bars):
        assert p.bar_spread_points[i] == pytest.approx(float(bars.spread[i]))
    assert p.max_spread_points == pytest.approx(50.0)


def test_gate_off_is_nan_not_zero() -> None:
    """Zero would gate out every bar; absent means absent."""
    assert np.isnan(_prep().max_spread_points)


# ---------------------------------------------------------------------------
# Swap schedule (swap.py:91, _nights_between)
# ---------------------------------------------------------------------------

def test_swap_multiplier_is_three_on_the_specs_triple_day() -> None:
    """swap.py:91 -- multiplier = 3 if d.weekday() == triple_dow."""
    from datetime import date, timedelta

    spec = spec_for(SYMBOL)
    p = _prep()
    for k in range(p.swap_multiplier.size):
        d = date(1970, 1, 1) + timedelta(days=p.day_index_min + k)
        expected = 3 if d.weekday() == spec.swap_rollover_triple_day else 1
        assert int(p.swap_multiplier[k]) == expected, d.isoformat()
    assert set(p.swap_multiplier.tolist()) == {1, 3}, (
        "fixture must span a triple day and ordinary days"
    )


def test_swap_multiplier_is_indexed_by_day_index() -> None:
    p = _prep()
    for i in _spot_indices(p.n_bars):
        k = int(p.day_index[i]) - p.day_index_min
        assert 0 <= k < p.swap_multiplier.size


# ---------------------------------------------------------------------------
# Per-leg tick values (W3, engine.py:215-217)
# ---------------------------------------------------------------------------

def test_per_leg_tick_factors_match_the_engine() -> None:
    spec = spec_for(SYMBOL)
    p = _prep()
    assert p.per_unit_profit == pytest.approx(
        spec.trade_tick_value_profit / spec.trade_tick_size, rel=0, abs=1e-12
    )
    assert p.per_unit_loss == pytest.approx(
        spec.trade_tick_value_loss / spec.trade_tick_size, rel=0, abs=1e-12
    )
    assert p.per_unit_profit != p.per_unit_loss, (
        "fixture symbol has equal tick values; the per-leg split is untested"
    )


def test_slippage_cap_defaults_to_the_asset_class() -> None:
    from engine.research.costs.slippage import slippage_cap_for_symbol

    assert _prep().slippage_cap_points == slippage_cap_for_symbol(SYMBOL)
