"""Tests for engine.research.data.mt5_source.

These tests talk to the real MT5 terminal of account acct1, attached by explicit
login through the mt5_acct1 fixture (tests/mt5_attach.py). An absent environment
(no package, no accounts config, no terminal) skips, so CI / non-MT5 hosts can
still run the suite; a wrong account on the attached terminal fails.
"""
# ruff: noqa: N813  # `import MetaTrader5 as mt5` is the upstream convention
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from engine.research.data.mt5_source import (
    BarData,
    SymbolInfo,
    Timeframe,
    detect_symbols,
    detect_symbols_by_category,
    fetch_bars,
    snapshot_symbol_spec,
)
from tests.mt5_attach import acct1_terminal

MT5_TERMINAL = acct1_terminal()

pytestmark = pytest.mark.usefixtures("mt5_acct1")


# --- catalog detection -------------------------------------------------------

def test_detect_symbols_returns_nonempty_catalog() -> None:
    syms = detect_symbols(MT5_TERMINAL)
    assert len(syms) > 0
    assert all(isinstance(s, SymbolInfo) for s in syms)


def test_detect_symbols_includes_usdcadm() -> None:
    syms = detect_symbols(MT5_TERMINAL)
    names = {s.name for s in syms}
    assert "USDCADm" in names, f"USDCADm not in catalog; got {len(names)} symbols"


def test_detect_symbols_categorises_usdcadm_as_forex() -> None:
    syms = detect_symbols(MT5_TERMINAL)
    usdcadm = next(s for s in syms if s.name == "USDCADm")
    assert usdcadm.category == "Forex"
    assert usdcadm.digits == 5
    assert usdcadm.point == pytest.approx(1e-5)


def test_detect_symbols_by_category_filters() -> None:
    syms = detect_symbols_by_category(MT5_TERMINAL, "Forex")
    assert all(s.category == "Forex" for s in syms)
    assert len(syms) > 10  # broker should have plenty


# --- spec snapshot -----------------------------------------------------------

def test_snapshot_usdcadm_has_profit_loss_tick_values() -> None:
    spec = snapshot_symbol_spec(MT5_TERMINAL, "USDCADm")
    # Exness quotes profit and loss tick values separately. They should be
    # finite, positive, and within 1% of each other.
    assert spec.trade_tick_value_profit > 0
    assert spec.trade_tick_value_loss > 0
    assert (
        abs(spec.trade_tick_value_profit - spec.trade_tick_value_loss)
        / spec.trade_tick_value_profit
        < 0.01
    )


def test_snapshot_usdcadm_contract_terms() -> None:
    spec = snapshot_symbol_spec(MT5_TERMINAL, "USDCADm")
    assert spec.contract_size == pytest.approx(100_000)
    assert spec.digits == 5
    assert spec.point == pytest.approx(1e-5)
    assert spec.currency_base == "USD"
    assert spec.currency_profit == "CAD"
    assert spec.volume_min > 0
    assert spec.volume_step > 0
    assert spec.volume_max > spec.volume_min
    # Exness charges weekend roll on Wednesday.
    assert spec.swap_rollover_triple_day == 2


def test_snapshot_unknown_symbol_raises() -> None:
    from engine.core.errors import DataError
    with pytest.raises(DataError, match="not found in catalog"):
        snapshot_symbol_spec(MT5_TERMINAL, "NOSUCH_SYMBOL_XYZ")


def test_snapshot_is_json_serialisable() -> None:
    import json
    spec = snapshot_symbol_spec(MT5_TERMINAL, "USDCADm")
    d = spec.to_dict()
    s = json.dumps(d, default=str)
    assert "USDCADm" in s


def test_normalise_volume_snaps_to_step() -> None:
    spec = snapshot_symbol_spec(MT5_TERMINAL, "USDCADm")
    v = spec.normalise_volume(0.123)
    assert abs(v - round(0.123 / spec.volume_step) * spec.volume_step) < 1e-9
    v_small = spec.normalise_volume(0.0001)
    assert v_small >= spec.volume_min
    v_huge = spec.normalise_volume(1e9)
    assert v_huge <= spec.volume_max


def test_spread_in_price_calculation() -> None:
    spec = snapshot_symbol_spec(MT5_TERMINAL, "USDCADm")
    expected = spec.spread_points * spec.point
    assert spec.spread_in_price() == pytest.approx(expected)


# --- bar fetch ---------------------------------------------------------------

def test_fetch_bars_returns_data_for_all_timeframes() -> None:
    # End on a completed trading week. A live MT5 terminal can legitimately
    # have no W1 bar for the current Sunday-to-Sunday window, even though its
    # history is healthy; this test is about timeframe mapping, not the
    # presence of an in-progress weekly candle.
    date_to = datetime.now(tz=UTC) - timedelta(days=14)
    date_from = date_to - timedelta(days=30)

    # Keep the endpoint aligned to a completed UTC day for deterministic
    # inclusion of the final daily/weekly bar across broker time zones.
    date_to = date_to.replace(hour=23, minute=59, second=59, microsecond=0)
    date_from = date_from.replace(hour=0, minute=0, second=0, microsecond=0)


    for tf in Timeframe:
        bars = fetch_bars(MT5_TERMINAL, "USDCADm", tf, date_from, date_to)
        assert isinstance(bars, BarData)
        assert bars.symbol == "USDCADm"
        assert bars.timeframe == tf
        assert bars.n_bars > 0, f"no bars for {tf.value}"


def test_fetch_bars_rejects_inverted_range() -> None:
    from engine.core.errors import DataError
    now = datetime.now(tz=UTC)
    with pytest.raises(DataError, match="date_from must be < date_to"):
        fetch_bars(MT5_TERMINAL, "USDCADm", Timeframe.H2, now, now - timedelta(days=1))


def test_fetch_bars_h2_returns_at_least_50_bars_in_30_days() -> None:
    date_to = datetime.now(tz=UTC)
    date_from = date_to - timedelta(days=30)
    bars = fetch_bars(MT5_TERMINAL, "USDCADm", Timeframe.H2, date_from, date_to)
    # 30 days * 12 H2 bars/day * 5/7 weekdays ~= 257; even partial gives > 50.
    assert bars.n_bars > 50


def test_fetch_bars_ohlc_consistent() -> None:
    date_to = datetime.now(tz=UTC)
    date_from = date_to - timedelta(days=30)
    bars = fetch_bars(MT5_TERMINAL, "USDCADm", Timeframe.H2, date_from, date_to)
    import numpy as np
    assert np.all(bars.low <= bars.open)
    assert np.all(bars.low <= bars.close)
    assert np.all(bars.high >= bars.open)
    assert np.all(bars.high >= bars.close)
