"""W3: the persistent symbol-spec store.

The backtester knew two symbols, hardcoded. Everything else raised, so
XAUUSDm -- the one calibrated symbol with a zero-spread hour -- could
never reach the engine, and F1's no-trade window had no production cell
to fire on.
"""
from __future__ import annotations

import json

import pytest

from engine.core.errors import DataError
from engine.research.backtest.engine import BacktestConfig, BacktestEngine
from engine.research.backtest.order import OrderIntent
from engine.research.data.mt5_source import BarData, Timeframe
from engine.research.data.symbol_specs import (
    STOPS_LEVEL_FALLBACK_PIPS,
    available_symbols,
    effective_stops_level_points,
    load_spec,
    mt5_dow_to_python_weekday,
    save_snapshot,
    to_symbol_spec,
    validate_snapshot,
)

CATALOG = (
    "EURUSDm", "GBPUSDm", "USDJPYm", "USDCADm", "AUDUSDm", "NZDUSDm",
    "USDCHFm", "GBPJPYm", "EURJPYm", "EURGBPm",
    "XAUUSDm", "XAGUSDm", "US30m", "US500m",
)


def _snapshot(**overrides):
    data = {
        "name": "TESTm",
        "path": "Forex\\TESTm",
        "visible": True,
        "digits": 5,
        "point": 1e-05,
        "spread": 12,
        "currency_base": "EUR",
        "currency_profit": "USD",
        "trade_contract_size": 100000.0,
        "trade_tick_size": 1e-05,
        "trade_tick_value": 1.0,
        "trade_tick_value_profit": 1.0,
        "trade_tick_value_loss": 1.2,
        "trade_stops_level": 0,
        "trade_freeze_level": 0,
        "volume_min": 0.01,
        "volume_max": 200.0,
        "volume_step": 0.01,
        "swap_long": -0.5,
        "swap_short": -8.4,
        "swap_mode": 1,
        "swap_rollover3days": 3,
        "snapshot_utc": "2026-09-23T00:00:00+00:00",
        "account_login": 1,
        "server": "test",
        "source": "MT5:test",
    }
    data.update(overrides)
    return data


# ---------------------------------------------------------------------------
# The day-of-week trap
# ---------------------------------------------------------------------------

def test_mt5_day_of_week_is_converted_to_python_weekday() -> None:
    """MT5 counts from Sunday, Python from Monday. costs/swap.py uses
    date.weekday(), so the raw enum would charge the 3x roll a day late.
    """
    # MT5: SUNDAY=0 MONDAY=1 ... SATURDAY=6
    # Python: MONDAY=0 ... SUNDAY=6
    assert mt5_dow_to_python_weekday(0) == 6   # Sunday
    assert mt5_dow_to_python_weekday(1) == 0   # Monday
    assert mt5_dow_to_python_weekday(3) == 2   # Wednesday
    assert mt5_dow_to_python_weekday(4) == 3   # Thursday
    assert mt5_dow_to_python_weekday(6) == 5   # Saturday

    spec = to_symbol_spec(_snapshot(swap_rollover3days=3))
    assert spec.swap_rollover_triple_day == 2, "Wednesday in Python terms"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "field", ["trade_tick_size", "trade_contract_size",
              "trade_tick_value_profit", "trade_tick_value_loss"],
)
def test_non_positive_multiplicative_inputs_are_refused(field) -> None:
    """A zero tick value silently zeroes P&L instead of failing."""
    check = validate_snapshot(_snapshot(**{field: 0.0}))
    assert not check.ok
    assert any(field in e for e in check.errors)

    with pytest.raises(DataError):
        to_symbol_spec(_snapshot(**{field: 0.0}))


def test_zero_stops_level_warns_and_uses_the_documented_floor() -> None:
    check = validate_snapshot(_snapshot(trade_stops_level=0))
    assert check.ok, "a zero stops level is a warning, not a refusal"
    assert any("fallback" in w for w in check.warnings)

    # 5-digit quote: a pip is 10 points, so the 2-pip floor is 20 points.
    assert effective_stops_level_points(_snapshot(digits=5)) == int(
        STOPS_LEVEL_FALLBACK_PIPS * 10
    )
    # 2-digit index: a pip is 1 point.
    assert effective_stops_level_points(
        _snapshot(digits=2, trade_stops_level=0)
    ) == int(STOPS_LEVEL_FALLBACK_PIPS)
    # A real broker minimum is used as-is.
    assert effective_stops_level_points(_snapshot(trade_stops_level=35)) == 35


def test_round_trip_through_disk(tmp_path) -> None:
    data = _snapshot()
    path = save_snapshot(data, tmp_path)
    assert json.loads(path.read_text(encoding="utf-8")) == data
    spec = load_spec("TESTm", tmp_path)
    assert spec is not None
    assert spec.name == "TESTm"
    assert spec.trade_tick_value_profit == 1.0
    assert spec.trade_tick_value_loss == 1.2
    assert load_spec("NOTHINGm", tmp_path) is None
    assert available_symbols(tmp_path) == ("TESTm",)


# ---------------------------------------------------------------------------
# The ingested catalog
# ---------------------------------------------------------------------------

def test_every_catalog_symbol_is_ingested_and_usable() -> None:
    """All 14, including the metals and indices that used to raise."""
    present = set(available_symbols())
    missing = [s for s in CATALOG if s not in present]
    if missing:
        pytest.skip(
            f"specs not ingested on this machine ({missing}); "
            "run scripts/ingest_symbol_specs.py"
        )
    for symbol in CATALOG:
        spec = load_spec(symbol)
        assert spec is not None, symbol
        assert spec.trade_tick_size > 0, symbol
        assert spec.contract_size > 0, symbol
        assert spec.trade_tick_value_profit > 0, symbol
        assert spec.trade_tick_value_loss > 0, symbol
        assert spec.snapshot_at_utc, f"{symbol} has no provenance stamp"


def test_xauusdm_is_backtestable() -> None:
    """The symbol F1's window exists for. It used to raise here."""
    from engine.research.search.real_backtest import spec_for

    if load_spec("XAUUSDm") is None:
        pytest.skip("XAUUSDm spec not ingested on this machine")
    spec = spec_for("XAUUSDm")
    assert spec.name == "XAUUSDm"
    assert spec.contract_size > 0


# ---------------------------------------------------------------------------
# Per-leg tick values
# ---------------------------------------------------------------------------

def _flat_bars(n: int, price: float) -> BarData:
    import numpy as np

    close = np.full(n, price, dtype=np.float64)
    return BarData(
        time=np.array([86400 + i * 3600 for i in range(n)], dtype=np.int64),
        open=close.copy(),
        high=close + 0.01,
        low=close - 0.01,
        close=close.copy(),
        tick_volume=np.ones(n),
        spread=np.zeros(n, dtype=np.int32),
        real_volume=np.zeros(n),
        symbol="TESTm",
        timeframe=Timeframe.H1,
    )


class _OpenThenClose:
    strategy_id = "open-close"

    def __init__(self, direction: str) -> None:
        self.direction = direction

    def on_bar(self, bar_index, bars, state):
        if bar_index == 1:
            return OrderIntent(
                direction=self.direction, volume=1.0, bar_index=bar_index
            )
        if bar_index == 4:
            return OrderIntent(
                direction=self.direction, volume=0.0, bar_index=bar_index
            )
        return None


def test_tick_value_profit_vs_loss() -> None:
    """A winning leg uses tick_value_profit, a losing leg uses _loss.

    Fixture: the two values differ by a factor of 2, far larger than any
    real symbol, so a single-value implementation cannot pass by
    rounding. Price moves +0.00100 (10 pips on a 5-digit quote).

        per_unit_profit = 1.0 / 1e-05 = 100_000
        per_unit_loss   = 2.0 / 1e-05 = 200_000

    LONG into a rise is a WIN:  +0.00100 * 100_000 * 1.0 lot = +100.00
    SHORT into a rise is a LOSS: -0.00100 * 200_000 * 1.0 lot = -200.00

    Taking max() for both, as the engine used to, would report -200.00
    for the win and -200.00 for the loss -- the loss value applied to
    every leg.
    """
    import numpy as np

    spec = to_symbol_spec(
        _snapshot(trade_tick_value_profit=1.0, trade_tick_value_loss=2.0)
    )
    assert spec.trade_tick_value_profit * 2 == spec.trade_tick_value_loss

    n = 8
    price = np.full(n, 1.10000)
    # Entry fills at bar 2's open (signal bar 1 + 1), exit at bar 5's
    # open (signal bar 4 + 1), so the step must sit between them.
    price[3:] = 1.10100          # +0.00100 from bar 3 onward
    bars = BarData(
        time=np.array([86400 + i * 3600 for i in range(n)], dtype=np.int64),
        open=price.copy(),
        high=price + 0.00001,
        low=price - 0.00001,
        close=price.copy(),
        tick_volume=np.ones(n),
        spread=np.zeros(n, dtype=np.int32),
        real_volume=np.zeros(n),
        symbol="TESTm",
        timeframe=Timeframe.H1,
    )
    config = BacktestConfig(
        strategy_id="legs",
        symbol="TESTm",
        timeframe="H1",
        initial_equity=10_000.0,
        charge_swap=False,
        charge_commission=False,
        slippage_mean_points=0.0,
        slippage_std_points=0.0,
    )

    long_result = BacktestEngine(spec=spec, config=config, rng_seed=0).run(
        bars, _OpenThenClose("long")
    )
    short_result = BacktestEngine(spec=spec, config=config, rng_seed=0).run(
        bars, _OpenThenClose("short")
    )

    long_trade = long_result.trades[0]
    short_trade = short_result.trades[0]
    assert long_trade.exit_fill is not None
    assert short_trade.exit_fill is not None

    assert long_trade.gross_price_pnl == pytest.approx(100.0, abs=1e-6), (
        "the winning leg did not use trade_tick_value_profit"
    )
    assert short_trade.gross_price_pnl == pytest.approx(-200.0, abs=1e-6), (
        "the losing leg did not use trade_tick_value_loss"
    )
    # And the two are not the same magnitude, which a single value would force.
    assert abs(long_trade.gross_price_pnl) != pytest.approx(
        abs(short_trade.gross_price_pnl), abs=1e-6
    )
