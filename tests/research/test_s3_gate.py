"""S3 GATE: backtester for USDCADm H2, MA-crossover strategy.

The hard gate property: a hand-computed MA-crossover backtest on the
SAME data must match the engine's output to the cent, trade by trade.

Hand computation rule:
    For each trade (i, j) where i is entry bar, j is exit bar:
        entry_fill = bars.open[i+1]  (next-bar-open execution)
        exit_fill  = bars.open[j+1]
        gross = sign * (exit_fill - entry_fill) * per_unit(leg) * volume
        net   = gross - entry_costs - exit_costs
    UPDATED IN S1/W3. This hand computation used
    max(tick_value_profit, tick_value_loss) for BOTH legs, matching the
    engine at the time. MT5 reports two values because they are not the
    same number, and S2 6.3 says per leg: a winning leg takes
    tick_value_profit, a losing one tick_value_loss. The engine now does
    that, so the expectation here follows. Only the tick-value selection
    changed; every price, bar index and cost term is untouched.

    where per_unit(leg) = (tick_value_profit if the leg won
                           else tick_value_loss) /
          trade_tick_size (MT5's trade_tick_size is in PRICE UNITS per
          the S2 finding).

The GATE compares:
  - Same N trades
  - Same trade direction sequence
  - Same entry/exit bar indices (within 1)
  - Same net P&L per trade to within 1e-6 USD
  - Same final equity to within 1e-6 USD
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from engine.research.backtest.engine import BacktestConfig, BacktestEngine
from engine.research.backtest.strategies.ma_crossover import (
    MACrossConfig,
    MACrossStrategy,
)
from engine.research.data.mt5_source import BarData, Timeframe, fetch_bars
from tests.mt5_attach import acct1_terminal

MT5_TERMINAL = acct1_terminal()  # acct1's terminal; the live test attaches by login (mt5_acct1)


def _per_unit_for(spec, signed_move: float) -> float:
    """Account ccy per 1.0 price move per lot, for the leg that happened.

    W3: MT5 reports trade_tick_value_profit and trade_tick_value_loss
    because they are not the same number, and S2 6.3 applies them per
    leg. A winning leg takes the profit value, a losing one the loss
    value. This hand computation previously used max() for both, which
    is what the engine did at the time.
    """
    value = (
        spec.trade_tick_value_profit if signed_move >= 0.0
        else spec.trade_tick_value_loss
    )
    return value / spec.trade_tick_size


def _build_synthetic_bars(n: int = 60, *, seed: int = 7, start_price: float = 1.36) -> BarData:
    """Build deterministic BarData with a known cross pattern.

    The walk is up-trend for the first half (so MA fast > MA slow
    eventually), then a noisy range, then a down-trend (so MA fast <
    MA slow). This guarantees at least one cross in each direction.
    """
    rng = np.random.default_rng(seed)
    base = start_price
    rets = np.zeros(n, dtype=np.float64)
    if n >= 20:
        rets[:20] = rng.normal(loc=0.0005, scale=0.0005, size=20)
    if n >= 40:
        rets[20:40] = rng.normal(loc=0.0, scale=0.001, size=20)
    if n > 40:
        rets[40:] = rng.normal(loc=-0.0005, scale=0.0005, size=n - 40)
    # Fill any tail noise if n is between the thresholds.
    if 1 <= n < 20:
        rets[:] = rng.normal(loc=0.0001, scale=0.0005, size=n)
    close = base * np.cumprod(1.0 + rets)
    open_ = np.concatenate([[base], close[:-1]])
    high = np.maximum(open_, close) + 0.0002
    low = np.minimum(open_, close) - 0.0002
    return BarData(
        time=(np.arange(n, dtype="int64") * 7200) + 1_700_000_000,
        open=open_.astype(np.float64),
        high=high.astype(np.float64),
        low=low.astype(np.float64),
        close=close.astype(np.float64),
        tick_volume=np.full(n, 1000, dtype="int64"),
        spread=np.full(n, 14, dtype="int32"),
        real_volume=np.zeros(n, dtype="int64"),
        symbol="USDCADm",
        timeframe=Timeframe.H2,
        source="synthetic:test",
    )


def _build_test_spec():
    from engine.research.data.symbol_spec import SymbolSpec
    return SymbolSpec(
        name="USDCADm",
        path="Standard\\Forex\\USDCADm",
        visible=True,
        digits=5,
        point=1e-5,
        contract_size=100_000,
        trade_tick_size=1e-5,        # in price units (MT5 convention)
        trade_tick_value=0.717891,
        trade_tick_value_profit=0.717891,
        trade_tick_value_loss=0.717963,
        spread_points=14,
        currency_base="USD",
        currency_profit="CAD",
        volume_min=0.01,
        volume_step=0.01,
        volume_max=100.0,
        swap_long=-2.5,
        swap_short=0.5,
        swap_rollover_triple_day=2,
        freeze_level=0,
        stop_level=0,
        snapshot_at_utc="2024-01-01T00:00:00+00:00",
        source="test",
    )


# ----------------------------------------------------------------------------
# Unit-level tests
# ----------------------------------------------------------------------------

def test_engine_rejects_too_few_bars() -> None:
    from engine.core.errors import DataError
    spec = _build_test_spec()
    cfg = BacktestConfig(strategy_id="x", symbol="USDCADm", timeframe="H2")
    bars = _build_synthetic_bars(n=1)
    eng = BacktestEngine(spec, cfg, rng_seed=0)
    strat = MACrossStrategy()
    with pytest.raises(DataError):
        eng.run(bars, strat)


def test_engine_rejects_symbol_mismatch() -> None:
    from engine.core.errors import DataError
    spec = _build_test_spec()
    cfg = BacktestConfig(strategy_id="x", symbol="USDCADm", timeframe="H2")
    bars = _build_synthetic_bars(n=60)
    # BarData is frozen; construct a new one with the wrong symbol.
    from engine.research.data.mt5_source import BarData
    bars = BarData(
        time=bars.time, open=bars.open, high=bars.high, low=bars.low,
        close=bars.close, tick_volume=bars.tick_volume, spread=bars.spread,
        real_volume=bars.real_volume,
        symbol="EURUSDm",  # mismatch on purpose
        timeframe=bars.timeframe, source=bars.source,
    )
    eng = BacktestEngine(spec, cfg, rng_seed=0)
    strat = MACrossStrategy()
    with pytest.raises(DataError):
        eng.run(bars, strat)


def test_engine_runs_on_synthetic_data() -> None:
    spec = _build_test_spec()
    cfg = BacktestConfig(
        strategy_id="ma_test",
        symbol="USDCADm",
        timeframe="H2",
        initial_equity=10_000.0,
        slippage_mean_points=0.0,    # zero slippage for determinism
        slippage_std_points=0.0,
    )
    bars = _build_synthetic_bars(n=60)
    eng = BacktestEngine(spec, cfg, rng_seed=0)
    strat = MACrossStrategy(MACrossConfig(fast=5, slow=15, volume=0.1, strategy_id="ma_test"))
    result = eng.run(bars, strat)
    assert result.bars_n == 60
    assert result.equity_curve.shape == (61,)
    # All trades must have volume 0.1 (lot step snaps).
    for t in result.trades:
        assert abs(t.volume - 0.1) < 1e-9


# ----------------------------------------------------------------------------
# THE S3 GATE: hand-computed verification to the cent
# ----------------------------------------------------------------------------

def _expected_trade_ledger(bars: BarData, spec, fast: int, slow: int, volume: float) -> list[dict]:
    """Re-derive the trade ledger by hand, mirroring the strategy + engine.

    Returns a list of dicts with:
        direction, entry_bar, exit_bar, entry_price, exit_price, net_pnl

    Includes BOTH closed and still-open trades (the open one has
    exit_price=None and net_pnl based on the last bar's close).
    """
    close = bars.close
    n = bars.n_bars
    # Costs are not a leg, so they keep a single factor.
    per_unit_per_lot = max(
        spec.trade_tick_value_profit, spec.trade_tick_value_loss
    ) / spec.trade_tick_size

    trades = []
    direction = None
    entry_bar = None
    for i in range(slow, n - 1):
        fast_now = float(np.mean(close[i - fast + 1 : i + 1]))
        slow_now = float(np.mean(close[i - slow + 1 : i + 1]))
        fast_prev = float(np.mean(close[i - fast : i]))
        slow_prev = float(np.mean(close[i - slow : i]))
        # Bullish cross
        if fast_prev <= slow_prev and fast_now > slow_now:
            if direction == "short":
                exit_price = float(bars.open[i + 1])
                entry_price = float(bars.open[entry_bar + 1])
                sign = -1.0
                gross = (
                    sign * (exit_price - entry_price)
                    * _per_unit_for(spec, sign * (exit_price - entry_price))
                    * volume
                )
                spread_ccy = (spec.spread_points * spec.point) * per_unit_per_lot * volume
                net = gross - spread_ccy
                trades.append({
                    "direction": "short",
                    "entry_bar": entry_bar,
                    "exit_bar": i,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "net_pnl": net,
                })
                entry_bar = None
            if direction != "long":
                entry_bar = i
                direction = "long"
        # Bearish cross
        elif fast_prev >= slow_prev and fast_now < slow_now:
            if direction == "long":
                exit_price = float(bars.open[i + 1])
                entry_price = float(bars.open[entry_bar + 1])
                sign = 1.0
                gross = (
                    sign * (exit_price - entry_price)
                    * _per_unit_for(spec, sign * (exit_price - entry_price))
                    * volume
                )
                spread_ccy = (spec.spread_points * spec.point) * per_unit_per_lot * volume
                net = gross - spread_ccy
                trades.append({
                    "direction": "long",
                    "entry_bar": entry_bar,
                    "exit_bar": i,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "net_pnl": net,
                })
                entry_bar = None
            if direction != "short":
                entry_bar = i
                direction = "short"
    # If there's an open position at the end, include it with the
    # mark-to-market price (last bar's close) and a net_pnl of
    # unrealised P&L (gross only; no exit costs in this simplified model).
    if direction is not None and entry_bar is not None:
        entry_price = float(bars.open[entry_bar + 1])
        last_close = float(close[-1])
        sign = 1.0 if direction == "long" else -1.0
        gross = (
            sign * (last_close - entry_price)
            * _per_unit_for(spec, sign * (last_close - entry_price))
            * volume
        )
        spread_ccy = (spec.spread_points * spec.point) * per_unit_per_lot * volume
        net = gross - spread_ccy
        trades.append({
            "direction": direction,
            "entry_bar": entry_bar,
            "exit_bar": None,
            "entry_price": entry_price,
            "exit_price": None,
            "net_pnl": net,
        })
    return trades


def test_gate_ma_crossover_matches_hand_computation_to_the_cent() -> None:
    """The S3 hard gate: a hand-computed MA-crossover backtest on the
    SAME data must match the engine's output to within 1e-6 USD per
    trade and in final equity.
    """
    spec = _build_test_spec()
    cfg = BacktestConfig(
        strategy_id="ma_gate",
        symbol="USDCADm",
        timeframe="H2",
        initial_equity=10_000.0,
        slippage_mean_points=0.0,
        slippage_std_points=0.0,
        charge_swap=False,           # synthetic data may not have valid dates
    )
    bars = _build_synthetic_bars(n=60)
    eng = BacktestEngine(spec, cfg, rng_seed=0)
    strat = MACrossStrategy(MACrossConfig(fast=5, slow=15, volume=0.1, strategy_id="ma_gate"))
    result = eng.run(bars, strat)

    expected = _expected_trade_ledger(bars, spec, fast=5, slow=15, volume=0.1)

    # Same N trades (open + closed).
    assert len(result.trades) == len(expected), (
        f"engine produced {len(result.trades)} trades, expected {len(expected)}"
    )

    # Trade-by-trade comparison.
    for k, (engine_trade, exp) in enumerate(zip(result.trades, expected, strict=False)):
        # Direction
        assert engine_trade.direction == exp["direction"]
        # Entry/exit bar indices (engine uses bar at which the cross was
        # detected; exit fill is at i+1 which is the next-bar-open)
        assert engine_trade.entry_bar_index == exp["entry_bar"], (
            f"trade {k}: entry bar engine={engine_trade.entry_bar_index} "
            f"expected={exp['entry_bar']}"
        )
        assert engine_trade.exit_bar_index == exp["exit_bar"], (
            f"trade {k}: exit bar engine={engine_trade.exit_bar_index} "
            f"expected={exp['exit_bar']}"
        )
        # For the per-trade P&L, closed trades use exit_fill; open trades
        # use the last bar's close (mark-to-market).
        if exp["exit_bar"] is not None:
            exit_price = engine_trade.exit_fill.filled_price
        else:
            exit_price = float(bars.close[-1])
        expected_net = exp["net_pnl"]
        engine_net = (
            (1.0 if engine_trade.direction == "long" else -1.0)
            * (exit_price - engine_trade.entry_fill.filled_price)
            * _per_unit_for(
                spec,
                (1.0 if engine_trade.direction == "long" else -1.0)
                * (exit_price - engine_trade.entry_fill.filled_price)
            )
            * engine_trade.volume
        ) - engine_trade.entry_fill.total_cost - engine_trade.swap_charged
        assert abs(engine_net - expected_net) < 1e-6, (
            f"trade {k}: engine_net={engine_net:.6f} expected={expected_net:.6f}"
        )

    # Final equity = initial_equity + sum(net_pnl for closed trades) +
    # mark-to-market on the open trade (its net_pnl above is unrealised).
    # Engine's reported final_equity uses the same formula: realised_total
    # (sum of closed PnL) + last-bar unrealised on the open position.
    closed_net = sum(t["net_pnl"] for t in expected if t["exit_bar"] is not None)
    open_unrealised = sum(t["net_pnl"] for t in expected if t["exit_bar"] is None)
    expected_final = cfg.initial_equity + closed_net + open_unrealised
    assert abs(result.final_equity - expected_final) < 1e-6, (
        f"final equity engine={result.final_equity:.6f} "
        f"expected={expected_final:.6f} "
        f"(closed_net={closed_net:.6f}, open_unrealised={open_unrealised:.6f})"
    )


# ----------------------------------------------------------------------------
# Smoke: engine works on REAL Exness data
# ----------------------------------------------------------------------------

@pytest.mark.usefixtures("mt5_acct1")
def test_gate_runs_on_real_usdcadm_h2() -> None:
    """Smoke test: the engine produces a non-empty result on real Exness
    H2 data. Per-trade P&L verification is the unit test above; here
    we only assert the engine does not crash and produces sensible
    shapes.
    """
    from engine.research.data.mt5_source import snapshot_symbol_spec
    spec = snapshot_symbol_spec(MT5_TERMINAL, "USDCADm")
    cfg = BacktestConfig(
        strategy_id="ma_smoke",
        symbol="USDCADm",
        timeframe="H2",
        initial_equity=10_000.0,
        slippage_mean_points=0.5,
        slippage_std_points=0.2,
    )
    date_to = datetime.now(tz=UTC)
    date_from = date_to - timedelta(days=30)
    bars = fetch_bars(MT5_TERMINAL, "USDCADm", Timeframe.H2, date_from, date_to)
    eng = BacktestEngine(spec, cfg, rng_seed=0)
    strat = MACrossStrategy(MACrossConfig(fast=5, slow=20, volume=0.1, strategy_id="ma_smoke"))
    result = eng.run(bars, strat)
    assert result.bars_n == bars.n_bars
    assert result.equity_curve.shape == (bars.n_bars + 1,)
    # No NaNs in equity.
    assert np.all(np.isfinite(result.equity_curve))
    # All trade volumes snap to lot step.
    for t in result.trades:
        assert abs(t.volume - round(t.volume / spec.volume_step) * spec.volume_step) < 1e-9
