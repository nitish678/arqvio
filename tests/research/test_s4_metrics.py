"""S4 GATE: research metrics module.

WHAT: Verifies that every metric in :class:`ResearchMetrics` is
      computable, deterministic, and matches hand-computed expected
      values on synthetic inputs.

WHY: The S3 engine has its own gate; this is the S4 gate. It runs
     in isolation - no MT5, no engine.run() - by constructing
     :class:`BacktestResult` objects with known shapes and
     asserting the orchestrator returns the right values.

HOW: A ``_make_result`` helper builds a BacktestResult from a
     hand-crafted equity curve, an optional closed-trade ledger,
     and a small set of config knobs. Tests then call
     ``compute_metrics(result)`` and assert against the expected
     hand-computed values.

TRADE-OFF: We do NOT exercise the engine here; the S3 gate
           (``tests/research/test_s3_gate.py``) already does. S4
           tests focus on the metrics math, not the engine wiring.
"""
from __future__ import annotations

from typing import Final

import numpy as np
import pytest

from engine.research.backtest.engine import BacktestConfig, BacktestResult
from engine.research.backtest.order import Fill, Trade
from engine.research.metrics import compute_metrics
from engine.research.metrics.compute import _derive_periods_per_year
from engine.research.metrics.cost_impact import cost_impact
from engine.research.metrics.distribution import (
    avg_loss,
    avg_r_multiple,
    avg_win,
    expectancy,
    profit_factor,
    win_rate,
)
from engine.research.metrics.drawdown import max_drawdown, time_to_recovery_bars
from engine.research.metrics.exposure import exposure_time, months_in_run
from engine.research.metrics.ratios import (
    annualised_sharpe,
    annualised_sortino,
    calmar,
)
from engine.research.metrics.returns import (
    bar_returns,
    closed_trade_pnl_split,
    trade_returns,
)
from engine.research.metrics.stability import rolling_sharpe_stability

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PPY_H2: Final[int] = 252 * 12   # USDCADm H2


def _make_fill(price: float, *, cost: float = 0.0) -> Fill:
    """A minimal Fill with a given filled_price and total cost."""
    return Fill(
        fill_id="f",
        order_id="1",
        direction="long",
        volume=0.1,
        requested_price=price,
        filled_price=price,
        spread_cost=cost / 2.0,
        slippage_cost=0.0,
        commission_cost=cost / 2.0,
        bar_index=0,
        bar_time=1_700_000_000,
    )


def _make_trade(
    *,
    direction: str = "long",
    entry_price: float = 1.0,
    exit_price: float | None = 1.01,
    gross: float = 10.0,
    net: float = 8.0,
    entry_bar: int = 0,
    exit_bar: int | None = 5,
    entry_cost: float = 2.0,
) -> Trade:
    """A closed (or still-open) Trade with explicit gross/net P&L."""
    entry_fill = _make_fill(entry_price, cost=entry_cost)
    exit_fill = _make_fill(exit_price) if exit_price is not None else None
    return Trade(
        trade_id="t",
        strategy_id="s",
        symbol="USDCADm",
        direction=direction,  # type: ignore[arg-type]
        entry_fill=entry_fill,
        exit_fill=exit_fill,
        entry_bar_index=entry_bar,
        exit_bar_index=exit_bar,
        volume=0.1,
        gross_price_pnl=gross,
        realized_pnl=net,
    )


def _make_result(
    equity: np.ndarray,
    *,
    trades: tuple[Trade, ...] = (),
    timeframe: str = "H2",
    initial: float = 10_000.0,
) -> BacktestResult:
    """Build a BacktestResult from a hand-crafted equity + ledger."""
    eq = np.asarray(equity, dtype=np.float64)
    cfg = BacktestConfig(
        strategy_id="s",
        symbol="USDCADm",
        timeframe=timeframe,
        initial_equity=initial,
    )
    final = float(eq[-1]) if eq.size > 0 else initial
    return BacktestResult(
        config=cfg,
        spec_symbol="USDCADm",
        bars_n=int(eq.size) - 1 if eq.size > 0 else 0,
        trades=trades,
        equity_curve=eq,
        final_equity=final,
        n_trades=len(trades),
        n_round_trips=sum(1 for t in trades if t.exit_fill is not None),
        n_open=sum(1 for t in trades if t.exit_fill is None),
    )


# ---------------------------------------------------------------------------
# Helpers / pure-function tests
# ---------------------------------------------------------------------------


def test_max_drawdown_known_answer() -> None:
    # Peak 120, trough 60 -> -0.5
    eq = np.array([100.0, 120.0, 90.0, 60.0, 80.0, 110.0, 120.0])
    assert max_drawdown(eq) == pytest.approx(-0.5, abs=1e-12)


def test_max_drawdown_monotonic_is_zero() -> None:
    eq = np.array([100.0, 101.0, 102.0, 103.0])
    assert max_drawdown(eq) == 0.0


def test_time_to_recovery_recovers() -> None:
    # Peak 120 at idx 1, trough 60 at idx 3, recovers at idx 6 (>=120)
    eq = np.array([100.0, 120.0, 90.0, 60.0, 80.0, 110.0, 120.0])
    assert time_to_recovery_bars(eq) == 6


def test_time_to_recovery_never_recovers() -> None:
    eq = np.array([100.0, 120.0, 90.0, 60.0, 50.0, 40.0])
    assert time_to_recovery_bars(eq) is None


def test_time_to_recovery_no_drawdown_is_none() -> None:
    eq = np.array([100.0, 110.0, 120.0, 130.0])
    assert time_to_recovery_bars(eq) is None


def test_annualised_sharpe_known_answer() -> None:
    # 252 daily returns of +0.001 with std 0.01 -> per-obs Sharpe ~0.1
    # The seed makes the sample mean 0.000516 and std 0.00938
    # (sample, ddof=1), so annualised = 0.0551 * sqrt(252) = 0.874.
    rng = np.random.default_rng(42)
    rets = rng.normal(loc=0.001, scale=0.01, size=252)
    sr = annualised_sharpe(rets, 252)
    assert sr is not None
    assert sr == pytest.approx(0.874, rel=0.02)


def test_annualised_sharpe_zero_variance_is_none() -> None:
    rets = np.full(100, 0.001)
    assert annualised_sharpe(rets, 252) is None


def test_annualised_sharpe_too_few_obs_is_none() -> None:
    rets = np.array([0.01])
    assert annualised_sharpe(rets, 252) is None


def test_annualised_sortino_no_downside_is_none() -> None:
    rets = np.array([0.01, 0.02, 0.015, 0.005])
    assert annualised_sortino(rets, 252) is None


def test_annualised_sortino_known_answer() -> None:
    # Symmetric +/- 0.01 with no bias: per-obs Sortino = mean/downside_dev
    rets = np.array([0.01, -0.01, 0.01, -0.01, 0.01, -0.01, 0.01, -0.01])
    # downside_dev = sqrt(mean((-r)^2 for r<0)) = sqrt(0.0001) = 0.01
    # mean = 0 -> Sortino = 0 (annualised = 0)
    assert annualised_sortino(rets, 252) == pytest.approx(0.0, abs=1e-12)


def test_calmar_zero_dd_is_none() -> None:
    assert calmar(0.10, 0.0) is None
    assert calmar(None, -0.1) is None


def test_calmar_known_answer() -> None:
    # 25% CAGR, -10% max DD -> Calmar = 2.5
    assert calmar(0.25, -0.10) == pytest.approx(2.5, rel=1e-9)


def test_win_rate_avg_win_avg_loss_expectancy() -> None:
    pnls = np.array([100.0, -50.0, 30.0, -20.0, 200.0])
    assert win_rate(pnls) == pytest.approx(0.6, rel=1e-9)
    assert avg_win(pnls) == pytest.approx(110.0, rel=1e-9)   # (100+30+200)/3
    assert avg_loss(pnls) == pytest.approx(-35.0, rel=1e-9)  # (-50-20)/2
    assert expectancy(pnls) == pytest.approx(52.0, rel=1e-9)  # mean


def test_profit_factor_known_answer() -> None:
    pnls = np.array([100.0, 100.0, 100.0, -50.0])
    # gains = 300, |losses| = 50 -> 6.0
    assert profit_factor(pnls) == pytest.approx(6.0, rel=1e-9)


def test_profit_factor_no_losses_is_none() -> None:
    assert profit_factor(np.array([10.0, 20.0, 30.0])) is None


def test_profit_factor_no_gains_is_zero() -> None:
    assert profit_factor(np.array([-10.0, -20.0])) == 0.0


def test_avg_r_multiple_known_answer() -> None:
    # avg win 100, |avg loss| 50 -> R = 2.0
    pnls = np.array([100.0, 100.0, -50.0, -50.0])
    assert avg_r_multiple(pnls) == pytest.approx(2.0, rel=1e-9)


def test_avg_r_multiple_no_losses_is_none() -> None:
    assert avg_r_multiple(np.array([10.0, 20.0])) is None


def test_cost_impact_known_answer() -> None:
    # gross = 1000, net = 800 -> cost_impact = 0.2
    g = np.array([200.0, 300.0, 200.0, 300.0])
    n = np.array([160.0, 240.0, 160.0, 240.0])
    assert cost_impact(g, n) == pytest.approx(0.2, abs=1e-12)


def test_cost_impact_no_trades_is_none() -> None:
    assert cost_impact(np.array([]), np.array([])) is None


def test_cost_impact_zero_gross_is_none() -> None:
    # gross sums to 0 -> ratio undefined
    g = np.array([10.0, -10.0])
    n = np.array([9.0, -11.0])
    assert cost_impact(g, n) is None


def test_stability_constant_returns_is_zero() -> None:
    # A flat equity curve -> per-bar returns are all 0 -> per-window
    # std is 0 -> all window-Sharpes are NaN -> stability is None.
    eq = np.full(300, 100.0)
    assert rolling_sharpe_stability(eq) is None


def test_stability_strictly_increasing_is_finite() -> None:
    # Strictly increasing -> per-bar returns are ~0.001, with small
    # floating-point error across the cumprod. Per-window Sharpe is
    # numerically large (tiny std) but finite. Stability should be
    # a finite float, not None.
    eq = 100.0 * np.cumprod(1.0 + np.full(300, 0.001))
    eq = np.concatenate([[100.0], eq])
    s = rolling_sharpe_stability(eq)
    assert s is not None
    assert np.isfinite(s)


def test_stability_known_window_size() -> None:
    # Custom window length. A purely random equity -> finite stability.
    rng = np.random.default_rng(0)
    eq = 100.0 * np.cumprod(1.0 + rng.normal(0, 0.01, 500))
    eq = np.concatenate([[100.0], eq])
    s = rolling_sharpe_stability(eq, window=50)
    assert s is not None
    assert np.isfinite(s)


def test_stability_too_few_bars_is_none() -> None:
    eq = np.linspace(100, 105, 50)  # < 2*63
    assert rolling_sharpe_stability(eq) is None


def test_exposure_time_all_in_market() -> None:
    # 10 bars, 1 open trade covering 0..9 -> 10/10 = 1.0
    t = _make_trade(entry_bar=0, exit_bar=9)
    result = _make_result(np.linspace(100, 110, 11), trades=(t,))
    assert exposure_time(result) == pytest.approx(1.0, abs=1e-12)


def test_exposure_time_no_trades() -> None:
    result = _make_result(np.linspace(100, 110, 11))
    assert exposure_time(result) == pytest.approx(0.0, abs=1e-12)


def test_months_in_run_h2() -> None:
    # 252 * 12 = 3024 bars of H2 = 12 months = 1 year
    assert months_in_run(_PPY_H2, _PPY_H2) == pytest.approx(12.0, abs=1e-12)


def test_bar_returns_drop_non_finite() -> None:
    eq = np.array([100.0, np.nan, 110.0, 120.0])
    r = bar_returns(_make_result(eq))
    # Should drop the nan so we have a clean 2-element series.
    assert r.shape == (2,)
    assert np.all(np.isfinite(r))


def test_trade_returns_drops_open_trades() -> None:
    closed = _make_trade(gross=10.0, net=8.0)
    open_t = _make_trade(gross=0.0, net=0.0, exit_price=None, exit_bar=None)
    result = _make_result(
        np.linspace(100, 110, 11), trades=(closed, open_t),
    )
    r = trade_returns(result)
    assert r.shape == (1,)
    assert r[0] == pytest.approx(8.0, abs=1e-12)


def test_closed_trade_pnl_split() -> None:
    closed = _make_trade(gross=10.0, net=8.0)
    open_t = _make_trade(gross=0.0, net=0.0, exit_price=None, exit_bar=None)
    result = _make_result(
        np.linspace(100, 110, 11), trades=(closed, open_t),
    )
    g, n = closed_trade_pnl_split(result)
    assert g.shape == (1,)
    assert n.shape == (1,)
    assert g[0] == pytest.approx(10.0, abs=1e-12)
    assert n[0] == pytest.approx(8.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Orchestrator tests
# ---------------------------------------------------------------------------


def test_compute_metrics_orchestrates_all_fields() -> None:
    # Hand-built equity: 60 bars, monotonic 1% gain, 1 closed trade.
    eq = np.linspace(100.0, 160.0, 61)
    trade = _make_trade(gross=60.0, net=50.0, entry_bar=0, exit_bar=59)
    result = _make_result(eq, trades=(trade,))
    m = compute_metrics(
        result, dataset_version_id="v1", data_hash="abc",
    )
    # Spot-check every category. Exact values depend on the equity
    # shape, but the SCHEMA must be fully populated.
    assert m.n_trades == 1
    assert m.n_round_trips == 1
    assert m.n_bars == 60
    assert m.initial_equity == 10_000.0
    assert m.dataset_version_id == "v1"
    assert m.data_hash == "abc"
    assert m.periods_per_year == _PPY_H2
    # Monotonic -> max DD = 0, recovery undefined, Calmar None.
    assert m.max_drawdown == 0.0
    assert m.time_to_recovery_bars is None
    assert m.calmar is None
    # Closed trade net = 50 -> win rate 1.0, expectancy 50, PF None
    # (no losses), avg_r None.
    assert m.win_rate == pytest.approx(1.0, abs=1e-12)
    assert m.expectancy == pytest.approx(50.0, abs=1e-12)
    assert m.profit_factor is None
    assert m.avg_r_multiple is None
    # Cost impact: gross 60, net 50 -> (60-50)/60 = 0.1667
    assert m.cost_impact == pytest.approx(10.0 / 60.0, abs=1e-9)
    # Exposure: 60 bars / 60 = 1.0
    assert m.exposure_time == pytest.approx(1.0, abs=1e-12)
    # Stability: monotonic -> windows have zero variance -> None
    assert m.stability is None
    # 60 H2 bars = 60/(_PPY_H2) years ~ 0.0198 < 1y, so CAGR is None.
    assert m.cagr is None
    # Sharpe over 60 monotonic +0.99% per bar -> huge Sharpe but
    # finite; we only assert it's not None.
    assert m.sharpe is not None


def test_compute_metrics_no_trades_edge_case() -> None:
    # Use a flat equity that matches the initial equity so the total
    # return is 0 (otherwise the assertion would be measuring the
    # equity drift, not the no-trades case).
    eq = np.full(11, 1_000.0)
    cfg = BacktestConfig(
        strategy_id="s", symbol="USDCADm", timeframe="H2",
        initial_equity=1_000.0,
    )
    result = BacktestResult(
        config=cfg, spec_symbol="USDCADm",
        bars_n=10, trades=(), equity_curve=eq, final_equity=1_000.0,
        n_trades=0, n_round_trips=0, n_open=0,
    )
    m = compute_metrics(result)
    assert m.n_trades == 0
    assert m.n_round_trips == 0
    assert m.total_return == 0.0
    assert m.cost_impact is None
    assert m.profit_factor is None
    assert m.avg_r_multiple is None
    assert m.expectancy == 0.0
    assert m.win_rate == 0.0
    # Ratios over a flat equity -> None.
    assert m.sharpe is None
    assert m.sortino is None


def test_compute_metrics_deterministic() -> None:
    eq = np.linspace(100.0, 110.0, 101)
    trade = _make_trade(gross=8.0, net=5.0)
    result = _make_result(eq, trades=(trade,))
    m1 = compute_metrics(result, dataset_version_id="x", data_hash="y")
    m2 = compute_metrics(result, dataset_version_id="x", data_hash="y")
    # Frozen dataclass: equality is structural.
    assert m1 == m2


def test_compute_metrics_annualisation_periods() -> None:
    # 252 daily returns drawn from N(0.001, 0.01). The realised sample
    # mean / std are NOT 0.001 / 0.01; we just assert the relationship
    # between two periods_per_year values scales as sqrt(ratio).
    rng = np.random.default_rng(7)
    rets_seq = rng.normal(loc=0.001, scale=0.01, size=252)
    eq = 10_000.0 * np.cumprod(1.0 + np.concatenate([[0.0], rets_seq]))
    result = _make_result(eq, timeframe="D1", initial=10_000.0)
    m = compute_metrics(result)
    m2 = compute_metrics(result, periods_per_year=12)
    # Same realised series; Sharpe should scale by sqrt(12/252).
    assert m.sharpe is not None
    assert m2.sharpe is not None
    ratio = m2.sharpe / m.sharpe
    assert ratio == pytest.approx(np.sqrt(12.0 / 252.0), rel=1e-9)
    assert m2.periods_per_year == 12


def test_derive_periods_per_year_known_map() -> None:
    # H2 -> 252*12, D1 -> 252, unknown -> 252 default.
    assert _derive_periods_per_year(_make_result(np.array([100.0, 101.0]), timeframe="H2")) == 252 * 12
    assert _derive_periods_per_year(_make_result(np.array([100.0, 101.0]), timeframe="D1")) == 252
    assert _derive_periods_per_year(_make_result(np.array([100.0, 101.0]), timeframe="custom")) == 252


def test_compute_metrics_rejects_too_few_bars() -> None:
    from engine.core.errors import DataError
    result = _make_result(np.array([100.0]))
    with pytest.raises(DataError):
        compute_metrics(result)


def test_engine_persists_gross_and_realized_on_closed_trades() -> None:
    """S3 retro: every closed Trade carries the gross AND net P&L
    fields. The S3 engine populates these at the moment of close;
    the metrics layer reads them. This is the S3-S4 seam.

    Pure-data version: build a BacktestResult with hand-crafted
    closed trades and verify compute_metrics reads the persisted
    fields (rather than re-deriving them). The engine wiring itself
    is covered by the S3 gate (``tests/research/test_s3_gate.py``).
    """
    # Hand-craft a 100-bar equity with 3 closed trades summing to a
    # known net P&L; assert that the S4 metrics read the persisted
    # gross/net rather than guessing.
    eq = np.linspace(100.0, 110.0, 101)
    t1 = _make_trade(gross=10.0, net=8.0, entry_bar=0, exit_bar=20, entry_cost=2.0)
    t2 = _make_trade(gross=-5.0, net=-7.0, entry_bar=25, exit_bar=50, entry_cost=2.0)
    t3 = _make_trade(gross=15.0, net=13.0, entry_bar=55, exit_bar=99, entry_cost=2.0)
    result = _make_result(eq, trades=(t1, t2, t3))
    # Sanity: the S4 metrics reflect the persisted fields.
    m = compute_metrics(result)
    # gross sum = 20, net sum = 14 -> cost_impact = 6/20 = 0.3
    assert m.cost_impact == pytest.approx(0.3, abs=1e-12)
    # 2 winners, 1 loser -> win_rate = 2/3
    assert m.win_rate == pytest.approx(2.0 / 3.0, abs=1e-12)
    # expectancy = (8 - 7 + 13) / 3 = 14/3
    assert m.expectancy == pytest.approx(14.0 / 3.0, abs=1e-9)
    # profit_factor = (8 + 13) / 7 = 3.0
    assert m.profit_factor == pytest.approx(21.0 / 7.0, abs=1e-9)
