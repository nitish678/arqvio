"""S3 PART B: the STOCH oracle's append-only series cache changes nothing.

StochCrossStrategy used to recompute stoch_series over the whole visible
window on every on_bar call -- O(n^2) per candidate, which put the 60-DNA
STOCH parity far over budget. StochPrefixCache extends the series at the
tail instead. This is a pure-performance change, so the proof is equality:

  * at EVERY window length the cache returns stoch_d_series(window),
    byte for byte, on bars carrying a flat stretch, a NaN close and a
    +inf high;
  * a window that is not an extension of the cached one (shorter, or a
    changed bar) restarts and is still exact;
  * whole engine runs are identical -- trade list, counters, equity curve --
    to the same strategy recomputing the full series each bar.
"""
from __future__ import annotations

import numpy as np
import pytest

from engine.research.backtest.engine import BacktestConfig, BacktestEngine
from engine.research.backtest.strategies.stoch_cross import (
    StochCrossConfig,
    StochCrossStrategy,
    stoch_series,
)
from engine.research.features.forward_block import bars_up_to
from engine.research.features.stoch import StochPrefixCache, stoch_d_series
from engine.research.search.real_backtest import spec_for

from tests.research.backtest.test_kernel_parity_steps import _bars, slow_trade_rows

PARAMS = [(2, 1, 1), (5, 3, 1), (9, 3, 2), (14, 3, 3), (21, 5, 5)]


def _ohlc(n: int = 400) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(29)
    close = 1.30 + np.cumsum(rng.normal(0.0, 0.001, n))
    high = close + np.abs(rng.normal(0.0, 0.0005, n))
    low = close - np.abs(rng.normal(0.0, 0.0005, n))
    close[120:150] = high[120:150] = low[120:150] = close[119]  # HH == LL
    close[200] = np.nan
    high[260] = np.inf
    return high, low, close


def _same(got: tuple[np.ndarray, np.ndarray], want: tuple[np.ndarray, np.ndarray]) -> bool:
    return all(g.tobytes() == w.tobytes() for g, w in zip(got, want))


@pytest.mark.parametrize("k,d,s", PARAMS)
def test_prefix_cache_equals_full_series_at_every_length(k: int, d: int, s: int) -> None:
    high, low, close = _ohlc()
    cache = StochPrefixCache(period=k, d_period=d, smoothing=s)
    for n in range(high.size + 1):
        got = cache.series(high[:n], low[:n], close[:n])
        want = stoch_d_series(high[:n], low[:n], close[:n], period=k, d_period=d, smoothing=s)
        assert _same(got, want), f"k={k} d={d} s={s}: window length {n} differs"


def test_prefix_cache_restarts_on_a_window_that_is_not_an_extension() -> None:
    high, low, close = _ohlc()
    k, d, s = 9, 3, 2
    cache = StochPrefixCache(period=k, d_period=d, smoothing=s)

    def check(h: np.ndarray, lo: np.ndarray, c: np.ndarray) -> None:
        want = stoch_d_series(h, lo, c, period=k, d_period=d, smoothing=s)
        assert _same(cache.series(h, lo, c), want)

    check(high, low, close)
    check(high[:100], low[:100], close[:100])       # shorter
    changed = close.copy()
    changed[50] += 0.01
    check(high[:300], low[:300], changed[:300])     # a cached bar changed
    check(high, low, changed)                       # extension of that
    check(high[:300], low[:300], close[:300])       # and back


class _FullRecompute(StochCrossStrategy):
    """The pre-cache oracle: the whole series again on every bar."""

    def _get_series(self, bars):
        c = self.config
        return stoch_series(
            np.asarray(bars.high), np.asarray(bars.low), np.asarray(bars.close),
            k_period=c.k_period, d_period=c.d_period, smoothing=c.smoothing,
        )


@pytest.mark.parametrize("sl_atr,tp_atr", [(0.0, 0.0), (1.0, 1.5)])
@pytest.mark.parametrize("k,d,s", [(5, 3, 1), (14, 3, 3)])
def test_engine_runs_identical_to_full_recompute(k, d, s, sl_atr, tp_atr) -> None:
    bars = _bars(n=700)
    cfg = StochCrossConfig(
        k_period=k, d_period=d, smoothing=s, k_oversold=30.0, k_overbought=70.0,
        volume=0.1, strategy_id="prefix", sl_atr=sl_atr, tp_atr=tp_atr,
    )

    def run(strategy):
        return BacktestEngine(
            spec=spec_for("USDCADm"),
            config=BacktestConfig(
                strategy_id="prefix", symbol="USDCADm", timeframe="H1",
                initial_equity=10_000.0, slippage_mean_points=0.3,
                slippage_std_points=0.2, hourly_spread_points=None,
            ),
            rng_seed=0,
        ).run(bars, strategy)

    new, old = run(StochCrossStrategy(cfg)), run(_FullRecompute(cfg))
    assert old.n_trades > 3, "the configuration never traded; nothing is compared"
    assert slow_trade_rows(new) == slow_trade_rows(old)
    assert np.asarray(new.equity_curve).tobytes() == np.asarray(old.equity_curve).tobytes()
    for name in ("n_trades", "n_round_trips", "final_equity", "cost_spread", "cost_slippage"):
        assert getattr(new, name) == getattr(old, name), name
    assert dict(new.exit_reasons) == dict(old.exit_reasons)


def test_strategy_series_equals_stoch_series_on_engine_views() -> None:
    """Driven exactly as the engine drives it (bars_up_to, one bar at a
    time), the strategy's cache holds stoch_series of the visible window."""
    bars = _bars(n=300)
    strat = StochCrossStrategy(StochCrossConfig(k_period=9, d_period=3, smoothing=2))
    state: dict[str, object] = {}
    for i in range(bars.close.size):
        strat.on_bar(i, bars_up_to(bars, i), state)
        if strat._k_cache is None:
            continue
        want = stoch_series(
            np.asarray(bars.high)[: i + 1], np.asarray(bars.low)[: i + 1],
            np.asarray(bars.close)[: i + 1], k_period=9, d_period=3, smoothing=2,
        )
        assert _same((strat._k_cache, strat._d_cache), want), i
