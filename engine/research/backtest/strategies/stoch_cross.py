"""Stochastic %K / %D crossover strategy.

Rule (bar-close decision, next-bar-open execution):
    %K = StochK(high, low, close, k_period)
    %D = SMA(%K, d_period)
    if %K crosses ABOVE %D:  go LONG
    if %K crosses BELOW %D: go SHORT
    if a position is open and the OPPOSITE cross happens: close it first.

Look-ahead-safe: uses bars[0..i] inclusive and emits for i+1.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from engine.core.errors import DataError
from engine.research.backtest.order import OrderIntent
from engine.research.data.mt5_source import BarData
from engine.research.features.atr import DEFAULT_ATR_PERIOD, atr_at
from engine.research.features.stoch import StochPrefixCache, stoch_d_series

DEFAULT_K_PERIOD = 14
DEFAULT_D_PERIOD = 3
DEFAULT_SMOOTHING = 3


def _check_params(k_period: int, d_period: int, smoothing: int) -> None:
    if k_period < 1 or d_period < 1 or smoothing < 1:
        raise DataError(
            f"stoch params must be >= 1; got k={k_period} d={d_period} smooth={smoothing}"
        )


def stoch_series(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    *,
    k_period: int = DEFAULT_K_PERIOD,
    d_period: int = DEFAULT_D_PERIOD,
    smoothing: int = DEFAULT_SMOOTHING,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (raw_K, slow_D) arrays, same length as input, NaN during warmup.

    raw_K is the smoothed %K: SMA over `smoothing` bars of the raw stochastic
    100*(C - LL(k))/(HH(k) - LL(k)), with LL/HH over the rolling window
    [t-k+1, t]; HH == LL leaves the raw value NaN (no signal). slow_D is the
    SMA of raw_K over `d_period` bars.

    Delegates to ``features.stoch.stoch_d_series`` so the strategy and the
    feature can never diverge again (S3 bug 5: the strategy copy anchored
    LL/HH on an expanding window from bar 0, leaving %K unbounded).
    """
    _check_params(k_period, d_period, smoothing)
    return stoch_d_series(
        high, low, close, period=k_period, d_period=d_period, smoothing=smoothing,
    )


@dataclass
class StochCrossConfig:
    k_period: int = DEFAULT_K_PERIOD
    d_period: int = DEFAULT_D_PERIOD
    smoothing: int = DEFAULT_SMOOTHING
    # C3-A': threshold gates. A bullish cross only fires when %K is at or
    # below k_oversold (confirming oversold confirmation); a bearish cross
    # only fires when %K is at or above k_overbought. Defaults preserve the
    # pure-crossover behaviour (0 = no gate, 100 = never gate).
    k_oversold: float = 0.0
    k_overbought: float = 100.0
    volume: float = 0.1
    strategy_id: str = "stoch_cross_14_3_3"
    sl_atr: float = 0.0
    tp_atr: float = 0.0
    atr_period: int = DEFAULT_ATR_PERIOD

    def __post_init__(self) -> None:
        if self.k_period < 2:
            raise DataError(f"stoch k_period ({self.k_period}) must be >= 2")
        if self.k_oversold >= self.k_overbought:
            raise DataError(
                f"stoch k_oversold ({self.k_oversold}) must be < "
                f"k_overbought ({self.k_overbought})"
            )

    @property
    def warmup(self) -> int:
        """First bar index on which on_bar can emit an intent."""
        return self.k_period + self.d_period + self.smoothing - 2


class StochCrossStrategy:
    """Stochastic %K/%D crossover. No look-ahead: uses bars[0..i] inclusive."""

    def __init__(self, config: StochCrossConfig | None = None) -> None:
        self.config = config or StochCrossConfig()
        self.strategy_id = self.config.strategy_id
        self._k_cache: np.ndarray | None = None
        self._d_cache: np.ndarray | None = None
        self._series: StochPrefixCache | None = None

    def _get_series(self, bars: BarData) -> tuple[np.ndarray, np.ndarray]:
        """(%K, %D) of the visible window, equal to stoch_series(window).

        The engine grows the window by one bar per on_bar call, so the
        series is extended at the tail (StochPrefixCache) instead of being
        recomputed from bar 0 each call -- that recompute made the slow
        oracle O(n^2) per candidate. The cache compares the window's
        contents, never an object identity (``id(bars.high)`` is reused
        across the fresh BarsView the engine passes each call).
        """
        c = self.config
        params = (c.k_period, c.d_period, c.smoothing)
        _check_params(*params)
        if self._series is None or self._series.params != params:
            self._series = StochPrefixCache(
                period=c.k_period, d_period=c.d_period, smoothing=c.smoothing,
            )
        self._k_cache, self._d_cache = self._series.series(bars.high, bars.low, bars.close)
        return self._k_cache, self._d_cache

    def on_bar(
        self,
        bar_index: int,
        bars: BarData,
        state: dict[str, object],
    ) -> OrderIntent | None:
        if bar_index < self.config.warmup:
            return None

        raw_k, slow_d = self._get_series(bars)
        k_now = float(raw_k[bar_index])
        d_now = float(slow_d[bar_index])
        if not np.isfinite(k_now) or not np.isfinite(d_now):
            return None
        k_prev = float(raw_k[bar_index - 1]) if bar_index >= 1 else np.nan
        d_prev = float(slow_d[bar_index - 1]) if bar_index >= 1 else np.nan
        if not np.isfinite(k_prev) or not np.isfinite(d_prev):
            return None

        currently_long = state.get("direction") == "long"
        currently_short = state.get("direction") == "short"

        pending = state.get("pending_direction")
        if pending is not None:
            suppress, sl_dist, tp_dist = self._stop_distances(bars, bar_index)
            if suppress:
                return None
            del state["pending_direction"]
            state["direction"] = pending
            return OrderIntent(
                direction=str(pending),
                volume=self.config.volume,
                bar_index=bar_index,
                sl_dist=sl_dist,
                tp_dist=tp_dist,
            )

        # Bullish cross: %K crosses above %D
        if k_prev <= d_prev and k_now > d_now:
            # Zone gate: BOTH %K and %D must be in oversold zone, confirming
            # the cross originates from oversold territory.
            if k_now <= self.config.k_oversold and d_now <= self.config.k_oversold:
                if currently_short:
                    state.pop("direction", None)
                    state["pending_direction"] = "long"
                    return OrderIntent(direction="long", volume=0.0, bar_index=bar_index)
                if not currently_long:
                    suppress, sl_dist, tp_dist = self._stop_distances(bars, bar_index)
                    if suppress:
                        return None
                    state["direction"] = "long"
                    return OrderIntent(
                        direction="long",
                        volume=self.config.volume,
                        bar_index=bar_index,
                        sl_dist=sl_dist,
                        tp_dist=tp_dist,
                    )
        # Bearish cross: %K crosses below %D
        elif k_prev >= d_prev and k_now < d_now:
            # Zone gate: BOTH %K and %D must be in overbought zone, confirming
            # the cross originates from overbought territory.
            if k_now >= self.config.k_overbought and d_now >= self.config.k_overbought:
                if currently_long:
                    state.pop("direction", None)
                    state["pending_direction"] = "short"
                    return OrderIntent(direction="short", volume=0.0, bar_index=bar_index)
                if not currently_short:
                    suppress, sl_dist, tp_dist = self._stop_distances(bars, bar_index)
                    if suppress:
                        return None
                    state["direction"] = "short"
                    return OrderIntent(
                        direction="short",
                        volume=self.config.volume,
                        bar_index=bar_index,
                        sl_dist=sl_dist,
                        tp_dist=tp_dist,
                    )
        return None

    def _stop_distances(
        self, bars: BarData, bar_index: int,
    ) -> tuple[bool, float | None, float | None]:
        want_sl = bool(self.config.sl_atr and self.config.sl_atr > 0.0)
        want_tp = bool(self.config.tp_atr and self.config.tp_atr > 0.0)
        if not want_sl and not want_tp:
            return False, None, None
        atr = atr_at(
            bars.high, bars.low, bars.close, bar_index,
            period=self.config.atr_period,
        )
        if atr is None:
            return True, None, None
        sl_dist = float(self.config.sl_atr) * atr if want_sl else None
        tp_dist = float(self.config.tp_atr) * atr if want_tp else None
        return False, sl_dist, tp_dist
