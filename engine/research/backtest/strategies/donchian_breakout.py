"""Donchian channel breakout strategy.

Rule (bar-close decision, next-bar-open execution):
    upper_channel = max(high[bar_index - period + 1 : bar_index + 1])  # n bars back
    lower_channel = min(low[bar_index - period + 1 : bar_index + 1])
    if close > upper_channel_prev and not long:  go LONG  (breakout above)
    if close < lower_channel_prev and not short: go SHORT (breakout below)
    if a position is open and the OPPOSITE breakout happens: close it first.

    "prev" means the channel from the bar BEFORE the breakout bar: a breakout
    is defined relative to the prior window. close[t] must exceed the
    channel formed by bars [t-period .. t-1] to qualify.

Look-ahead-safe: uses bars[0..i] inclusive and emits for i+1.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from engine.core.errors import DataError
from engine.research.backtest.order import OrderIntent
from engine.research.data.mt5_source import BarData
from engine.research.features.atr import DEFAULT_ATR_PERIOD, atr_at

DEFAULT_PERIOD = 20


def donchian_channels(
    high: np.ndarray, low: np.ndarray, period: int,
) -> tuple[np.ndarray, np.ndarray]:
    """(upper, lower): max(high) / min(low) over [i-period+1, i], NaN before
    index period-1. Causal, so the fast path computes it once on the full
    arrays and gets the strategy's per-bar values bit for bit."""
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    n = high.size
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        upper[i] = float(np.max(high[i - period + 1 : i + 1]))
        lower[i] = float(np.min(low[i - period + 1 : i + 1]))
    return upper, lower


@dataclass
class DonchianBreakoutConfig:
    period: int = DEFAULT_PERIOD
    volume: float = 0.1
    strategy_id: str = "donchian_20"
    sl_atr: float = 0.0
    tp_atr: float = 0.0
    atr_period: int = DEFAULT_ATR_PERIOD

    def __post_init__(self) -> None:
        if self.period < 2:
            raise DataError(f"donchian period ({self.period}) must be >= 2")

    @property
    def warmup(self) -> int:
        """First bar index on which on_bar can emit an intent."""
        return self.period


class DonchianBreakoutStrategy:
    """Donchian channel breakout. No look-ahead: bars[0..i] inclusive."""

    def __init__(self, config: DonchianBreakoutConfig | None = None) -> None:
        self.config = config or DonchianBreakoutConfig()
        self.strategy_id = self.config.strategy_id
        self._rolling_cache: tuple[np.ndarray, np.ndarray] | None = None
        self._cache_key: int | None = None

    def _get_rolling(self, bars: BarData) -> tuple[np.ndarray, np.ndarray]:
        """Precompute rolling upper/lower channels once per visible length.

        Keyed on visible length, not ``id(bars.high)``, matching the fix
        in rsi.py and stoch_cross.py: the engine hands on_bar a fresh BarsView
        each call, so id() is not a stable identity.
        """
        high_arr = np.asarray(bars.high, dtype=np.float64)
        low_arr = np.asarray(bars.low, dtype=np.float64)
        key = high_arr.size
        if self._cache_key != key:
            self._rolling_cache = donchian_channels(high_arr, low_arr, self.config.period)
            self._cache_key = key
        return self._rolling_cache

    def on_bar(
        self,
        bar_index: int,
        bars: BarData,
        state: dict[str, object],
    ) -> OrderIntent | None:
        p = self.config.period
        if p < 1:
            raise DataError(f"donchian period must be >= 1; got {p}")
        if bar_index < self.config.warmup:
            return None
        close = np.asarray(bars.close, dtype=np.float64)

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

        # Channel from the PRIOR window: bars [t-p .. t-1] (p bars ending yesterday).
        upper, lower = self._get_rolling(bars)
        upper_prev = float(upper[bar_index - 1])
        lower_prev = float(lower[bar_index - 1])
        close_now = float(close[bar_index])

        # Bullish breakout: close breaks above the prior upper channel.
        if close_now > upper_prev:
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
        # Bearish breakout: close breaks below the prior lower channel.
        elif close_now < lower_prev:
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
