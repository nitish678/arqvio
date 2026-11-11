"""MACD crossover strategy.

Rule (bar-close decision, next-bar-open execution):
    fast_ema = EMA(close, fast_period)
    slow_ema = EMA(close, slow_period)
    macd_line = fast_ema - slow_ema
    signal_line = EMA(macd_line, signal_period)
    if macd_line crosses ABOVE signal_line:  go LONG
    if macd_line crosses BELOW signal_line: go SHORT
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

DEFAULT_FAST_PERIOD = 12
DEFAULT_SLOW_PERIOD = 26
DEFAULT_SIGNAL_PERIOD = 9


def ema_series(close: np.ndarray, period: int) -> np.ndarray:
    """EMA with SMA seed (alpha = 2/(period+1)). NaN during warmup.

    Seeds from the first ``period`` FINITE values, so it can be called on
    a series with a NaN warmup (e.g. EMA of a MACD line that starts as NaN).
    Matches ``engine.research.features.macd.ema_series``.
    """
    if period < 1:
        raise DataError(f"ema period must be >= 1; got {period}")
    close = np.asarray(close, dtype=np.float64)
    n = close.size
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0 or period > n:
        return out
    alpha = 2.0 / (period + 1)
    finite_idx = np.where(np.isfinite(close))[0]
    if finite_idx.size < period:
        return out
    seed_start = int(finite_idx[0])
    seed_end = int(finite_idx[period - 1])
    seed_vals = close[seed_start : seed_end + 1]
    out[seed_end] = float(np.mean(seed_vals))
    for i in range(seed_end + 1, n):
        if not np.isfinite(close[i]):
            continue
        prev = out[i - 1]
        if not np.isfinite(prev):
            prev = out[seed_end]
        out[i] = alpha * close[i] + (1.0 - alpha) * prev
    return out


def macd_series(
    close: np.ndarray,
    *,
    fast_period: int = DEFAULT_FAST_PERIOD,
    slow_period: int = DEFAULT_SLOW_PERIOD,
    signal_period: int = DEFAULT_SIGNAL_PERIOD,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (macd_line, signal_line, histogram) arrays.

    macd_line  = EMA(close, fast) - EMA(close, slow)
    signal_line = EMA(macd_line, signal_period)
    histogram   = macd_line - signal_line
    """
    fast_ema = ema_series(close, fast_period)
    slow_ema = ema_series(close, slow_period)
    macd_line = fast_ema - slow_ema
    signal_line = ema_series(macd_line, signal_period)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


@dataclass
class MACDCrossConfig:
    fast_period: int = DEFAULT_FAST_PERIOD
    slow_period: int = DEFAULT_SLOW_PERIOD
    signal_period: int = DEFAULT_SIGNAL_PERIOD
    volume: float = 0.1
    strategy_id: str = "macd_cross_12_26_9"
    sl_atr: float = 0.0
    tp_atr: float = 0.0
    atr_period: int = DEFAULT_ATR_PERIOD

    def __post_init__(self) -> None:
        if self.fast_period >= self.slow_period:
            raise DataError(
                f"MACD fast_period ({self.fast_period}) must be < "
                f"slow_period ({self.slow_period})"
            )
        if self.slow_period - self.fast_period < 5:
            raise DataError(
                f"MACD slow-fast gap ({self.slow_period - self.fast_period}) "
                f"must be >= 5 to avoid degenerate near-zero MACD line "
                f"(fast={self.fast_period}, slow={self.slow_period})"
            )

    @property
    def warmup(self) -> int:
        """First bar index on which on_bar can emit an intent."""
        return self.slow_period + self.signal_period - 1


class MACDCrossStrategy:
    """MACD line / signal line crossover. No look-ahead: bars[0..i] inclusive."""

    def __init__(self, config: MACDCrossConfig | None = None) -> None:
        self.config = config or MACDCrossConfig()
        self.strategy_id = self.config.strategy_id
        self._macd_cache: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._cache_key: tuple[int, int, int, int] | None = None

    def _get_series(self, bars: BarData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute macd_series once per (bars identity, config) and cache."""
        close_arr = np.asarray(bars.close, dtype=np.float64)
        key = (close_arr.size, self.config.fast_period,
               self.config.slow_period, self.config.signal_period)
        if self._cache_key != key:
            self._macd_cache = macd_series(
                close_arr,
                fast_period=self.config.fast_period,
                slow_period=self.config.slow_period,
                signal_period=self.config.signal_period,
            )
            self._cache_key = key
        return self._macd_cache

    def on_bar(
        self,
        bar_index: int,
        bars: BarData,
        state: dict[str, object],
    ) -> OrderIntent | None:
        if bar_index < self.config.warmup:
            return None

        macd_line, signal_line, _ = self._get_series(bars)
        macd_now = float(macd_line[bar_index])
        sig_now = float(signal_line[bar_index])
        if not np.isfinite(macd_now) or not np.isfinite(sig_now):
            return None
        macd_prev = float(macd_line[bar_index - 1]) if bar_index >= 1 else np.nan
        sig_prev = float(signal_line[bar_index - 1]) if bar_index >= 1 else np.nan
        if not np.isfinite(macd_prev) or not np.isfinite(sig_prev):
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

        # Bullish cross: MACD crosses above signal line
        if macd_prev <= sig_prev and macd_now > sig_now:
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
        # Bearish cross: MACD crosses below signal line
        elif macd_prev >= sig_prev and macd_now < sig_now:
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
