"""RSI mean-reversion strategy.

Rule (bar-close decision, next-bar-open execution):
    rsi = RSI(close, period=14)  [Wilder smoothing]
    if  rsi < oversold and not long:    -> go LONG
    if  rsi > overbought and not short: -> go SHORT
    if  a position is open and the OPPOSITE signal fires: close it first.

Like MACrossStrategy, this is look-ahead-safe: it sees bars 0..i inclusive
and emits for i+1.

W1: the first strategy family shipped after the MA-cross gate. Its feature
(rsi_14) is registered in the feature registry; the strategy itself is the
ORACL's real RSI — the kernel cannot express RSI and must refuse it (fast_path.expressible).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from engine.core.errors import DataError
from engine.research.backtest.order import OrderIntent
from engine.research.data.mt5_source import BarData
from engine.research.features.atr import DEFAULT_ATR_PERIOD, atr_at
from engine.research.features.rsi import DEFAULT_RSI_PERIOD, rsi_series


@dataclass
class RSIConfig:
    period: int = DEFAULT_RSI_PERIOD
    oversold: float = 30.0
    overbought: float = 70.0
    # C3-A': when True, an open position exits when RSI crosses the 50 midline
    # (mean-reversion signal that the move has run its course). Default False
    # preserves backwards compatibility for any existing config construction.
    exit_midline: bool = False
    volume: float = 0.1
    strategy_id: str = "rsi_14_os30_ob70"
    sl_atr: float = 0.0
    tp_atr: float = 0.0
    atr_period: int = DEFAULT_ATR_PERIOD

    def __post_init__(self) -> None:
        if self.oversold >= self.overbought:
            raise DataError(
                f"RSI oversold ({self.oversold}) must be < "
                f"overbought ({self.overbought})"
            )
        if self.overbought - self.oversold < 5.0:
            raise DataError(
                f"RSI zone gap overbought-oversold "
                f"({self.overbought - self.oversold}) must be >= 5"
            )

    @property
    def warmup(self) -> int:
        """First bar index on which on_bar can emit an intent."""
        return self.period


class RSIStrategy:
    """RSI mean-reversion. No look-ahead: uses bars[0..i] inclusive."""

    def __init__(self, config: RSIConfig | None = None) -> None:
        self.config = config or RSIConfig()
        # Invariant enforced by RSIConfig.__post_init__.
        self.strategy_id = self.config.strategy_id
        self._rsi_cache: np.ndarray | None = None
        self._cache_key: tuple[int, ...] | None = None

    def _get_rsi(self, bars: BarData) -> np.ndarray:
        """Compute rsi_series, cached per visible-window length.

        The engine hands ``on_bar`` a fresh BarsView each call, each
        wrapping a new ForwardAccessBlocker. ``id(bars.close)`` is therefore
        not a stable identity -- Python may reuse a freed object's address,
        returning a stale cache from a shorter window and triggering an
        IndexError. We key on the visible length instead: the only thing
        the window's contents can differ in is how many bars it contains.
        """
        close_arr = np.asarray(bars.close, dtype=np.float64)
        key = (close_arr.size, self.config.period)
        if self._cache_key != key:
            self._rsi_cache = rsi_series(close_arr, period=self.config.period)
            self._cache_key = key
        return self._rsi_cache

    def on_bar(
        self,
        bar_index: int,
        bars: BarData,
        state: dict[str, object],
    ) -> OrderIntent | None:
        # Need at least `period` bars to compute RSI (the first valid RSI
        # value appears at index = period, since period-1 bars seed the
        # Wilder average and bar_index is the period-th).
        if bar_index < self.config.warmup:
            return None

        rsi = self._get_rsi(bars)
        val = float(rsi[bar_index])
        if not np.isfinite(val):
            return None

        # If a reversal was pended last bar (close intent was sent),
        # open the new position now.
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

        currently_long = state.get("direction") == "long"
        currently_short = state.get("direction") == "short"

        # C3-A': midline exit — when enabled, a long exits on RSI crossing
        # below 50, a short exits on RSI crossing above 50. This is a
        # mean-reversion "the move has run its course" signal.
        if self.config.exit_midline and (currently_long or currently_short):
            prev_val = float(rsi[bar_index - 1]) if bar_index >= 1 else np.nan
            if np.isfinite(prev_val):
                if currently_long and prev_val >= 50.0 and val < 50.0:
                    state.pop("direction", None)
                    return OrderIntent(direction="long", volume=0.0,
                                       bar_index=bar_index)
                if currently_short and prev_val <= 50.0 and val > 50.0:
                    state.pop("direction", None)
                    return OrderIntent(direction="short", volume=0.0,
                                       bar_index=bar_index)

        if val < self.config.oversold:
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
        elif val > self.config.overbought:
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
        """(suppress, sl_dist, tp_dist) from ATR at the SIGNAL bar.

        Mirrors MACrossStrategy._stop_distances (G9 rulings 2 and 3).
        """
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
