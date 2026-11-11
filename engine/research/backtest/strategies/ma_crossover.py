"""Simple moving-average crossover strategy.

The S3 GATE strategy. Hand-computed, deterministic, easy to verify.

Rule (bar-close decision, next-bar-open execution):
    fast_ma = SMA(close, fast)
    slow_ma = SMA(close, slow)
    if  fast_ma > slow_ma and previous_bar fast_ma <= previous_bar slow_ma:
        -> go LONG
    if  fast_ma < slow_ma and previous_bar fast_ma >= previous_bar slow_ma:
        -> go SHORT
    if  a position is open and the OPPOSITE cross happens, close it.

The strategy emits a single OrderIntent per bar (or None). It only
ever holds one position at a time.

For the GATE, the strategy uses fixed (fast, slow) parameters and a
fixed lot size. The P&L of each trade is computable by hand:
    gross = sign * (exit - entry) * per_unit_per_lot * volume
    net   = gross - entry_costs - exit_costs

The GATE test re-derives these from the same data the backtester
sees, and compares trade-by-trade.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from engine.research.backtest.order import OrderIntent
from engine.research.data.mt5_source import BarData
from engine.research.features.atr import DEFAULT_ATR_PERIOD, atr_at


@dataclass
class MACrossConfig:
    fast: int = 20
    slow: int = 50
    volume: float = 0.1       # lots
    strategy_id: str = "ma_cross_20_50"
    # SL/TP as multiples of ATR at the signal bar. 0.0 (or None) means
    # "no stop of this kind" -- the position then exits only on the
    # opposite cross, which is the pre-S1 behaviour.
    sl_atr: float = 0.0
    tp_atr: float = 0.0
    atr_period: int = DEFAULT_ATR_PERIOD

    @property
    def warmup(self) -> int:
        """First bar index on which on_bar can emit an intent."""
        return self.slow


class MACrossStrategy:
    """Simple MA-crossover. No look-ahead: uses bars[0..i] inclusive."""

    def __init__(self, config: MACrossConfig | None = None) -> None:
        self.config = config or MACrossConfig()
        self.strategy_id = self.config.strategy_id

    def on_bar(
        self,
        bar_index: int,
        bars: BarData,
        state: dict[str, object],
    ) -> OrderIntent | None:
        # We need at least `slow` bars to compute slow MA, plus one
        # prior bar to detect a CROSS. So bar_index >= slow (we have
        # bars 0..bar_index, i.e. bar_index+1 bars).
        if bar_index < self.config.warmup:
            return None
        close = bars.close

        # If a reversal was pended last bar (close intent was sent),
        # open the new position now.
        pending = state.get("pending_direction")
        if pending is not None:
            # Distances are computed BEFORE the state mutation. A
            # suppressed bar must leave the strategy exactly as it was:
            # consuming pending_direction and setting direction without
            # emitting an intent would leave it believing it holds a
            # position that was never opened.
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

        # SMA up to and including bar_index (inclusive).
        fast_now = float(np.mean(close[bar_index - self.config.fast + 1 : bar_index + 1]))
        slow_now = float(np.mean(close[bar_index - self.config.slow + 1 : bar_index + 1]))
        fast_prev = float(np.mean(close[bar_index - self.config.fast : bar_index]))
        slow_prev = float(np.mean(close[bar_index - self.config.slow : bar_index]))

        currently_long = state.get("direction") == "long"
        currently_short = state.get("direction") == "short"

        # Bullish cross: fast was below or equal, now above.
        if fast_prev <= slow_prev and fast_now > slow_now:
            if currently_short:
                # Send close intent; open long next bar.
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
        # Bearish cross.
        if fast_prev >= slow_prev and fast_now < slow_now:
            if currently_long:
                # Send close intent; open short next bar.
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

        ``suppress`` is True when this DNA asked for ATR-derived stops
        and ATR cannot be computed here -- during the 13-bar warmup, or
        on a degenerate flat series where ATR is 0. The bar then emits
        NO intent at all (G9 ruling 2 and 3).

        Suppressing rather than trading without a stop matters: a DNA
        with sl_atr set is a different strategy from the same DNA
        without one, and quietly dropping the stop would file the second
        strategy's results under the first one's hash. A zero-distance
        stop is worse still -- it would fill instantly at entry.

        A DNA that asked for no stops never consults ATR, so it is never
        suppressed.
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
