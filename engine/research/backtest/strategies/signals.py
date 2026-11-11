"""The SIGNALS family: research entry signals as switches (backtest/signal_switches.py), traded like a breakout.

Rule (bar-close decision, next-bar-open execution), the same position logic as donchian_breakout.py:
    vote = signal_switches.combine(each switch's signal on bar i)
    vote +1 and not long:  go LONG  (a short is closed first and the long pends to the next bar)
    vote -1 and not short: go SHORT (mirrored)

The switches step one bar at a time, bars[0..i] only, in the order the bars arrive; the kernel reads
``signal_arrays`` built from the same classes, so oracle and kernel take the same entries bar for bar.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from engine.core.errors import DataError
from engine.research.backtest.order import OrderIntent
from engine.research.backtest.signal_switches import READY_SIGNALS, combine, make
from engine.research.data.mt5_source import BarData
from engine.research.features.atr import DEFAULT_ATR_PERIOD, atr_at

WARMUP = 1


@dataclass
class SignalsConfig:
    switches: tuple[str, ...] = field(default_factory=tuple)
    volume: float = 0.1
    strategy_id: str = "signals"
    sl_atr: float = 0.0
    tp_atr: float = 0.0
    atr_period: int = DEFAULT_ATR_PERIOD

    def __post_init__(self) -> None:
        if not self.switches:
            raise DataError("a SIGNALS strategy needs at least one signal switch")
        bad = [s for s in self.switches if s not in READY_SIGNALS]
        if bad:
            raise DataError(f"signal switches {bad} are not READY (READY: {list(READY_SIGNALS)})")

    @property
    def warmup(self) -> int:
        return WARMUP


class SignalsStrategy:
    def __init__(self, config: SignalsConfig) -> None:
        self.config = config
        self.strategy_id = config.strategy_id
        self._sigs = make(tuple(config.switches))
        self._stepped = -1
        self._vote = 0

    def _step_to(self, bars: BarData, bar_index: int) -> int:
        if bar_index < self._stepped:  # a fresh run over the same object: start again
            self._sigs = make(tuple(self.config.switches))
            self._stepped = -1
        t = np.asarray(bars.time, dtype=np.int64)
        o = np.asarray(bars.open, dtype=np.float64)
        h = np.asarray(bars.high, dtype=np.float64)
        lo = np.asarray(bars.low, dtype=np.float64)
        c = np.asarray(bars.close, dtype=np.float64)
        v = np.asarray(bars.tick_volume, dtype=np.float64)
        for j in range(self._stepped + 1, bar_index + 1):
            for s in self._sigs:
                s.update(int(t[j]), float(o[j]), float(h[j]), float(lo[j]), float(c[j]), float(v[j]))
            self._vote = combine([s.signal() for s in self._sigs])
        self._stepped = max(self._stepped, bar_index)
        return self._vote

    def on_bar(self, bar_index: int, bars: BarData, state: dict[str, object]) -> OrderIntent | None:
        vote = self._step_to(bars, bar_index)
        if bar_index < self.config.warmup:
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
            return OrderIntent(direction=str(pending), volume=self.config.volume, bar_index=bar_index,
                               sl_dist=sl_dist, tp_dist=tp_dist)

        if vote > 0:
            if currently_short:
                state.pop("direction", None)
                state["pending_direction"] = "long"
                return OrderIntent(direction="long", volume=0.0, bar_index=bar_index)
            if not currently_long:
                suppress, sl_dist, tp_dist = self._stop_distances(bars, bar_index)
                if suppress:
                    return None
                state["direction"] = "long"
                return OrderIntent(direction="long", volume=self.config.volume, bar_index=bar_index,
                                   sl_dist=sl_dist, tp_dist=tp_dist)
        elif vote < 0:
            if currently_long:
                state.pop("direction", None)
                state["pending_direction"] = "short"
                return OrderIntent(direction="short", volume=0.0, bar_index=bar_index)
            if not currently_short:
                suppress, sl_dist, tp_dist = self._stop_distances(bars, bar_index)
                if suppress:
                    return None
                state["direction"] = "short"
                return OrderIntent(direction="short", volume=self.config.volume, bar_index=bar_index,
                                   sl_dist=sl_dist, tp_dist=tp_dist)
        return None

    def _stop_distances(self, bars: BarData, bar_index: int) -> tuple[bool, float | None, float | None]:
        want_sl = bool(self.config.sl_atr and self.config.sl_atr > 0.0)
        want_tp = bool(self.config.tp_atr and self.config.tp_atr > 0.0)
        if not want_sl and not want_tp:
            return False, None, None
        atr = atr_at(bars.high, bars.low, bars.close, bar_index, period=self.config.atr_period)
        if atr is None:
            return True, None, None
        return False, (float(self.config.sl_atr) * atr if want_sl else None), \
            (float(self.config.tp_atr) * atr if want_tp else None)


__all__ = ["SignalsConfig", "SignalsStrategy", "WARMUP"]
