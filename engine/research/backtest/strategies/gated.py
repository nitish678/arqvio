"""The oracle's entry gates: any family's strategy, with ARQVIO's session and trend gates on its openings.

``GatedStrategy`` wraps a family strategy. Each bar it lets the family decide, then applies the gates
(engine/research/backtest/gates.py) to an OPENING intent only: a refused opening is dropped and the strategy's own
state is made flat (``direction`` popped), exactly what the kernel does at its emit point. It reads bars 0..i only
-- the trend EMA advances one close at a time through the look-ahead-blocked view the engine hands it.
"""
from __future__ import annotations

from engine.research.backtest import switch_gates
from engine.research.backtest.gates import TrendEma, session_open
from engine.research.backtest.order import OrderIntent
from engine.research.data.mt5_source import BarData


class GatedStrategy:
    def __init__(self, inner: object, *, session: str, trend_ema: int, bias_switches: tuple[str, ...] = (),
                 filter_switches: tuple[str, ...] = ()) -> None:
        self.inner = inner
        sw = "".join(f"_b{b}" for b in bias_switches) + "".join(f"_f{f}" for f in filter_switches)
        self.strategy_id = f"{inner.strategy_id}_g{session or 'all'}_e{trend_ema}{sw}"
        self.session = session
        self.trend = TrendEma(trend_ema) if trend_ema else None
        self.switches = switch_gates.make(tuple(bias_switches), tuple(filter_switches))
        self._seen = -1  # the last bar index the EMA has taken

    def _advance(self, bar_index: int, bars: BarData) -> None:
        if self.trend is None and not self.switches:
            return
        for j in range(self._seen + 1, bar_index + 1):
            if self.trend is not None:
                self.trend.update(float(bars.close[j]))
            for g in self.switches:
                g.update(int(bars.time[j]), float(bars.open[j]), float(bars.high[j]), float(bars.low[j]),
                         float(bars.close[j]), float(bars.tick_volume[j]))
        self._seen = max(self._seen, bar_index)

    def allows(self, direction: int, bar_index: int, bars: BarData) -> bool:
        if self.session and not session_open(int((int(bars.time[bar_index]) // 3600) % 24), self.session):
            return False
        if self.trend is not None and not self.trend.allows(direction, float(bars.close[bar_index])):
            return False
        return all(g.allows(direction) for g in self.switches)

    def on_bar(self, bar_index: int, bars: BarData, state: dict[str, object]) -> OrderIntent | None:
        self._advance(bar_index, bars)
        intent = self.inner.on_bar(bar_index, bars, state)
        if intent is None or intent.volume == 0:
            return intent
        if not self.allows(1 if intent.direction == "long" else -1, bar_index, bars):
            state.pop("direction", None)  # the opening did not happen: the strategy is flat
            return None
        return intent


__all__ = ["GatedStrategy"]
