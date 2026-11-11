"""Sham strategy: random entries carrying a real DNA's risk and exit genes.

The control a real candidate is judged against. It trades at bars chosen
without looking at prices, on sides chosen by a coin, and keeps everything
else the candidate has: lot size, ATR stops and targets, the engine, the
costs and the gates. Whatever it earns is what the candidate's risk and exit
shape earns with no signal behind it.

Rule (bar-close decision, next-bar-open execution), for each scheduled bar
in order:
    flat              -> enter the drawn side
    a position is open -> close it; enter the drawn side on the next bar
A scheduled bar that arrives while a reversal's entry is still pending is
taken on the first bar after that entry, so no draw is lost to a collision.
There are no other signals: positions otherwise exit on SL, TP, the gates or
the end of data.

ATR-derived stops are suppressed exactly as the family strategies suppress
them: a DNA that wants a stop emits no entry where ATR is undefined.
"""
from __future__ import annotations

from dataclasses import dataclass

from engine.core.errors import DataError
from engine.research.backtest.order import OrderIntent
from engine.research.data.mt5_source import BarData
from engine.research.features.atr import DEFAULT_ATR_PERIOD, atr_at

_SIDES = ("long", "short")


@dataclass(frozen=True)
class ShamConfig:
    entry_bars: tuple[int, ...]
    sides: tuple[str, ...]
    volume: float
    strategy_id: str
    sl_atr: float = 0.0
    tp_atr: float = 0.0
    atr_period: int = DEFAULT_ATR_PERIOD

    def __post_init__(self) -> None:
        if len(self.entry_bars) != len(self.sides):
            raise DataError(
                f"sham: {len(self.entry_bars)} entry bars but "
                f"{len(self.sides)} sides"
            )
        if len(set(self.entry_bars)) != len(self.entry_bars):
            raise DataError("sham: entry bars must be distinct")
        bad = sorted(set(self.sides) - set(_SIDES))
        if bad:
            raise DataError(f"sham: sides must be long/short, got {bad}")


class ShamStrategy:
    """Scheduled random entries. Reads prices only for ATR at the signal bar."""

    def __init__(self, config: ShamConfig) -> None:
        self.config = config
        self.strategy_id = config.strategy_id
        self._schedule = dict(zip(config.entry_bars, config.sides, strict=True))

    def on_bar(
        self,
        bar_index: int,
        bars: BarData,
        state: dict[str, object],
    ) -> OrderIntent | None:
        owed: list[str] = state.setdefault("owed_sides", [])  # type: ignore[assignment]
        side = self._schedule.get(bar_index)
        if side is not None:
            owed.append(side)

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

        if not owed:
            return None
        side = owed.pop(0)
        if state.get("direction") is not None:
            state.pop("direction", None)
            state["pending_direction"] = side
            return OrderIntent(direction=side, volume=0.0, bar_index=bar_index)
        suppress, sl_dist, tp_dist = self._stop_distances(bars, bar_index)
        if suppress:
            return None
        state["direction"] = side
        return OrderIntent(
            direction=side,
            volume=self.config.volume,
            bar_index=bar_index,
            sl_dist=sl_dist,
            tp_dist=tp_dist,
        )

    def _stop_distances(
        self, bars: BarData, bar_index: int,
    ) -> tuple[bool, float | None, float | None]:
        """(suppress, sl_dist, tp_dist) from ATR at the signal bar, with the
        family strategies' rule: stops wanted and ATR undefined -> suppress."""
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


__all__ = ["ShamConfig", "ShamStrategy"]
