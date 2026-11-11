"""Account heat (DR §4): the money at risk at the stops, as a share of equity.

A position's stop-distance risk is ``|entry - sl| * per_unit_loss * volume``, where ``per_unit_loss`` is the frozen
engine's own ``_per_unit_loss`` (trade_tick_value_loss / trade_tick_size) for that symbol. A position without a stop
has no bounded risk: it is ``None`` here and makes the heat ``unbounded``. Account heat is the sum over open positions
divided by equity, with a per-symbol rollup. ``HEAT_ALERT`` upward crossings become ``HEAT_HIGH`` events; ``HEAT_CAP``
is the replay's refusal cap (DR §2).
"""
from __future__ import annotations

from dataclasses import dataclass

HEAT_ALERT = 0.06
HEAT_CAP = 0.10


class HeatError(ValueError):
    """Heat cannot be computed (non-positive equity, negative risk inputs)."""


@dataclass(frozen=True)
class OpenRisk:
    member: str
    symbol: str
    risk: float | None  # None: no stop


@dataclass(frozen=True)
class HeatPoint:
    heat: float | None               # None when unbounded
    unbounded: bool
    per_symbol: dict[str, float | None]


def position_risk(entry: float, sl: float | None, volume: float, per_unit_loss: float) -> float | None:
    if sl is None:
        return None
    if volume < 0 or per_unit_loss < 0:
        raise HeatError(f"negative volume {volume} or per-unit loss {per_unit_loss}")
    return abs(float(entry) - float(sl)) * float(per_unit_loss) * float(volume)


def account_heat(positions: list[OpenRisk], equity: float) -> HeatPoint:
    if not equity > 0:
        raise HeatError(f"equity {equity} is not positive")
    per: dict[str, float | None] = {}
    total = 0.0
    unbounded = False
    for p in positions:
        if p.risk is None:
            unbounded = True
            per[p.symbol] = None
            continue
        total += p.risk
        if per.get(p.symbol, 0.0) is not None:
            per[p.symbol] = per.get(p.symbol, 0.0) + p.risk
    rollup = {s: (None if v is None else v / equity) for s, v in sorted(per.items())}
    return HeatPoint(None if unbounded else total / equity, unbounded, rollup)


def upward_crossings(series: list[float | None], threshold: float = HEAT_ALERT) -> list[int]:
    """Indices where heat goes from <= threshold (or the start) to > threshold; unbounded counts as above."""
    out, above = [], False
    for k, h in enumerate(series):
        now = h is None or h > threshold
        if now and not above:
            out.append(k)
        above = now
    return out


__all__ = ["HEAT_ALERT", "HEAT_CAP", "HeatError", "HeatPoint", "OpenRisk", "account_heat", "position_risk",
           "upward_crossings"]
