"""Swap (overnight roll) cost model.

For a position held past the broker's rollover time (usually 22:00 or
23:00 broker-local, 00:00 UTC for most), the broker charges (or pays)
the swap rate. Exness USDCADm long/short swap rates are points per lot
per night.

Spec requirement: the Wednesday roll is 3x to cover the weekend (the
broker does not actually settle on Sat/Sun). Exness implements this
identically to most brokers.

A position opened on Wednesday at 22:01 and held to Thursday 22:01
accrues 3 nights of swap (Wed, Sat, Sun). A position opened on
Thursday at 22:01 and held to Friday 22:01 accrues 1 night.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

from engine.research.data.symbol_spec import SymbolSpec


@dataclass(frozen=True)
class SwapCost:
    """Total swap cost for a position held across N nights."""

    nights: int                  # number of nights the position was rolled
    effective_nights: int        # weighted by triple-day rule
    points_charged: float        # total points per lot
    account_charged: float       # account currency, for the given volume
    breakdown: list[tuple[date, int, float]]   # (date, multiplier, points)

    def to_dict(self) -> dict[str, object]:
        return {
            "nights": self.nights,
            "effective_nights": self.effective_nights,
            "points_charged": self.points_charged,
            "account_charged": self.account_charged,
            "breakdown": [(d.isoformat(), m, p) for d, m, p in self.breakdown],
        }


def _nights_between(open_d: date, close_d: date) -> list[date]:
    """Return the dates (broker-local) on which a swap was charged.

    A position opened at 22:00 on day X and held to 22:00 on day Y is
    rolled on the days (X+1, X+2, ..., Y). We use open_d+1 through close_d
    inclusive as the roll dates; the open day itself is not charged
    because the position was put on after the roll.
    """
    if close_d <= open_d:
        return []
    out = []
    d = open_d + timedelta(days=1)
    while d <= close_d:
        out.append(d)
        d += timedelta(days=1)
    return out


def swap_cost(
    spec: SymbolSpec,
    volume: float,
    *,
    open_date: date,
    close_date: date,
    direction: Literal["long", "short"] = "long",
) -> SwapCost:
    """Compute total swap for a position held from open_date to close_date.

    direction: "long" charges spec.swap_long; "short" charges spec.swap_short.
    The Wednesday roll (or whatever day spec.swap_rollover_triple_day is
    set to) is 3x the normal rate.
    """
    if close_date <= open_date:
        return SwapCost(0, 0, 0.0, 0.0, [])
    if volume <= 0:
        raise ValueError(f"volume must be > 0; got {volume}")
    if direction not in ("long", "short"):
        raise ValueError(f"direction must be long|short; got {direction!r}")
    vol = spec.normalise_volume(volume)
    base_points = spec.swap_long if direction == "long" else spec.swap_short
    triple_dow = spec.swap_rollover_triple_day
    roll_dates = _nights_between(open_date, close_date)
    breakdown: list[tuple[date, int, float]] = []
    effective = 0
    total_points = 0.0
    for d in roll_dates:
        multiplier = 3 if d.weekday() == triple_dow else 1
        points_for_day = base_points * multiplier
        breakdown.append((d, multiplier, points_for_day))
        effective += multiplier
        total_points += points_for_day
    tick_value = max(spec.trade_tick_value_profit, spec.trade_tick_value_loss)
    if spec.trade_tick_size <= 0:
        raise ValueError("invalid trade_tick_size")
    per_unit_per_lot = tick_value / spec.trade_tick_size
    # points * point = price units; price_units * vol * per_unit_per_lot = $CCY
    # (no extra contract_size: tick_value is already per-lot)
    account_charged = total_points * spec.point * vol * per_unit_per_lot
    return SwapCost(
        nights=len(roll_dates),
        effective_nights=effective,
        points_charged=total_points,
        account_charged=account_charged,
        breakdown=breakdown,
    )