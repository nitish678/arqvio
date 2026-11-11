"""Spread cost model.

For a round-trip trade the spread cost is:

    cost = spread_price * contract_size * volume * account_per_price_per_lot

where spread_price = spread_points * point. For a one-way entry (no
exit yet), we charge half the round-trip spread at entry; the backtester
will charge the other half on exit.

The cost is in the account currency.
"""
from __future__ import annotations

from dataclasses import dataclass

from engine.research.data.symbol_spec import SymbolSpec


@dataclass(frozen=True)
class SpreadCost:
    """Cost of the spread for one trade leg."""

    one_way: float       # cost in account currency, one leg only
    round_trip: float    # both legs; the value the backtester should subtract from P&L
    spread_points: int   # source spread
    price_spread: float  # spread expressed in price units

    def to_dict(self) -> dict[str, float | int]:
        return {
            "one_way": self.one_way,
            "round_trip": self.round_trip,
            "spread_points": self.spread_points,
            "price_spread": self.price_spread,
        }


def spread_cost(spec: SymbolSpec, volume: float) -> SpreadCost:
    """One-way spread cost in account currency for the given volume.

    The profit-side tick value is used (worst case for the trader: every
    trade starts in the spread, so the "loss" side is the right
    conservator). If the broker has different profit/loss tick values,
    the larger of the two is charged.

    Math: trade_tick_value is "account-currency per 1.0 lot per 1 tick".
    Empirically (verified against Exness symbols), trade_tick_size is in
    PRICE UNITS, not points - it equals `point` for every Exness symbol
    we tested. So per 1.0 price-unit per 1.0 lot:
        tick_value / trade_tick_size
    Multiply by spread_price (in price units) and by volume (in lots).
    contract_size is NOT in the formula because tick_value is already
    per-lot (it includes the contract_size scaling).
    """
    if volume <= 0:
        raise ValueError(f"volume must be > 0; got {volume}")
    vol = spec.normalise_volume(volume)
    spread_price = spec.spread_in_price()
    tick_value = max(spec.trade_tick_value_profit, spec.trade_tick_value_loss)
    if spec.trade_tick_size <= 0:
        raise ValueError("invalid trade_tick_size")
    per_unit_per_lot = tick_value / spec.trade_tick_size
    one_way = spread_price * vol * per_unit_per_lot
    return SpreadCost(
        one_way=one_way,
        round_trip=2.0 * one_way,
        spread_points=spec.spread_points,
        price_spread=spread_price,
    )
