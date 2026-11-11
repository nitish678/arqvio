"""Commission cost model.

Exness Standard account: $0 commission, all cost is in the spread.
Exness Raw account: $7 per lot per side ($14 round-trip).

The model takes a per-lot-per-side commission and the volume; the
backtester charges one side at entry, one at exit.
"""
from __future__ import annotations

from dataclasses import dataclass

from engine.research.data.symbol_spec import SymbolSpec


@dataclass(frozen=True)
class CommissionCost:
    one_way: float      # account currency per side
    round_trip: float   # both sides
    per_lot_per_side: float
    volume: float

    def to_dict(self) -> dict[str, float]:
        return {
            "one_way": self.one_way,
            "round_trip": self.round_trip,
            "per_lot_per_side": self.per_lot_per_side,
            "volume": self.volume,
        }


def commission_cost(
    spec: SymbolSpec,
    volume: float,
    *,
    per_lot_per_side: float = 0.0,
) -> CommissionCost:
    """Round-trip commission in account currency.

    per_lot_per_side: account-currency cost per 1.0 lot per side.
        Exness Standard: 0.0 (cost is in spread)
        Exness Raw: ~7.0 USD per lot per side
    """
    if volume <= 0:
        raise ValueError(f"volume must be > 0; got {volume}")
    vol = spec.normalise_volume(volume)
    one_way = per_lot_per_side * vol
    return CommissionCost(
        one_way=one_way,
        round_trip=2.0 * one_way,
        per_lot_per_side=per_lot_per_side,
        volume=vol,
    )
