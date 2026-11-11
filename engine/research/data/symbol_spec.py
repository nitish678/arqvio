"""Symbol spec snapshot.

A frozen, JSON-serialisable record of a broker symbol's contract parameters,
frozen at a specific instant. The cost model reads this snapshot - it never
queries the broker mid-strategy - so that backtests, portfolio sims, and live
deployments all use identical numbers.

The Exness quirk this file is built to handle: trade_tick_value_profit !=
trade_tick_value_loss for several symbols. The cost model needs both
separately; a single "tick_value" loses information.

Conventions
-----------
* All prices are in the symbol's price currency.
* `point` is the smallest price increment (e.g. 1e-5 for USDCADm with
  digits=5).
* `digits` is the number of decimal places in displayed prices.
* `contract_size` is the lot multiplier (e.g. 100_000 for a standard FX lot).
* `tick_value_profit` / `tick_value_loss` are account-currency P&L per
  *one tick* of price movement, in the direction of profit or loss
  respectively. Exness quotes these separately; most brokers do not.
* `swap_long` / `swap_short` are overnight roll costs in points per lot.
* `swap_rollover_triple_day` is the day of the week (Mon=0..Sun=6) on which
  the rollover is charged 3x to cover the weekend. Exness = Wednesday = 2.
* `volume_min` / `volume_step` / `volume_max` are MT5's lot bounds.
* `freeze_level` / `stop_level` are the broker's order-validation distances
  in points; a MARKET2 order must be at least `stop_level` away from the
  current price and the broker freezes its books for `freeze_level` points
  around rollover.
* `currency_base` / `currency_profit` are the three-letter ISO codes for
  the base and profit currencies (e.g. USDCADm: USD base, CAD profit).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SymbolSpec:
    name: str
    path: str
    visible: bool
    digits: int
    point: float
    contract_size: float
    trade_tick_size: float
    trade_tick_value: float           # broker's default tick value (kept for compat)
    trade_tick_value_profit: float    # P&L per tick when trade is profitable
    trade_tick_value_loss: float      # P&L per tick when trade is at a loss
    spread_points: int                # current spread in points
    currency_base: str
    currency_profit: str
    volume_min: float
    volume_step: float
    volume_max: float
    swap_long: float                  # points per lot per night
    swap_short: float                 # points per lot per night
    swap_rollover_triple_day: int     # 0=Mon ... 6=Sun (Exness = 2 = Wed)
    freeze_level: int                 # points
    stop_level: int                   # points
    snapshot_at_utc: str              # ISO-8601, when the snapshot was taken
    source: str                       # 'MT5:<broker>' for now

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def profit_per_price_unit_per_lot(self) -> float:
        """P&L per 1.0 price move, per 1.0 lot, in account currency (profit side)."""
        if self.trade_tick_size <= 0 or self.point <= 0:
            raise ValueError("invalid tick_size/point for profit calculation")
        return self.trade_tick_value_profit * (self.point / self.trade_tick_size)

    def loss_per_price_unit_per_lot(self) -> float:
        """P&L per 1.0 price move, per 1.0 lot, in account currency (loss side)."""
        if self.trade_tick_size <= 0 or self.point <= 0:
            raise ValueError("invalid tick_size/point for loss calculation")
        return self.trade_tick_value_loss * (self.point / self.trade_tick_size)

    def spread_in_price(self) -> float:
        """Current spread expressed in price units (e.g. 0.00014 for USDCADm)."""
        return self.spread_points * self.point

    def normalise_volume(self, volume: float) -> float:
        """Round volume to the broker's lot step. Clamps to [volume_min, volume_max]."""
        if self.volume_step <= 0:
            raise ValueError("invalid volume_step")
        steps = round(volume / self.volume_step)
        v = steps * self.volume_step
        v = max(self.volume_min, min(self.volume_max, v))
        return float(v)

    def validate_sl_tp_distance(
        self, sl_distance_points: float, tp_distance_points: float
    ) -> tuple[bool, str]:
        """Check that SL/TP are at least stop_level points from the order price."""
        if sl_distance_points < self.stop_level:
            return False, f"SL distance {sl_distance_points:.1f} below stop_level {self.stop_level}"
        if tp_distance_points < self.stop_level:
            return False, f"TP distance {tp_distance_points:.1f} below stop_level {self.stop_level}"
        return True, "ok"

    def __post_init__(self) -> None:
        # Defensive checks. SymbolSpec is a frozen record; invalid inputs
        # should fail loudly here so callers never get a bad snapshot.
        if self.digits < 0 or self.digits > 10:
            raise ValueError(f"unreasonable digits: {self.digits}")
        if not math.isfinite(self.point) or self.point <= 0:
            raise ValueError(f"invalid point: {self.point}")
        if self.contract_size <= 0:
            raise ValueError(f"invalid contract_size: {self.contract_size}")
        if not math.isfinite(self.trade_tick_value_profit):
            raise ValueError("trade_tick_value_profit is not finite")
        if not math.isfinite(self.trade_tick_value_loss):
            raise ValueError("trade_tick_value_loss is not finite")
        if self.volume_min <= 0 or self.volume_step <= 0:
            raise ValueError("invalid volume bounds")
        if self.swap_rollover_triple_day not in range(7):
            raise ValueError("swap_rollover_triple_day must be 0..6")
        # Tick sizes may not be cleanly divisible by point; this is allowed
        # (some symbols have e.g. 0.00032 ticks on a 1e-5 point grid).
