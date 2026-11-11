"""Slippage cost model.

Slippage is the difference between the expected fill price and the
actual fill price. It is random per fill. ARQVIO uses a deterministic
seeded model: the caller supplies a mean (in points) and a std (in
points); the realised slippage is drawn from a half-normal distribution
(with mean and std) to ensure it is always non-negative.

For a backtest without live data we treat slippage as a constant
expected cost per fill - the std is for sensitivity analysis.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from engine.research.data.symbol_spec import SymbolSpec


@dataclass(frozen=True)
class SlippageCost:
    expected_price: float   # expected adverse move in price units
    one_way: float          # account currency
    round_trip: float       # both sides
    mean_points: float
    std_points: float

    def to_dict(self) -> dict[str, float]:
        return {
            "expected_price": self.expected_price,
            "one_way": self.one_way,
            "round_trip": self.round_trip,
            "mean_points": self.mean_points,
            "std_points": self.std_points,
        }


def slippage_cost(
    spec: SymbolSpec,
    volume: float,
    *,
    mean_points: float = 0.5,
    std_points: float = 0.3,
    side: str = "both",
) -> SlippageCost:
    """Expected slippage cost in account currency for a fill.

    Args:
        spec: the symbol's contract terms.
        volume: lots.
        mean_points: average slippage in points. Exness NDD: ~0.3-0.5 pips on USDCADm.
        std_points: per-fill std (used for sensitivity, not for the expectation).
        side: "entry", "exit", or "both" (default).
    """
    if mean_points < 0:
        raise ValueError("mean_points must be >= 0")
    if std_points < 0:
        raise ValueError("std_points must be >= 0")
    if volume <= 0:
        raise ValueError(f"volume must be > 0; got {volume}")
    vol = spec.normalise_volume(volume)
    expected_price = mean_points * spec.point
    tick_value = max(spec.trade_tick_value_profit, spec.trade_tick_value_loss)
    if spec.trade_tick_size <= 0:
        raise ValueError("invalid trade_tick_size")
    per_unit_per_lot = tick_value / spec.trade_tick_size
    one_way = expected_price * vol * per_unit_per_lot
    if side == "entry":
        return SlippageCost(expected_price, one_way, one_way, mean_points, std_points)
    if side == "exit":
        return SlippageCost(expected_price, one_way, one_way, mean_points, std_points)
    if side == "both":
        return SlippageCost(expected_price, one_way, 2.0 * one_way, mean_points, std_points)
    raise ValueError(f"side must be entry|exit|both; got {side!r}")


# Sanctioned per-asset-class caps, in POINTS, from the spec text.
# Nothing else is invented here; a symbol outside these two classes
# uses the FX cap, which is the tighter of the two.
SLIPPAGE_CAP_POINTS_FX = 10.0
SLIPPAGE_CAP_POINTS_METALS = 30.0

_METAL_PREFIXES = ("XAU", "XAG", "XPT", "XPD")


def slippage_cap_for_symbol(
    symbol: str,
    *,
    fx_cap: float = SLIPPAGE_CAP_POINTS_FX,
    metals_cap: float = SLIPPAGE_CAP_POINTS_METALS,
) -> float:
    """The cap in points for this symbol's asset class."""
    upper = symbol.upper()
    if any(upper.startswith(p) for p in _METAL_PREFIXES):
        return float(metals_cap)
    return float(fx_cap)


def realised_slippage_points(
    *,
    mean_points: float = 0.5,
    std_points: float = 0.3,
    rng: np.random.Generator | None = None,
    cap_points: float | None = None,
) -> float:
    """Draw a single realised slippage value, half-normal, capped.

    Always >= 0 (adverse move only). Used by the backtester to perturb
    a fill price.

    A5: the half-normal draw is unbounded above. Without ``cap_points``
    a single tail sample can charge an entry hundreds of points, which
    is not slippage, it is a different instrument. The cap is applied
    to the DRAW, so the fill price can never move further than a real
    venue would let it.
    """
    if rng is None:
        rng = np.random.default_rng()
    if std_points == 0:
        value = float(mean_points)
    else:
        # half-normal: |N(mean, std)|
        value = float(abs(rng.normal(loc=mean_points, scale=std_points)))
    if cap_points is not None:
        value = min(value, float(cap_points))
    return value
