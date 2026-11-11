"""Order, Fill, Trade, OrderIntent dataclasses.

These are the event-driven backtester's vocabulary. An order INTENDS
something; a FILL records what actually happened. A trade is the
matched (entry_fill, exit_fill) pair. The position tracker keeps the
running state.

Naming follows the spec's event-driven model (no SQL polling, no broker
dependency - the engine simulates fills deterministically).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from engine.core.ids import new_id

Direction = Literal["long", "short"]

# Why a position closed. Recorded per trade and aggregated into the
# per-strategy exit-reason distribution the report carries.
EXIT_SIGNAL = "SIGNAL"                                  # opposite cross / close intent
EXIT_SL = "SL"                                          # stop-loss touched
EXIT_TP = "TP"                                          # take-profit touched
EXIT_EOD = "EOD"                                        # still open at end of data
EXIT_FRIDAY_CLOSE = "FRIDAY_CLOSE"                      # B4 guardrail: flattened before the weekend
EXIT_SLTP_DEFERRED_ZERO_SPREAD = "SLTP_DEFERRED_ZERO_SPREAD"  # owed from inside a no-trade hour
EXIT_REASONS = (
    EXIT_SIGNAL,
    EXIT_SL,
    EXIT_TP,
    EXIT_EOD,
    EXIT_FRIDAY_CLOSE,
    EXIT_SLTP_DEFERRED_ZERO_SPREAD,
)
OrderType = Literal["MARKET2"]  # S3 supports only MARKET2; more in S6+


@dataclass(frozen=True)
class OrderIntent:
    """Strategy output: the strategy decided to do something.

    The backtester converts this into a Fill on the NEXT bar's open
    (no look-ahead). volume==0 means "close current position", not
    "do nothing" - this matches Algory's convention and is the only
    way a strategy can exit without an explicit reverse signal.
    """

    direction: Direction
    volume: float          # lots; 0 means "close any open position"
    # Absolute levels. Only an order that knows its own price can set
    # these; a market order cannot, because its fill price is the NEXT
    # bar's open and the strategy is not allowed to see it. They stay
    # None on the MARKET2 path and are here for S6's LIMIT/STOP.
    sl_price: float | None = None
    tp_price: float | None = None
    # DISTANCES in price units, set by the strategy at signal time from
    # ATR at the signal bar. The engine anchors them to the actual fill
    # price when the position opens: LONG sl = fill - sl_dist.
    #
    # The spec writes the rule as "sl = entry_open - sl_dist". The
    # subtraction cannot happen in the strategy: entry_open is the next
    # bar's open, and the whole structural look-ahead block exists to
    # keep that out of the strategy's hands. So the strategy emits the
    # distance -- which depends only on bars it may see -- and the
    # engine does the arithmetic at fill time. Same formula, placed on
    # the only side of the boundary that can evaluate it honestly.
    sl_dist: float | None = None
    tp_dist: float | None = None
    intent_id: str = field(default_factory=new_id)
    bar_index: int = 0     # bar at which the signal was generated (close time)


@dataclass(frozen=True)
class Fill:
    """A single execution. The backtester emits these; the strategy never does."""

    fill_id: str
    order_id: str            # backtest-assigned sequence number
    direction: Direction
    volume: float            # lots, always > 0
    requested_price: float   # the bar's open that the strategy wanted
    filled_price: float      # after slippage and spread adjustment
    spread_cost: float       # account currency, this fill's half
    slippage_cost: float     # account currency, this fill
    commission_cost: float   # account currency, this fill
    bar_index: int           # the bar whose OPEN produced this fill
    bar_time: int             # UNIX seconds
    intent_id: str | None = None   # the OrderIntent that caused this Fill

    @property
    def total_cost(self) -> float:
        """Per-fill cost charged against the trade P&L (in account ccy)."""
        return self.spread_cost + self.slippage_cost + self.commission_cost


@dataclass
class Trade:
    """A matched (entry_fill, exit_fill) pair. The atomic unit of P&L.

    The engine creates a Trade on entry (exit_fill=None) and mutates
    exit_fill when the position closes. The Trade is therefore NOT
    frozen; downstream code must not mutate it after the backtest
    finishes.
    """

    trade_id: str
    strategy_id: str
    symbol: str
    direction: Direction
    entry_fill: Fill
    exit_fill: Fill | None  # None while trade is still open
    entry_bar_index: int
    exit_bar_index: int | None
    volume: float           # lots
    swap_charged: float = 0.0   # account ccy
    # P&L fields populated by the engine at the moment the trade closes.
    # Both are in account currency. ``gross_price_pnl`` is the price move
    # only (no costs, no swap); ``realized_pnl`` is gross minus entry-
    # side total cost minus swap. Zero on a still-open trade.
    gross_price_pnl: float = 0.0
    realized_pnl: float = 0.0
    notes: str = ""
    # Absolute protective levels, anchored to the ENTRY FILL price when
    # the position opened. None when the genes asked for no stop, or
    # when ATR was undefined at the signal bar.
    sl_price: float | None = None
    tp_price: float | None = None
    # Why the position closed. One of ExitReason; None while open.
    exit_reason: str | None = None

    @property
    def is_open(self) -> bool:
        return self.exit_fill is None

    @property
    def entry_costs(self) -> float:
        return self.entry_fill.total_cost

    @property
    def exit_costs(self) -> float:
        return self.exit_fill.total_cost if self.exit_fill else 0.0

    @property
    def total_costs(self) -> float:
        return self.entry_costs + self.exit_costs + self.swap_charged
