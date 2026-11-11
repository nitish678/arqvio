"""Backtester: event-driven strategy execution simulator.

This is the S3 module. It implements:

  - One bar at a time, oldest first.
  - On each bar close, the strategy emits OrderIntent(s).
  - On the NEXT bar's open, intents are converted to Fills (slippage +
    spread + commission applied).
  - SL/TP are checked intra-bar using the bar's high/low. The
    conservative rule: if BOTH are inside the bar, SL hits first.
  - Swap is charged at each daily roll (EET 22:00 broker-local).
  - Weekend gap rule: positions held over Fri->Sun are valued at
    Monday's open (not Friday's close).
  - Strategy is a black box that returns OrderIntents given the state
    visible up to the current bar's close.

Only MARKET2 exec mode is supported in S3. LIMIT/LIMIT2/STOP come in S6.

P&L accounting (closed trade):
    gross = sign * (exit_price - entry_price) * per_unit_per_lot * volume
    net   = gross - entry_costs - exit_costs - swap

per_unit_per_lot = max(tick_value_profit, tick_value_loss) / trade_tick_size
(the MT5 trade_tick_size is in price units per the S2 finding).

Equity is marked-to-market on each bar's close. Realised P&L only
flips when a trade closes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import TYPE_CHECKING, Protocol

import numpy as np

from engine.core.errors import DataError
from engine.core.ids import new_id
from engine.research.backtest.order import (
    EXIT_EOD,
    EXIT_FRIDAY_CLOSE,
    EXIT_SIGNAL,
    EXIT_SL,
    EXIT_SLTP_DEFERRED_ZERO_SPREAD,
    EXIT_TP,
    Fill,
    OrderIntent,
    Trade,
)
from engine.research.costs.commission import commission_cost
from engine.research.costs.slippage import (
    realised_slippage_points,
    slippage_cap_for_symbol,
)
from engine.research.costs.spread import spread_cost
from engine.research.costs.swap import swap_cost
from engine.research.data.mt5_source import BarData
from engine.research.data.symbol_spec import SymbolSpec

if TYPE_CHECKING:
    from numpy import ndarray


from engine.research.features.atr import DEFAULT_ATR_PERIOD, sma_atr_series, true_range_series
from engine.research.features.forward_block import bars_up_to


class Strategy(Protocol):
    """A strategy is a callable: on_bar(bar_index, bars, state) -> OrderIntent | None."""

    strategy_id: str

    def on_bar(
        self,
        bar_index: int,
        bars: BarData,
        state: dict[str, object],
    ) -> OrderIntent | None:
        ...


@dataclass(frozen=True)
class BacktestConfig:
    """Backtest configuration. Pure data; no I/O."""

    strategy_id: str
    symbol: str
    timeframe: str
    initial_equity: float = 10_000.0
    risk_per_trade_pct: float = 0.01
    commission_per_lot_per_side: float = 0.0   # Exness Standard
    slippage_mean_points: float = 0.3
    slippage_std_points: float = 0.2
    charge_swap: bool = True
    charge_commission: bool = True
    seed: int = 0
    notes: str = ""
    # UTC hours in which NOTHING fills: no entries, no pending-order
    # triggers, no SL/TP. Sourced from the calibrated spread matrix, where
    # a median spread of zero means the broker quoted no cost -- which is
    # not a cheap hour, it is an untradeable one.
    no_trade_hours: tuple[int, ...] = ()
    # A5. Entry slippage cap in POINTS. None resolves per asset class
    # from the symbol name (FX 10, metals 30 -- the sanctioned values).
    slippage_cap_points: float | None = None
    # A5. Entries are refused on a bar whose spread exceeds this, in
    # POINTS. None disables the gate. The campaign layer sets it from
    # the p95 of the symbol's own calibrated spread distribution, so
    # the threshold is data-driven rather than a hardcoded pip count.
    max_spread_points: float | None = None
    # v4 spread-regime gate: a per-bar cap in POINTS (costs/spread_regime.py; NaN = no gate on that bar).
    # When set it replaces max_spread_points for the entry gate.
    spread_cap_by_bar: ndarray | None = None
    # B4 guardrail 1. No NEW entries once the day's realised loss
    # reaches this percentage of starting equity. None disables it.
    # Open positions are left alone: a daily loss limit stops you
    # digging, it does not liquidate what you already hold.
    daily_loss_limit_pct: float | None = None
    # B4 guardrail 2. Flatten open positions at this broker hour on a
    # Friday, at the last TRADEABLE bar at or before it -- a no-trade
    # hour cannot be used to flatten. None disables it.
    friday_close_hour: int | None = None
    # W0.1: 24-element tuple of hourly spread points from the calibrated
    # matrix. When set, the entry-side spread cost on each bar uses the
    # bucket for that bar's UTC hour (hour = (bar_time // 3600) % 24)
    # instead of the flat spec.spread_points. None falls back to the flat
    # snapshot spread (the pre-W0.1 behaviour).
    hourly_spread_points: tuple[float, ...] | None = None
    # v4 (ARQVIO_V4.md 10.1): the spread in POINTS charged on a fill at each bar -- the signal candle's own
    # historical spread (costs/spread_regime.fill_spread_points). NaN on a bar: the hourly / flat rule above.
    fill_spread_points: ndarray | None = None
    # v4 management gene breakeven: once a bar CLOSES with the position ahead by this many times its initial stop
    # distance (entry fill to the placed stop), the stop moves to the entry fill price, quantized, from the next
    # bar on -- once per position, and only ever towards the trade. None: off. Needs a stop to measure against.
    breakeven_at_r: float | None = None
    # v4 management genes that move the stop (step 4b, after breakeven; judged on the bar's close, acting from the next
    # bar, only ever towards the trade, measured in the initial stop distance R). None: off.
    #   sl_reduce_at_r  once ahead by this x R: the stop to entry - 0.5 R (half the risk), once per position
    #   sl_lock_at_r    once ahead by this x R: the stop to entry + half of that gain, once per position
    #   atr_trail       (k, act): once ahead by act x R, on every close the stop trails close - k x ATR(14)
    sl_reduce_at_r: float | None = None
    sl_lock_at_r: float | None = None
    atr_trail: tuple[float, float] | None = None


@dataclass(frozen=True)
class BacktestResult:
    """The output of a backtest run."""

    config: BacktestConfig
    spec_symbol: str
    bars_n: int
    trades: tuple[Trade, ...]
    equity_curve: ndarray
    final_equity: float
    n_trades: int
    n_round_trips: int
    n_open: int
    dataset_id: str = ""
    # Entries refused and exits deferred because the bar fell in a
    # zero-spread hour. Surfaced per candidate so the cost of the window is
    # visible instead of silently reshaping the equity curve.
    blocked_zero_spread_events: int = 0
    # Of those blocked events, the ones that later FILLED, on the first
    # tradeable bar after the window (R1). The difference between this
    # and blocked_zero_spread_events is work the window absorbed rather
    # than merely delayed.
    deferred_zero_spread_fills: int = 0
    # SL/TP that came due inside a no-trade hour and was taken at the
    # first tradeable open instead of at its own level.
    sltp_deferred_zero_spread: int = 0
    # A5. Entries refused because the bar's spread exceeded the gate.
    blocked_max_spread: int = 0
    # B4. Entries refused by the daily loss limit, and positions
    # flattened by the Friday-close rule.
    blocked_daily_loss: int = 0
    friday_flattens: int = 0
    # A4/W4. Where the money went, split four ways, in account currency.
    # "fee_total = commission + spread + slippage + swap" only means
    # something if the four are reported separately: a strategy killed
    # by spread and one killed by swap need different fixes.
    cost_commission: float = 0.0
    cost_spread: float = 0.0
    cost_slippage: float = 0.0
    cost_swap: float = 0.0

    @property
    def fee_total(self) -> float:
        return (
            self.cost_commission + self.cost_spread
            + self.cost_slippage + self.cost_swap
        )

    def cost_decomposition(self) -> dict[str, float]:
        """The four cost components and each one's share of the total."""
        total = self.fee_total
        out: dict[str, float] = {
            "commission": self.cost_commission,
            "spread": self.cost_spread,
            "slippage": self.cost_slippage,
            "swap": self.cost_swap,
            "fee_total": total,
        }
        # Shares are of the FEE TOTAL, not of notional: notional is not
        # a number this result carries, and dividing by a quantity we
        # would have to guess is how a plausible-looking fraction gets
        # invented.
        for name in ("commission", "spread", "slippage", "swap"):
            out[f"{name}_pct_of_fees"] = (
                (out[name] / total * 100.0) if total else 0.0
            )
        return out
    # Distribution over ExitReason. Empty when nothing closed.
    exit_reasons: dict[str, int] = field(default_factory=dict)

    @property
    def total_pnl(self) -> float:
        return self.final_equity - self.config.initial_equity

    @property
    def closed_trades(self) -> tuple[Trade, ...]:
        return tuple(t for t in self.trades if t.exit_fill is not None)

    def to_dict(self) -> dict[str, object]:
        return {
            "strategy_id": self.config.strategy_id,
            "symbol": self.spec_symbol,
            "timeframe": self.config.timeframe,
            "bars_n": self.bars_n,
            "n_trades": self.n_trades,
            "n_round_trips": self.n_round_trips,
            "n_open": self.n_open,
            "final_equity": self.final_equity,
            "initial_equity": self.config.initial_equity,
            "total_pnl": self.total_pnl,
            "dataset_id": self.dataset_id,
        }


def _is_friday(bar_time: int) -> bool:
    """1970-01-01 was a Thursday, so day 1 is a Friday: day % 7 == 1."""
    return (int(bar_time) // 86400) % 7 == 1


def _tally_exit_reasons(
    trades: list[Trade], counted: dict[str, int],
) -> dict[str, int]:
    """Exit-reason distribution over every trade.

    ``counted`` holds the reasons the run loop already tallied (SL, TP,
    the deferred exit, EOD). SIGNAL closes are recorded on the Trade by
    _fill_intents but not tallied there, so they are counted here. Any
    reason already in ``counted`` is trusted and not recounted.
    """
    out = dict(counted)
    tallied_elsewhere = set(counted)
    for t in trades:
        reason = t.exit_reason
        if reason is None or reason in tallied_elsewhere:
            continue
        out[reason] = out.get(reason, 0) + 1
    return out


# ----------------------------------------------------------------------------
# Internal: position state
# ----------------------------------------------------------------------------

@dataclass
class _Position:
    trade: Trade
    # P&L cache updated when the trade closes.
    realised_pnl: float = 0.0
    # breakeven: the stop distance the position opened with, and whether the stop has been moved to entry.
    initial_sl_dist: float | None = None
    breakeven_done: bool = False
    reduce_done: bool = False
    lock_done: bool = False

    def sign(self) -> float:
        return 1.0 if self.trade.direction == "long" else -1.0


# ----------------------------------------------------------------------------
# The engine
# ----------------------------------------------------------------------------

class BacktestEngine:
    """Event-driven MARKET2 backtester. S3 scope only."""

    def __init__(self, spec: SymbolSpec, config: BacktestConfig, *, rng_seed: int = 0) -> None:
        self.spec = spec
        self.config = config
        self._rng = np.random.default_rng(rng_seed)
        if spec.trade_tick_size <= 0:
            raise ValueError("invalid trade_tick_size")
        # Per-lot-per-price-unit factor (account ccy per 1.0 price move
        # per 1.0 lot). MT5 trade_tick_size is in price units (per S2).
        # MT5 reports two tick values because they are not the same
        # number: trade_tick_value_profit applies to a winning leg and
        # trade_tick_value_loss to a losing one. Taking max() for both
        # (as this did) overstates every loss by the difference. On
        # USDCADm that is ~7e-5 per tick, small; the point is that it is
        # a systematic bias in the pessimistic direction for winners and
        # the optimistic direction for losers, and the spec (S2 6.3)
        # says per leg.
        #
        # This one is kept for the slippage charge, which is a cost and
        # has no winning side.
        self._per_unit_per_lot = max(
            spec.trade_tick_value_profit, spec.trade_tick_value_loss
        ) / spec.trade_tick_size
        self._per_unit_profit = spec.trade_tick_value_profit / spec.trade_tick_size
        self._per_unit_loss = spec.trade_tick_value_loss / spec.trade_tick_size
        self._slippage_cap_points = (
            self.config.slippage_cap_points
            if self.config.slippage_cap_points is not None
            else slippage_cap_for_symbol(spec.name)
        )

    # -- public ---------------------------------------------------------------

    def run(self, bars: BarData, strategy: Strategy) -> BacktestResult:
        if bars.n_bars < 2:
            raise DataError(f"need >= 2 bars; got {bars.n_bars}")
        if bars.symbol != self.spec.name:
            raise DataError(
                f"bar symbol {bars.symbol!r} != spec {self.spec.name!r}"
            )
        if not hasattr(strategy, "strategy_id") or not strategy.strategy_id:
            raise DataError("strategy must define a non-empty strategy_id")

        equity: list[float] = [self.config.initial_equity]
        trades: list[Trade] = []
        realised_total = 0.0
        position: _Position | None = None
        pending_intents: list[OrderIntent] = []
        strategy_state: dict[str, object] = {}
        order_seq = 0
        last_swap_day: int | None = None
        no_trade_hours = frozenset(int(h) for h in self.config.no_trade_hours)
        blocked_events = 0
        # Set when SL/TP would have fired inside a no-trade window. The exit
        # is owed and is taken on the first tradeable bar, at that bar's
        # open -- the same price entries fill at, not the stop price, which
        # was never available.
        deferred_exit = False
        # Intents that were refused a fill inside a no-trade hour, by id.
        deferred_ids: set[str] = set()
        deferred_fills = 0
        # SL/TP owed from inside a no-trade hour, and the tally of them.
        deferred_exit_reason: str | None = None
        sltp_deferred_zero_spread = 0
        blocked_max_spread = 0
        max_spread_points = self.config.max_spread_points
        spread_cap = self.config.spread_cap_by_bar
        if spread_cap is not None and len(spread_cap) != len(bars.spread):
            raise ValueError(f"spread_cap_by_bar has {len(spread_cap)} values for {len(bars.spread)} bars")
        fill_spread = self.config.fill_spread_points
        if fill_spread is not None and len(fill_spread) != len(bars.spread):
            raise ValueError(f"fill_spread_points has {len(fill_spread)} values for {len(bars.spread)} bars")
        # v4 stop rules: the ATR the trail reads is prep_cell's atr14 (SMA of the true range, NaN in the warmup).
        stop_rules = (self.config.sl_reduce_at_r is not None or self.config.sl_lock_at_r is not None
                      or self.config.atr_trail is not None)
        atr_series = (sma_atr_series(true_range_series(np.asarray(bars.high, dtype=np.float64),
                                                       np.asarray(bars.low, dtype=np.float64),
                                                       np.asarray(bars.close, dtype=np.float64)),
                                     period=DEFAULT_ATR_PERIOD)
                      if self.config.atr_trail is not None else None)
        # B4 guardrails.
        daily_loss_limit_pct = self.config.daily_loss_limit_pct
        friday_close_hour = self.config.friday_close_hour
        blocked_daily_loss = 0
        friday_flattens = 0
        day_start_realised = 0.0
        current_day: int | None = None
        exit_reasons: dict[str, int] = {}

        for i in range(bars.n_bars - 1):
            bar_time = int(bars.time[i])
            bar_open = float(bars.open[i])
            bar_high = float(bars.high[i])
            bar_low = float(bars.low[i])
            bar_close = float(bars.close[i])
            hour = (bar_time // 3600) % 24
            blocked = hour in no_trade_hours

            # 0. An exit owed from inside the window is taken now, before
            #    anything else, at this bar's open.
            if deferred_exit and not blocked and position is not None:
                gross_pnl, net_pnl = self._close_trade_pnl_split(
                    position, bar_open,
                )
                order_seq += 1
                position.trade.exit_fill = self._protective_exit_fill(
                    position, bar_open, bar_time, i, order_seq,
                )
                position.trade.exit_bar_index = i
                position.trade.gross_price_pnl = gross_pnl
                position.trade.realized_pnl = net_pnl
                # The owed exit is its own reason, not SL/TP: it did not
                # get the stop price, it got this bar's open.
                position.trade.exit_reason = EXIT_SLTP_DEFERRED_ZERO_SPREAD
                exit_reasons[EXIT_SLTP_DEFERRED_ZERO_SPREAD] = (
                    exit_reasons.get(EXIT_SLTP_DEFERRED_ZERO_SPREAD, 0) + 1
                )
                sltp_deferred_zero_spread += 1
                realised_total += net_pnl
                position.realised_pnl = net_pnl
                position = None
                deferred_exit = False
                deferred_exit_reason = None
            elif deferred_exit and not blocked:
                deferred_exit = False
                deferred_exit_reason = None

            # 1. Fill pending intents on THIS bar's open.
            #
            # R1 -- zero-spread window policy. An intent blocked inside a
            # no-trade hour is NEITHER dropped NOR discarded. It stays
            # pending and fills at the open of the first tradeable bar after
            # the window. This mirrors the SL/TP deferral below and
            # approximates live MT5: an order sent across a quote freeze is
            # not rejected, it executes at reopen, at the reopened spread.
            # The alternative -- dropping the signal -- would silently
            # delete trades from the equity curve and make the window look
            # cheaper than it is. Only MARKET2 exists today (EXEC_VALUES);
            # when LIMIT/LIMIT2/STOP arrive in S6 they keep their own
            # expiry, so an intent whose expiry passes before the window
            # reopens is dropped rather than deferred, and that drop must
            # get its own counter here.
            # B4. Day boundary: reset the daily loss baseline.
            bar_day = bar_time // 86400
            if current_day is None or bar_day != current_day:
                current_day = bar_day
                day_start_realised = realised_total

            # B4 guardrail 2: Friday close. Flatten at the last
            # TRADEABLE bar at or before the configured hour -- a
            # no-trade hour cannot be used to flatten, so the position
            # carries to the next tradeable bar rather than filling in
            # a window where nothing fills.
            if (
                friday_close_hour is not None
                and position is not None
                and not blocked
                and _is_friday(bar_time)
                and hour >= friday_close_hour
            ):
                gross_pnl, net_pnl = self._close_trade_pnl_split(
                    position, bar_open,
                )
                order_seq += 1
                position.trade.exit_fill = self._protective_exit_fill(
                    position, bar_open, bar_time, i, order_seq,
                )
                position.trade.exit_bar_index = i
                position.trade.gross_price_pnl = gross_pnl
                position.trade.realized_pnl = net_pnl
                position.trade.exit_reason = EXIT_FRIDAY_CLOSE
                exit_reasons[EXIT_FRIDAY_CLOSE] = (
                    exit_reasons.get(EXIT_FRIDAY_CLOSE, 0) + 1
                )
                realised_total += net_pnl
                position.realised_pnl = net_pnl
                position = None
                friday_flattens += 1
                strategy_state.pop("direction", None)
                # Nothing new opens on a bar we just flattened on.
                entries = [x for x in pending_intents if x.volume > 0]
                if entries:
                    pending_intents = [
                        x for x in pending_intents if x.volume <= 0
                    ]

            # B4 guardrail 1: daily loss limit. No NEW entries once the
            # day's realised loss reaches the limit. Open positions are
            # untouched -- this stops you digging, it does not
            # liquidate what you already hold.
            if daily_loss_limit_pct is not None and pending_intents:
                day_loss = day_start_realised - realised_total
                limit = (
                    self.config.initial_equity
                    * float(daily_loss_limit_pct) / 100.0
                )
                if day_loss >= limit > 0:
                    entries = [x for x in pending_intents if x.volume > 0]
                    if entries:
                        blocked_daily_loss += len(entries)
                        pending_intents = [
                            x for x in pending_intents if x.volume <= 0
                        ]

            # A5. A bar whose spread blew out is not a bar an entry
            # would have been filled on at anything like the modelled
            # cost. ENTRIES are refused; exits are NOT, because
            # refusing to close a position because spreads widened
            # traps it in exactly the conditions it should be leaving.
            if spread_cap is not None:
                cap_i = float(spread_cap[i])  # NaN: no gate on this bar (compares False)
            else:
                cap_i = max_spread_points
            if (
                cap_i is not None
                and not blocked
                and pending_intents
                and float(bars.spread[i]) > cap_i
            ):
                entries = [x for x in pending_intents if x.volume > 0]
                if entries:
                    blocked_max_spread += len(entries)
                    pending_intents = [
                        x for x in pending_intents if x.volume <= 0
                    ]

            if blocked:
                blocked_events += len(pending_intents)
                # By identity, not by count. On the reopening bar the
                # pending list also holds the intent the strategy emitted
                # at the END of the blocked bar (step 5 runs even when the
                # window is shut); that one was never refused a fill, so
                # counting the list length would over-report.
                deferred_ids.update(x.intent_id for x in pending_intents)
            else:
                if deferred_ids:
                    deferred_fills += sum(
                        1 for x in pending_intents
                        if x.intent_id in deferred_ids
                    )
                    deferred_ids.clear()
                order_seq, position, realised_total, filled = self._fill_intents(
                    pending_intents, bar_open, bar_time, i, order_seq,
                    position, realised_total,
                )
                trades.extend(filled)
                pending_intents = []

            # 2. Charge swap at day-roll.
            day = bar_time // 86400
            if (position is not None and last_swap_day is not None
                    and day != last_swap_day and self.config.charge_swap):
                position = self._charge_swap(position, last_swap_day, day)
            last_swap_day = day if position is not None else None

            # 3. SL/TP against THIS bar's range, from the entry bar
            # onward and BEFORE the strategy sees the bar (step 5). A
            # position that was stopped out cannot also act on a signal
            # from the same bar.
            if position is not None and position.trade.entry_fill is not None:
                touched = self._check_sl_tp(
                    position, bar_high, bar_low, bar_index=i
                )
                if touched is not None and blocked:
                    # The level was touched but nothing could execute. The
                    # position carries through and the exit is owed.
                    if not deferred_exit:
                        blocked_events += 1
                    deferred_exit = True
                    deferred_exit_reason = touched
                elif touched is not None:
                    exit_price = self._exit_price_for(position, touched, bar_open)
                    gross_pnl, net_pnl = self._close_trade_pnl_split(
                        position, exit_price,
                    )
                    order_seq += 1
                    position.trade.exit_fill = self._protective_exit_fill(
                        position, exit_price, bar_time, i, order_seq,
                    )
                    position.trade.exit_bar_index = i
                    position.trade.gross_price_pnl = gross_pnl
                    position.trade.realized_pnl = net_pnl
                    position.trade.exit_reason = touched
                    exit_reasons[touched] = exit_reasons.get(touched, 0) + 1
                    realised_total += net_pnl
                    position.realised_pnl = net_pnl
                    position = None
                    # The strategy's own state still thinks it holds this
                    # position; clear it so the next cross opens cleanly.
                    strategy_state.pop("direction", None)

            # 4. Mark-to-market equity.
            unrealised = self._unrealised(position, bar_close)
            equity.append(self.config.initial_equity + realised_total + unrealised)

            # 4b. Management (v4 gene breakeven): judged on this bar's close, the stop it sets acts from the
            # next bar's step 3 -- the live EA moves the stop when it sees the closed bar.
            if position is not None and self.config.breakeven_at_r is not None:
                self._breakeven(position, bar_close)
            if position is not None and stop_rules:
                self._stop_rules(position, bar_close, float("nan") if atr_series is None else float(atr_series[i]))

            # 5. Strategy emits an intent for the NEXT bar's open.
            # Structural look-ahead block: the strategy sees bars up to
            # and including i, never past it. Statistical correction does
            # not catch leakage (a planted oracle at Sharpe 35 survived
            # both DSR and PBO), so this is enforced at the data-access
            # boundary rather than by convention.
            intent = strategy.on_bar(i, bars_up_to(bars, i), strategy_state)
            if intent is not None and not (intent.volume == 0 and position is None):
                pending_intents.append(intent)

        # 6. Final bar.
        #
        # This is a SECOND fill path and it obeys the same window as the
        # first. Without the check below, an intent pending into a final
        # bar that falls in a no-trade hour filled at that bar's open --
        # inside the window the gate exists to close. There is no bar after
        # the last one, so a blocked intent here cannot be deferred: it
        # simply never executes, which is the correct end-of-data outcome
        # and the only one that keeps "nothing fills in a no-trade hour"
        # true for every path.
        last_idx = bars.n_bars - 1
        if bars.n_bars >= 1:
            last_open = float(bars.open[last_idx])
            last_time = int(bars.time[last_idx])
            last_blocked = ((last_time // 3600) % 24) in no_trade_hours
            if last_blocked:
                blocked_events += len(pending_intents)
                pending_intents = []
            if deferred_ids:
                deferred_fills += sum(
                    1 for x in pending_intents if x.intent_id in deferred_ids
                )
                deferred_ids.clear()
            order_seq, position, realised_total, filled = self._fill_intents(
                pending_intents, last_open, last_time, last_idx, order_seq,
                position, realised_total,
            )
            trades.extend(filled)
            unrealised = self._unrealised(position, float(bars.close[last_idx]))
            equity.append(self.config.initial_equity + realised_total + unrealised)
            if position is not None and position.trade.exit_reason is None:
                # Still open when the data ran out. Not an exit the
                # strategy chose, and it must not be counted as one.
                position.trade.exit_reason = EXIT_EOD
                exit_reasons[EXIT_EOD] = exit_reasons.get(EXIT_EOD, 0) + 1

        return BacktestResult(
            config=self.config,
            spec_symbol=self.spec.name,
            bars_n=bars.n_bars,
            trades=tuple(trades),
            equity_curve=np.asarray(equity, dtype=np.float64),
            final_equity=float(equity[-1]),
            n_trades=len(trades),
            n_round_trips=sum(1 for t in trades if t.exit_fill is not None),
            n_open=sum(1 for t in trades if t.exit_fill is None),
            blocked_zero_spread_events=blocked_events,
            deferred_zero_spread_fills=deferred_fills,
            sltp_deferred_zero_spread=sltp_deferred_zero_spread,
            blocked_max_spread=blocked_max_spread,
            blocked_daily_loss=blocked_daily_loss,
            friday_flattens=friday_flattens,
            cost_commission=sum(
                t.entry_fill.commission_cost for t in trades if t.entry_fill
            ),
            cost_spread=sum(
                t.entry_fill.spread_cost for t in trades if t.entry_fill
            ),
            cost_slippage=sum(
                t.entry_fill.slippage_cost for t in trades if t.entry_fill
            ),
            cost_swap=sum(t.swap_charged for t in trades),
            # SIGNAL closes happen inside _fill_intents, which has no
            # access to the tally, so the distribution is completed from
            # the trades themselves. Counting one way for all reasons
            # would be cleaner but would mean threading the dict through
            # a signature that four other call sites share.
            exit_reasons=_tally_exit_reasons(trades, exit_reasons),
        )

    # -- P&L helpers ----------------------------------------------------------

    def _unrealised(self, position: _Position | None, price: float) -> float:
        if position is None:
            return 0.0
        price_move = price - float(position.trade.entry_fill.filled_price)
        gross = self._gross_pnl(position, price_move)
        # Subtract the entry-side costs that were charged on the open
        # fill, so the MTM equity matches the closed-trade accounting.
        # (Spread/commission charged on entry; slippage was already
        # baked into the filled_price.)
        return gross - position.trade.entry_fill.total_cost - position.trade.swap_charged

    def _close_trade_pnl(self, position: _Position, exit_price: float) -> float:
        """Net P&L for closing the position at exit_price (in account ccy).

        Kept for callers that only need the net value; the gross/net
        split is in :meth:`_close_trade_pnl_split`.
        """
        _, net = self._close_trade_pnl_split(position, exit_price)
        return net

    def _gross_pnl(self, position: _Position, price_move: float) -> float:
        """Price move in account currency, using the correct leg's tick value.

        The sign of the move TIMES the direction decides which one: a
        long that moved up and a short that moved down are both winning
        legs and take trade_tick_value_profit.
        """
        signed = position.sign() * price_move
        per_unit = self._per_unit_profit if signed >= 0.0 else self._per_unit_loss
        return signed * per_unit * position.trade.volume

    def _close_trade_pnl_split(
        self, position: _Position, exit_price: float,
    ) -> tuple[float, float]:
        """Return (gross_price_pnl, realized_pnl) for closing at exit_price.

        Both values are in account currency. ``gross_price_pnl`` is the
        price move only (no costs, no swap); ``realized_pnl`` is gross
        minus entry-side total cost minus accumulated swap. The split
        is what S4's ``cost_impact`` metric consumes; persisting it on
        the Trade means the metrics layer never re-derives P&L.
        """
        entry_price = float(position.trade.entry_fill.filled_price)
        price_move = exit_price - entry_price
        gross = self._gross_pnl(position, price_move)
        # Costs: entry_fill already charged round-trip; exit charges zero in
        # the simplified S3 model (we charged full spread on entry). For
        # commission, we charge half on entry, half on exit in a more
        # accurate model; for S3 we charge round-trip on entry only.
        net = gross - position.trade.entry_fill.total_cost - position.trade.swap_charged
        return gross, net

    # -- intent handling ------------------------------------------------------

    def _fill_intents(
        self,
        intents: list[OrderIntent],
        bar_open: float,
        bar_time: int,
        bar_index: int,
        order_seq_start: int,
        position: _Position | None,
        realised_total: float,
    ) -> tuple[int, _Position | None, float, list[Trade]]:
        trades: list[Trade] = []
        order_seq = order_seq_start
        for intent in intents:
            order_seq += 1
            fill = self._simulate_fill(intent, bar_open, bar_time, bar_index, order_seq)
            if intent.volume == 0:
                # Close intent.
                if position is None:
                    continue
                gross_pnl, net_pnl = self._close_trade_pnl_split(
                    position, fill.filled_price,
                )
                # Mutate the open Trade to record the exit. The Trade was
                # already appended to the trades list when it was opened.
                # exit_bar_index is the SIGNAL bar (when the strategy
                # emitted the close intent). For volume==0 closes the
                # caller passed bar_index as the bar at which the
                # strategy emitted the intent. We use intent.bar_index
                # if set, otherwise the current bar_index.
                position.trade.exit_fill = fill
                position.trade.exit_bar_index = intent.bar_index
                position.trade.gross_price_pnl = gross_pnl
                position.trade.realized_pnl = net_pnl
                position.trade.exit_reason = EXIT_SIGNAL
                realised_total += net_pnl
                position.realised_pnl = net_pnl
                position = None
            else:
                # Open or reverse. For S3, we close any existing position
                # first, then open the new one (reverse = close + open).
                if position is not None:
                    gross_pnl, net_pnl = self._close_trade_pnl_split(
                        position, fill.filled_price,
                    )
                    position.trade.exit_fill = fill
                    position.trade.exit_bar_index = intent.bar_index
                    position.trade.gross_price_pnl = gross_pnl
                    position.trade.realized_pnl = net_pnl
                    position.trade.exit_reason = EXIT_SIGNAL
                    realised_total += net_pnl
                    position.realised_pnl = net_pnl
                    position = None
                position = self._open_position(fill, intent, bar_index)
                # Append the newly opened trade to the trades list.
                trades.append(position.trade)
        return order_seq, position, realised_total, trades

    def _protective_exit_fill(
        self,
        position: _Position,
        exit_price: float,
        bar_time: int,
        bar_index: int,
        order_seq: int,
    ) -> Fill:
        """The exit Fill for an SL/TP (or deferred) close.

        Without one the Trade keeps exit_fill=None, `is_open` stays
        True, and n_round_trips counts only signal closes -- a
        stopped-out trade would read as still running.

        Costs are ZERO here on purpose: the S3 convention charges the
        full round trip on the ENTRY fill (see _simulate_fill), so
        charging again on exit would double-count. The price is the
        level (or the gapped open); the cost model already applied.
        """
        return Fill(
            fill_id=new_id(),
            order_id=str(order_seq),
            # The closing side is opposite the position.
            direction="short" if position.trade.direction == "long" else "long",
            volume=position.trade.volume,
            requested_price=exit_price,
            filled_price=exit_price,
            spread_cost=0.0,
            slippage_cost=0.0,
            commission_cost=0.0,
            bar_index=bar_index,
            bar_time=bar_time,
            intent_id=None,
        )

    def _simulate_fill(
        self,
        intent: OrderIntent,
        bar_open: float,
        bar_time: int,
        bar_index: int,
        order_seq: int,
    ) -> Fill:
        slip_pts = realised_slippage_points(
            mean_points=self.config.slippage_mean_points,
            std_points=self.config.slippage_std_points,
            rng=self._rng,
            # A5: the half-normal draw is unbounded above; one tail
            # sample would otherwise charge an entry hundreds of points.
            cap_points=self._slippage_cap_points,
        )
        slip_price = slip_pts * self.spec.point
        # MARKET2: long entry fills at ask (above mid), short at bid.
        if intent.direction == "long":
            filled_price = bar_open + slip_price
        else:
            filled_price = bar_open - slip_price

        if intent.volume == 0:
            volume = self.spec.volume_min  # unused for close fills
        else:
            volume = self.spec.normalise_volume(max(intent.volume, self.spec.volume_min))
        # Costs: in S3 we charge the ENTRY side of the spread (one_way)
        # on entry, and zero on exit. Commission is round-trip on entry
        # (since Exness charges per lot per side at 0 for Standard, the
        # distinction is moot here). The simplification is in S3 only;
        # a per-side split with proper exit-side charges lives in S6+
        # once SL/TP fills matter.
        bar_pts = (float(self.config.fill_spread_points[bar_index])
                   if self.config.fill_spread_points is not None else float("nan"))
        if intent.volume > 0:
            if bar_pts == bar_pts:  # not NaN: this bar's own historical spread
                spread_price = bar_pts * self.spec.point
                one_way = spread_price * volume * self._per_unit_per_lot
                sp = _ZeroCost(one_way=one_way, round_trip=2.0 * one_way)
            elif self.config.hourly_spread_points is not None:
                hour = (bar_time // 3600) % 24
                spread_price = self.config.hourly_spread_points[hour] * self.spec.point
                one_way = spread_price * volume * self._per_unit_per_lot
                sp = _ZeroCost(one_way=one_way, round_trip=2.0 * one_way)
            else:
                sp = spread_cost(self.spec, volume)
        else:
            sp = _zero_spread()
        commission = commission_cost(
            self.spec, volume, per_lot_per_side=self.config.commission_per_lot_per_side
        ) if self.config.charge_commission else _zero_commission()
        slippage_ccy = slip_price * self._per_unit_per_lot * volume
        # Charge one_way on entry (the entry side of the spread).
        spread_charge = sp.one_way if intent.volume > 0 else 0.0
        commission_charge = commission.round_trip if intent.volume > 0 else 0.0
        return Fill(
            fill_id=new_id(),
            order_id=str(order_seq),
            direction=intent.direction,
            volume=volume,
            requested_price=bar_open,
            filled_price=filled_price,
            spread_cost=spread_charge,
            slippage_cost=slippage_ccy,
            commission_cost=commission_charge,
            bar_index=bar_index,
            bar_time=bar_time,
            intent_id=intent.intent_id,
        )

    def _open_position(
        self, fill: Fill, intent: OrderIntent, bar_index: int,
    ) -> _Position:
        trade = Trade(
            trade_id=new_id(),
            strategy_id=self.config.strategy_id,
            symbol=self.spec.name,
            direction=intent.direction,
            entry_fill=fill,
            exit_fill=None,
            # entry_bar_index is the SIGNAL bar (where the strategy
            # detected the cross), NOT the fill bar. The fill bar is
            # entry_bar_index + 1. This convention matches the strategy's
            # perspective.
            entry_bar_index=intent.bar_index,
            exit_bar_index=None,
            volume=fill.volume,
        )
        # Anchor the protective levels to the price the trade ACTUALLY
        # got, not to the price the strategy hoped for. The strategy
        # supplied distances because it could not see this number.
        entry = float(fill.filled_price)
        long_side = intent.direction == "long"
        if intent.sl_dist is not None and intent.sl_dist > 0:
            raw = entry - intent.sl_dist if long_side else entry + intent.sl_dist
            trade.sl_price = self._quantize(raw)
        if intent.tp_dist is not None and intent.tp_dist > 0:
            raw = entry + intent.tp_dist if long_side else entry - intent.tp_dist
            trade.tp_price = self._quantize(raw)
        return _Position(trade=trade,
                         initial_sl_dist=None if trade.sl_price is None else abs(entry - trade.sl_price))

    def _breakeven(self, position: _Position, bar_close: float) -> None:
        trade = position.trade
        dist = position.initial_sl_dist
        if position.breakeven_done or dist is None or dist <= 0.0 or trade.sl_price is None:
            return
        entry = float(trade.entry_fill.filled_price)
        if (bar_close - entry) * position.sign() < float(self.config.breakeven_at_r) * dist:
            return
        level = self._quantize(entry)
        if (level - trade.sl_price) * position.sign() > 0.0:  # only ever towards the trade
            trade.sl_price = level
        position.breakeven_done = True

    def _stop_rules(self, position: _Position, bar_close: float, atr: float) -> None:
        """sl_reduce, sl_lock, atr_trail (BacktestConfig), in that order; the kernel mirrors this step for step."""
        trade = position.trade
        dist = position.initial_sl_dist
        if dist is None or dist <= 0.0 or trade.sl_price is None:
            return
        entry = float(trade.entry_fill.filled_price)
        sign = position.sign()
        gain = (bar_close - entry) * sign
        cfg = self.config
        if cfg.sl_reduce_at_r is not None and not position.reduce_done and gain >= float(cfg.sl_reduce_at_r) * dist:
            self._tighten(position, entry - sign * (0.5 * dist))
            position.reduce_done = True
        if cfg.sl_lock_at_r is not None and not position.lock_done and gain >= float(cfg.sl_lock_at_r) * dist:
            self._tighten(position, entry + sign * (0.5 * float(cfg.sl_lock_at_r) * dist))
            position.lock_done = True
        if cfg.atr_trail is not None and not np.isnan(atr) and gain >= float(cfg.atr_trail[1]) * dist:
            self._tighten(position, bar_close - sign * (float(cfg.atr_trail[0]) * atr))

    def _tighten(self, position: _Position, raw: float) -> None:
        level = self._quantize(raw)
        if (level - position.trade.sl_price) * position.sign() > 0.0:  # only ever towards the trade
            position.trade.sl_price = level

    def _quantize(self, price: float) -> float:
        """Round to the symbol's quoted precision. A stop the broker
        could not represent is not a stop that could have been placed."""
        return round(float(price), int(self.spec.digits))

    def _check_sl_tp(
        self,
        position: _Position,
        bar_high: float,
        bar_low: float,
        bar_index: int,
    ) -> str | None:
        """Which protective level this bar touched, or None.

        Returns an exit reason (EXIT_SL / EXIT_TP), not a position: the
        caller owns closing, because the price it closes at depends on
        whether the window is open (SL/TP price) or shut (deferred to
        the next tradeable open).

        PESSIMISM. When one bar's range spans BOTH levels, SL wins. An
        OHLC bar does not say which came first, and assuming the good
        one is how a backtest invents profit it never had. This is not
        tunable.
        """
        trade = position.trade
        sl = trade.sl_price
        tp = trade.tp_price
        if sl is None and tp is None:
            return None
        long_side = trade.direction == "long"
        sl_hit = sl is not None and (bar_low <= sl if long_side else bar_high >= sl)
        tp_hit = tp is not None and (bar_high >= tp if long_side else bar_low <= tp)
        if sl_hit:
            return EXIT_SL
        if tp_hit:
            return EXIT_TP
        return None

    def _exit_price_for(self, position: _Position, reason: str, bar_open: float) -> float:
        """The price a protective exit actually gets.

        The level itself, EXCEPT when the bar gapped past it: price was
        never available at the level, so the fill is the open. For a
        stop that is worse than the level, for a target it is better --
        both are just "the first price that existed".
        """
        trade = position.trade
        level = trade.sl_price if reason == EXIT_SL else trade.tp_price
        if level is None:
            return bar_open
        long_side = trade.direction == "long"
        if reason == EXIT_SL:
            gapped = bar_open < level if long_side else bar_open > level
        else:
            gapped = bar_open > level if long_side else bar_open < level
        return float(bar_open) if gapped else float(level)

    def _charge_swap(
        self, position: _Position, from_day: int, to_day: int,
    ) -> _Position:
        """Charge the nights between two day-rolls.

        ONLY those nights. This used to compute the swap from the ENTRY
        date to today and ADD the whole thing at every roll, so night 1
        was charged N times, night 2 N-1 times and so on -- a quadratic
        overcharge. On USDCADm at the real -8.4pt short rate, seven
        rolls came to -22.64 instead of -5.36, and it grew from there.

        It hid because the hardcoded spec carried a -0.4pt rate, 21x
        smaller than the broker's actual one, so the error stayed in the
        noise until W3 ingested the live numbers.
        """
        open_d = date(1970, 1, 1) + timedelta(days=int(from_day))
        close_d = date(1970, 1, 1) + timedelta(days=int(to_day))
        sc = swap_cost(
            self.spec,
            position.trade.volume,
            open_date=open_d,
            close_date=close_d,
            direction=position.trade.direction,
        )
        # Mutate in place. Rebuilding the Trade here REPLACED the object
        # that _fill_intents had already appended to the trades list, so
        # every position held across a day roll kept a stale copy in that
        # list while the close was recorded on the orphan. The run then
        # reported the trade as never closed: on USDCADm D1 that was 121
        # of 125 trades, n_round_trips 4 instead of 59, and every
        # trade-distribution metric computed over 4 samples. Trade is
        # documented as not frozen precisely so the engine can do this
        # (order.py:84-87).
        # NEGATED. swap_cost returns the MT5-signed adjustment, where a
        # negative rate means the trader PAYS (the s2 gate tests pin
        # that convention). Trade.swap_charged is a COST, positive when
        # money leaves, because _close_trade_pnl_split SUBTRACTS it.
        # Without the negation a -8.4pt short swap paid the strategy:
        # the cost decomposition read swap = -10663 on a 10k account,
        # i.e. a credit larger than the account.
        position.trade.swap_charged += -sc.account_charged
        return position


# ----------------------------------------------------------------------------
# Module helpers
# ----------------------------------------------------------------------------

@dataclass
class _ZeroCost:
    round_trip: float = 0.0
    one_way: float = 0.0


def _zero_spread() -> _ZeroCost:
    return _ZeroCost()


def _zero_commission() -> _ZeroCost:
    return _ZeroCost()
