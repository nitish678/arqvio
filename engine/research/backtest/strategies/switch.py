"""V4 switch-genome strategy: bias + filter gates wrapping a family strategy.

W4-2.2 (DONE). W4-2.5 (DONE): management exit gates. The research genome
switches ON/OFF indicator conditions:
  - BIAS switches (group `bias.`) = directional side gates. A long is taken
    only when the bias condition is true; a short only when its mirror is.
    Zero or more bias switches are AND-ed: if ALL on bias conditions fire for
    a direction, that direction is allowed.
  - FILTER switches (group `filter.`) = no-trade bars. If any filter is ON and
    its condition is FALSE, no entry on that bar (close and skip).
  - SIGNAL switches (group `signal.`) = the entry family (already wired).
  - MANAGEMENT switches (group `management.`) = exit guards applied to an
    OPEN position (breakeven, partial TP, friday close, EOD).
  - RISK switches (group `risk.`) = session/day DD guards applied at entry.

SwitchStrategy wraps an inner family strategy (the `signal` family). On each
bar it:
  1. asks the inner strategy for its intent,
  2. if intent is None -> return None (no change),
  3. if intent is an OPENING order -> apply bias gate (direction), filter gate
     (bar veto), and risk gate (session / day DD); if any rejects, drop it and
     pop `direction` from state (mirrors GatedStrategy),
  4. if intent is a CLOSING/management action -> apply management/risk exits.

The bias and filter conditions are pure functions of the visible bar slice
(bars[0..i]); no future bar is read. Each gate is independent and testable.

Idea source: research §4.1, §5.1 (BREED), APPENDIX-D seeds,
CB2 (bias = "direction" indicators, filter = entry gate).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from engine.research.backtest.order import OrderIntent
from engine.research.backtest.gates import TrendEma, session_open
from engine.research.features.atr import DEFAULT_ATR_PERIOD, atr_at

if TYPE_CHECKING:
    from engine.research.data.mt5_source import BarData


# ---------------------------------------------------------------------------
# Bias gates: each returns True when the bar supports a LONG (short = mirror).
# ---------------------------------------------------------------------------
# TODO 2.7: implement each bias gate. Stubs raise NotImplementedError until wired
# (W4-2.2 skeleton): the structure exists, gates are opt-in by switch name.
# A switch that is PLANNED (no implementation) must raise loudly at construction
# (fail hard, not silently ignore) so a PLANNED gene can never reach a trial.

class SwitchGateNotImplemented(NotImplementedError):
    """A switch gene named in the DNA has no evaluator yet (still PLANNED)."""


def _bias_gate_sma(bars, bar_index, params):
    """bias.sma: long if close above slow SMA, short if below (trend side).

    W4-2.7: implemented when the `sma` feature exists. Until then this function
    is kept as a placeholder — it is NOT registered in _BIAS_GATES (sma stays
    PLANNED), so it never runs during a trial. Kept so the gate body is ready
    to register the moment `engine.research.features.ema.sma` lands.
    """
    from engine.research.features.ema import sma  # noqa: F401  # W4-2.7: not yet present
    raise SwitchGateNotImplemented("sma feature not yet implemented (W4-2.7)")


_BIAS_GATES = {
    # name -> callable(bars, bar_index, params) -> (long_ok, short_ok)
    # PLANNED bias switches raise at call; READY ones compute.
}


def _bias_not_ready(name):
    def _raise(bars, bar_index, params):
        raise SwitchGateNotImplemented(
            f"bias switch {name!r} is PLANNED (no evaluator yet) — cannot breed"
        )
    return _raise


# W4-2.2: no bias gate is READY yet (sma needs an SMA feature that doesn't exist).
# bias.sma stays PLANNED -> raises SwitchGateNotImplemented if named in a DNA.
# This gate will be registered in W4-2.7 when sma is implemented.


def _bias_gate(name, bars, bar_index, params):
    gate = _BIAS_GATES.get(name)
    if gate is None:
        raise SwitchGateNotImplemented(
            f"bias switch {name!r} has no evaluator in the genome (PLANNED). "
            "Add it to _BIAS_GATES when its gate is wired."
        )
    return gate(bars, bar_index, params)


# ---------------------------------------------------------------------------
# Filter gates: each returns True when the bar is TRADEABLE (entry allowed).
# ---------------------------------------------------------------------------
def _filter_not_ready(name):
    def _raise(bars, bar_index, params):
        raise SwitchGateNotImplemented(
            f"filter switch {name!r} is PLANNED (no evaluator yet) — cannot breed"
        )
    return _raise


_FILTER_GATES: dict[str, object] = {}


def _filter_gate(name, bars, bar_index, params):
    gate = _FILTER_GATES.get(name)
    if gate is None:
        raise SwitchGateNotImplemented(
            f"filter switch {name!r} has no evaluator in the genome (PLANNED). "
            "Add it to _FILTER_GATES when its gate is wired."
        )
    return gate  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Management / risk gates (exits + guards). PLANNED until wired per batch.
# ---------------------------------------------------------------------------
#
# W4-2.5: Batch A management exits. The switch-genome management genes are
# (registry.py `_MANAGEMENT`):
#   breakeven   (be_trigger_pct)   -> move SL to entry            [engine loop: BacktestConfig.breakeven_at_r]
#   sl_reduce   ()                  -> shrink SL toward lock-in   [needs SL-mutation: PLANNED]
#   sl_lock     ()                  -> lock SL at current level   [needs SL-mutation: PLANNED]
#   atr_trail   (trail_k, trail_act) -> trailing stop behind price [needs SL-mutation: PLANNED]
#   partial_tp  (pt_trigger_pct, pt_close_pct) -> partial close    [needs partial-close: PLANNED]
#   friday_close_profit (fcp_trigger_pct) -> close whole pos if in profit on Friday  [Wired]
#   eod_close   ()                  -> close at daily session end [Wired]
#
# Parity contract (CLAUDE.md law 7): exits that mutate a protective level
# (breakeven/sl_reduce/sl_lock/atr_trail) must be engine-loop features because
# the engine owns the position's sl_price and only the engine can move it
# without a level-dependent close -- a strategy-side SL move priced from
# bars.close would diverge from the engine's fill-based P&L by slippage+spread
# and silently break kernel parity. partial_tp needs a partial-close intent
# shape the engine does not expose (volume=0 closes the whole position).
#
# The two WIRED gates below are full-close intents (volume=0): the engine
# closes whatever is open at the next bar's open, at a price that depends on
# NO level the strategy could guess. Both are therefore bit-stable between the
# slow oracle and (later) a kernel implementation -- expressible() already
# routes any active management switch to the slow oracle, so the fallback IS
# the parity guarantee until the kernel catches up.
#
# A management gate that is PLANNED raises SwitchGateNotImplemented at eval
# time (fail hard, never a phantom). A gate that is READY returns an
# OrderIntent(volume=0, direction=<opposite>) when it fires, else None.

def _not_implemented(name: str, why: str = "requires engine-loop SL/partial-close support"):
    def _raise(bars, bar_index, state, params):
        raise SwitchGateNotImplemented(
            f"mgmt switch {name!r} is PLANNED ({why}) — cannot breed. "
            f"Add it to _MGMT_GATES when the engine loop supports it."
        )
    return _raise


# SL_LOCK / SL_REDUCE / ATR_TRAIL / PARTIAL_TP are PLANNED.
# They mutate protective levels or halve volume; neither is expressible as a
# volume=0 close intent without engine-loop support, so a DNA naming them
# raises at eval and can never reach a trial. BREAKEVEN is engine-owned: the
# engine loop moves the stop (BacktestConfig.breakeven_at_r = BREAKEVEN_AT_R),
# so its gate here emits nothing.
BREAKEVEN_AT_R = 1.0  # research's be_trigger_pct default 100: ahead by the full stop distance
# The other engine-owned stop movers (BacktestConfig, engine.py _stop_rules; ARQVIO's fixed values, in R = the
# initial stop distance): sl_reduce halves the risk once 0.5 R ahead; sl_lock locks +1 R once 2 R ahead; atr_trail
# (research trail_k, trail_act) trails the close by 2 ATR(14) once 1 R ahead.
SL_REDUCE_AT_R = 0.5
SL_LOCK_AT_R = 2.0
ATR_TRAIL = (2.0, 1.0)
# Management switches the engine loop (and the kernel's step 4b) carries.
ENGINE_MGMT: tuple[str, ...] = ("breakeven", "sl_reduce", "sl_lock", "atr_trail")
# Management switches a campaign may breed: an oracle rule = the kernel AND a live-EA implementation
# (deploy/live_ea.LIVE_MGMT). Order matters: the sampler draws one coin per switch, in this order.
READY_MGMT: tuple[str, ...] = ("breakeven", "sl_reduce", "sl_lock", "atr_trail", "friday_close_profit", "eod_close")
# The SwitchStrategy close rules the kernel mirrors in its step 5 (kernel.py mg_exit), with their oracle defaults.
KERNEL_CLOSE_MGMT: tuple[str, ...] = ("friday_close_profit", "eod_close")
FCP_TRIGGER_PCT = 100.0
EOD_HOUR_UTC = 20


def close_rule_kwargs(mgmt_switches: tuple[str, ...]) -> dict[str, object]:
    """The kernel-mirrored close rules as fast_path.run_fast keyword arguments (the oracle runs them in
    SwitchStrategy with the same defaults)."""
    on = set(mgmt_switches)
    return {"friday_close_profit_pct": FCP_TRIGGER_PCT if "friday_close_profit" in on else None,
            "eod_close_hour": EOD_HOUR_UTC if "eod_close" in on else None}


def stop_rule_kwargs(mgmt_switches: tuple[str, ...]) -> dict[str, object]:
    """The engine-owned management switches as BacktestConfig / fast_path.run_fast keyword arguments."""
    on = set(mgmt_switches)
    return {"breakeven_at_r": BREAKEVEN_AT_R if "breakeven" in on else None,
            "sl_reduce_at_r": SL_REDUCE_AT_R if "sl_reduce" in on else None,
            "sl_lock_at_r": SL_LOCK_AT_R if "sl_lock" in on else None,
            "atr_trail": ATR_TRAIL if "atr_trail" in on else None}


def _engine_owned(bars, bar_index, state, params) -> None:
    return None


_MGMT_GATES: dict[str, object] = {
    **{name: _engine_owned for name in ENGINE_MGMT},
    "partial_tp": _not_implemented("partial_tp", "partial close requires a volume-reduction intent the engine does not expose"),
}


def _mgmt_gate(name, bars, bar_index, state, params):
    """Dispatch a management switch to its evaluator. PLANNED gates raise."""
    gate = _MGMT_GATES.get(name)
    if gate is None:
        raise SwitchGateNotImplemented(
            f"mgmt switch {name!r} is PLANNED (no evaluator yet) — cannot breed. "
            f"Add it to _MGMT_GATES when its gate is wired."
        )
    return gate(bars, bar_index, state, params)


# --- Wired management exits --------------------------------------------------

def _position_context(state: dict) -> dict | None:
    """The position fields a management gate needs from `state`.

    Tracked by SwitchStrategy._observe_open: entry_price (next-open, the
    price the engine will fill the SIGNAL bar's intent at), direction
    ('long'/'short'), entry_bar (the bar the signal fired on), and volume.
    Absent when the strategy is flat.
    """
    if state.get("direction") not in ("long", "short"):
        return None
    return state


def _close_intent(state: dict, bar_index: int, reason: str) -> OrderIntent:
    """A full-close intent for the current position.

    Parity note: the engine closes whatever position is open on a volume==0
    intent at the next bar's open (see engine._fill_intents), and records the
    exit reason itself (EXIT_SIGNAL for a close intent). The strategy does NOT
    need to name the reason -- `reason` here is only used to build a distinct
    intent_id label for debugging. The close price is the next bar's open,
    decided solely by the engine, so slow-oracle and kernel agree.

    `direction` on a volume==0 intent is the position's own direction; the
    engine ignores intent.direction when volume==0 and simply flattens the
    open position, so this is not a reversal signal."""

    direction = state["direction"]
    return OrderIntent(direction=direction, volume=0.0, bar_index=bar_index)


def _friday_close_profit(bars, bar_index, state, params) -> OrderIntent | None:
    """Close on Friday when floating P&L >= fcp_trigger_pct% of the risk stake.

    Risk stake is the price distance the strategy is risking on the trade:
    ``risk_distance = sl_atr * atr`` (the stop distance in price units). The
    floating move in price units is ``(close - entry) * sign``. The gate fires
    when ``move >= (fcp_trigger_pct / 100) * risk_distance`` -- i.e. the
    unrealised gain has reached the configured fraction of the stop distance.

    Parity note: this is a full-close intent (volume=0). The engine closes at
    the next bar's open, at a price determined solely by the next open --
    nothing in this gate is level-dependent, so slow-oracle and a future
    kernel agree. fcp_trigger_pct is read from params (default 100: close
    only when in profit by the full stop distance).
    """
    ctx = _position_context(state)
    if ctx is None:
        return None
    fcp_trigger_pct = float(params.get("fcp_trigger_pct", FCP_TRIGGER_PCT))
    if fcp_trigger_pct <= 0:
        return None
    # Friday in broker server time. Bars arrive as UTC after convert_times;
    # the server is EET/EEST, so a Friday in UTC is also a Friday server-side
    # (EET+2 / EEST+3 never crosses the UTC day boundary at the date level).
    bar_time = int(bars.time[bar_index])
    weekday = (bar_time // 86400 + 3) % 7  # Mon=0..Sun=6; epoch(Thu=3)+3 -> Fri=4
    if weekday != 4:
        return None
    at = float(bars.close[bar_index])
    entry = float(state["entry_price"])
    sign = 1.0 if ctx["direction"] == "long" else -1.0
    move = (at - entry) * sign  # price units, positive = in profit
    if move <= 0.0:
        return None
    sl_atr = float(params.get("sl_atr", 0.0))
    atr_period = int(params.get("atr_period", DEFAULT_ATR_PERIOD))
    atr = atr_at(bars.high, bars.low, bars.close, bar_index, period=atr_period)
    if atr is None or atr <= 0.0:
        # No ATR stop to reference: compare against a single-point floor so
        # the gate is still usable when the DNA carries no stop genes.
        risk_distance = max(move, 1e-9)
    else:
        risk_distance = sl_atr * atr
    if move >= (fcp_trigger_pct / 100.0) * risk_distance:
        return _close_intent(state, bar_index, "friday_close_profit")
    return None


def _eod_close(bars, bar_index, state, params) -> OrderIntent | None:
    """Close at the end of the broker's trading day.

    Broker daily roll is EET 22:00 (engine.py: swap charged at day roll).
    In UTC that is 20:00 (winter) / 19:00 (summer). A position held past the
    day boundary would ride the weekend gap, so this closes on the last bar
    whose UTC hour is >= the EOD threshold. The threshold is read from params
    as `eod_hour_utc` (default 20, the winter EET-close); the slow oracle and
    a future kernel read the same param, so parity holds through fallback.

    This is a full-close intent (volume=0): the engine closes at next bar's
    open, price-independent. Distinct from the engine's final-bar EXIT_EOD
    (data exhaustion); this fires every session end.
    """
    ctx = _position_context(state)
    if ctx is None:
        return None
    eod_hour = int(params.get("eod_hour_utc", EOD_HOUR_UTC))
    bar_time = int(bars.time[bar_index])
    hour = (bar_time // 3600) % 24
    if hour >= eod_hour:
        return _close_intent(state, bar_index, "eod_close")
    return None


# The wired exits, registered by name so _mgmt_exit can dispatch.
_MGMT_GATES["friday_close_profit"] = _friday_close_profit
_MGMT_GATES["eod_close"] = _eod_close


# ---------------------------------------------------------------------------
# SwitchStrategy
# ---------------------------------------------------------------------------
@dataclass
class SwitchConfig:
    """V4 switch genome config for the slow oracle."""
    bias_switches: tuple[str, ...] = ()
    filter_switches: tuple[str, ...] = ()
    mgmt_switches: tuple[str, ...] = ()
    risk_switches: tuple[str, ...] = ()
    # parameter bag (DNA scalar genes the gates read; research §4.1/APPENDIX-D)
    params: dict = field(default_factory=dict)
    inner_strategy_id: str = ""


class SwitchStrategy:
    """Wraps a family (signal) strategy with V4 bias/filter/management/risk gates.

    The inner strategy is the entry signal; the gates decide whether that entry
    is allowed and whether an open position is exited early.

    `params` carries the DNA numeric genes the gates need (sma_slow_period,
    rsi_period, consec_count, be_trigger_pct, ...). Only READY gates read
    params; a PLANNED gate raises SwitchGateNotImplemented at eval time so it
    can never silently affect a trial.
    """

    def __init__(self, inner: object, config: SwitchConfig) -> None:
        self.inner = inner
        self.config = config
        self.strategy_id = (
            f"{getattr(inner, 'strategy_id', 'switch')}"
            f"+bias={','.join(config.bias_switches) or '-'}"
            f"+filt={','.join(config.filter_switches) or '-'}"
            f"+mgmt={','.join(config.mgmt_switches) or '-'}"
            f"+risk={','.join(config.risk_switches) or '-'}"
        )
        # session / trend EMA are separate gates (already in DNA v3).
        self.session = config.params.get("session", "") or ""
        self.trend_ema = int(config.params.get("trend_ema", 0) or 0)
        self._trend = TrendEma(self.trend_ema) if self.trend_ema else None
        self._seen = -1

    def _advance_trend(self, bar_index: int, bars: "BarData") -> None:
        if self._trend is None:
            return
        for j in range(self._seen + 1, bar_index + 1):
            self._trend.update(float(bars.close[j]))
        self._seen = max(self._seen, bar_index)

    def _session_ok(self, bar_index: int, bars: "BarData", direction: int) -> bool:
        if not self.session:
            return True
        hour = int(int(bars.time[bar_index]) // 3600) % 24
        return session_open(hour, self.session)

    def _bias_ok(self, bar_index: int, bars: "BarData", direction: int) -> bool:
        """All ON bias switches must agree with the intended direction."""
        if not self.config.bias_switches:
            return True
        for name in self.config.bias_switches:
            long_ok, short_ok = _bias_gate(name, bars, bar_index, self.config.params)
            if direction > 0 and not long_ok:
                return False
            if direction < 0 and not short_ok:
                return False
        return True

    def _filters_ok(self, bar_index: int, bars: "BarData") -> bool:
        """All ON filter switches must pass (bar is tradeable)."""
        for name in self.config.filter_switches:
            if not _filter_gate(name, bars, bar_index, self.config.params):
                return False
        return True

    def _risk_ok(self, bar_index: int, bars: "BarData", direction: int) -> bool:
        """ON risk switches may veto an entry."""
        # TODO 2.9: daily_dd, symbol_lock, news_filter, no_open_news_day
        return True

    def _mgmt_exit(self, bar_index: int, bars: "BarData", state: dict) -> OrderIntent | None:
        """Return a close intent if a management/risk gate closes the position, else None.

        Iterates the ON management switches in order; the first to fire wins
        (an earlier gate that closes the position makes later gates moot).
        Each gate reads `state` (kept stocked with entry_price/entry_bar by
        _observe_open) and the visible bar slice up to bar_index. PLANNED
        switches raise SwitchGateNotImplemented so they never affect a trial.
        """
        if not self.config.mgmt_switches:
            return None
        for name in self.config.mgmt_switches:
            intent = _mgmt_gate(name, bars, bar_index, state, self.config.params)
            if intent is not None:
                return intent
        return None

    def _observe_open(self, intent: OrderIntent, bar_index: int, bars: "BarData", state: dict) -> None:
        """Track the position context a management gate needs.

        Records the signal-bar close as the entry reference price. The engine
        fills the opening intent at NEXT bar's open (slippage-adjusted), but
        the strategy cannot see that open without look-ahead; the signal-bar
        close is the honest, causal reference both this gate and the SL/TP
        distance computation use. entry_bar is the bar the signal fired on.
        """
        if intent.volume > 0:
            state["direction"] = intent.direction  # "long"/"short"
            state["entry_price"] = float(bars.close[bar_index])
            state["entry_bar"] = bar_index

    def on_bar(self, bar_index: int, bars: "BarData", state: dict) -> OrderIntent | None:
        self._advance_trend(bar_index, bars)

        # Management exits run FIRST, on every bar a position is open,
        # independent of whether the inner strategy emits a signal. An exit
        # intent short-circuits: the engine closes the position, so the inner
        # signal (if any) would only re-enter on a later bar.
        if state.get("direction") in ("long", "short"):
            exit_intent = self._mgmt_exit(bar_index, bars, state)
            if exit_intent is not None:
                state.pop("direction", None)  # closed: flat
                return exit_intent

        intent = self.inner.on_bar(bar_index, bars, state)
        if intent is None or intent.volume == 0:
            return intent

        # CLOSING / management intent passes through; risk may still block a re-entry.
        if intent.direction in ("close", "exit"):
            return intent

        # OPENING intent: gate it.
        direction = 1 if intent.direction == "long" else -1
        if not self._session_ok(bar_index, bars, direction):
            state.pop("direction", None)  # dropped opening: flat
            return None
        if not self._bias_ok(bar_index, bars, direction):
            state.pop("direction", None)
            return None
        if not self._filters_ok(bar_index, bars):
            state.pop("direction", None)
            return None
        if not self._risk_ok(bar_index, bars, direction):
            state.pop("direction", None)
            return None
        # Gate passed: observe the new position for future management exits.
        self._observe_open(intent, bar_index, bars, state)
        return intent


__all__ = [
    "SwitchConfig",
    "SwitchStrategy",
    "SwitchGateNotImplemented",
]
