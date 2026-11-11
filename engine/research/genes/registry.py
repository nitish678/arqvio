"""Gene registry: the one gene vocabulary both engines read (worker4 W4-1).

ENGINE V1 (research mode) breeds only genes whose origin is "research"; ENGINE V2 (ARQVIO mode) breeds every gene.
Neither engine draws a gene that is not READY.

Origins
    research  the gene list of the research pack (CLAUDE_BUILD_INSTRUCTIONS.md PART 4.1, verbatim seed JSON): 63
              on/off switches (13 bias, 21 signals, 14 filters, 7 trade management, 8 risk guards), the execution
              choice, the session/stop core and the per-class spread/slippage limits. Concepts only: every name
              here is ours and no research code or byte format is used (worker4/02_PLAN.md F6, F7).
    arqvio    genes ARQVIO has that the research genome does not (the MA cross signal, the named session gate,
              risk per trade).

Status -- what is true of the gene TODAY, never what it is meant to become
    READY     slow oracle + numba kernel bit-identical + MQL5 EA, with parity tests; the GA may vary it.
    FIXED     modelled in the oracle, the kernel and the EA, but set per campaign/cell, never bred.
    EA_ONLY   a live-EA guardrail the backtest does not model: a known backtest-vs-live gap until it is built.
    PLANNED   not built. Research gives its name and parameter names only; its ranges are chosen when it is
              built (W4-2), so none are invented here.

A READY gene names what implements it: ``family`` (a DNA signal family, its genes and ranges are
gene_pool.FAMILY_GENES -- one source, not copied here) or ``dna_field`` (a DNA field). A FIXED or EA_ONLY gene
names the cell setting or EA input in ``backing``.

REGISTRY TRUTH (2026-10-28): a gene is READY only when the code proves it. tests/research/genes/test_registry.py
fails if a DNA signal family has no READY gene, if a READY gene names a family or DNA field that does not exist,
or if the research switch counts drift from PART 4.1.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Any

from engine.research.search.dna import DNA, SIGNAL_VALUES
from engine.research.search.gene_pool import FAMILY_GENES

REGISTRY_VERSION = 1

RESEARCH = "research"
ARQVIO = "arqvio"
ORIGINS: tuple[str, ...] = (RESEARCH, ARQVIO)

GROUPS: tuple[str, ...] = ("core", "bias", "signal", "filter", "exec", "management", "risk", "limits", "meta")
ENGINES: tuple[str, ...] = ("v1", "v2")

# DNA fields no crossover or mutation may touch (the GA's pinned set reads this; ga/operators.py).
LOCKED_DNA_FIELDS = frozenset({"mgmt", "exec_mode", "risk", "uses_future_bar", "schema_version"})


class Status(str, Enum):
    READY = "READY"
    FIXED = "FIXED"
    EA_ONLY = "EA_ONLY"
    PLANNED = "PLANNED"


@dataclass(frozen=True)
class Gene:
    name: str                       # "<group>.<gene>", unique
    group: str
    kind: str                       # "switch" (on/off + params) | "choice" (one of a group) | "number" | "tag"
    origin: str
    status: Status
    params: tuple[str, ...] = ()    # the gene's parameter names
    family: str = ""                # READY signal: the DNA signal family that implements it
    dna_field: str = ""             # READY/FIXED: the DNA field that carries it
    backing: str = ""               # FIXED / EA_ONLY: the cell setting or EA input that enforces it
    research_hidden: bool = False   # research never shows it in its inspector; ARQVIO shows every gene
    note: str = ""

    def public(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name, "group": self.group, "kind": self.kind, "origin": self.origin,
            "status": self.status.value, "params": list(self.params),
        }
        for key in ("family", "dna_field", "backing", "note"):
            if getattr(self, key):
                out[key] = getattr(self, key)
        if self.research_hidden:
            out["research_hidden"] = True
        if self.family:
            out["ranges"] = [spec.public() for spec in FAMILY_GENES[self.family]]
        return out


R, P, F, E = Status.READY, Status.PLANNED, Status.FIXED, Status.EA_ONLY


def _switches(group: str, rows: tuple[tuple[str, tuple[str, ...]], ...], **ready: dict[str, Any]) -> list[Gene]:
    """Research on/off genes of one group; ``ready`` maps a gene name to the fields that make it READY."""
    out = []
    for name, params in rows:
        extra = dict(ready.get(name, {}))
        out.append(Gene(f"{group}.{name}", group, "switch", RESEARCH, extra.pop("status", P), params, **extra))
    return out


_BIAS = (
    ("sma", ("sma_fast_period", "sma_slow_period")), ("rsi", ("rsi_period",)), ("trailing", ("trailing_period",)),
    ("momentum", ("momentum_period",)), ("chandelier", ("atr_ma_period", "chand_mult")), ("donchian_mid", ()),
    ("market_struct", ()), ("daily_mid", ()), ("vwap", ("bias_vwap_hour",)), ("adx", ("adx_period",)),
    ("ema", ("ema_period",)), ("psar", ("psar_step", "psar_max")), ("htf", ("htf_sma_period",)),
)
_SIGNALS = (
    ("macd", ()), ("cci", ("cci_period", "cci_limit")), ("mom_break", ("mom_break_atr",)),
    ("engulfing", ("engulfing_type",)), ("inside_break", ()), ("wick_rejection", ("wick_ratio",)),
    ("fib", ("fib_level", "fib_lookback")), ("breakout", ("breakout_type", "breakout_lookback")),
    ("liqsweep", ("liqsweep_lookback", "liqsweep_atr")), ("atr_run", ("atrrun_bars", "atrrun_atr")),
    ("fade", ("fade_bars", "fade_atr", "fade_window")), ("vwap", ("vwap_anchor_hour", "vwap_dev_atr")),
    ("orb", ("orb_session_hour", "orb_range_bars")), ("squeeze", ("sqz_bars", "sqz_ratio")),
    ("three_soldiers", ("soldiers_min_body_atr",)), ("rsi", ("rsi_period",)), ("bb", ("bb_period", "bb_std")),
    ("pin_bar", ("pin_bar_body_pct",)), ("stoch", ()), ("williams", ()), ("kalman", ()),
)
_FILTERS = (
    ("sma", ()), ("rsi", ()), ("bb", ()), ("cci", ()), ("receding", ()), ("doji", ()), ("no_open_friday", ()),
    ("volatility", ()), ("consec", ("consec_count",)), ("adr_exhaust", ("adr_period", "adr_exhaust_pct")),
    ("adx", ("adx_threshold",)), ("keltner", ("keltner_period", "keltner_mult")),
    ("regime", ("regime_mode", "chop_period", "chop_threshold", "trend_threshold")),
    ("event", ("event_atr_mult", "gap_atr_mult")),
)
_MANAGEMENT = (
    ("breakeven", ("be_trigger_pct",)), ("sl_reduce", ()), ("sl_lock", ()), ("atr_trail", ("trail_k", "trail_act")),
    ("partial_tp", ("pt_trigger_pct", "pt_close_pct")), ("friday_close_profit", ("fcp_trigger_pct",)),
    ("eod_close", ()),
)
_RISK = (
    ("daily_dd", ("daily_dd_limit",)), ("friday_close", ()), ("max_agg_risk", ("max_agg_risk_pct",)),
    ("news_filter", ("news_mins_before", "news_all_high_impact")), ("no_open_news_day", ()),
    ("profit_target", ("profit_target_pct",)), ("slippage", ()), ("symbol_lock", ()),
)
# Research's execution choices. PART 4.1 lists six; the owner's own research-product vault (docs/ARQVIO_V4.md
# Appendix B: market2 99, limit 83, limit2 71, stop 28) shows a seventh, limit2, so it is listed.
_EXEC = (
    ("market", ()), ("market2", ("market2_window_bars", "market2_pullback_atr")),
    ("limit", ("limit_offset_atr", "limit_expiry_bars")), ("limit2", ("limit2_offset_atr",)),
    ("stop", ("stop_offset_atr", "stop_expiry_bars", "stop_strict_offset")), ("scalein", (
        "scalein_max_adds", "scalein_step_atr", "scalein_expiry_bars")), ("swing", ("swing_expiry_bars",)),
)
_HIDDEN = frozenset({"management.breakeven", "bias.ema", "filter.adx", "management.sl_lock", "filter.bb",
                     "exec.swing", "exec.scalein"})
_META = ("tribe_origin", "pg_source", "family_penalty", "family_reward", "locked_grade", "oos_validated",
         "smart_generated", "cached_dd", "cached_return", "survived_full_tribe")

_GENES: list[Gene] = [
    # core: the trading window and the stop/target distances
    Gene("core.session_hours", "core", "number", RESEARCH, P, ("start_hour", "end_hour"),
         note="DNA window_start/end_hour_utc exist but no engine reads them"),
    Gene("core.sl_mult", "core", "number", RESEARCH, R, dna_field="sl_atr", note="stop = sl_atr x ATR"),
    Gene("core.tp_mult", "core", "number", RESEARCH, R, dna_field="tp_atr", note="target = tp_atr x ATR"),
    Gene("core.risk_pct", "core", "number", ARQVIO, R, dna_field="risk_pct"),
    Gene("core.session_gate", "core", "choice", ARQVIO, R, dna_field="session",
         note="named UTC windows ASIA / LONDON / NEWYORK / LONDON_NY (backtest/gates.py)"),
    *_switches("bias", _BIAS, ema={"status": R, "dna_field": "trend_ema",
                                   "note": "long only above the EMA, short only below (backtest/gates.py)"},
               sma={"status": R, "dna_field": "bias_switches",
                    "note": "fixed SMA(100) of the close: long above, short below (backtest/switch_gates.py)"},
               momentum={"status": R, "dna_field": "bias_switches",
                         "note": "fixed 20 bars: long when the close is above the close 20 bars ago (switch_gates.py)"},
               donchian_mid={"status": R, "dna_field": "bias_switches",
                             "note": "long above the midpoint of the 20-bar high/low, short below (switch_gates.py)"},
               rsi={"status": R, "dna_field": "bias_switches", "note": "long when RSI(14) > 50, short < 50"},
               adx={"status": R, "dna_field": "bias_switches", "note": "long when +DI(14) > -DI(14), short when below"},
               chandelier={"status": R, "dna_field": "bias_switches",
                           "note": "long above HH(22) - 3 ATR(22), short below LL(22) + 3 ATR(22)"},
               daily_mid={"status": R, "dna_field": "bias_switches",
                          "note": "long above the previous UTC day's high/low midpoint, short below"},
               htf={"status": R, "dna_field": "bias_switches", "note": "long above SMA(200) of the close, short below"},
               trailing={"status": R, "dna_field": "bias_switches",
                         "note": "the side of a 3 x ATR(14) trailing stop that flips when a close crosses it"},
               market_struct={"status": R, "dna_field": "bias_switches",
                              "note": "long when the last 10 bars made a higher high and a higher low than the 10 "
                                      "before; short on a lower high and lower low"},
               vwap={"status": R, "dna_field": "bias_switches",
                     "note": "long above the day's tick-volume VWAP (from 00:00 of the bar day), short below; "
                             "research's bias_vwap_hour anchor is fixed at 0"},
               psar={"status": R, "dna_field": "bias_switches",
                     "note": "the side of Parabolic SAR (step 0.02, max 0.2; Wilder)"}),
    *_switches("signal", _SIGNALS,
               macd={"status": R, "family": "MACD_CROSS"},
               rsi={"status": R, "family": "RSI_REVERSION"},
               stoch={"status": R, "family": "STOCH_CROSS"},
               breakout={"status": R, "family": "DONCHIAN_BREAKOUT",
                         "note": "Donchian channel break; research's breakout_type is not modelled"},
               cci={"status": R, "dna_field": "signal_switches", "note": "SIGNALS family: CCI(14) crosses up through -100 / down through +100 (signal_switches.py)"},
               mom_break={"status": R, "dna_field": "signal_switches", "note": "SIGNALS family: a bar's move beyond 0.5 ATR(14) (signal_switches.py)"},
               engulfing={"status": R, "dna_field": "signal_switches", "note": "SIGNALS family: a body engulfing the previous opposite body (signal_switches.py)"},
               inside_break={"status": R, "dna_field": "signal_switches", "note": "SIGNALS family: the close breaks an inside bar (signal_switches.py)"},
               wick_rejection={"status": R, "dna_field": "signal_switches", "note": "SIGNALS family: a wick >= 2x the body, close in the far third (signal_switches.py)"},
               liqsweep={"status": R, "dna_field": "signal_switches", "note": "SIGNALS family: takes the prior 20-bar low/high and reclaims it (signal_switches.py)"},
               fade={"status": R, "dna_field": "signal_switches", "note": "SIGNALS family: close beyond SMA(10) +- 2 ATR(14), faded (signal_switches.py)"},
               squeeze={"status": R, "dna_field": "signal_switches", "note": "SIGNALS family: Bollinger inside Keltner, then a band break (signal_switches.py)"},
               three_soldiers={"status": R, "dna_field": "signal_switches", "note": "SIGNALS family: three rising bullish bars, bodies >= 0.4 ATR (signal_switches.py)"},
               bb={"status": R, "dna_field": "signal_switches", "note": "SIGNALS family: the close re-enters Bollinger(20, 2) (signal_switches.py)"},
               pin_bar={"status": R, "dna_field": "signal_switches", "note": "SIGNALS family: body <= 33 % of the range, a 2/3 wick (signal_switches.py)"},
               williams={"status": R, "dna_field": "signal_switches", "note": "SIGNALS family: %R(14) crosses up through -80 / down through -20 (signal_switches.py)"},
               fib={"status": R, "dna_field": "signal_switches", "note": "SIGNALS family: a dip to the 61.8 % retracement of the 50-bar swing, closed back beyond it (signal_switches.py)"},
               atr_run={"status": R, "dna_field": "signal_switches", "note": "SIGNALS family: three closes in a row moving more than 1.5 ATR(14) (signal_switches.py)"},
               vwap={"status": R, "dna_field": "signal_switches", "note": "SIGNALS family: the close crosses the day's tick-volume VWAP (signal_switches.py)"},
               orb={"status": R, "dna_field": "signal_switches", "note": "SIGNALS family: the first close beyond the first 4 bars' range of the day (signal_switches.py)"},
               kalman={"status": R, "dna_field": "signal_switches", "note": "SIGNALS family: the slope of a Kalman level filter of the close turns (signal_switches.py)"}),
    Gene("signal.ma_cross", "signal", "switch", ARQVIO, R, family="MA_CROSS"),
    *_switches("filter", _FILTERS,
               no_open_friday={"status": R, "dna_field": "filter_switches",
                               "note": "no opening on a Friday (UTC) (backtest/switch_gates.py)"},
               volatility={"status": R, "dna_field": "filter_switches",
                           "note": "ATR(14)/ATR(100) in [0.5, 2.0] (backtest/switch_gates.py)"},
               rsi={"status": R, "dna_field": "filter_switches",
                    "note": "RSI(14) in [30, 70]: no opening into an extreme (backtest/switch_gates.py)"},
               bb={"status": R, "dna_field": "filter_switches", "note": "close inside SMA(20) +- 2 sd"},
               cci={"status": R, "dna_field": "filter_switches", "note": "CCI(20) in [-100, 100]"},
               consec={"status": R, "dna_field": "filter_switches",
                       "note": "not the 4th close in a row in the trade's direction"},
               adr_exhaust={"status": R, "dna_field": "filter_switches",
                            "note": "today's range < 80 % of the 14-day mean range (UTC days)"},
               adx={"status": R, "dna_field": "filter_switches", "note": "ADX(14) >= 20"},
               keltner={"status": R, "dna_field": "filter_switches", "note": "close inside SMA(20) +- 1.5 ATR(20)"},
               sma={"status": R, "dna_field": "filter_switches",
                    "note": "SMA(10) and SMA(30) at least 0.25 ATR(14) apart (no opening in a tangle)"},
               receding={"status": R, "dna_field": "filter_switches",
                         "note": "volatility not receding: mean TR of the last 5 bars >= the 5 before"},
               doji={"status": R, "dna_field": "filter_switches", "note": "no opening on a doji (body <= 10 % of range)"},
               regime={"status": R, "dna_field": "filter_switches",
                       "note": "Choppiness(14) < 61.8 (trending only); research's regime_mode is not modelled"},
               event={"status": R, "dna_field": "filter_switches",
                      "note": "no price shock: the bar's range <= 3 ATR and its gap <= 1 ATR of the 14 bars before "
                              "(research event_atr_mult / gap_atr_mult); not a news calendar"}),
    Gene("exec.market", "exec", "choice", RESEARCH, R, dna_field="exec_mode",
         note="a market order at the next bar's open (DNA exec_mode MARKET2 -- ARQVIO's label, not research's "
              "market2)"),
    # PLANNED: pending orders (limit / stop / scale-in / swing expiry) need a pending-order fill model; engine.py
    # (frozen this stage, CLAUDE.md law 3) fills only market intents at the next bar's open.
    *[Gene(f"exec.{name}", "exec", "choice", RESEARCH, P, params,
           note="PLANNED: needs pending-order fills; engine.py fills market intents only and is frozen this stage")
      for name, params in _EXEC[1:]],
    *_switches("management", _MANAGEMENT,
               breakeven={"status": R, "dna_field": "mgmt_switches",
                          "note": "stop to the entry fill once a bar closes 1x the stop distance ahead (engine loop: "
                                  "BacktestConfig.breakeven_at_r; kernel step 4b; live EA InpBreakevenAtR)"},
               sl_reduce={"status": R, "dna_field": "mgmt_switches",
                          "note": "stop to entry - 0.5 R once a close is 0.5 R ahead (R = the initial stop distance; "
                                  "BacktestConfig.sl_reduce_at_r; kernel 4b; live EA InpSlReduceAtR)"},
               sl_lock={"status": R, "dna_field": "mgmt_switches",
                        "note": "stop to entry + 1 R once a close is 2 R ahead (BacktestConfig.sl_lock_at_r; kernel 4b; "
                                "live EA InpSlLockAtR)"},
               atr_trail={"status": R, "dna_field": "mgmt_switches",
                          "note": "once a close is 1 R ahead the stop trails the close by 2 ATR(14), every bar "
                                  "(research trail_act / trail_k fixed; BacktestConfig.atr_trail; live EA InpAtrTrail*)"},
               friday_close_profit={"status": R, "dna_field": "mgmt_switches",
                                    "note": "close on a Friday (UTC) bar once the close is 100 % of the stop distance "
                                            "past the signal-bar close (SwitchStrategy; kernel step 5; live EA "
                                            "InpFridayCloseProfitPct)"},
               eod_close={"status": R, "dna_field": "mgmt_switches",
                          "note": "close on a bar whose bar-time hour is >= 20 (SwitchStrategy; kernel step 5; live EA "
                                  "InpEodCloseHour)"},
               partial_tp={"note": "PLANNED: the engine and kernel hold one trade per position and a close intent "
                                   "closes it whole; a partial close needs a split-trade model (engine.py is frozen)"}),
    *_switches("risk", _RISK,
               daily_dd={"status": E, "backing": "EA InpDailyLossPct",
                         "note": "the backtest's daily_loss_limit_pct blocks entries on REALISED loss vs the initial "
                                 "equity; the EA flattens on EQUITY vs the day-start balance: not the same rule, so "
                                 "not bred"},
               friday_close={"status": E, "backing": "EA InpFridayCloseHour",
                             "note": "the backtest's friday_close_hour reads the bar-label hour; the EA's is an "
                                     "owner guardrail on server time (Sat/Sun included), not a DNA gene"},
               news_filter={"status": E, "backing": "EA InpNewsBlockMinutes",
                            "note": "no economic-calendar data in the engine"},
               profit_target={"status": E, "backing": "EA InpProfitTargetPct",
                              "note": "account-wide, from the balance at start, stops every bot for good: an owner "
                                      "guardrail, not modelled per strategy"},
               slippage={"status": E, "backing": "EA InpSlippagePoints",
                         "note": "the EA's order deviation; the backtest's slippage model is limits.max_slip (FIXED)"},
               max_agg_risk={"note": "PLANNED: across every bot; one strategy's backtest holds one position, so it "
                                     "cannot be modelled honestly there (the EA caps positions, InpMaxAccountPositions)"},
               symbol_lock={"note": "PLANNED: one trade per symbol across bots; a no-op in a one-strategy backtest"},
               no_open_news_day={"note": "PLANNED: no economic-calendar data in the engine"}),
    Gene("limits.max_spread", "limits", "number", RESEARCH, F, ("major", "cross", "metal", "index", "crypto"),
         backing="cell max_spread_points + EA InpMaxSpreadPoints",
         note="one cap per cell (p95 of recent ticks); W4-4 replaces it with the spread-regime gate"),
    Gene("limits.max_slip", "limits", "number", RESEARCH, F, ("major", "cross", "metal", "index", "crypto"),
         backing="cell slippage model (kernel slippage_* inputs)"),
    Gene("limits.broker_stops_level", "limits", "number", RESEARCH, P, ("broker_stops_level_pts",)),
    Gene("limits.friday_close_jitter", "limits", "number", RESEARCH, P, ("friday_close_jitter_min",)),
    *[Gene(f"meta.{name}", "meta", "tag", RESEARCH, P, note="GA bookkeeping, added with engine V1 (W4-3)")
      for name in _META],
]
GENES: tuple[Gene, ...] = tuple(
    dataclasses.replace(g, research_hidden=True) if g.name in _HIDDEN else g for g in _GENES
)
BY_NAME: dict[str, Gene] = {g.name: g for g in GENES}


def genes(*, engine: str | None = None, group: str | None = None, status: Status | None = None) -> tuple[Gene, ...]:
    """The registry filtered; ``engine`` "v1" keeps research genes only, "v2" keeps all."""
    if engine is not None and engine not in ENGINES:
        raise ValueError(f"unknown engine {engine!r}; engines are {list(ENGINES)}")
    return tuple(
        g for g in GENES
        if (engine != "v1" or g.origin == RESEARCH)
        and (group is None or g.group == group)
        and (status is None or g.status == status)
    )


def breedable_families(engine: str) -> tuple[str, ...]:
    """The DNA signal families an engine may draw, in SIGNAL_VALUES order."""
    allowed = {g.family for g in genes(engine=engine, group="signal", status=Status.READY)}
    return tuple(f for f in SIGNAL_VALUES if f in allowed)


def counts() -> dict[str, Any]:
    by_status = Counter(g.status.value for g in GENES)
    return {
        "total": len(GENES),
        "by_status": {s.value: by_status.get(s.value, 0) for s in Status},
        "by_group": dict(Counter(g.group for g in GENES)),
        "engines": {
            e: {"genes": len(genes(engine=e)), "ready": len(genes(engine=e, status=Status.READY)),
                "families": list(breedable_families(e))}
            for e in ENGINES
        },
    }


def registry_hash() -> str:
    payload = json.dumps([g.public() for g in GENES], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def public() -> dict[str, Any]:
    return {"version": REGISTRY_VERSION, "hash": registry_hash(), "counts": counts(),
            "genes": [g.public() for g in GENES]}


def problems() -> list[str]:
    """Every way the registry disagrees with the code; empty when it is true."""
    out: list[str] = []
    dna_fields = set(DNA.__dataclass_fields__)
    for name, n in Counter(g.name for g in _GENES).items():
        if n > 1:
            out.append(f"{name}: listed {n} times")
    for g in GENES:
        if g.group not in GROUPS or g.origin not in ORIGINS:
            out.append(f"{g.name}: unknown group {g.group!r} or origin {g.origin!r}")
        if g.status is Status.READY and not (g.family or g.dna_field):
            out.append(f"{g.name}: READY but names no family or DNA field")
        if g.family and g.family not in SIGNAL_VALUES:
            out.append(f"{g.name}: family {g.family!r} is not a DNA signal family")
        if g.dna_field and g.dna_field not in dna_fields:
            out.append(f"{g.name}: DNA has no field {g.dna_field!r}")
        if g.status in (Status.FIXED, Status.EA_ONLY) and not g.backing:
            out.append(f"{g.name}: {g.status.value} but names no backing")
    implemented = {g.family for g in GENES if g.status is Status.READY and g.family}
    out += [f"DNA signal family {f} has no READY registry gene" for f in SIGNAL_VALUES if f not in implemented]
    return out


__all__ = [
    "ARQVIO", "BY_NAME", "ENGINES", "GENES", "GROUPS", "Gene", "LOCKED_DNA_FIELDS", "REGISTRY_VERSION", "RESEARCH",
    "Status", "breedable_families", "counts", "genes", "problems", "public", "registry_hash",
]
