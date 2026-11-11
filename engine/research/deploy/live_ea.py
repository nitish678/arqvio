"""DNA -> a LIVE MetaTrader 5 Expert Advisor, chart-locked, with owner-set guardrails.

The tester EA (engine/research/mql5/codegen.py) replays a pinned cell bar by bar for the parity battery and
refuses to run outside the Strategy Tester. This module renders ``templates/ea_live.mq5.j2`` instead: the SAME
family partial (the signal logic the battery checks against the oracle), fed by the live chart's closed bars,
with execution and guardrails of its own. The strategy parameters come from ``real_backtest.dna_to_strategy``,
exactly as the tester EA's do, so the two can never read a DNA differently.

The live template lives outside ``engine/research/mql5/templates`` on purpose: the tester EA stamps a digest of
every template in that directory, and a live template there would change every parity EA's source.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

import jinja2

from engine.research.backtest.gates import SESSIONS
from engine.research.backtest.strategies.switch import (
    ENGINE_MGMT,
    KERNEL_CLOSE_MGMT,
    close_rule_kwargs,
    stop_rule_kwargs,
)
from engine.research.backtest.switch_gates import CHOP_RATIO, READY_BIAS, READY_FILTERS
from engine.research.mql5.codegen import CodegenError, _validated, bits
from engine.research.mql5.magic import load_table
from engine.core.errors import DataError
from engine.research.search.dna import DNA, SWITCH_FAMILY, validate_gene_space
from engine.research.search.real_backtest import dna_to_strategy

TEMPLATE_VERSION = "arqvio-live-v1"
# Management switches the live EA carries itself (the rest are refused until they have an EA implementation).
LIVE_MGMT: tuple[str, ...] = ENGINE_MGMT + KERNEL_CLOSE_MGMT
DEPLOY_RANGE = "arqvio-deploy"
# Incubation ladder (Algory-style): a new bot trades a quarter of its lot, then half, then the full lot.
INCUBATION_STAGES: tuple[float, ...] = (0.25, 0.5, 1.0)
PERIOD_ENUMS: dict[str, str] = {"M1": "PERIOD_M1", "M5": "PERIOD_M5", "M15": "PERIOD_M15", "M30": "PERIOD_M30",
                                "H1": "PERIOD_H1", "H2": "PERIOD_H2", "H4": "PERIOD_H4", "D1": "PERIOD_D1"}
PERIOD_MINUTES: dict[str, int] = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H2": 120, "H4": 240, "D1": 1440}


@dataclass(frozen=True)
class Guardrails:
    """The EA's owner-set limits. Each becomes an ``input`` the owner can still change in MT5's EA dialog."""

    lots: float = 0.01
    lot_scale: float = INCUBATION_STAGES[0]
    max_spread_points: int = 0
    daily_loss_pct: float = 3.0
    max_account_positions: int = 5
    friday_close_hour: int = 20
    profit_target_pct: float = 0.0
    news_block_minutes: int = 30
    slippage_points: int = 20
    history_bars: int = 3000
    # v4 spread-regime gate (costs/spread_regime.py): 0 = off; the deploy sets 3.0 and the cell's 20-day bar count.
    spread_regime_multiple: float = 0.0
    spread_regime_bars: int = 0

    def problems(self) -> list[str]:
        out = []
        if not (math.isfinite(self.lots) and self.lots > 0):
            out.append("lots must be > 0")
        if self.lot_scale not in INCUBATION_STAGES:
            out.append(f"lot_scale must be an incubation stage {INCUBATION_STAGES}")
        if self.max_spread_points < 0:
            out.append("max_spread_points must be >= 0 (0 = off)")
        if not (math.isfinite(self.spread_regime_multiple) and self.spread_regime_multiple >= 0):
            out.append("spread_regime_multiple must be >= 0 (0 = off)")
        if self.spread_regime_bars < 0 or (self.spread_regime_multiple > 0 and self.spread_regime_bars < 2):
            out.append("spread_regime_bars must be >= 2 when the regime gate is on")
        if not (math.isfinite(self.daily_loss_pct) and 0 <= self.daily_loss_pct <= 100):
            out.append("daily_loss_pct must be in [0, 100] (0 = off)")
        if self.max_account_positions < 1:
            out.append("max_account_positions must be >= 1")
        if not -1 <= self.friday_close_hour <= 23:
            out.append("friday_close_hour must be -1 (off) or a server hour 0..23")
        if not (math.isfinite(self.profit_target_pct) and self.profit_target_pct >= 0):
            out.append("profit_target_pct must be >= 0 (0 = off)")
        if not 0 <= self.news_block_minutes <= 720:
            out.append("news_block_minutes must be in [0, 720] (0 = off)")
        if not 0 <= self.slippage_points <= 10_000:
            out.append("slippage_points must be in [0, 10000]")
        if not 100 <= self.history_bars <= 100_000:
            out.append("history_bars must be in [100, 100000]")
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Guardrails:
        known = {f.name for f in fields(cls)}
        d = {k: v for k, v in d.items() if k != "allow_real"}  # retired in v3: the EA trades any account it is on
        unknown = sorted(set(d) - known)
        if unknown:
            raise CodegenError(f"unknown guardrail fields {unknown}; known {sorted(known)}")
        return cls(**d)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LiveEA:
    source: str
    name: str
    magic: int
    dna_hash: str
    family: str
    strategy_id: str
    symbol: str
    timeframe: str
    template_version: str
    inputs: dict[str, str]

    @property
    def source_sha256(self) -> str:
        return hashlib.sha256(self.source.encode("utf-8")).hexdigest()


def _environment() -> jinja2.Environment:
    tester = Path(__file__).resolve().parents[1] / "mql5" / "templates"
    env = jinja2.Environment(
        loader=jinja2.ChoiceLoader([jinja2.FileSystemLoader(str(Path(__file__).resolve().parent / "templates")),
                                    jinja2.FileSystemLoader(str(tester))]),
        undefined=jinja2.StrictUndefined, keep_trailing_newline=True, trim_blocks=True, lstrip_blocks=True,
        newline_sequence="\n", autoescape=False)
    env.filters["bits"] = bits
    return env


def bot_name(dna: DNA, symbol: str, timeframe: str) -> str:
    return f"arqvio_{dna.signal.lower()}_{dna.dna_hash}_{symbol.lower()}_{timeframe.lower()}"


def ea_inputs(g: Guardrails) -> dict[str, str]:
    """The EA's input values as MT5 writes them in a template or preset (``name=value``)."""
    return {"InpLots": f"{g.lots:.8f}", "InpLotScale": f"{g.lot_scale:.8f}",
            "InpMaxSpreadPoints": str(g.max_spread_points), "InpDailyLossPct": f"{g.daily_loss_pct:.8f}",
            "InpMaxAccountPositions": str(g.max_account_positions), "InpFridayCloseHour": str(g.friday_close_hour),
            "InpProfitTargetPct": f"{g.profit_target_pct:.8f}", "InpNewsBlockMinutes": str(g.news_block_minutes),
            "InpSlippagePoints": str(g.slippage_points),
            "InpHistoryBars": str(g.history_bars),
            "InpSpreadRegimeMultiple": f"{g.spread_regime_multiple:.8f}", "InpSpreadRegimeBars": str(g.spread_regime_bars)}


def _validated_signals(dna: DNA) -> tuple[Any, str]:
    """The SIGNALS family (v4 batch 2.6) for the live EA only: the Strategy Tester parity EA refuses it until its
    parity subject exists (Phase 6), so it is validated here rather than by mql5/codegen._validated."""
    from engine.research.backtest.signal_switches import READY_SIGNALS

    if dna.exec_mode != "MARKET2":
        raise CodegenError(f"exec_mode {dna.exec_mode!r}: the oracle executes MARKET2 only")
    bad = [s for s in dna.signal_switches if s not in READY_SIGNALS]
    if bad:
        raise CodegenError(f"signal switches {bad} have no live-EA implementation yet (live: {list(READY_SIGNALS)})")
    try:
        validate_gene_space(dna.to_dict())
        cfg, _ = dna_to_strategy(dna)
    except (ValueError, DataError) as exc:
        raise CodegenError(f"DNA {dna.dna_hash} refused: {exc}") from exc
    return cfg, SWITCH_FAMILY


def _stop_inputs(mgmt_switches: tuple[str, ...]) -> dict[str, float]:
    """The management switches as the EA's inputs (0 / -1 = off), the oracle's own values."""
    kw = stop_rule_kwargs(mgmt_switches)
    trail = kw["atr_trail"] or (0.0, 0.0)
    close = close_rule_kwargs(mgmt_switches)
    eod = close["eod_close_hour"]
    return {"breakeven_at_r": kw["breakeven_at_r"] or 0.0, "sl_reduce_at_r": kw["sl_reduce_at_r"] or 0.0,
            "sl_lock_at_r": kw["sl_lock_at_r"] or 0.0, "atr_trail_k": trail[0], "atr_trail_act": trail[1],
            "fcp_pct": close["friday_close_profit_pct"] or 0.0, "eod_hour": -1 if eod is None else int(eod)}


def generate_live(dna: DNA, *, symbol: str, timeframe: str, magic: int, guardrails: Guardrails,
                  no_trade_hours: tuple[int, ...] = ()) -> LiveEA:
    """Render the live EA for ``dna`` locked to ``symbol`` ``timeframe``; CodegenError on any refusal."""
    unknown = [m for m in dna.mgmt_switches if m not in LIVE_MGMT]
    if unknown:
        raise CodegenError(f"management switches {unknown} have no live-EA implementation yet (live: {list(LIVE_MGMT)})")
    gates = [f"bias {b}" for b in dna.bias_switches if b not in READY_BIAS]
    gates += [f"filter {f}" for f in dna.filter_switches if f not in READY_FILTERS]
    if gates:  # the template renders only READY switches: an unknown one would silently trade without its gate
        raise CodegenError(f"switches {gates} have no live-EA implementation yet")
    base = replace(dna.ungated() if dna.gated else dna, mgmt_switches=())
    cfg, family = _validated_signals(base) if base.signal == SWITCH_FAMILY else _validated(base)
    if timeframe not in PERIOD_ENUMS:
        raise CodegenError(f"timeframe {timeframe!r}: a live EA runs on one of {sorted(PERIOD_ENUMS)}")
    if not symbol or any(c in symbol for c in '"\\\n\r'):
        raise CodegenError(f"symbol {symbol!r} cannot be written into an EA")
    problems = guardrails.problems()
    if problems:
        raise CodegenError("guardrails refused: " + "; ".join(problems))
    deploy = load_table().range(DEPLOY_RANGE)
    if not deploy.contains(magic):
        raise CodegenError(f"magic {magic} is outside the {DEPLOY_RANGE} range [{deploy.base}, {deploy.last}]")
    if any(not 0 <= h <= 23 for h in no_trade_hours):
        raise CodegenError(f"no-trade hours {no_trade_hours} must be server hours 0..23")
    name = bot_name(dna, symbol, timeframe)
    d = dna.to_dict()
    ctx = {
        "template_version": TEMPLATE_VERSION, "name": name, "family": family, "dna_hash": dna.dna_hash,
        "dna_json": json.dumps(d, sort_keys=True, separators=(",", ":")), "strategy_id": cfg.strategy_id,
        "cfg": cfg, "symbol": symbol, "timeframe": timeframe, "period_enum": PERIOD_ENUMS[timeframe],
        "magic": int(magic), "magic_min": deploy.base, "magic_max": deploy.last, "g": guardrails,
        "no_trade": [h in set(no_trade_hours) for h in range(24)],
        "session": dna.session, "session_lo": SESSIONS.get(dna.session, (0, 24))[0],
        "session_hi": SESSIONS.get(dna.session, (0, 24))[1], "trend_ema": int(dna.trend_ema),
        "bias": list(dna.bias_switches), "filters": list(dna.filter_switches),
        **_stop_inputs(dna.mgmt_switches),
        "switches": list(dna.signal_switches), "chop_ratio": CHOP_RATIO,
    }
    source = _environment().get_template("ea_live.mq5.j2").render(**ctx)
    if "\r" in source:
        raise CodegenError("generated source carries a carriage return")
    return LiveEA(source=source, name=name, magic=int(magic), dna_hash=dna.dna_hash, family=family,
                  strategy_id=cfg.strategy_id, symbol=symbol, timeframe=timeframe, template_version=TEMPLATE_VERSION,
                  inputs=ea_inputs(guardrails))


def attach_script_source() -> str:
    return _environment().get_template("attach.mq5.j2").render(template_version=TEMPLATE_VERSION)


__all__ = ["DEPLOY_RANGE", "Guardrails", "INCUBATION_STAGES", "LiveEA", "PERIOD_ENUMS",
           "PERIOD_MINUTES", "TEMPLATE_VERSION", "attach_script_source", "bot_name", "ea_inputs", "generate_live"]
