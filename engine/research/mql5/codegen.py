"""DNA -> MQL5 Expert Advisor, deterministic, through the oracle's own DNA mapping.

``generate(dna, cell, magic=...)`` renders ``templates/ea.mq5.j2`` (one
parametrized template; the family logic is a partial per family) into ONE
self-contained ``.mq5``:

* the strategy parameters are what ``real_backtest.dna_to_strategy`` makes of
  the DNA -- the codegen never re-derives them, so the mapping cannot drift;
* every double the EA computes with (cell costs, tick values, thresholds,
  gene values) is emitted as its exact IEEE-754 bit pattern and rebuilt through
  a union, never as a decimal literal;
* DNA genes the oracle does not consume are listed in the header as
  inert-in-oracle and are NOT generated;
* the same DNA, cell and template version give a byte-identical file: no
  clock, no environment, sorted iteration, ``\\n`` endings.

Refusals raise ``CodegenError``: an unknown family, another DNA schema, a DNA
outside the gene space, a non-MARKET2 exec mode, or a config the oracle's
config class rejects.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jinja2

from engine.core.errors import DataError
from engine.research.mql5.cell_config import INITIAL_EQUITY, CellConfig
from engine.research.search.dna import DNA, DNA_SCHEMA_VERSION, SIGNAL_VALUES, validate_gene_space
from engine.research.search.real_backtest import dna_to_strategy

TEMPLATE_VERSION = "astra-mql5-v1"  # frozen: recorded in every parity subject (the pre-ARQVIO name)

# The genes each family's dna_to_*_config reads (real_backtest.py), beyond signal / sl_atr / tp_atr / risk_pct.
FAMILY_GENES: dict[str, tuple[str, ...]] = {
    "MA_CROSS": ("max_feature_lookback",),
    "RSI_REVERSION": ("rsi_period", "rsi_oversold", "rsi_overbought", "rsi_exit_midline"),
    "STOCH_CROSS": ("stoch_k_period", "stoch_d_period", "stoch_smoothing", "stoch_oversold", "stoch_overbought"),
    "MACD_CROSS": ("macd_fast", "macd_slow", "macd_signal_period"),
    "DONCHIAN_BREAKOUT": ("donchian_period",),
}
COMMON_GENES = ("schema_version", "signal", "sl_atr", "tp_atr", "risk_pct")


class CodegenError(ValueError):
    """A DNA (or cell) the generator refuses to turn into an EA."""


@dataclass(frozen=True)
class GeneratedEA:
    source: str
    name: str
    magic: int
    dna_hash: str
    family: str
    strategy_id: str
    cell_name: str
    template_version: str
    template_digest: str

    @property
    def source_sha256(self) -> str:
        return hashlib.sha256(self.source.encode("utf-8")).hexdigest()


def _templates_dir() -> Path:
    return Path(__file__).resolve().parent / "templates"


def template_digest() -> str:
    """sha256 over the template sources, in sorted name order (name and text). Line endings are
    normalised to LF first: a checkout that converts to CRLF must stamp the same digest (and so
    produce the same .mq5) as one that does not."""
    h = hashlib.sha256()
    for p in sorted(_templates_dir().glob("*.j2")):
        h.update(p.name.encode("utf-8") + b"\0" + p.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return h.hexdigest()


def bits(x: float) -> str:
    """The IEEE-754 bit pattern of a double, as an MQL5 ulong literal."""
    return f"0x{struct.unpack('<Q', struct.pack('<d', float(x)))[0]:016X}"


def _environment() -> jinja2.Environment:
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(_templates_dir())), undefined=jinja2.StrictUndefined,
        keep_trailing_newline=True, trim_blocks=True, lstrip_blocks=True, newline_sequence="\n",
        autoescape=False,
    )
    env.filters["bits"] = bits
    return env


def early_gates() -> tuple[int, float, float]:
    """(min_trades, max_dd_limit, return_threshold): the funnel's early-gate defaults, read from
    ``runner.run_campaign``'s own signature so the EA's DIAG label can never use other numbers."""
    from engine.research.search.runner import run_campaign

    p = inspect.signature(run_campaign).parameters
    return int(p["min_trades"].default), float(p["max_dd_limit"].default), float(p["return_threshold"].default)


def _validated(dna: DNA) -> tuple[Any, str]:
    d = dna.to_dict()
    # A schema-3 DNA is a schema-4 DNA with every switch off (dna.py: the v4 bump left v3 bodies and hashes intact),
    # so the owner's vaulted and deployed v3 strategies still render; refusing them broke every deploy (2026-10-31).
    if d["schema_version"] not in (3, DNA_SCHEMA_VERSION):
        raise CodegenError(f"DNA schema_version {d['schema_version']} is not 3 or {DNA_SCHEMA_VERSION}")
    if dna.signal not in SIGNAL_VALUES or dna.signal not in FAMILY_GENES:
        raise CodegenError(f"unknown signal family {dna.signal!r}")
    if dna.exec_mode != "MARKET2":
        raise CodegenError(f"exec_mode {dna.exec_mode!r}: the oracle executes MARKET2 only")
    if dna.gated:
        raise CodegenError("entry gates (session / trend_ema / bias and filter switches) are not in the Strategy Tester parity EA; "
                           "the live EA carries them (engine/research/deploy/live_ea.py)")
    # W4-2: a V4 switch-genome DNA (bias/filter/mgmt/risk switches ON) is not
    # yet representable in the parity EA template. Refuse loudly until the
    # switch gates are wired into engine/research/mql5/templates/ea.mq5.j2
    # (each switch becomes a tester/EA input toggle — W4-2.4 batch).
    if dna.bias_switches or dna.filter_switches or dna.mgmt_switches or dna.risk_switches:
        raise CodegenError(f"dna {dna.dna_hash}: V4 switch genome not yet codegen-able "
                           f"(switches={sorted(dna.bias_switches + dna.filter_switches + dna.mgmt_switches + dna.risk_switches)})")
    try:
        validate_gene_space(d)
        cfg, _ = dna_to_strategy(dna)
    except (ValueError, DataError) as exc:
        raise CodegenError(f"DNA {dna.dna_hash} refused: {exc}") from exc
    return cfg, dna.signal


def generate(dna: DNA, cell: CellConfig, *, magic: int) -> GeneratedEA:
    cfg, family = _validated(dna)
    d = dna.to_dict()
    used = set(COMMON_GENES) | set(FAMILY_GENES[family])
    inert = sorted(k for k in d if k not in used)
    spec = cell.spec
    if cell.hourly_spread_points is None or len(cell.hourly_spread_points) != 24:
        raise CodegenError(f"cell {cell.name} has no 24-hour spread matrix; every S6 cell prices from one")
    if cell.max_spread_points is not None and math.isnan(cell.max_spread_points):
        raise CodegenError(f"cell {cell.name}: max_spread_points is NaN")
    name = f"arqvio_{family.lower()}_{dna.dna_hash}_{cell.name}"
    ctx = {
        "template_version": TEMPLATE_VERSION,
        "template_digest": template_digest(),
        "name": name,
        "family": family,
        "dna_hash": dna.dna_hash,
        "dna_json": json.dumps(d, sort_keys=True, separators=(",", ":")),
        "inert_genes": inert,
        "strategy_id": cfg.strategy_id,
        "cfg": cfg,
        "magic": int(magic),
        "cell": cell,
        "replica_l": cell.replica_names[0],
        "replica_h": cell.replica_names[1],
        "digits": int(spec.digits),
        "point": spec.point,
        "tick_size": spec.trade_tick_size,
        "tvp": spec.trade_tick_value_profit,
        "tvl": spec.trade_tick_value_loss,
        "vmin": spec.volume_min,
        "vstep": spec.volume_step,
        "vmax": spec.volume_max,
        "swap_long": spec.swap_long,
        "swap_short": spec.swap_short,
        "triple_day": int(spec.swap_rollover_triple_day),
        "initial_equity": INITIAL_EQUITY,
        "hourly": list(cell.hourly_spread_points),
        "no_trade": [h in set(cell.no_trade_hours) for h in range(24)],
        "max_spread": cell.max_spread_points,
        "commission": cell.commission_per_lot_per_side,
        "gates": early_gates(),
        "n_bars": cell.n_bars,
        "first_time": cell.first_time,
        "last_time": cell.last_time,
    }
    source = _environment().get_template("ea.mq5.j2").render(**ctx)
    if "\r" in source:
        raise CodegenError("generated source carries a carriage return")
    return GeneratedEA(source=source, name=name, magic=int(magic), dna_hash=dna.dna_hash, family=family,
                       strategy_id=cfg.strategy_id, cell_name=cell.name, template_version=TEMPLATE_VERSION,
                       template_digest=ctx["template_digest"])


__all__ = ["CodegenError", "FAMILY_GENES", "GeneratedEA", "TEMPLATE_VERSION", "bits", "generate", "template_digest"]
