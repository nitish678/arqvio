"""Gene pool: which families the sampler may draw, how often, and which
genes it holds fixed.

The pool narrows random search; it never widens it. Every range it can
express is the sampler's own (engine/research/search/sampler.py), read
from the sampler's constants, so a fixed value the pool accepts is a value
the sampler could have drawn.

Canonical form (what is stored, hashed and returned by the API):

    {"families": {FAMILY: {"enabled": bool, "weight": float}},  # all five
     "genes":    {FAMILY: {gene: "enabled" | {"fixed": value}}},  # every gene
     "hash":     16 hex of the canonical {families, genes}}

Enabled weights are renormalised to sum to 1; disabled families carry 0.
A pool whose enabled weights cannot be normalised to finite values > 0
(their total overflows, or one underflows to 0 against the rest) is
rejected rather than stored with a weight the sampler cannot draw.

ARQVIO entry gates: an optional ``"gates": {"session": bool, "trend_ema": bool}``
turns each gate gene on for the sampler (engine/research/backtest/gates.py).
Gates are off by default and an all-off pool has no "gates" key at all, so
every pool stored before them keeps its canonical form and its hash. A PUT
without "gates" keeps the stored gates (they have their own control); a
PATCH merges them like any other section.

PATCH (``merge``) applies its values onto the STORED canonical pool, so a
patched weight is relative to the stored normalised weights, not to the
weights last PUT: on a pool at 0.25 each, weight 0.5 makes that family
twice as likely as each of the others. A patch that enables a disabled
family without giving a weight gives it the mean of the currently enabled
weights, so it joins at an equal share. Like PUT, PATCH ignores "hash".
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from engine.core import atomic
from engine.core.paths import get_configs_dir
from engine.research.search.dna import SIGNAL2_VALUES, SIGNAL_VALUES
from engine.research.search.sampler import (
    DONCHIAN_PERIOD_VALUES,
    MACD_FAST_VALUES,
    MACD_MIN_GAP,
    MACD_SIGNAL_PERIOD_VALUES,
    MACD_SLOW_VALUES,
    MAX_FEATURE_LOOKBACK_RANGE,
    RSI_OVERBOUGHT_RANGE,
    RSI_OVERSOLD_RANGE,
    RSI_PERIOD_VALUES,
    STOCH_D_PERIOD_VALUES,
    STOCH_K_PERIOD_VALUES,
    STOCH_OVERBOUGHT_RANGE,
    STOCH_OVERSOLD_RANGE,
    STOCH_SMOOTHING_VALUES,
    _macd_fast_choices,
    _macd_slow_choices,
)

ENABLED = "enabled"
GENE_POOL_FILENAME = "gene_pool.json"
MACD_CONSTRAINT = f"MACD_CROSS: macd_slow - macd_fast >= {MACD_MIN_GAP}"


class GenePoolError(ValueError):
    """A gene pool that cannot be run. ``problems`` lists every reason."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = list(problems)


@dataclass(frozen=True)
class GeneSpec:
    name: str
    kind: str  # "int" | "float" | "bool"
    lo: float | None = None
    hi: float | None = None

    def public(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name, "kind": self.kind}
        if self.kind != "bool":
            out["min"] = self.lo
            out["max"] = self.hi
        return out


def _int_values(name: str, values: tuple[int, ...]) -> GeneSpec:
    return GeneSpec(name, "int", values[0], values[-1])


FAMILY_GENES: dict[str, tuple[GeneSpec, ...]] = {
    "MA_CROSS": (
        GeneSpec("max_feature_lookback", "int", *MAX_FEATURE_LOOKBACK_RANGE),
    ),
    "RSI_REVERSION": (
        _int_values("rsi_period", RSI_PERIOD_VALUES),
        GeneSpec("rsi_oversold", "float", *RSI_OVERSOLD_RANGE),
        GeneSpec("rsi_overbought", "float", *RSI_OVERBOUGHT_RANGE),
        GeneSpec("rsi_exit_midline", "bool"),
    ),
    "STOCH_CROSS": (
        _int_values("stoch_k_period", STOCH_K_PERIOD_VALUES),
        _int_values("stoch_d_period", STOCH_D_PERIOD_VALUES),
        _int_values("stoch_smoothing", STOCH_SMOOTHING_VALUES),
        GeneSpec("stoch_oversold", "float", *STOCH_OVERSOLD_RANGE),
        GeneSpec("stoch_overbought", "float", *STOCH_OVERBOUGHT_RANGE),
    ),
    "MACD_CROSS": (
        _int_values("macd_fast", MACD_FAST_VALUES),
        _int_values("macd_slow", MACD_SLOW_VALUES),
        _int_values("macd_signal_period", MACD_SIGNAL_PERIOD_VALUES),
    ),
    "DONCHIAN_BREAKOUT": (
        _int_values("donchian_period", DONCHIAN_PERIOD_VALUES),
    ),
}


GATES: tuple[str, ...] = ("session", "trend_ema", "bias_switches", "filter_switches", "mgmt_switches",
                         "signal_switches")
# Owner 2026-11-01 ("genomes setup karo"): single switches can be turned off inside an open gate. ``off`` lists them
# by registry name ("bias.sma", "signal.fib", "management.breakeven"); the sampler still draws the full READY list
# (same random stream) and drops the ones turned off, so a pool with no ``off`` key samples exactly as before.
SWITCH_FIELDS: dict[str, str] = {"bias": "bias_switches", "filter": "filter_switches",
                                 "management": "mgmt_switches", "signal": "signal_switches"}


def switch_universe() -> dict[str, tuple[str, ...]]:
    """DNA field -> the READY switch names the sampler can draw for it."""
    from engine.research.backtest.signal_switches import READY_SIGNALS
    from engine.research.backtest.switch_gates import READY_BIAS, READY_FILTERS
    from engine.research.backtest.strategies.switch import READY_MGMT

    return {"bias_switches": READY_BIAS, "filter_switches": READY_FILTERS, "mgmt_switches": READY_MGMT,
            "signal_switches": READY_SIGNALS}


@dataclass(frozen=True)
class GenePool:
    """A validated, canonical gene pool. Build it with ``canonicalise``."""

    families: dict[str, dict[str, Any]]
    genes: dict[str, dict[str, Any]]
    gates: tuple[str, ...] = ()  # the entry gates the sampler draws, sorted; () = none
    off: tuple[str, ...] = ()    # switches turned off inside their gate, registry names, sorted; () = none

    def canonical(self) -> dict[str, Any]:
        base: dict[str, Any] = {"families": self.families, "genes": self.genes}
        if self.gates:
            base["gates"] = {g: True for g in self.gates}
        if self.off:
            base["off"] = list(self.off)
        return base

    def allowed(self, field: str, names: tuple[str, ...]) -> tuple[str, ...]:
        """``names`` (a drawn switch tuple of DNA ``field``) without the switches this pool turns off."""
        if not self.off:
            return names
        group = next(g for g, f in SWITCH_FIELDS.items() if f == field)
        return tuple(n for n in names if f"{group}.{n}" not in self.off)

    @property
    def hash(self) -> str:
        return pool_hash(self.canonical())

    def to_dict(self) -> dict[str, Any]:
        return {**self.canonical(), "hash": self.hash}

    def signal_weights(self) -> tuple[float, ...]:
        return tuple(self.families[f]["weight"] for f in SIGNAL_VALUES)

    def signal2_weights(self) -> tuple[float, ...]:
        """Uniform over the enabled families plus "" (no second signal)."""
        allowed = [v == "" or self.families[v]["enabled"] for v in SIGNAL2_VALUES]
        share = 1.0 / sum(allowed)
        return tuple(share if ok else 0.0 for ok in allowed)

    def fixed_genes(self) -> dict[str, Any]:
        return {
            gene: setting["fixed"]
            for fam in SIGNAL_VALUES
            for gene, setting in self.genes[fam].items()
            if setting != ENABLED
        }


def pool_hash(canonical: dict[str, Any]) -> str:
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def default_gene_pool() -> GenePool:
    return canonicalise(
        {"families": {f: {"enabled": True, "weight": 1.0} for f in SIGNAL_VALUES}}
    )


def schema() -> dict[str, Any]:
    return {
        "families": list(SIGNAL_VALUES),
        "genes": {
            fam: [spec.public() for spec in FAMILY_GENES[fam]] for fam in SIGNAL_VALUES
        },
        "constraints": [MACD_CONSTRAINT],
        "gates": list(GATES),
    }


def _is_finite_number(value: Any) -> bool:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return False
    # A JSON integer such as 10**400 parses to an int too large for a
    # float, and math.isfinite raises on it instead of answering False.
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _fixed_value(spec: GeneSpec, value: Any) -> tuple[Any, str | None]:
    """Coerce a fixed value to its gene's kind, or say why it cannot be."""
    if spec.kind == "bool":
        if not isinstance(value, bool):
            return None, f"fixed value {value!r} must be a bool"
        return value, None
    if spec.kind == "int":
        if not isinstance(value, int) or isinstance(value, bool):
            return None, f"fixed value {value!r} must be an int"
    elif not _is_finite_number(value):
        return None, f"fixed value {value!r} must be a finite number"
    if not (spec.lo <= value <= spec.hi):
        return None, f"fixed value {value!r} outside [{spec.lo}, {spec.hi}]"
    return (int(value) if spec.kind == "int" else float(value)), None


def _families(families_in: Any, problems: list[str]) -> dict[str, tuple[bool, float]]:
    if not isinstance(families_in, dict):
        problems.append("families: must be an object naming every family")
        return {}
    for name in sorted(set(families_in) - set(SIGNAL_VALUES), key=str):
        problems.append(f"unknown family {name!r}")
    parsed: dict[str, tuple[bool, float]] = {}
    for fam in SIGNAL_VALUES:
        entry = families_in.get(fam)
        if entry is None:
            problems.append(f"{fam}: missing from families")
            continue
        if not isinstance(entry, dict):
            problems.append(f"{fam}: must be an object {{enabled, weight}}")
            continue
        for key in sorted(set(entry) - {"enabled", "weight"}, key=str):
            problems.append(f"{fam}: unknown key {key!r}")
        enabled = entry.get("enabled")
        weight = entry.get("weight", 1.0)
        if not isinstance(enabled, bool):
            problems.append(f"{fam}: enabled must be a bool, got {enabled!r}")
            continue
        if not _is_finite_number(weight):
            problems.append(f"{fam}: weight must be a finite number, got {weight!r}")
            continue
        if enabled and weight <= 0:
            problems.append(f"{fam}: an enabled family needs weight > 0, got {weight!r}")
            continue
        parsed[fam] = (enabled, float(weight))
    if len(parsed) == len(SIGNAL_VALUES) and not any(on for on, _ in parsed.values()):
        problems.append("at least one family must be enabled (all-off rejected)")
    return parsed


def _genes(genes_in: Any, problems: list[str]) -> dict[str, dict[str, Any]]:
    if not isinstance(genes_in, dict):
        problems.append("genes: must be an object keyed by family")
        genes_in = {}
    for name in sorted(set(genes_in) - set(SIGNAL_VALUES), key=str):
        problems.append(f"genes: unknown family {name!r}")
    genes: dict[str, dict[str, Any]] = {}
    for fam in SIGNAL_VALUES:
        fam_in = genes_in.get(fam, {})
        if not isinstance(fam_in, dict):
            problems.append(f"genes.{fam}: must be an object keyed by gene")
            fam_in = {}
        specs = FAMILY_GENES[fam]
        for name in sorted(set(fam_in) - {s.name for s in specs}, key=str):
            problems.append(f"genes.{fam}: unknown gene {name!r}")
        out: dict[str, Any] = {}
        for spec in specs:
            setting = fam_in.get(spec.name, ENABLED)
            if setting == ENABLED:
                out[spec.name] = ENABLED
            elif isinstance(setting, dict) and set(setting) == {"fixed"}:
                value, err = _fixed_value(spec, setting["fixed"])
                if err:
                    problems.append(f"genes.{fam}.{spec.name}: {err}")
                else:
                    out[spec.name] = {"fixed": value}
            else:
                problems.append(
                    f'genes.{fam}.{spec.name}: must be "enabled" or {{"fixed": value}}, '
                    f"got {setting!r}"
                )
        genes[fam] = out
    _check_macd(genes.get("MACD_CROSS", {}), problems)
    return genes


def _check_macd(macd: dict[str, Any], problems: list[str]) -> None:
    fast = macd.get("macd_fast")
    slow = macd.get("macd_slow")
    fast = fast["fixed"] if isinstance(fast, dict) else None
    slow = slow["fixed"] if isinstance(slow, dict) else None
    if fast is not None and slow is not None:
        if slow - fast < MACD_MIN_GAP:
            problems.append(
                f"MACD_CROSS: fixed macd_slow - macd_fast = {slow - fast} "
                f"violates {MACD_CONSTRAINT}"
            )
    elif fast is not None and not _macd_slow_choices(fast):
        problems.append(f"MACD_CROSS: no legal macd_slow for fixed macd_fast={fast}")
    elif slow is not None and not _macd_fast_choices(slow):
        problems.append(f"MACD_CROSS: no legal macd_fast for fixed macd_slow={slow}")


def canonicalise(data: Any) -> GenePool:
    """Validate a full pool and return its canonical form.

    Raises GenePoolError carrying every problem found, not just the first.
    A "hash" key is accepted and ignored: the hash is derived, never input.
    """
    if not isinstance(data, dict):
        raise GenePoolError(["gene pool must be a JSON object"])
    problems: list[str] = []
    for key in sorted(set(data) - {"families", "genes", "gates", "off", "hash"}, key=str):
        problems.append(f"unknown key {key!r}")
    parsed = _families(data.get("families"), problems)
    genes = _genes(data.get("genes", {}), problems)
    gates = _gates(data.get("gates", {}), problems)
    off = _off(data.get("off", []), problems)
    if problems:
        raise GenePoolError(problems)
    return GenePool(families=_normalised(parsed), genes=genes, gates=gates, off=off)


def _off(raw: Any, problems: list[str]) -> tuple[str, ...]:
    if not isinstance(raw, list) or not all(isinstance(x, str) for x in raw):
        problems.append("off: must be a list of switch names such as 'bias.sma'")
        return ()
    universe = switch_universe()
    known = {f"{g}.{n}" for g, f in SWITCH_FIELDS.items() for n in universe[f]}
    for name in sorted(set(raw) - known):
        problems.append(f"off: {name!r} is not a READY switch")
    for group, field in SWITCH_FIELDS.items():
        if group == "signal" and all(f"signal.{n}" in raw for n in universe[field]):
            problems.append("off: the SIGNALS family needs at least one signal switch on (turn its gate off instead)")
    return tuple(sorted(set(raw) & known))


def _gates(raw: Any, problems: list[str]) -> tuple[str, ...]:
    if not isinstance(raw, dict):
        problems.append("gates: must be an object of gate -> bool")
        return ()
    for key in sorted(set(raw) - set(GATES), key=str):
        problems.append(f"gates: unknown gate {key!r}; gates are {list(GATES)}")
    for key, on in raw.items():
        if key in GATES and not isinstance(on, bool):
            problems.append(f"gates.{key}: {on!r} must be a bool")
    return tuple(g for g in GATES if raw.get(g) is True)


def _normalised(parsed: dict[str, tuple[bool, float]]) -> dict[str, dict[str, Any]]:
    total = sum(w for on, w in parsed.values() if on)
    if not math.isfinite(total):
        raise GenePoolError([f"enabled weights sum to {total}; the total must be finite"])
    # An already-normalised pool is left as-is so a GET -> PUT round trip
    # keeps its hash instead of drifting by an ulp per pass.
    scale = 1.0 if math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9) else total
    families = {
        fam: {"enabled": on, "weight": (w / scale) if on else 0.0}
        for fam, (on, w) in parsed.items()
    }
    underflow = [
        f"{fam}: weight {parsed[fam][1]!r} is too small against a total of "
        f"{total!r}; it normalises to 0"
        for fam, entry in families.items() if entry["enabled"] and entry["weight"] <= 0.0
    ]
    if underflow:
        raise GenePoolError(underflow)
    return families


def merge(pool: GenePool, patch: Any) -> GenePool:
    """Apply a partial pool onto ``pool`` and validate the result as a whole.

    Weights are relative to ``pool``'s normalised weights; see the module
    docstring for the rest of the PATCH semantics.
    """
    if not isinstance(patch, dict):
        raise GenePoolError(["gene pool patch must be a JSON object"])
    unknown = sorted(set(patch) - {"families", "genes", "gates", "off", "hash"}, key=str)
    if unknown:
        raise GenePoolError([f"unknown key {key!r}" for key in unknown])
    merged = json.loads(json.dumps(pool.canonical()))
    if "off" in patch:  # the whole list: what the UI shows is what is stored
        merged["off"] = patch["off"]
    if "gates" in patch:
        if not isinstance(patch["gates"], dict):
            raise GenePoolError(["gates: must be an object of gate -> bool"])
        merged["gates"] = {**{g: True for g in pool.gates}, **patch["gates"]}
    enabled = [e["weight"] for e in pool.families.values() if e["enabled"]]
    joining_weight = sum(enabled) / len(enabled)
    problems: list[str] = []
    for section in ("families", "genes"):
        updates = patch.get(section, {})
        if not isinstance(updates, dict):
            problems.append(f"{section}: must be an object keyed by family")
            continue
        for fam, update in updates.items():
            if not isinstance(update, dict):
                problems.append(f"{section}.{fam}: must be an object")
                continue
            entry = merged[section].setdefault(fam, {})
            if (
                section == "families"
                and entry.get("enabled") is False
                and update.get("enabled") is True
                and "weight" not in update
            ):
                update = {**update, "weight": joining_weight}
            entry.update(update)
    if problems:
        raise GenePoolError(problems)
    return canonicalise(merged)


def summary(pool: GenePool | None) -> str:
    """One line for the campaign manifest."""
    if pool is None:
        return "none (uniform over every family; pre-W2 draw)"
    on = " ".join(
        f"{f}={pool.families[f]['weight']:.4g}"
        for f in SIGNAL_VALUES if pool.families[f]["enabled"]
    )
    off = ",".join(f for f in SIGNAL_VALUES if not pool.families[f]["enabled"]) or "none"
    fixed = ",".join(f"{g}={v}" for g, v in pool.fixed_genes().items()) or "none"
    gates = f" | gates: {','.join(pool.gates)}" if pool.gates else ""
    return f"{pool.hash} | {on} | off: {off} | fixed: {fixed}{gates}"


def default_path() -> Path:
    return get_configs_dir() / GENE_POOL_FILENAME


def load(path: Path | None = None) -> GenePool:
    """The stored pool, or the default pool when none is stored.

    A stored pool that no longer validates, or whose recorded hash does not
    match its content, raises rather than quietly running a different pool.
    """
    pool_path = default_path() if path is None else path
    if not pool_path.is_file():
        return default_gene_pool()
    try:
        data = json.loads(pool_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise GenePoolError([f"{pool_path}: unreadable ({exc})"]) from exc
    pool = canonicalise(data)
    stored_hash = data.get("hash")
    if stored_hash is not None and stored_hash != pool.hash:
        raise GenePoolError([
            f"{pool_path}: recorded hash {stored_hash!r} does not match its "
            f"content ({pool.hash!r})"
        ])
    return pool


def save(pool: GenePool, path: Path | None = None) -> Path:
    if path is None:
        path = get_configs_dir(create=True) / GENE_POOL_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(pool.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    atomic.replace(tmp, path)
    return path


__all__ = [
    "ENABLED",
    "FAMILY_GENES",
    "GenePool",
    "GenePoolError",
    "GeneSpec",
    "canonicalise",
    "default_gene_pool",
    "default_path",
    "load",
    "merge",
    "pool_hash",
    "save",
    "schema",
    "summary",
]
