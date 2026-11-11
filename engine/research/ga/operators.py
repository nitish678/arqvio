"""GA operators: tournament selection, family-rule crossover, sampler mutation.

Every operator draws from the one generator it is handed, in a fixed order,
so a seed fixes the whole search. Every child is validated (population.checked)
before it can be admitted: DNA construction checks no ranges, so an operator
bug surfaces as an error, never as a silently repaired genome.

Gene roles (effective_hash.py): the FAMILY genes of a DNA are
FAMILY_GENES[dna.signal]; the COMMON engine-reaching genes are risk_pct,
sl_atr and tp_atr; everything else is inert (bias, filter, the hours, ...),
another family's latent genes, or PINNED (mgmt, exec_mode, risk,
uses_future_bar, schema_version -- the sampler pins them, and a non-NONE mgmt
would reach the slow engine as a mgmt=NONE backtest under another hash).
"""
from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from engine.research.backtest.strategies.switch import READY_MGMT
from engine.research.backtest.switch_gates import READY_BIAS, READY_FILTERS
from engine.research.ga.population import Individual
from engine.research.genes.registry import LOCKED_DNA_FIELDS
from engine.research.search.dna import DNA, SESSION_VALUES, SIGNAL_VALUES, SWITCH_FAMILY, TREND_EMA_VALUES
from engine.research.search.gene_pool import FAMILY_GENES, GenePool
from engine.research.search.sampler import (
    draw_signal_switches,
    draw_switches,
    DONCHIAN_PERIOD_VALUES,
    MACD_SIGNAL_PERIOD_VALUES,
    MAX_FEATURE_LOOKBACK_RANGE,
    RISK_PCT_RANGE,
    RSI_OVERBOUGHT_RANGE,
    RSI_OVERSOLD_RANGE,
    RSI_PERIOD_VALUES,
    SL_ATR_RANGE,
    STOCH_D_PERIOD_VALUES,
    STOCH_K_PERIOD_VALUES,
    STOCH_OVERBOUGHT_RANGE,
    STOCH_OVERSOLD_RANGE,
    STOCH_SMOOTHING_VALUES,
    TP_ATR_RANGE,
    _draw_choice,
    _macd_fast_choices,
    _macd_slow_choices,
)

PINNED_GENES = LOCKED_DNA_FIELDS
COMMON_ENGINE_GENES: tuple[str, ...] = ("risk_pct", "sl_atr", "tp_atr")
MACD_PAIR: tuple[str, str] = ("macd_fast", "macd_slow")
# signal_switches is the SIGNALS family's one family gene (v4 batch 2.6): crossed and mutated as a family gene, so
# it never joins the common genes (whose per-gene coins every older run's draws depend on).
_ALL_FAMILY_GENES = frozenset(s.name for specs in FAMILY_GENES.values() for s in specs) | {"signal_switches"}
# The ARQVIO entry gates: crossed and mutated only when a gate is on (a parent carries one, or the pool draws one),
# so an ungated population consumes exactly the random draws it always did.
GATE_GENES: tuple[str, ...] = ("session", "trend_ema", "bias_switches", "filter_switches")
# Inert and common genes, in DNA field order: crossover takes each from either
# parent. Family genes (of any family) and the signal are handled by the family
# rule; pinned genes are never touched.
CROSSOVER_COMMON_GENES: tuple[str, ...] = tuple(
    f.name for f in dataclasses.fields(DNA)
    if f.name != "signal" and f.name not in PINNED_GENES and f.name not in _ALL_FAMILY_GENES
    and f.name not in GATE_GENES
)


def family_genes(signal: str) -> tuple[str, ...]:
    if signal == SWITCH_FAMILY:
        return ("signal_switches",)
    return tuple(s.name for s in FAMILY_GENES[signal])


# One draw per gene from the sampler's own distribution for it
# (sampler.random_dna), so a mutated value is one random search could produce.
# The MACD pair draws from the gap-constrained set for the partner's value.
_DRAWS: dict[str, Callable[[np.random.Generator, DNA], Any]] = {
    "max_feature_lookback": lambda rng, d: int(
        rng.integers(MAX_FEATURE_LOOKBACK_RANGE[0], MAX_FEATURE_LOOKBACK_RANGE[1] + 1)
    ),
    "rsi_period": lambda rng, d: int(rng.choice(RSI_PERIOD_VALUES)),
    "rsi_oversold": lambda rng, d: float(rng.uniform(*RSI_OVERSOLD_RANGE)),
    "rsi_overbought": lambda rng, d: float(rng.uniform(*RSI_OVERBOUGHT_RANGE)),
    "rsi_exit_midline": lambda rng, d: bool(rng.random() < 0.5),
    "stoch_k_period": lambda rng, d: int(rng.choice(STOCH_K_PERIOD_VALUES)),
    "stoch_d_period": lambda rng, d: int(rng.choice(STOCH_D_PERIOD_VALUES)),
    "stoch_smoothing": lambda rng, d: int(rng.choice(STOCH_SMOOTHING_VALUES)),
    "stoch_oversold": lambda rng, d: float(rng.uniform(*STOCH_OVERSOLD_RANGE)),
    "stoch_overbought": lambda rng, d: float(rng.uniform(*STOCH_OVERBOUGHT_RANGE)),
    "macd_fast": lambda rng, d: int(rng.choice(_macd_fast_choices(int(d.macd_slow)))),
    "macd_slow": lambda rng, d: int(rng.choice(_macd_slow_choices(int(d.macd_fast)))),
    "macd_signal_period": lambda rng, d: int(rng.choice(MACD_SIGNAL_PERIOD_VALUES)),
    "donchian_period": lambda rng, d: int(rng.choice(DONCHIAN_PERIOD_VALUES)),
    "risk_pct": lambda rng, d: float(rng.uniform(*RISK_PCT_RANGE)),
    "sl_atr": lambda rng, d: float(rng.uniform(*SL_ATR_RANGE)),
    "tp_atr": lambda rng, d: float(rng.uniform(*TP_ATR_RANGE)),
    "session": lambda rng, d: str(rng.choice(SESSION_VALUES)),
    "trend_ema": lambda rng, d: int(rng.choice(TREND_EMA_VALUES)),
    "bias_switches": lambda rng, d: draw_switches(rng, READY_BIAS),
    "filter_switches": lambda rng, d: draw_switches(rng, READY_FILTERS),
    # crossed with the common genes (it always was a DNA field); mutated only when the pool turns it on
    "mgmt_switches": lambda rng, d: draw_switches(rng, READY_MGMT),
    "signal_switches": lambda rng, d: draw_signal_switches(rng),
}


SWITCH_FIELD_NAMES = ("bias_switches", "filter_switches", "mgmt_switches", "signal_switches")


def rank_key(ind: Individual) -> tuple[float, str]:
    """Sort key: higher fitness first, ties -> lower dna_hash."""
    return (-float(ind.fitness.fitness), ind.dna_hash)


def tournament(rng: np.random.Generator, scored: Sequence[Individual], k: int) -> Individual:
    """The fittest of k individuals drawn with replacement; ties -> lower dna_hash."""
    picks = rng.integers(0, len(scored), size=int(k))
    return min((scored[int(i)] for i in picks), key=rank_key)


def crossover(rng: np.random.Generator, a: DNA, b: DNA) -> DNA:
    """Uniform crossover under the family rule.

    The child takes its signal from parent A. Parents of different families:
    A's family genes wholesale, never mixed with B's. Same family: each family
    gene from either parent, the (macd_fast, macd_slow) pair as one gene so
    its gap survives. Every other family's (latent) genes come from A, so each
    family's genes stay one parent's consistent set. Common genes: uniform per
    gene. One coin per considered gene, in a fixed order.
    """
    genes: dict[str, Any] = {}
    fam = family_genes(a.signal)
    if a.signal == b.signal:
        units: list[tuple[str, ...]] = []
        for name in fam:
            if name == MACD_PAIR[1]:
                continue
            units.append(MACD_PAIR if name == MACD_PAIR[0] else (name,))
        for unit in units:
            donor = b if rng.random() < 0.5 else a
            for name in unit:
                genes[name] = getattr(donor, name)
    for name in CROSSOVER_COMMON_GENES:
        donor = b if rng.random() < 0.5 else a
        genes[name] = getattr(donor, name)
    if a.gated or b.gated:
        for name in GATE_GENES:
            donor = b if rng.random() < 0.5 else a
            genes[name] = getattr(donor, name)
    return dataclasses.replace(a, **genes)


def mutate(
    rng: np.random.Generator, dna: DNA, pool: GenePool, *, p: float, scale: float,
) -> tuple[DNA, tuple[str, ...]]:
    """Mutate each engine-reaching gene with probability ``p``.

    Order: signal, then the (possibly new) signal's family genes, then the
    common genes; pool-fixed and pinned genes are skipped. A signal redraw uses
    the pool's family weights (a disabled family has weight 0). A numeric gene
    moves toward a fresh sampler draw, v + scale * (draw - v): scale 1.0 is a
    plain redraw, and any scale stays inside the gene's range (a convex blend
    of two legal values, and the MACD sets are contiguous). An integer blend
    that rounds back to its old value takes one step toward the draw instead.
    Returns the child and the genes whose value changed.
    """
    fixed = pool.fixed_genes()
    changed: list[str] = []
    current = dna
    if rng.random() < p:
        signal = _draw_choice(rng, SIGNAL_VALUES, pool.signal_weights())
        if signal != current.signal:
            # leaving SIGNALS drops its switches (only SIGNALS carries them); the pool draw never enters SIGNALS
            current = dataclasses.replace(current, signal=signal, signal_switches=())
            changed.append("signal")
    for name in (*family_genes(current.signal), *COMMON_ENGINE_GENES, *pool.gates):
        if name == "signal_switches" and current.signal != SWITCH_FAMILY:
            continue  # the pool gate lets the sampler draw SIGNALS; only a SIGNALS genome mutates its switches
        if name in fixed or rng.random() >= p:
            continue
        old = getattr(current, name)
        draw = _DRAWS[name](rng, current)
        if name in SWITCH_FIELD_NAMES:  # switches the pool turns off are never bred back in
            draw = pool.allowed(name, draw)
            if name == "signal_switches" and not draw:
                # Same deterministic fallback as draw_signal_switches(pool): no extra RNG draw.
                from engine.research.backtest.signal_switches import READY_SIGNALS

                draw = pool.allowed(name, READY_SIGNALS)[:1]
        # a gate (and any switch tuple, e.g. mgmt_switches, crossed with the common genes) is a category: redrawn,
        # never blended (2026-11-01: blending the mgmt_switches tuple raised TypeError at scale < 1)
        if isinstance(old, bool | tuple) or scale == 1.0 or name in GATE_GENES:
            new: Any = draw
        elif isinstance(old, int):
            new = int(round(old + scale * (draw - old)))
            if new == old and draw != old:
                # The blend rounded back to where it started: one step toward
                # the draw, so a narrow integer range cannot freeze at a small scale.
                new = old + (1 if draw > old else -1)
        else:
            new = float(old + scale * (draw - old))
        if new != old:
            current = dataclasses.replace(current, **{name: new})
            changed.append(name)
    return current, tuple(changed)


__all__ = [
    "COMMON_ENGINE_GENES",
    "CROSSOVER_COMMON_GENES",
    "MACD_PAIR",
    "PINNED_GENES",
    "crossover",
    "family_genes",
    "mutate",
    "rank_key",
    "tournament",
]
