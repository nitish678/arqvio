"""Degeneracy report: is the search actually exploring, or collapsing?

A campaign can look healthy -- a thousand trials, a full funnel -- while
every candidate is the same strategy in disguise, one family never trades,
or the best candidates all sit on a gene's boundary (the range, not the
data, is choosing them). These fields make each of those visible from the
report alone. Every number is counted from the campaign's own trials.

Boundary saturation is measured on the funnel's survivors when there are
at least ``MIN_SATURATION_SET`` of them, otherwise on the top quartile (by
Sharpe) of the eligible candidates; ``basis`` names which. A gene is at a
boundary when an integer gene equals its range end, or a float gene lies
within ``FLOAT_EDGE_FRACTION`` of the range width of either end (a uniform
draw lands there with probability 2 * FLOAT_EDGE_FRACTION). ``flagged``
lists every gene above ``SATURATION_FLAG``.

Each per-gene entry also carries its uniform-draw reference (additive; the
flag and every other field are unchanged): an integer gene with a k-value
legal range has 2 boundary values, so a uniform draw sits on one with
probability 2/k -- a raw share of 0.72 means little when k = 3 (baseline
0.667). ``small_value_set_caveat`` (k < SMALL_VALUE_SET) is a display aid
for reading the flag, not an input to any verdict.

The baseline is a uniform draw over the gene's FULL legal range, not the
sampler's marginal. The two differ where the draw is constrained: a gene the
gene pool fixes (one value drawn, yet read against 2/k), and the MACD pair,
whose MACD_MIN_GAP makes macd_slow's lower end unreachable for the largest
fast values and narrows macd_fast when macd_slow is fixed. For those genes
``adjusted_margin`` is not a measure of search pressure.

``sampler_marginal_baselines`` is that missing reference: the probability,
estimated from seeded draws of the sampler itself under the run's gene pool,
that a gene sits on a boundary. Drawing from the sampler respects the pool's
fixed genes (exactly 0 or 1) and the MACD gap by construction. It is OPT-IN:
passed to boundary_saturation / degeneracy_report as ``marginal_baselines``,
it adds ``sampler_marginal_baseline`` and ``sampler_marginal_margin`` beside
each entry. Without it the census is byte-identical to before.
"""
from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

import numpy as np

from engine.research.search.dna import SIGNAL_VALUES, FailureTaxonomy
from engine.research.search.gene_pool import FAMILY_GENES

if TYPE_CHECKING:  # runner imports this module
    from engine.research.search.runner import TrialResult

MIN_SATURATION_SET = 4
FLOAT_EDGE_FRACTION = 0.025
SATURATION_FLAG = 0.5
SMALL_VALUE_SET = 10
COST_LINES = ("spread", "commission", "slippage", "swap")
SAMPLER_MARGINAL_DRAWS = 20_000
SAMPLER_MARGINAL_SEED = 20261001


def sampler_marginal_baselines(
    gene_pool=None, *, n_draws: int = SAMPLER_MARGINAL_DRAWS, seed: int = SAMPLER_MARGINAL_SEED,
) -> dict[str, float]:
    """P(gene on a boundary) under the sampler and ``gene_pool``, per "FAMILY.gene".

    Estimated from ``n_draws`` seeded random_dna draws (the sampler's genes are
    drawn independently of the signal, so one draw set serves every family).
    Bool genes carry no boundary and are left out, as in boundary_saturation.
    """
    from engine.research.search.sampler import random_dna

    rng = np.random.default_rng(seed)
    draws = [random_dna(rng, gene_pool=gene_pool) for _ in range(int(n_draws))]
    out: dict[str, float] = {}
    for family in SIGNAL_VALUES:
        for spec in FAMILY_GENES[family]:
            if spec.kind == "bool":
                continue
            hits = sum(_at_boundary(spec, float(getattr(d, spec.name))) for d in draws)
            out[f"{family}.{spec.name}"] = hits / len(draws)
    return out


def _at_boundary(spec, value: float) -> bool:
    if spec.kind == "int":
        return value in (spec.lo, spec.hi)
    edge = FLOAT_EDGE_FRACTION * (spec.hi - spec.lo)
    return value <= spec.lo + edge or value >= spec.hi - edge


def saturation_context(
    raw_share: float, *, distinct_values: int, value_set_size: int | None,
) -> dict[str, Any]:
    """A boundary share beside what a uniform draw over the legal range gives.

    ``value_set_size`` is the gene's legal value count k (None for a float
    gene). The baseline counts the boundary points ``_at_boundary`` counts:
    2 of k values for k >= 2 (none for a one-value gene), and the two edge
    bands, 2 * FLOAT_EDGE_FRACTION, for a float gene. ``distinct_values`` is
    how many different values the measured set actually holds.
    """
    if value_set_size is None:
        baseline: float | None = 2 * FLOAT_EDGE_FRACTION
    elif value_set_size >= 2:
        baseline = 2 / value_set_size
    else:
        baseline = None
    return {
        "raw_share": raw_share,
        "distinct_values": distinct_values,
        "value_set_size": value_set_size,
        "uniform_baseline": baseline,
        "adjusted_margin": None if baseline is None else raw_share - baseline,
        "small_value_set_caveat": value_set_size is not None and value_set_size < SMALL_VALUE_SET,
    }


def boundary_saturation(
    trials: list[TrialResult], outcomes: list[str], *, min_trades: int,
    marginal_baselines: dict[str, float] | None = None,
) -> dict[str, Any]:
    survivors = [
        t for t, o in zip(trials, outcomes, strict=True)
        if o == FailureTaxonomy.VAULTED.value
    ]
    if len(survivors) >= MIN_SATURATION_SET:
        basis, chosen = f"survivors (n={len(survivors)})", survivors
    else:
        eligible = sorted(
            (
                t for t, o in zip(trials, outcomes, strict=True)
                if o != FailureTaxonomy.Crash.value
                and int(t.n_trades or 0) >= min_trades
                and t.sharpe is not None and np.isfinite(t.sharpe)
            ),
            key=lambda t: float(t.sharpe), reverse=True,
        )
        chosen = eligible[: max(1, len(eligible) // 4)] if eligible else []
        basis = (
            f"top quartile of eligible by Sharpe (n={len(chosen)} of "
            f"{len(eligible)}; survivors={len(survivors)} < {MIN_SATURATION_SET})"
        )
    per_gene: dict[str, dict[str, Any]] = {}
    flagged: list[str] = []
    for family in SIGNAL_VALUES:
        members = [t for t in chosen if t.dna.signal == family]
        if not members:
            continue
        for spec in FAMILY_GENES[family]:
            if spec.kind == "bool":
                continue
            values = [float(getattr(t.dna, spec.name)) for t in members]
            hits = sum(_at_boundary(spec, v) for v in values)
            share = hits / len(members)
            key = f"{family}.{spec.name}"
            per_gene[key] = {
                "n": len(members), "at_boundary": hits, "share": share,
                **saturation_context(
                    share,
                    distinct_values=len(set(values)),
                    value_set_size=(
                        int(spec.hi) - int(spec.lo) + 1 if spec.kind == "int" else None
                    ),
                ),
            }
            if marginal_baselines is not None:
                marginal = marginal_baselines[key]
                per_gene[key]["sampler_marginal_baseline"] = marginal
                per_gene[key]["sampler_marginal_margin"] = share - marginal
            if share > SATURATION_FLAG:
                flagged.append(key)
    return {"basis": basis, "per_gene": per_gene, "flagged": flagged}


def degeneracy_report(
    trials: list[TrialResult],
    outcomes: list[str],
    *,
    min_trades: int,
    trials_raw: int,
    trials_effective: int,
    marginal_baselines: dict[str, float] | None = None,
) -> dict[str, Any]:
    per_family = Counter(t.dna.signal for t in trials)
    zero_trade = Counter(t.dna.signal for t in trials if int(t.n_trades or 0) == 0)
    low_trades = Counter(
        t.dna.signal for t in trials if int(t.n_trades or 0) < min_trades
    )
    return {
        "trials_per_family": {f: per_family.get(f, 0) for f in SIGNAL_VALUES},
        "effective_over_real": (trials_effective / trials_raw) if trials_raw else None,
        "dedupe_hits": trials_raw - trials_effective,
        "zero_trade_rate_per_family": {
            f: (zero_trade.get(f, 0) / per_family[f]) if per_family.get(f) else None
            for f in SIGNAL_VALUES
        },
        "min_trades_failures_per_family": {
            f: low_trades.get(f, 0) for f in SIGNAL_VALUES
        },
        "min_trades": min_trades,
        "boundary_saturation": boundary_saturation(
            trials, outcomes, min_trades=min_trades, marginal_baselines=marginal_baselines,
        ),
    }


def cost_totals(trials: list[TrialResult]) -> dict[str, float]:
    """Sum of every cost line over the campaign's candidates."""
    return {
        line: float(sum(float(t.costs.get(line, 0.0)) for t in trials))
        for line in COST_LINES
    }


__all__ = [
    "COST_LINES",
    "FLOAT_EDGE_FRACTION",
    "MIN_SATURATION_SET",
    "SAMPLER_MARGINAL_DRAWS",
    "SAMPLER_MARGINAL_SEED",
    "SATURATION_FLAG",
    "SMALL_VALUE_SET",
    "boundary_saturation",
    "cost_totals",
    "degeneracy_report",
    "saturation_context",
    "sampler_marginal_baselines",
]
