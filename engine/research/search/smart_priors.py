"""Smart priors (worker4 W4-9 9.5; idea: research "SMART MODE" -- breed more of what the vault shows works, with half
the draws left random as the control).

Family weights for the gene pool from the vault: each enabled signal family gets
    w = 0.5 / k  +  0.5 * (its share of the vault's surviving strategies)
over the k enabled families, so half the draws stay uniform (the control: smart priors can never starve a family) and
half follow the evidence. Survivors = vault strategies that are CERTIFIED, or CANDIDATE / UNCERTIFIED with a grade
other than F / WEAK. A suggestion only: the owner applies it to the gene pool (PATCH /api/v1/gene-pool), and
the pool hash -- part of every campaign's identity -- records that it changed.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

from engine.research.search.dna import SIGNAL_VALUES
from engine.research.search.gene_pool import GenePool

CONTROL_SHARE = 0.5


def survivors(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if r.get("tier") == "CERTIFIED" or r.get("grade") not in ("F", "WEAK", None)]


def suggest(pool: GenePool, rows: list[dict[str, Any]]) -> dict[str, Any]:
    enabled = [f for f in SIGNAL_VALUES if (pool.families.get(f) or {}).get("enabled")]
    alive = [r for r in survivors(rows) if r.get("family") in enabled]
    counts = Counter(r["family"] for r in alive)
    k = len(enabled)
    total = sum(counts.values())
    weights = {f: round(CONTROL_SHARE / k + (1 - CONTROL_SHARE) * (counts[f] / total if total else 1 / k), 6)
               for f in enabled}
    return {"weights": weights, "basis": {"survivors": total, "by_family": dict(counts), "vault": len(rows),
                                          "control_share": CONTROL_SHARE},
            "patch": {"families": {f: {"enabled": True, "weight": w} for f, w in weights.items()}}}


__all__ = ["CONTROL_SHARE", "suggest", "survivors"]
