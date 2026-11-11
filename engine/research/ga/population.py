"""GA population: individuals, their lineage, and dedupe by effective hash.

An individual is one DNA plus where it came from. Two DNAs with the same
effective_gene_hash are one backtest under two names (effective_hash.py),
so the GA never evaluates a second copy: every DNA is unique by effective
hash within its generation AND against every DNA the run has evaluated (the
run registry). A collision is replaced by the next sampler draw, and every
replacement is counted.

Generation 0 is drawn by the S3 sampler exactly as random search draws it --
random_dna(rng, gene_pool=pool) -- so the gene pool's enabled families,
weights and fixed genes (and the MACD gap) hold by construction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from engine.research.search.dna import DNA, validate_gene_space
from engine.research.search.effective_hash import effective_gene_hash
from engine.research.search.gene_pool import GenePool
from engine.research.search.sampler import random_dna

if TYPE_CHECKING:
    from engine.research.ga.fitness import FitnessRecord
    from engine.research.search.runner import TrialResult

# How many consecutive sampler draws one slot may spend on collisions before
# the space is declared exhausted. A pool narrow enough to hit this cannot
# fill a population of distinct backtests, and saying so beats looping.
MAX_RESAMPLES_PER_SLOT = 1000


class DedupeExhausted(RuntimeError):
    """The sampler could not produce a DNA the run has not already evaluated."""


@dataclass
class Individual:
    """One DNA in the search, with its lineage and (once scored) its result.

    ``parents`` are the parents' effective hashes; ``operators`` is the chain
    that produced the DNA ("init", "crossover", "clone", "mutate:<genes>",
    "resample"); ``generation`` is the generation it was first evaluated in.
    """

    dna: DNA
    effective_hash: str
    generation: int
    parents: tuple[str, ...] = ()
    operators: tuple[str, ...] = ()
    result: TrialResult | None = None
    fitness: FitnessRecord | None = None
    engine: str | None = None

    @property
    def dna_hash(self) -> str:
        return self.dna.dna_hash

    def lineage(self) -> dict[str, Any]:
        return {
            "generation_found": self.generation,
            "parents": list(self.parents),
            "operators": list(self.operators),
        }


@dataclass
class Registry:
    """Every effective hash this run has admitted."""

    seen: set[str] = field(default_factory=set)
    collisions: int = 0

    def admit(self, effective_hash: str) -> bool:
        """Admit a new hash; False (and a counted collision) if already seen."""
        if effective_hash in self.seen:
            self.collisions += 1
            return False
        self.seen.add(effective_hash)
        return True


def checked(dna: DNA) -> str:
    """Validate a DNA's gene space and return its effective hash.

    DNA construction checks neither ranges nor the MACD gap, so every DNA the
    GA admits goes through validate_gene_space; effective_gene_hash raises for
    a family with no projection. An operator producing an invalid DNA is a
    bug, raised here rather than repaired.
    """
    validate_gene_space(dna.to_dict())
    from engine.research.genes.capabilities import require_breedable

    require_breedable(dna)
    return effective_gene_hash(dna)


def fresh_draw(
    rng: np.random.Generator,
    pool: GenePool,
    registry: Registry,
    *,
    generation: int,
    operators: tuple[str, ...],
) -> Individual:
    """The next sampler draw the run has not seen, as an Individual."""
    for _ in range(MAX_RESAMPLES_PER_SLOT):
        dna = random_dna(rng, gene_pool=pool)
        eh = checked(dna)
        if registry.admit(eh):
            return Individual(dna=dna, effective_hash=eh, generation=generation, operators=operators)
    raise DedupeExhausted(
        f"{MAX_RESAMPLES_PER_SLOT} consecutive sampler draws collided with the "
        f"{len(registry.seen)} effective hashes already evaluated; the gene pool "
        f"({pool.hash}) cannot supply a distinct DNA"
    )


def initial_population(
    rng: np.random.Generator, pool: GenePool, size: int, registry: Registry,
) -> list[Individual]:
    """Generation 0: ``size`` distinct DNAs from the sampler, in draw order."""
    return [
        fresh_draw(rng, pool, registry, generation=0, operators=("init",))
        for _ in range(int(size))
    ]


__all__ = [
    "MAX_RESAMPLES_PER_SLOT",
    "DedupeExhausted",
    "Individual",
    "Registry",
    "checked",
    "fresh_draw",
    "initial_population",
]
