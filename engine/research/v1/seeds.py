"""The 16 starter strategies (research PART 4.2 / APPENDIX-D: 16 seeds, tribe origins Wild 10, Tribe A 3,
Tribe B 1, FINAL WAR 2). The research ships its own seed values; it also says to keep the schema and choose one's
own values, so these are ARQVIO's: drawn once from the V1 gene pool with a fixed salt, cycling the research
families so every family starts with at least one seed.
"""
from __future__ import annotations

import dataclasses
import hashlib

import numpy as np

from engine.research.search.dna import DNA
from engine.research.search.gene_pool import GenePool
from engine.research.search.sampler import random_dna
from engine.research.v1.score import FINAL_WAR, TRIBE_A, TRIBE_B, WILD

SEED_SALT = "arqvio-v1-seeds-1"
ORIGINS: tuple[str, ...] = (WILD,) * 10 + (TRIBE_A,) * 3 + (TRIBE_B,) + (FINAL_WAR,) * 2
MAX_DRAWS = 10_000


def seeds(pool: GenePool, risk_pct: float) -> list[tuple[str, DNA]]:
    """(tribe origin, DNA) x 16, the same for the same pool and risk."""
    families = [f for f, spec in pool.families.items() if spec.get("enabled")]
    rng = np.random.default_rng(int(hashlib.sha256(SEED_SALT.encode()).hexdigest()[:16], 16))
    out: list[tuple[str, DNA]] = []
    for i, origin in enumerate(ORIGINS):
        want = families[i % len(families)]
        for _ in range(MAX_DRAWS):
            dna = random_dna(rng, gene_pool=pool)
            if dna.signal == want:
                break
        else:  # pragma: no cover - an enabled family with weight > 0 is always drawn
            raise RuntimeError(f"no seed of family {want} in {MAX_DRAWS} draws")
        out.append((origin, dataclasses.replace(dna, risk_pct=float(risk_pct))))
    return out


__all__ = ["ORIGINS", "SEED_SALT", "seeds"]
