"""Engine V1 settings: the research numbers, and ARQVIO's where the research names no number.

Research numbers (CLAUDE_BUILD_INSTRUCTIONS PART 4): proving grounds 3000 (750 per exec mode x 4), 2 tribes x 20
generations + Final War, score weights 20/15/15/15/12/10/10/3 with GOOD at 55, held-out < 15 trades rejected, 100
random strategies, 1000 reshuffled runs with a bad-luck drawdown under 40 %, 16 seeds.
Everything marked ``# ARQVIO`` is not in the pack: a tribe's size, selection pressure, the floors' values.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

from engine.research.genes import registry
from engine.research.search.dna import SIGNAL_VALUES
from engine.research.search.gene_pool import GenePool, canonicalise

ENGINE = "v1"

# The V1 entry gates: research genes the sampler draws as switches (registry engine="v1", READY). The ARQVIO
# session gate is not one of them.
V1_GATES: tuple[str, ...] = ("bias_switches", "filter_switches", "mgmt_switches", "signal_switches", "trend_ema")


@dataclass(frozen=True)
class V1Config:
    preset: str = "research"
    seed: int = 0
    risk_pct: float = 0.01               # the campaign's risk % (a campaign setting in research, not a gene)
    proving_grounds: int = 3000          # research PART 4.10
    exec_modes: tuple[str, ...] = ("MARKET2",)  # research: 4 modes x 750; only market is READY today
    generations: int = 20                # research: 20 per tribe
    tribe_size: int = 40                 # ARQVIO: survivors of the proving grounds per tribe
    elite: int = 4                       # ARQVIO: best of a tribe kept unchanged each generation
    tournament_k: int = 3                # ARQVIO
    crossover_p: float = 0.7             # ARQVIO
    mutation_p: float = 0.2              # ARQVIO
    mutation_scale: float = 0.5          # ARQVIO
    low_fitness_share: float = 0.25      # ARQVIO: LOW_FITNESS culls this bottom share each generation
    drawdown_limit: float = 0.30         # ARQVIO default of the prop guardrail: DRAWDOWN_BREACH above it
    stagnation_after: int = 4            # ARQVIO: generations without a better best -> STAGNATION + revolution
    revolution_share: float = 0.5        # ARQVIO: share of a stagnant tribe replaced by fresh draws
    family_cap: float = 0.5              # ARQVIO: a family above this share of a tribe pays the family penalty
    family_penalty: float = 0.9          # ARQVIO: fitness x this while over the cap
    family_reward: float = 1.05          # ARQVIO: fitness x this for a family below 1/(2 x families) of the tribe
    min_trades: int = 30                 # ARQVIO floor: fewer in-sample trades -> not alive
    finalists: int = 8                   # ARQVIO: Final War ranks; the best this many go to the vault gates
    # vault gates (research names them; numbers marked ARQVIO are ours)
    held_out_min_trades: int = 15        # research
    random_n: int = 100                  # research
    random_beat_min: float = 0.95        # ARQVIO: beat 95 of the 100 random strategies
    shuffle_n: int = 1000                # research
    shuffle_dd_max: float = 0.40         # research: the unluckiest 1 in 20 reshuffled runs under 40 % drawdown
    nudge_hold_min: float = 0.5          # ARQVIO: at least half the nudged settings still make money
    floor_max_dd: float = 0.35           # ARQVIO
    floor_pf: float = 1.1                # ARQVIO
    floor_trades: int = 30               # ARQVIO

    def block(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["exec_modes"] = list(self.exec_modes)
        return d


PRESETS: dict[str, dict[str, Any]] = {
    "research": {},
    # a short run for tests and a first look; not the research numbers
    "quick": {"proving_grounds": 200, "generations": 4, "tribe_size": 16, "elite": 2, "finalists": 3,
              "stagnation_after": 2, "random_n": 30, "shuffle_n": 300},
}


def v1_config(preset: str = "research", **overrides: Any) -> V1Config:
    if preset not in PRESETS:
        raise ValueError(f"unknown V1 preset {preset!r}; presets are {sorted(PRESETS)}")
    cfg = V1Config(preset=preset, **{**PRESETS[preset], **overrides})
    problems = []
    if cfg.tribe_size * 2 > cfg.proving_grounds:
        problems.append("proving_grounds must hold two tribes")
    if not 0 < cfg.elite < cfg.tribe_size:
        problems.append("elite must be in (0, tribe_size)")
    if not 0.0 < cfg.risk_pct <= 0.05:
        problems.append("risk_pct must be in (0, 5 %]")
    if problems:
        raise ValueError("; ".join(problems))
    return cfg


def v1_gene_pool() -> GenePool:
    """Research signal families only (registry breedable_families("v1")), equal weights, the V1 switch gates."""
    allowed = set(registry.breedable_families(ENGINE))
    # the owner's switch choices (BREEDING tab) apply to V1 too; everything else stays research's
    from engine.research.search.gene_pool import GenePoolError
    from engine.research.search.gene_pool import load as load_gene_pool

    try:
        off = list(load_gene_pool().off)
    except (GenePoolError, OSError, ValueError):
        off = []
    return canonicalise({
        "families": {f: {"enabled": f in allowed, "weight": 1.0} for f in SIGNAL_VALUES},
        "genes": {},
        "gates": {g: True for g in V1_GATES},
        **({"off": off} if off else {}),
    })


__all__ = ["ENGINE", "PRESETS", "V1Config", "V1_GATES", "v1_config", "v1_gene_pool"]
