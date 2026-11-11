"""GA configuration: the ``ga`` block hashed into a GA campaign's config.

Every knob that changes what the GA draws or keeps is in the block, with
its value written out -- never left to a default a reader has to know. The
block joins run_campaign_cli.campaign_config_for's configuration, so
config_hash separates two GA runs that differ in any of them, and a random
search (no ``ga`` key) keeps the hash it always had.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from engine.core.errors import ConfigError
from engine.research.registry.cells import PinnedCellsView, cross_of
from engine.research.search.gene_pool import GenePool

GA_CONFIG_VERSION = 1

# The pinned cells, name -> (symbol, timeframe, dataset_id): a read-only view of
# the VALIDATED entries of the canonical cell registry (configs/cells_registry.json,
# DR-2026-10-07-u1), resolved at every access. A GA runs on one and re-tests its
# finalists on that entry's cross_cell.
PINNED_CELLS: Mapping[str, tuple[str, str, str]] = PinnedCellsView()


def cross_cell_for(cell: str) -> str:
    """The cell's registry cross_cell: where a GA on ``cell`` re-tests its finalists."""
    if cell not in PINNED_CELLS:
        raise GAConfigError([f"unknown cell {cell!r}; pinned cells are {sorted(PINNED_CELLS)}"])
    return cross_of(cell)

# The operator semantics the block stands for. A change to any of them is a
# change of experiment, so the text travels in the hash.
RULES: dict[str, str] = {
    "selection": "tournament of tournament_k drawn with replacement; ties -> lower dna_hash",
    "crossover": (
        "child takes signal from parent A; parents of different families -> A's family "
        "genes wholesale; same family -> family genes uniform per gene with the "
        "(macd_fast, macd_slow) pair as one gene; common genes uniform per gene"
    ),
    "mutation": (
        "per engine-reaching gene with probability mutation_p: blend toward a draw from "
        "the sampler's own range, v + mutation_scale * (draw - v); an integer blend that "
        "rounds back to v takes one step toward the draw; pool-fixed genes never mutate"
    ),
    "dedupe": (
        "effective_gene_hash unique within the generation and against every DNA this run "
        "evaluated; a collision is replaced by the next sampler draw"
    ),
    "fitness": (
        "crash, trades < min_trades or sharpe <= sharpe_min -> 0; else sharpe, "
        "+ 0.25 * sharpe if dsr >= dsr_min, + 0.25 * sham_margin if sham_margin >= "
        "sham_margin_min"
    ),
    "elitism": "the elitism fittest of a generation are copied verbatim, not re-evaluated",
}


class GAConfigError(ConfigError):
    """A GA configuration that cannot be run. ``problems`` lists every reason."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = list(problems)


@dataclass(frozen=True)
class Gates:
    """The fitness gates. No defaults: a campaign states its own."""

    min_trades: int
    sharpe_min: float
    dsr_min: float
    sham_margin_min: float


# The gates the GA CLI runs with, each an existing repo value, none invented:
#   min_trades 30  -- run_cell's eligibility bar (run_campaign_cli.campaign_config_for,
#                     runner.run_campaign's default min_trades);
#   dsr_min 0.5    -- run_campaign's d_sharpe_threshold, the FailedDSR line;
#   sharpe_min 0.0 and sham_margin_min 0.0 -- the no-edge line: a candidate
#                     must make money and must beat its own sham.
CAMPAIGN_GATES = Gates(min_trades=30, sharpe_min=0.0, dsr_min=0.5, sham_margin_min=0.0)


@dataclass(frozen=True)
class Selection:
    tournament_k: int = 3
    elitism: int = 10
    crossover_p: float = 0.7
    mutation_p: float = 0.1
    mutation_scale: float = 1.0


@dataclass(frozen=True)
class IslandConfig:
    """The island model's knobs (docs/decisions/DR-2026-10-01-ga-island-model.md).

    Present only when islands are enabled; GAConfig.islands None is the S4
    Stage 2 core, unchanged and hashed unchanged.
    """

    k: int
    migration_every: int = 2
    migrants: int = 2
    cryo_size: int = 10
    defrost_after: int = 3
    revolution_after: int = 3
    revolution_fraction: float = 0.25
    war_every: int = 4

    def block(self) -> dict[str, Any]:
        return {
            "k": int(self.k),
            "migration_every": int(self.migration_every),
            "migrants": int(self.migrants),
            "cryo_size": int(self.cryo_size),
            "defrost_after": int(self.defrost_after),
            "revolution_after": int(self.revolution_after),
            "revolution_fraction": float(self.revolution_fraction),
            "war_every": int(self.war_every),
            "rules": dict(ISLAND_RULES),
        }


ISLAND_RULES: dict[str, str] = {
    "split": "population split into k islands in admission order, sizes differing by at most one",
    "breeding": "each island breeds alone with the core operators; elitism max(1, elitism // k)",
    "migration": (
        "every migration_every generations, ring i -> i+1: migrants chosen by seeded tournaments "
        "replace the destination's worst; copies, not trials"
    ),
    "cryo": "the cryo_size fittest survivors ever, never bred, never culled",
    "defrost": (
        "global best unimproved for defrost_after generations -> cryo copies (best first, at most "
        "migrants) replace the weakest island's worst; counter resets"
    ),
    "revolution": (
        "island best unimproved for revolution_after generations -> ceil(revolution_fraction * size) "
        "worst replaced by fresh sampler draws (new trials); counter resets"
    ),
    "war": "every war_every generations the weakest island (lowest mean fitness) keeps its best, the rest re-seeded fresh",
    "order": "breed, evaluate, cryo, migration, revolutions, war, defrost",
}


@dataclass(frozen=True)
class GAConfig:
    """A validated GA configuration. Build it with ``ga_config``."""

    seed: int
    gates: Gates
    gene_pool: GenePool
    population: int = 200
    generations: int = 25
    trials_budget: int = 8000
    kernel_share_min: float = 0.95
    selection: Selection = field(default_factory=Selection)
    final_n: int = 20
    islands: IslandConfig | None = None

    def block(self) -> dict[str, Any]:
        """The hashed ``ga`` block: plain JSON values only."""
        # Numbers are normalised to their kind, so an int 0 and a float 0.0
        # for the same knob hash as the same experiment.
        return {
            "version": GA_CONFIG_VERSION,
            "population": int(self.population),
            "generations": int(self.generations),
            "seed": int(self.seed),
            "trials_budget": int(self.trials_budget),
            "kernel_share_min": float(self.kernel_share_min),
            # From the gene pool, and linked to it by hash: the families the
            # sampler may draw are the pool's, never a second copy to drift.
            "families": {
                f: dict(entry) for f, entry in self.gene_pool.families.items()
            },
            "gene_pool_hash": self.gene_pool.hash,
            "gates": {
                "min_trades": int(self.gates.min_trades),
                "sharpe_min": float(self.gates.sharpe_min),
                "dsr_min": float(self.gates.dsr_min),
                "sham_margin_min": float(self.gates.sham_margin_min),
            },
            "selection": {
                "tournament_k": int(self.selection.tournament_k),
                "elitism": int(self.selection.elitism),
                "crossover_p": float(self.selection.crossover_p),
                "mutation_p": float(self.selection.mutation_p),
                "mutation_scale": float(self.selection.mutation_scale),
            },
            "final_n": int(self.final_n),
            "rules": dict(RULES),
            # Absent when disabled, so every pre-island config hashes unchanged.
            **({"islands": self.islands.block()} if self.islands is not None else {}),
        }


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _problems(cfg: GAConfig) -> list[str]:
    problems: list[str] = []
    for name in ("population", "generations", "trials_budget", "final_n"):
        value = getattr(cfg, name)
        if not _is_int(value) or value <= 0:
            problems.append(f"{name} must be an int > 0, got {value!r}")
    if not _is_int(cfg.seed) or cfg.seed < 0:
        problems.append(f"seed must be an int >= 0 and is recorded in the block, got {cfg.seed!r}")
    if not _is_number(cfg.kernel_share_min) or not 0.0 <= cfg.kernel_share_min <= 1.0:
        problems.append(f"kernel_share_min must be in [0, 1], got {cfg.kernel_share_min!r}")
    if _is_int(cfg.population) and _is_int(cfg.trials_budget) and cfg.trials_budget < cfg.population:
        problems.append(
            f"trials_budget {cfg.trials_budget} cannot evaluate one population of {cfg.population}"
        )
    g = cfg.gates
    if not _is_int(g.min_trades) or g.min_trades < 0:
        problems.append(f"gates.min_trades must be an int >= 0, got {g.min_trades!r}")
    for name in ("sharpe_min", "dsr_min", "sham_margin_min"):
        if not _is_number(getattr(g, name)):
            problems.append(f"gates.{name} must be a finite number, got {getattr(g, name)!r}")
    s = cfg.selection
    if not _is_int(s.tournament_k) or s.tournament_k < 1:
        problems.append(f"selection.tournament_k must be an int >= 1, got {s.tournament_k!r}")
    if not _is_int(s.elitism) or s.elitism < 0:
        problems.append(f"selection.elitism must be an int >= 0, got {s.elitism!r}")
    elif _is_int(cfg.population) and s.elitism >= cfg.population:
        problems.append(
            f"selection.elitism {s.elitism} must leave room for offspring in a "
            f"population of {cfg.population}"
        )
    for name in ("crossover_p", "mutation_p"):
        value = getattr(s, name)
        if not _is_number(value) or not 0.0 <= value <= 1.0:
            problems.append(f"selection.{name} must be in [0, 1], got {value!r}")
    if not _is_number(s.mutation_scale) or not 0.0 < s.mutation_scale <= 1.0:
        problems.append(f"selection.mutation_scale must be in (0, 1], got {s.mutation_scale!r}")
    if not isinstance(cfg.gene_pool, GenePool):
        problems.append("gene_pool must be a canonical GenePool (gene_pool.canonicalise)")
    if cfg.islands is not None:
        problems += _island_problems(cfg, cfg.islands)
    return problems


def _island_problems(cfg: GAConfig, isl: IslandConfig) -> list[str]:
    problems: list[str] = []
    if not _is_int(isl.k) or isl.k < 2:
        problems.append(f"islands.k must be an int >= 2, got {isl.k!r}")
    elif _is_int(cfg.population) and cfg.population < 2 * isl.k:
        problems.append(f"islands.k {isl.k} needs a population of at least {2 * isl.k}, got {cfg.population}")
    for name in ("migration_every", "defrost_after", "revolution_after", "war_every"):
        value = getattr(isl, name)
        if not _is_int(value) or value < 1:
            problems.append(f"islands.{name} must be an int >= 1, got {value!r}")
    for name in ("migrants", "cryo_size"):
        value = getattr(isl, name)
        if not _is_int(value) or value < 0:
            problems.append(f"islands.{name} must be an int >= 0, got {value!r}")
    if not _is_number(isl.revolution_fraction) or not 0.0 < isl.revolution_fraction <= 1.0:
        problems.append(f"islands.revolution_fraction must be in (0, 1], got {isl.revolution_fraction!r}")
    return problems


def ga_config(**kwargs: Any) -> GAConfig:
    """Build and validate a GAConfig; GAConfigError carries every problem found."""
    cfg = GAConfig(**kwargs)
    problems = _problems(cfg)
    if problems:
        raise GAConfigError(problems)
    return cfg


__all__ = [
    "CAMPAIGN_GATES",
    "GA_CONFIG_VERSION",
    "GAConfig",
    "GAConfigError",
    "Gates",
    "ISLAND_RULES",
    "IslandConfig",
    "PINNED_CELLS",
    "RULES",
    "Selection",
    "cross_cell_for",
    "ga_config",
]
