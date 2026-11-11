"""The island model: K islands, migration, cryo, revolution, war.

docs/decisions/DR-2026-10-01-ga-island-model.md, implemented. Engaged only
when GAConfig.islands is set; with it None the runner never calls this module
and the S4 Stage 2 core runs unchanged.

Every mechanism is a pure function of (state, rng). The one generator seeded
from the config drives every draw, in a fixed island order. Kill gates,
fitness, dedupe (one run registry for all islands), shams, DSR, finalists and
the abort rules are the core's, reused rather than copied.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from engine.research.ga.config import GAConfig, IslandConfig
from engine.research.ga.fitness import candidate_dsr, dsr_trials
from engine.research.ga.operators import rank_key, tournament
from engine.research.ga.population import DedupeExhausted, Individual, Registry, fresh_draw, initial_population
from engine.research.search.effective_hash import count_effective_trials


def split(population: list[Individual], k: int) -> list[list[Individual]]:
    """k contiguous islands in admission order, sizes differing by at most one."""
    n = len(population)
    sizes = [n // k + (1 if i < n % k else 0) for i in range(k)]
    out, start = [], 0
    for size in sizes:
        out.append(population[start:start + size])
        start += size
    return out


def _mean_fitness(island: list[Individual]) -> float:
    return float(np.mean([i.fitness.fitness for i in island])) if island else float("-inf")


def weakest(islands: list[list[Individual]]) -> int:
    """The island with the lowest mean fitness; ties -> the lower index."""
    means = [_mean_fitness(isl) for isl in islands]
    return min(range(len(islands)), key=lambda i: (means[i], i))


def update_cryo(cryo: list[Individual], new: list[Individual], size: int) -> list[Individual]:
    """The ``size`` fittest survivors ever seen; members are never bred or culled."""
    pool = {i.effective_hash: i for i in cryo}
    for ind in new:
        if ind.fitness.alive:
            pool.setdefault(ind.effective_hash, ind)
    return sorted(pool.values(), key=rank_key)[:size]


def _replace_worst(island: list[Individual], incoming: list[Individual]) -> list[Individual]:
    """Swap the island's worst members (rank_key) for ``incoming``, skipping any already present."""
    present = {i.effective_hash for i in island}
    incoming = [i for i in incoming if i.effective_hash not in present][: len(island)]
    if not incoming:
        return island
    keep = sorted(island, key=rank_key)[: len(island) - len(incoming)]
    kept = {i.effective_hash for i in keep}
    return [i for i in island if i.effective_hash in kept] + incoming


def migrate(rng: np.random.Generator, islands: list[list[Individual]], isl: IslandConfig,
            tournament_k: int) -> tuple[list[list[Individual]], int]:
    """Ring i -> i+1: ``migrants`` picks by seeded tournaments, chosen before any
    island changes. Returns the new islands and how many individuals moved."""
    picks: list[list[Individual]] = []
    for island in islands:
        chosen: dict[str, Individual] = {}
        for _ in range(isl.migrants):
            winner = tournament(rng, island, tournament_k)
            chosen.setdefault(winner.effective_hash, winner)
        picks.append(list(chosen.values()))
    k = len(islands)
    out, moved = [], 0
    for i in range(k):
        before = {x.effective_hash for x in islands[i]}
        dest = _replace_worst(islands[i], picks[(i - 1) % k])
        moved += sum(1 for x in dest if x.effective_hash not in before)
        out.append(dest)
    return out, moved


def island_search(
    cfg: GAConfig,
    ev,
    *,
    on_generation: Callable[[dict[str, Any]], None] | None = None,
    on_evaluated: Callable[[list[Individual], int], None] | None = None,
):
    """The generation loop with islands. Returns the core's SearchResult."""
    from engine.research.ga.runner import (
        STOP_ABORTED,
        STOP_BUDGET,
        STOP_COMPLETED,
        SearchResult,
        _finalists,
        _generation_log,
        _score_new,
        breed,
    )

    isl = cfg.islands
    rng = np.random.default_rng(cfg.seed)
    registry = Registry()
    archive: list[Individual] = []
    evaluations: dict[str, Any] = {}
    generations: list[dict[str, Any]] = []
    stop, aborted, trials_used = STOP_COMPLETED, None, 0
    islands: list[list[Individual]] = []
    cryo: list[Individual] = []
    island_best = [float("-inf")] * isl.k
    island_stag = [0] * isl.k
    global_best, global_stag = float("-inf"), 0
    keep: set[str] = set()
    elitism = max(1, cfg.selection.elitism // isl.k)

    def run_batch(batch: list[Individual], g: int) -> tuple[int, dict[str, str]]:
        nonlocal trials_used
        for ind in batch:
            e = ev.evaluate(ind.dna)
            evaluations[ind.effective_hash] = e
            ind.result, ind.engine = e.result, e.engine
        trials_used += len(batch)
        archive.extend(batch)
        n_trials, errors = _score_new(batch, archive, cfg, ev, evaluations)
        if on_evaluated is not None:
            on_evaluated(batch, g)
        return n_trials, errors

    def reseed(island: list[Individual], n: int, g: int, label: str) -> tuple[list[Individual], list[Individual]]:
        n = min(n, max(0, cfg.trials_budget - trials_used))
        if n <= 0:
            return island, []
        fresh = [fresh_draw(rng, cfg.gene_pool, registry, generation=g, operators=(label,)) for _ in range(n)]
        return _replace_worst(island, fresh), fresh

    for g in range(cfg.generations):
        t0 = time.perf_counter()
        collisions_before = registry.collisions
        events: dict[str, Any] = {"migrated": 0, "revolutions": [], "war": None, "defrosted": 0, "budget_skips": 0}
        try:
            if g == 0:
                new = initial_population(rng, cfg.gene_pool, cfg.population, registry)
                elites_n = 0
            else:
                n_new = sum(len(i) - min(elitism, len(i) - 1) for i in islands)
                if trials_used + n_new > cfg.trials_budget:
                    stop = STOP_BUDGET
                    break
                new, next_islands, elites_n = [], [], 0
                for island in islands:
                    e = min(elitism, len(island) - 1)
                    elites = sorted(island, key=rank_key)[:e]
                    children = breed(rng, island, len(island) - e, cfg, registry, g)
                    next_islands.append(elites + children)
                    new += children
                    elites_n += e
                islands = next_islands
        except DedupeExhausted as exc:
            stop, aborted = STOP_ABORTED, {"generation": g, "reason": f"dedupe exhausted: {exc}"}
            break
        t1 = time.perf_counter()
        n_trials, sham_errors = run_batch(new, g)
        if g == 0:
            islands = split(new, isl.k)
        gen_new = list(new)
        cryo = update_cryo(cryo, new, isl.cryo_size)
        if g > 0 and g % isl.migration_every == 0 and isl.migrants > 0:
            islands, events["migrated"] = migrate(rng, islands, isl, cfg.selection.tournament_k)
        for i, island in enumerate(islands):
            best = max(x.fitness.fitness for x in island)
            if best > island_best[i]:
                island_best[i], island_stag[i] = best, 0
            else:
                island_stag[i] += 1
            if island_stag[i] >= isl.revolution_after:
                islands[i], fresh = reseed(island, math.ceil(isl.revolution_fraction * len(island)), g, "revolution")
                if fresh:
                    n_trials, errs = run_batch(fresh, g)
                    sham_errors.update(errs)
                    gen_new += fresh
                    events["revolutions"].append({"island": i, "reseeded": len(fresh)})
                else:
                    events["budget_skips"] += 1
                island_stag[i] = 0
        if g > 0 and g % isl.war_every == 0:
            w = weakest(islands)
            islands[w], fresh = reseed(islands[w], len(islands[w]) - 1, g, "war")
            if fresh:
                n_trials, errs = run_batch(fresh, g)
                sham_errors.update(errs)
                gen_new += fresh
                cryo = update_cryo(cryo, fresh, isl.cryo_size)
            events["war"] = {"island": w, "reseeded": len(fresh)}
        pop_best = max(x.fitness.fitness for island in islands for x in island)
        if pop_best > global_best:
            global_best, global_stag = pop_best, 0
        else:
            global_stag += 1
        if global_stag >= isl.defrost_after and cryo:
            w = weakest(islands)
            before = {x.effective_hash for x in islands[w]}
            islands[w] = _replace_worst(islands[w], cryo[: max(1, isl.migrants)])
            events["defrosted"] = sum(1 for x in islands[w] if x.effective_hash not in before)
            global_stag = 0
        t2 = time.perf_counter()
        pop = [x for island in islands for x in island]
        log = _generation_log(
            g, pop, gen_new, elites_n, registry.collisions - collisions_before, trials_used, cfg, n_trials,
            {"breed": t1 - t0, "evaluate_and_score": t2 - t1},
        )
        log["sham_errors"] = sham_errors
        log["islands"] = {
            "per_island": [
                {"size": len(island), "best": max(x.fitness.fitness for x in island),
                 "mean": _mean_fitness(island), "alive": sum(1 for x in island if x.fitness.alive),
                 "stagnation": island_stag[i]}
                for i, island in enumerate(islands)
            ],
            "cryo_size": len(cryo),
            "global_stagnation": global_stag,
            **events,
        }
        generations.append(log)
        if on_generation is not None:
            on_generation(log)
        if log["crashes"]:
            stop, aborted = STOP_ABORTED, {"generation": g, "reason": f"{log['crashes']} crash(es); crash rate must be 0"}
        elif not log["kernel_share_ok"]:
            stop, aborted = STOP_ABORTED, {
                "generation": g, "reason": f"generation INVALID: kernel share {log['kernel_share']} "
                f"< kernel_share_min {cfg.kernel_share_min}; engines {log['engines']}",
            }
        excluded = aborted["generation"] if aborted is not None else None
        hof = {i.effective_hash for i in _finalists(archive, cfg.final_n, excluded)}
        for ind in [*gen_new, *(i for i in archive if i.effective_hash in keep - hof)]:
            if ind.effective_hash not in hof and ind.result is not None:
                ind.result.per_period_returns = None
        keep = hof
        if aborted is not None:
            break
    excluded = aborted["generation"] if aborted is not None else None
    finalists = _finalists(archive, cfg.final_n, excluded)
    n_final, sharpes = dsr_trials(
        [i.result.sharpe for i in archive], count_effective_trials(i.dna for i in archive),
    )
    final_dsr = {
        i.effective_hash: candidate_dsr(i.result, n_trials=n_final, trial_sharpes=sharpes,
                                        symbol=ev.symbol, timeframe=ev.timeframe)
        for i in finalists
    }
    return SearchResult(archive, generations, finalists, final_dsr, n_final, stop, aborted,
                        trials_used, registry.collisions, excluded)


__all__ = ["island_search", "migrate", "split", "update_cryo", "weakest"]
