"""GA runner: the generation loop, its budgets, the cross-cell pass, the outputs.

    python -m engine.research.ga.runner --cell usdcadm_h1 --pop 50 --gen 5 --seed 7 --budget 400

One run searches a pinned cell and re-tests its finalists on the other one:

  1. preflight each cell the way run_cell does -- spread gate, target audit,
     time-basis check, the pinned snapshot and its depth, the planted-edge
     canary on the cell's own bars -- and build its backtest once
     (kernel first: engine_mode "auto" is "fast" on a pinned dataset);
  2. generation 0 from the sampler, then per generation: elites copied
     verbatim, offspring by tournament -> crossover -> mutation under dedupe,
     every new individual backtested, gated and scored (fitness.py), the
     funnel, kernel share and degeneracy census logged;
  3. a generation with any crash, or with kernel share below
     kernel_share_min, aborts the run and says so; the trials budget stops it
     before a generation it cannot pay for;
  4. the final_n fittest SURVIVORS are re-run on the other pinned cell under
     the identical gates and sham controls, and written as manifest v2.

The search is deterministic: one generator seeded from the config drives
every draw, backtests and shams re-seed per candidate, DSR inputs are
sorted. Nothing here is a verdict: a run with no survivor, or none that
survives the second cell, is reported as exactly that.
"""
from __future__ import annotations

import hashlib
import statistics
import sys
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from engine.core.errors import ConfigError
from engine.research.campaign.manifest import Manifest, config_hash, dataset_provenance_for
from engine.research.ga import manifest as mv2
from engine.research.ga.config import (
    CAMPAIGN_GATES,
    GA_CONFIG_VERSION,
    PINNED_CELLS,
    GAConfig,
    IslandConfig,
    Selection,
    cross_cell_for,
    ga_config,
)
from engine.research.ga.determinism import individuals_table, run_hash
from engine.research.ga.fitness import (
    KILLED_CRASH,
    KILLED_MIN_TRADES,
    KILLED_SHARPE_MIN,
    DSRReading,
    FitnessRecord,
    candidate_dsr,
    dsr_trials,
    score,
)
from engine.research.ga.operators import crossover, mutate, rank_key, tournament
from engine.research.ga.parallel import POWER_MODES, workers_for
from engine.research.ga.population import (
    DedupeExhausted,
    Individual,
    Registry,
    checked,
    fresh_draw,
    initial_population,
)
from engine.research.search.degeneracy import cost_totals, degeneracy_report
from engine.research.search import live_view
from engine.research.search.dna import DNA
from engine.research.search.effective_hash import count_effective_trials
from engine.research.search.runner import CrashRecord, TrialResult, evaluate_one

ENGINE_KERNEL = "kernel"
ENGINE_SLOW_FALLBACK = "slow_fallback"
ENGINE_SLOW = "slow"
ENGINE_REFUSED = "refused"
# Census outcome labels: the GA's own gates, never a funnel verdict such as
# VAULTED (the census then reads the top quartile of eligible candidates).
OUTCOME_LABELS = {KILLED_CRASH: "Crash", KILLED_MIN_TRADES: "LowTrades", KILLED_SHARPE_MIN: "LowSharpe", None: "ALIVE"}
STOP_COMPLETED = "completed"
STOP_BUDGET = "trials_budget"
STOP_ABORTED = "aborted"


class GARunError(RuntimeError):
    """A GA run that cannot start: a cell refused at preflight."""


@dataclass(frozen=True)
class Evaluation:
    result: TrialResult
    crashed: bool
    engine: str
    crash: CrashRecord | None = None


class Evaluator(Protocol):
    name: str
    symbol: str
    timeframe: str
    engine_mode_actual: str

    def evaluate(self, dna: DNA) -> Evaluation: ...

    def sham_sharpe(self, result: TrialResult) -> float: ...


@dataclass
class CellEvaluator:
    """A prepared pinned cell: its backtest (built once), its sham, its record."""

    name: str
    symbol: str
    timeframe: str
    dataset_id: str
    backtest_fn: Callable[[DNA], TrialResult]
    sham_fn: Callable[[DNA, int], TrialResult]
    engine_mode_actual: str
    context: dict[str, Any]
    # (dna_hash, entries requested) -> the sham's Sharpe and fills; the sham's
    # per-period returns are not kept (one slow run each, ~60k bars).
    shams: dict[tuple[str, int], dict[str, Any]] = field(default_factory=dict)

    def evaluate(self, dna: DNA) -> Evaluation:
        """runner.evaluate_one (lookahead refusal, crash capture), labelled by
        engine from the fast factory's fallback counter."""
        state = getattr(self.backtest_fn, "fallback_state", None)
        before = None if state is None else int(state["fallbacks"])
        result, crash = evaluate_one(dna, backtest_fn=self.backtest_fn)
        if crash is not None and crash.exception_type == "LookaheadDetected":
            engine = ENGINE_REFUSED
        elif state is None:
            engine = ENGINE_SLOW
        elif int(state["fallbacks"]) > before:
            engine = ENGINE_SLOW_FALLBACK
        else:
            engine = ENGINE_KERNEL
        return Evaluation(result, crash is not None, engine, crash)

    def sham_hit(self, dna: DNA, entries: int) -> dict[str, Any]:
        """One sham run for ``dna`` at ``entries`` requested entries: its Sharpe and fills."""
        sham = self.sham_fn(dna, entries)
        # Fewer eligible bars than requested entries: sham_backtest_factory
        # draws them all and records (requested, drawn) in ``truncated``.
        drawn = getattr(self.sham_fn, "truncated", {}).get(dna.dna_hash, (entries, entries))[1]
        return {"sharpe": float(sham.sharpe), "requested": entries, "drawn": int(drawn),
                "filled": int(sham.n_trades or 0), "truncated": int(drawn) < entries}

    def sham_sharpe(self, result: TrialResult) -> float:
        from engine.research.search.sham_control import attempted_entries

        key = (result.dna.dna_hash, attempted_entries(result))
        hit = self.shams.get(key)
        if hit is None:
            hit = self.shams[key] = self.sham_hit(result.dna, key[1])
        return float(hit["sharpe"])

    def sham_fills(self, result: TrialResult) -> dict[str, Any] | None:
        from engine.research.search.sham_control import attempted_entries

        return self.shams.get((result.dna.dna_hash, attempted_entries(result)))


@dataclass
class SearchResult:
    archive: list[Individual]
    generations: list[dict[str, Any]]
    finalists: list[Individual]
    final_dsr: dict[str, DSRReading]
    dsr_n_final: int
    stop_reason: str
    aborted: dict[str, Any] | None
    trials_used: int
    collisions: int
    # The generation that aborted the run, if one did: never a finalist source.
    excluded_generation: int | None = None


def kernel_share(engines: Counter) -> float | None:
    """Kernel runs over runs that reached a backtest; None if none did."""
    ran = engines[ENGINE_KERNEL] + engines[ENGINE_SLOW_FALLBACK] + engines[ENGINE_SLOW]
    return engines[ENGINE_KERNEL] / ran if ran else None


def _outcome(ind: Individual) -> str:
    return OUTCOME_LABELS[ind.fitness.killed_by]


def breed(
    rng: np.random.Generator, parents: list[Individual], n: int, cfg: GAConfig,
    registry: Registry, generation: int,
) -> list[Individual]:
    """``n`` offspring: tournament -> crossover (p) -> mutation, then dedupe.

    A child whose effective hash the run has already evaluated (a clone, a
    neutral mutation, a repeat) is replaced by the next sampler draw.
    """
    sel: Selection = cfg.selection
    children: list[Individual] = []
    while len(children) < n:
        a = tournament(rng, parents, sel.tournament_k)
        b = tournament(rng, parents, sel.tournament_k)
        # A parent crossed with itself is a clone, and is recorded as one.
        if a is not b and rng.random() < sel.crossover_p:
            dna, ops, lineage = crossover(rng, a.dna, b.dna), ["crossover"], (a.effective_hash, b.effective_hash)
        else:
            dna, ops, lineage = a.dna, ["clone"], (a.effective_hash,)
        dna, changed = mutate(rng, dna, cfg.gene_pool, p=sel.mutation_p, scale=sel.mutation_scale)
        if changed:
            ops.append("mutate:" + ",".join(changed))
        eh = checked(dna)
        if registry.admit(eh):
            children.append(Individual(dna=dna, effective_hash=eh, generation=generation,
                                       parents=lineage, operators=tuple(ops)))
        else:
            children.append(fresh_draw(rng, cfg.gene_pool, registry, generation=generation,
                                       operators=("resample",)))
    return children


def _gated(
    result: TrialResult, *, crashed: bool, cfg: GAConfig,
    dsr_of: Callable[[TrialResult], DSRReading], sham_of: Callable[[TrialResult], float],
) -> tuple[FitnessRecord, str | None]:
    """score(), with a sham that raises recorded as a crash of its candidate.

    The sham is a slow-engine run; an exception there is a crash like any
    other -- counted, and aborting the run -- not a traceback that loses it.
    """
    try:
        return score(result, crashed=crashed, gates=cfg.gates, dsr_of=dsr_of, sham_of=sham_of), None
    except Exception as exc:
        sharpe = None if result.sharpe is None else float(result.sharpe)
        return FitnessRecord(0.0, KILLED_CRASH, int(result.n_trades or 0), sharpe), f"{type(exc).__name__}: {exc}"


def _score_new(
    new: list[Individual], archive: list[Individual], cfg: GAConfig, ev: Evaluator,
    evaluations: dict[str, Evaluation],
) -> tuple[int, dict[str, str]]:
    """Gate and score this generation's new individuals.

    The luck bar is set by every distinct backtest the run has done so far,
    this generation included. Returns the DSR N used and, per dna_hash, the
    error of every sham that raised.
    """
    n_trials, sharpes = dsr_trials(
        [i.result.sharpe for i in archive], count_effective_trials(i.dna for i in archive),
    )

    def dsr_of(r: TrialResult) -> DSRReading:
        return candidate_dsr(r, n_trials=n_trials, trial_sharpes=sharpes, symbol=ev.symbol, timeframe=ev.timeframe)

    sham_errors: dict[str, str] = {}
    for ind in new:
        ind.fitness, error = _gated(ind.result, crashed=evaluations[ind.effective_hash].crashed,
                                    cfg=cfg, dsr_of=dsr_of, sham_of=ev.sham_sharpe)
        if error is not None:
            sham_errors[ind.dna_hash] = error
    return n_trials, sham_errors


def _generation_log(
    g: int, pop: list[Individual], new: list[Individual], n_elites: int, collisions: int,
    trials_used: int, cfg: GAConfig, n_trials: int, walls: dict[str, float],
) -> dict[str, Any]:
    engines = Counter(i.engine for i in new)
    share = kernel_share(engines)
    outcomes = [_outcome(i) for i in new]
    fitnesses = [float(i.fitness.fitness) for i in pop]
    best = min(pop, key=rank_key)
    return {
        "generation": g,
        "n_new": len(new),
        "n_elites": n_elites,
        "collisions": collisions,
        "trials_used": trials_used,
        "trials_budget": cfg.trials_budget,
        "engines": dict(sorted(engines.items())),
        "kernel_share": share,
        "kernel_share_min": cfg.kernel_share_min,
        "kernel_share_ok": share is not None and share >= cfg.kernel_share_min,
        "crashes": outcomes.count("Crash"),
        "funnel": {
            **{label: outcomes.count(label) for label in ("Crash", "LowTrades", "LowSharpe", "ALIVE")},
            "dsr_passed": sum(1 for i in new if i.fitness.dsr_passed),
            "sham_passed": sum(1 for i in new if i.fitness.sham_passed),
        },
        "population_fitness": {
            "best": max(fitnesses), "median": statistics.median(fitnesses),
            "mean": statistics.fmean(fitnesses), "alive": sum(1 for i in pop if i.fitness.alive),
        },
        "best": {"dna_hash": best.dna_hash, "signal": best.dna.signal, "fitness": best.fitness.fitness},
        "dsr_n_trials": n_trials,
        # Over this generation's new trials: trials_raw counts every DNA
        # proposed, collisions included, so dedupe_hits is the collision count.
        "census": degeneracy_report(
            [i.result for i in new], outcomes, min_trades=cfg.gates.min_trades,
            trials_raw=len(new) + collisions, trials_effective=len(new),
        ),
        "wall_seconds": walls,
    }


def search(
    cfg: GAConfig,
    ev: Evaluator,
    *,
    on_generation: Callable[[dict[str, Any]], None] | None = None,
    on_evaluated: Callable[[list[Individual], int], None] | None = None,
    checkpoint: Callable[[dict[str, Any]], None] | None = None,
    resume: dict[str, Any] | None = None,
) -> SearchResult:
    """The generation loop on one cell. Deterministic for a given cfg and cell.

    ``checkpoint`` receives the complete loop state after every generation
    (checkpoint.save writes it atomically); ``resume`` is such a state
    (checkpoint.load) and the loop continues from the generation after it,
    exactly as if it had never stopped. Both None: the loop is unchanged.
    """
    evaluations: dict[str, Evaluation] = {}
    if resume is None:
        rng = np.random.default_rng(cfg.seed)
        registry = Registry()
        archive: list[Individual] = []
        generations: list[dict[str, Any]] = []
        stop, aborted, trials_used = STOP_COMPLETED, None, 0
        pop: list[Individual] = []
        keep: set[str] = set()  # effective hashes whose per-period returns are kept
        start = 0
    else:
        rng = resume["rng"]
        registry = Registry(seen=set(resume["seen"]), collisions=resume["collisions"])
        archive, generations = resume["archive"], resume["generations"]
        stop, aborted, trials_used = resume["stop"], resume["aborted"], resume["trials_used"]
        pop, keep = resume["population"], set(resume["keep"])
        # A run that had already stopped resumes into no further generation.
        start = resume["generation"] + 1 if stop == STOP_COMPLETED and aborted is None else cfg.generations
    for g in range(start, cfg.generations):
        t0 = time.perf_counter()
        collisions_before = registry.collisions
        try:
            if g == 0:
                new, elites = initial_population(rng, cfg.gene_pool, cfg.population, registry), []
            else:
                n_new = cfg.population - cfg.selection.elitism
                if trials_used + n_new > cfg.trials_budget:
                    stop = STOP_BUDGET
                    break
                elites = sorted(pop, key=rank_key)[: cfg.selection.elitism]
                new = breed(rng, pop, n_new, cfg, registry, g)
        except DedupeExhausted as exc:
            stop, aborted = STOP_ABORTED, {"generation": g, "reason": f"dedupe exhausted: {exc}"}
            break
        t1 = time.perf_counter()
        # A power-mode evaluator runs the generation on worker processes, results in input order.
        batch = getattr(ev, "evaluate_batch", None)
        evals = batch([ind.dna for ind in new]) if batch is not None else [ev.evaluate(ind.dna) for ind in new]
        for ind, e in zip(new, evals, strict=True):
            evaluations[ind.effective_hash] = e
            ind.result, ind.engine = e.result, e.engine
        trials_used += len(new)
        archive.extend(new)
        t2 = time.perf_counter()
        prefetch = getattr(ev, "prefetch_shams", None)
        if prefetch is not None:
            prefetch([(ind.result, evaluations[ind.effective_hash].crashed) for ind in new], cfg.gates)
        n_trials, sham_errors = _score_new(new, archive, cfg, ev, evaluations)
        t3 = time.perf_counter()
        pop = elites + new
        if on_evaluated is not None:
            on_evaluated(new, g)
        log = _generation_log(
            g, pop, new, len(elites), registry.collisions - collisions_before, trials_used, cfg, n_trials,
            {"breed": t1 - t0, "evaluate": t2 - t1, "score": t3 - t2},
        )
        log["sham_errors"] = sham_errors
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
        # A generation that aborted the run is not a source of finalists: its
        # trials count (they were run) but its results are not trusted.
        excluded = aborted["generation"] if aborted is not None else None
        # Keep per-period returns only for the current final_n eligible
        # survivors: the finalists' DSR inputs. Everyone else needs a Sharpe.
        hof = {i.effective_hash for i in _finalists(archive, cfg.final_n, excluded)}
        for ind in [*new, *(i for i in archive if i.effective_hash in keep - hof)]:
            if ind.effective_hash not in hof and ind.result is not None:
                ind.result.per_period_returns = None
        keep = hof
        if checkpoint is not None:
            checkpoint({"generation": g, "rng": rng, "registry": registry, "archive": archive, "population": pop,
                        "keep": keep, "generations": generations, "trials_used": trials_used, "stop": stop,
                        "aborted": aborted})
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


def searcher_for(cfg: GAConfig) -> Callable[..., SearchResult]:
    """Islands disabled (the default): the S4 Stage 2 core loop, unchanged."""
    if cfg.islands is None:
        return search
    from engine.research.ga.islands import island_search

    return island_search


def _finalists(archive: list[Individual], n: int, excluded_generation: int | None) -> list[Individual]:
    """The n fittest survivors, none from a generation that aborted the run."""
    eligible = (i for i in archive if i.fitness.alive and i.generation != excluded_generation)
    return sorted(eligible, key=rank_key)[:n]


@dataclass(frozen=True)
class CrossResult:
    evaluation: Evaluation
    fitness: FitnessRecord
    sham_error: str | None = None


def cross_check(finalists: list[Individual], cfg: GAConfig, ev: Evaluator) -> tuple[list[CrossResult], int]:
    """The finalists on the other cell, identical gates and sham controls.

    Its luck bar counts the backtests run THERE: the finalists themselves.
    """
    evaluations = [ev.evaluate(ind.dna) for ind in finalists]
    n_trials, sharpes = dsr_trials([e.result.sharpe for e in evaluations],
                                   count_effective_trials(ind.dna for ind in finalists))

    def dsr_of(r: TrialResult) -> DSRReading:
        return candidate_dsr(r, n_trials=n_trials, trial_sharpes=sharpes, symbol=ev.symbol, timeframe=ev.timeframe)

    out = []
    for e in evaluations:
        record, error = _gated(e.result, crashed=e.crashed, cfg=cfg, dsr_of=dsr_of, sham_of=ev.sham_sharpe)
        out.append(CrossResult(e, record, error))
    return out, n_trials


# --- pinned cells -------------------------------------------------------------

def prepare_cell(name: str, *, engine_mode: str = "auto", flat_cost: bool = False,
                 dataset_id: str | None = None) -> CellEvaluator:
    """Preflight one pinned cell as run_cell does, and build its evaluator. ``dataset_id`` re-uses an earlier
    snapshot of the cell (a vault strategy's own bars after its QUICK cell was refreshed with new data)."""
    from engine.data.spread_calibration import load as load_matrix
    from engine.research.campaign.preflight import refuse_on_basis_mismatch, spread_preflight
    from engine.research.search.canary import run_planted_edge_canary
    from engine.research.search.real_backtest import (
        fast_backtest_factory,
        load_cell_bars_from_snapshot,
        real_backtest_factory,
    )
    from engine.research.search.run_campaign_cli import (
        SpreadBlockedError,
        artifact_path,
        load_target_audit,
        resolve_engine_mode,
        snapshot_depth_check,
    )
    from engine.research.search.sham_control import sham_backtest_factory

    if name not in PINNED_CELLS:
        raise GARunError(f"unknown cell {name!r}; pinned cells are {sorted(PINNED_CELLS)}")
    if engine_mode not in {"auto", "fast", "slow"}:
        raise GARunError(f"engine_mode must be auto, fast or slow, got {engine_mode!r}")
    symbol, timeframe, pinned = PINNED_CELLS[name]
    dataset_id = dataset_id or pinned
    spread_dir = artifact_path("spread")
    spread = spread_preflight(symbol, spread_dir)
    if spread["blocks"]:
        raise SpreadBlockedError(symbol, timeframe, spread)
    header, audit = load_target_audit()
    row = audit.get((symbol, timeframe))
    if row is None:
        raise GARunError(f"{symbol} {timeframe} has no row in the target audit")
    matrix = load_matrix(symbol, spread_dir)
    if matrix is None:
        raise GARunError(f"{symbol}: no spread matrix; the manifest could not name the one costing it")
    basis = (row.get("time_basis", "") or header.get("TIME_BASIS", "")).strip()
    refuse_on_basis_mismatch(matrix, basis, f"{symbol}:{timeframe}")
    cell = load_cell_bars_from_snapshot(symbol, timeframe, dataset_id)
    depth_ok, depth_note = snapshot_depth_check(dataset_id, cell.n_returns)
    if not depth_ok:
        raise GARunError(f"snapshot {dataset_id} too shallow: {depth_note}")
    actual = resolve_engine_mode(engine_mode, dataset_id)
    backtest_fn = (fast_backtest_factory(cell, flat_cost=flat_cost) if actual == "fast"
                   else real_backtest_factory(cell, flat_cost=flat_cost))
    canary = run_planted_edge_canary(cell, fast=True, tmp_dir=None)
    if not canary.passed:
        raise GARunError(f"CANARY FAILED on {symbol} {timeframe}: {canary.reason}")
    context = {
        "cell": name,
        "symbol": symbol,
        "timeframe": timeframe,
        "dataset_id": dataset_id,
        "dataset_hash": cell.dataset_hash,
        "dataset_provenance": dataset_provenance_for(dataset_id),
        "dataset_reproducible": True,
        "bars_loaded": int(cell.n_bars),
        "depth": depth_note,
        "date_range_utc": [
            datetime.fromtimestamp(int(cell.bars.time[0]), tz=UTC).isoformat(),
            datetime.fromtimestamp(int(cell.bars.time[-1]), tz=UTC).isoformat(),
        ],
        # The audit header's values end in the newline load_target_audit keeps.
        # A QUICK cell's row carries its own as_of and source (a fresh app home has no audit CSV header).
        "audit": {
            "as_of": (row.get("as_of") or header.get("AS_OF", "")).strip(),
            "time_basis": basis,
            "source": (header.get("SOURCE") or row.get("source", "")).strip(),
        },
        "spread_check": spread,
        "spread_artifact_sha256": hashlib.sha256((spread_dir / f"{symbol}.json").read_bytes()).hexdigest(),
        "spread_cost_source": getattr(backtest_fn, "spread_cost_source", "none"),
        "cost_vector_hash": getattr(backtest_fn, "cost_vector_hash", ""),
        "engine_mode_configured": engine_mode,
        "engine_mode_actual": actual,
        "canary": {"passed": canary.passed, "verdict": canary.exit_verdict,
                   "dsr_within": canary.dsr_within, "vs_random_p_max": canary.vs_random_p_max,
                   "control": canary.control},
    }
    return CellEvaluator(name=name, symbol=symbol, timeframe=timeframe, dataset_id=dataset_id,
                         backtest_fn=backtest_fn, sham_fn=sham_backtest_factory(cell, flat_cost=flat_cost),
                         engine_mode_actual=actual, context=context)


def prepare_worker_cell(name: str, *, engine_mode_actual: str, flat_cost: bool, dataset_hash: str,
                        cost_vector_hash: str) -> CellEvaluator:
    """A power-mode worker's evaluator: the parent's pinned bars and costs, checked by hash.

    The parent ran the preflight and the canary on this cell (``prepare_cell``); a worker rebuilds only
    the backtest and the sham from the same snapshot and refuses to start if either hash differs.
    """
    from engine.research.search.real_backtest import (
        fast_backtest_factory,
        load_cell_bars_from_snapshot,
        real_backtest_factory,
    )
    from engine.research.search.sham_control import sham_backtest_factory

    symbol, timeframe, dataset_id = PINNED_CELLS[name]
    cell = load_cell_bars_from_snapshot(symbol, timeframe, dataset_id)
    if cell.dataset_hash != dataset_hash:
        raise GARunError(f"worker {name}: dataset hash {cell.dataset_hash} != the parent's {dataset_hash}")
    backtest_fn = (fast_backtest_factory(cell, flat_cost=flat_cost) if engine_mode_actual == "fast"
                   else real_backtest_factory(cell, flat_cost=flat_cost))
    if getattr(backtest_fn, "cost_vector_hash", "") != cost_vector_hash:
        raise GARunError(f"worker {name}: cost vector {getattr(backtest_fn, 'cost_vector_hash', '')} "
                         f"!= the parent's {cost_vector_hash}")
    return CellEvaluator(name=name, symbol=symbol, timeframe=timeframe, dataset_id=dataset_id,
                         backtest_fn=backtest_fn, sham_fn=sham_backtest_factory(cell, flat_cost=flat_cost),
                         engine_mode_actual=engine_mode_actual, context={})


def cache_key(dna: DNA) -> tuple:
    """The fast backtest's indicator-cache key for ``dna`` (power-mode routing affinity only)."""
    from engine.research.backtest.prep import family_key
    from engine.research.search.real_backtest import dna_to_strategy

    cfg, _ = dna_to_strategy(dna)
    if dna.signal == "MA_CROSS":
        return ("MA_CROSS", cfg.fast, cfg.slow)
    return family_key(cfg)


def with_power(ev: CellEvaluator, workers: int, *, flat_cost: bool = False) -> Any:
    """``ev`` itself on one worker; otherwise a PoolEvaluator over ``workers`` processes of the same cell."""
    if workers < 2:
        return ev
    from functools import partial

    from engine.research.ga.parallel import PoolEvaluator

    factory = partial(prepare_worker_cell, ev.name, engine_mode_actual=ev.engine_mode_actual, flat_cost=flat_cost,
                      dataset_hash=ev.context["dataset_hash"], cost_vector_hash=ev.context["cost_vector_hash"])
    return PoolEvaluator(ev, factory, workers, affinity=cache_key)


def cell_manifest(
    ctx: dict[str, Any], *, campaign_id: str, git_sha: str, cfg_hash: str, pool_summary: str, seed: int,
) -> dict[str, Any]:
    """The campaign Manifest for one cell -- the one construction site."""
    from engine.research.provenance import DataSource

    dataset_id = ctx["dataset_id"]
    m = Manifest(
        campaign_id=campaign_id,
        symbol=ctx["symbol"],
        timeframe=ctx["timeframe"],
        dataset_hash=ctx["dataset_hash"],
        dataset_source=DataSource.MT5_LIVE.value,
        dataset_provenance=dataset_provenance_for(dataset_id),
        bars_loaded=int(ctx["bars_loaded"]),
        date_range_utc=tuple(ctx["date_range_utc"]),
        git_sha=git_sha,
        config_hash=cfg_hash,
        gene_pool=pool_summary,
        seed=int(seed),
        completed_at_utc=datetime.now(tz=UTC).isoformat(),
    )
    if m.missing_fields():
        raise GARunError(f"campaign manifest for {ctx['cell']} incomplete: {list(m.missing_fields())}")
    return m.to_dict()


def _cell_metrics(ev_: CellEvaluator, e: Evaluation, record: FitnessRecord) -> dict[str, Any]:
    r = e.result
    return {
        "fitness": record.to_dict(),
        "engine": e.engine,
        "n_trades": r.n_trades,
        "sharpe": r.sharpe,
        "total_return": r.total_return,
        "max_dd": r.max_dd,
        "blocked_max_spread": r.blocked_max_spread,
        "costs": dict(r.costs),
        "sham": ev_.sham_fills(r),
    }


def _sham_fills(ev_: CellEvaluator) -> dict[str, int]:
    """W3's controls totals for this cell's shams: requested, drawn, filled, truncated."""
    hits = list(ev_.shams.values())
    return {
        "shams": len(hits),
        "entries_requested": sum(h["requested"] for h in hits),
        "entries_drawn": sum(h["drawn"] for h in hits),
        "entries_filled": sum(h["filled"] for h in hits),
        "truncated": sum(1 for h in hits if h["truncated"]),
    }


def _registry(path: Path):
    """A fresh TrialRegistry: a stale file would count earlier rows as trials."""
    from engine.validation.trial_registry import TrialRegistry

    for suffix in ("", "-wal", "-shm"):
        stale = path.with_name(path.name + suffix)
        if stale.exists():
            stale.unlink()
            if stale.exists():
                raise GARunError(f"{stale} still present after unlink()")
    path.parent.mkdir(parents=True, exist_ok=True)
    return TrialRegistry(path)


def _record(registry, campaign_id: str, ev: CellEvaluator, individuals: list[Individual], as_of: str) -> None:
    from engine.validation.op_types import OpType

    registry.record_many(
        campaign_id,
        [
            (i.dna_hash, "Crash" if i.fitness.killed_by == KILLED_CRASH else "EVALUATED",
             i.result.sharpe, i.result.n_trades, i.result.total_return, i.result.max_dd)
            for i in individuals
        ],
        symbol=ev.symbol, timeframe=ev.timeframe, op_type=OpType.SEARCH_TRIAL.value,
        dataset_hash=ev.context["dataset_hash"], source="MT5_LIVE", as_of=as_of,
    )


OUTPUT_FILES = ("manifest_v2.json", "individuals.json", "generations.json")


def _record_cross(registry, campaign_id: str, ev: CellEvaluator, crossed: list[CrossResult], as_of: str) -> None:
    """Already-selected finalists re-run on another cell are VALIDATION_RUN
    rows, which the registry keeps out of the search's trial budget (op_types).

    Cross-cell luck bar (Q2, owner-ratified 2026-09-30): N = final_n, under the
    selection-use rule -- only re-tests USED IN A SELECTION decision join the
    multiplicity count. The cross-cell finalists are final_n pre-chosen
    hypotheses re-tested, not chosen there, so their bar is N = final_n
    (cross_check), while the search's bar stays the number of distinct
    backtests the search ran, where selection actually happened.
    """
    from engine.validation.op_types import OpType

    registry.record_many(
        campaign_id,
        [(c.evaluation.result.dna.dna_hash, "Crash" if c.fitness.killed_by == KILLED_CRASH else "EVALUATED",
          c.evaluation.result.sharpe, c.evaluation.result.n_trades, c.evaluation.result.total_return,
          c.evaluation.result.max_dd) for c in crossed],
        symbol=ev.symbol, timeframe=ev.timeframe, op_type=OpType.VALIDATION_RUN.value,
        dataset_hash=ev.context["dataset_hash"], source="MT5_LIVE", as_of=as_of,
    )


def campaign_id_for(cell: str, cfg: GAConfig, cfg_hash: str) -> str:
    """Names the run's directory and registry. It carries the config hash, so
    two differently configured runs can never overwrite each other."""
    return f"ga-{cell}-s{cfg.seed}-{cfg_hash}"


MONITOR_FILE = "monitor_snapshot.json"


def _registry_rows(path: Path, campaign_id: str) -> int:
    import sqlite3

    con = sqlite3.connect(path)
    try:
        return int(con.execute("SELECT COUNT(*) FROM trials WHERE campaign_id = ?", (campaign_id,)).fetchone()[0])
    finally:
        con.close()


def _trim_registry(path: Path, campaign_id: str, keep_rows: int) -> int:
    """Delete the campaign's rows written after the checkpoint (rowid order); returns how many."""
    import sqlite3

    con = sqlite3.connect(path)
    try:
        cur = con.execute(
            "DELETE FROM trials WHERE campaign_id = ? AND rowid NOT IN "
            "(SELECT rowid FROM trials WHERE campaign_id = ? ORDER BY rowid LIMIT ?)",
            (campaign_id, campaign_id, int(keep_rows)),
        )
        con.commit()
        return int(cur.rowcount)
    finally:
        con.close()


def run_pinned(
    cfg: GAConfig, cell: str, *, engine_mode: str = "auto", echo: Callable[[str], None] = print,
    checkpoint_path: Path | None = None, resume_path: Path | None = None,
    hold_after_generation: int | None = None, monitor_every: float | None = None, workers: int = 1,
) -> dict[str, Any]:
    """Search ``cell``, re-test the finalists on the other pinned cell, write
    the individuals table, generation log, manifest v2 and run summary.

    Operational arguments -- outside the hashed config, recorded in the
    run's "operational" block (which the canonical form excludes, like clock
    readings, so they never change a run's identity):
      checkpoint_path        write the loop state atomically after every generation;
      resume_path            continue a checkpointed run of the SAME config exactly;
      hold_after_generation  after writing that generation's checkpoint, wait
                             (for the kill/resume proof) before going on;
      monitor_every          write monitor_snapshot.json at most every N seconds;
      workers                evaluate each generation on this many processes (power mode); the run's
                             results are identical for any value (engine/research/ga/parallel.py).
    """
    from engine.research.ga import checkpoint as ckpt
    from engine.core.git_sha import resolve_git_sha
    from engine.research.search.dna import DNA_SCHEMA_VERSION
    from engine.research.search.effective_hash import EFFECTIVE_HASH_VERSION
    from engine.research.search.gene_pool import summary as pool_summary
    from engine.research.search.run_campaign_cli import CAMPAIGN_TARGET_SHARPE, artifact_path, campaign_config_for
    from engine.research.search.sham_control import CONTROLS_REGIME, SHAM_CONTROL_KIND, SHAM_SALT
    from engine.data.spread_calibration import CALIBRATOR_VERSION

    if engine_mode == "slow" and cfg.kernel_share_min > 0:
        raise GARunError(
            f"engine_mode slow gives kernel share 0 < kernel_share_min {cfg.kernel_share_min}: every "
            "generation would be INVALID. A GA run is kernel-first (engine_mode auto)."
        )
    if cfg.islands is not None and (checkpoint_path is not None or resume_path is not None):
        raise GARunError("checkpoint/resume covers the core loop only; an island run cannot checkpoint yet")
    started = time.perf_counter()
    started_at = datetime.now(tz=UTC).isoformat()
    cross = cross_cell_for(cell)
    symbol, timeframe, dataset_id = PINNED_CELLS[cell]
    campaign_config = campaign_config_for(
        symbol=symbol, timeframe=timeframe, population=cfg.population, target_sharpe=CAMPAIGN_TARGET_SHARPE,
        seed=cfg.seed, dataset_id=dataset_id, gene_pool=cfg.gene_pool, min_trades=cfg.gates.min_trades,
        ga=cfg.block(),
    )
    cfg_hash = config_hash(campaign_config)
    campaign_id = campaign_id_for(cell, cfg, cfg_hash)
    out_dir = artifact_path("ga", campaign_id)
    # Refused here, before any work, if the checkpoint cannot be trusted or
    # belongs to another configuration (checkpoint.CheckpointError).
    resume_state = None if resume_path is None else ckpt.load(resume_path, config_hash=cfg_hash)
    if resume_state is not None:
        started_at = resume_state["extra"]["started_at"]
    # An earlier run's outputs must not survive next to this run's fresh
    # registry, whatever happens before this run writes its own.
    for name in (*OUTPUT_FILES, MONITOR_FILE):
        stale = artifact_path("ga", campaign_id, name)
        if stale.exists():
            stale.unlink()
    echo(f"GA {campaign_id}: search {cell}, cross-check {cross}; outputs {out_dir}")
    primary_ev = prepare_cell(cell, engine_mode=engine_mode)
    cross_ev = prepare_cell(cross, engine_mode=engine_mode)
    for ev_ in (primary_ev, cross_ev):
        c = ev_.context
        echo(f"  {ev_.name}: {c['depth']}; engine {c['engine_mode_configured']} -> {c['engine_mode_actual']}; "
             f"canary passed={c['canary']['passed']}")
    t_prep = time.perf_counter() - started
    # The cross cell is its own campaign: its own id, and a config naming its
    # own symbol and dataset, linked to the search it validates.
    cross_symbol, cross_timeframe, cross_dataset = PINNED_CELLS[cross]
    cross_id = f"{campaign_id}-cross"
    cross_config = campaign_config_for(
        symbol=cross_symbol, timeframe=cross_timeframe, population=cfg.final_n,
        target_sharpe=CAMPAIGN_TARGET_SHARPE, seed=cfg.seed, dataset_id=cross_dataset, gene_pool=cfg.gene_pool,
        min_trades=cfg.gates.min_trades,
        ga={**cfg.block(), "role": "cross_check", "search_campaign_id": campaign_id, "search_config_hash": cfg_hash},
    )
    cross_cfg_hash = config_hash(cross_config)
    git_sha = resolve_git_sha()
    reg_path = artifact_path("ga", campaign_id, "trials.sqlite")
    if resume_state is None:
        registry = _registry(reg_path)
    else:
        # The interrupted run's registry carries on; rows written after its
        # last checkpoint are not part of the state resumed, so they go.
        from engine.validation.trial_registry import TrialRegistry

        registry = TrialRegistry(reg_path)
        trimmed = _trim_registry(reg_path, campaign_id, resume_state["extra"]["registry_rows"])
        primary_ev.shams = {(h, int(n)): hit for h, n, hit in resume_state["extra"]["shams_primary"]}
        state = getattr(primary_ev.backtest_fn, "fallback_state", None)
        if state is not None:
            state["fallbacks"] = resume_state["extra"]["fallback_state"]["fallbacks"]
            state["reasons"][:] = resume_state["extra"]["fallback_state"]["reasons"]
        echo(f"  RESUME from {resume_path} after generation {resume_state['generation']}; "
             f"registry rows after it trimmed: {trimmed}")
    registry.open_campaign(campaign_id, symbol, timeframe, primary_ev.context["dataset_hash"], git_sha,
                           cfg_hash, cfg.seed, notes="GA search (engine.research.ga)")
    registry.open_campaign(cross_id, cross_symbol, cross_timeframe, cross_ev.context["dataset_hash"],
                           git_sha, cross_cfg_hash, cfg.seed, notes=f"GA finalists of {campaign_id} on {cross}")

    monitor = {"last": None, "walls": []}

    def write_monitor(log: dict[str, Any]) -> None:
        """Read-only sidecar: what the run has done, nothing it decides on."""
        now = time.monotonic()
        monitor["walls"].append(sum(log["wall_seconds"].values()))
        if monitor["last"] is not None and now - monitor["last"] < monitor_every:
            return
        monitor["last"] = now
        remaining = cfg.generations - 1 - log["generation"]
        mv2.write_json(artifact_path("ga", campaign_id, MONITOR_FILE), {
            "campaign_id": campaign_id, "generation": log["generation"], "generations": cfg.generations,
            "trials_used": log["trials_used"], "trials_budget": log["trials_budget"], "funnel": log["funnel"],
            "best_fitness": log["population_fitness"]["best"], "alive": log["population_fitness"]["alive"],
            "kernel_share": log["kernel_share"],
            "eta_seconds": statistics.fmean(monitor["walls"]) * remaining,
            "written_at_utc": datetime.now(tz=UTC).isoformat(),
        })

    def write_checkpoint(loop: dict[str, Any]) -> None:
        extra = {
            "started_at": started_at,
            "shams_primary": [[h, n, hit] for (h, n), hit in sorted(primary_ev.shams.items())],
            "fallback_state": {k: v for k, v in (getattr(primary_ev.backtest_fn, "fallback_state", None)
                                                 or {"fallbacks": 0, "reasons": []}).items()},
            "registry_rows": _registry_rows(reg_path, campaign_id),
        }
        ckpt.save(checkpoint_path, loop=loop, extra=extra, config_hash=cfg_hash)
        echo(f"  CHECKPOINT after generation {loop['generation']} -> {checkpoint_path}")
        if hold_after_generation is not None and loop["generation"] == hold_after_generation:
            echo(f"  HOLD after generation {hold_after_generation} (operational): waiting to be stopped")
            time.sleep(3600)

    # the CAMPAIGN tab's sidecar (live_view): the best alive strategies so far, with their curves and trades
    live: dict[str, tuple[float, Individual, list[float]]] = {}

    def on_evaluated(new: list[Individual], g: int) -> None:
        _record(registry, campaign_id, primary_ev, new, started_at)

        def keep() -> None:
            for ind in new:
                f, r = ind.fitness, ind.result
                if f is None or f.killed_by is not None or r is None:
                    continue
                c = live_view.curve_of(r.per_period_returns)
                if c:
                    live[ind.effective_hash] = (float(f.fitness), ind, c)
            if len(live) > live_view.MAX_BOTS * 2:
                for h in sorted(live, key=lambda h: live[h][0])[: len(live) - live_view.MAX_BOTS]:
                    del live[h]
        live_view.safe(keep)

    def write_live(log: dict[str, Any]) -> None:
        ranked = sorted(live.values(), key=lambda x: (-x[0], x[1].effective_hash))[: live_view.MAX_BOTS]
        bots = [{"id": ind.effective_hash[:6].upper(), "tribe": "", "ret": round(float(ind.result.total_return or 0) * 100, 2),
                 "dd": round(float(ind.result.max_dd or 0) * 100, 2), "fit": round(fit, 3), "curve": c}
                for fit, ind, c in ranked]
        top = [(ind.effective_hash[:6].upper(), ind.dna) for _, ind, _ in ranked
               if ind.engine == ENGINE_KERNEL][: live_view.TAPE_BOTS]
        tape = live_view.merged_tape(getattr(primary_ev, "backtest_fn", None), top)
        live_view.write(out_dir, generation=int(log["generation"]), generations=cfg.generations, phase="BREEDING",
                        survivors=int(log["population_fitness"]["alive"]), bots=bots, tape=tape)

    def on_generation(log: dict[str, Any]) -> None:
        live_view.safe(lambda: write_live(log))
        if monitor_every is not None:
            write_monitor(log)
        f, pf, cen = log["funnel"], log["population_fitness"], log["census"]
        echo(f"  gen {log['generation']:>2}: new={log['n_new']} elites={log['n_elites']} "
             f"collisions={log['collisions']} trials={log['trials_used']}/{log['trials_budget']} "
             f"kernel_share={log['kernel_share']} crashes={log['crashes']} "
             f"funnel={f} best={pf['best']:.4f} median={pf['median']:.4f} alive={pf['alive']}")
        echo(f"          census: per_family={cen['trials_per_family']} dedupe_hits={cen['dedupe_hits']} "
             f"zero_trade={cen['zero_trade_rate_per_family']} flagged={cen['boundary_saturation']['flagged']}")
        if "islands" in log:
            il = log["islands"]
            echo(f"          islands: best={[round(x['best'], 4) for x in il['per_island']]} "
                 f"stagnation={[x['stagnation'] for x in il['per_island']]} migrated={il['migrated']} "
                 f"revolutions={il['revolutions']} war={il['war']} cryo={il['cryo_size']} defrosted={il['defrosted']}")

    if workers < 1:
        raise GARunError(f"workers must be at least 1, got {workers}")
    search_ev = with_power(primary_ev, workers)
    echo(f"  power: {workers} worker process(es)")
    loop_ops: dict[str, Any] = {}
    if checkpoint_path is not None:
        loop_ops["checkpoint"] = write_checkpoint
    if resume_state is not None:
        loop_ops["resume"] = resume_state
    try:
        result = searcher_for(cfg)(cfg, search_ev, on_generation=on_generation,
                                   on_evaluated=on_evaluated,
                                   **loop_ops)
    finally:
        if search_ev is not primary_ev:
            search_ev.close()
    t_search = time.perf_counter() - started - t_prep
    crossed, cross_n = cross_check(result.finalists, cfg, cross_ev)
    _record_cross(registry, cross_id, cross_ev, crossed, started_at)
    for ev_ in (primary_ev, cross_ev):
        # Every fallback and its reason (CLAUDE.md s5), as the random-search report records them.
        state = getattr(ev_.backtest_fn, "fallback_state", None) or {}
        ev_.context["fast_fallback_count"] = int(state.get("fallbacks", 0))
        ev_.context["fast_fallback_reasons"] = list(state.get("reasons", []))
    t_cross = time.perf_counter() - started - t_prep - t_search

    table = individuals_table(result.archive)
    r_hash = run_hash(table)
    outcomes = [_outcome(i) for i in result.archive]
    census = degeneracy_report([i.result for i in result.archive], outcomes, min_trades=cfg.gates.min_trades,
                               trials_raw=len(result.archive) + result.collisions,
                               trials_effective=len(result.archive))
    cells = (cell, cross)
    controls = {"kind": SHAM_CONTROL_KIND, "salt": SHAM_SALT, "regime": CONTROLS_REGIME, "engine": "slow",
                "seed": "sha256(dna_hash:salt)"}
    schema_versions = {"dna": DNA_SCHEMA_VERSION, "effective_hash": EFFECTIVE_HASH_VERSION,
                       "spread_calibrator": CALIBRATOR_VERSION, "ga_config": GA_CONFIG_VERSION,
                       "ga_manifest": mv2.MANIFEST_VERSION}
    ctxs = {cell: primary_ev.context, cross: cross_ev.context}
    candidates = []
    for ind, c in zip(result.finalists, crossed, strict=True):
        primary_metrics = _cell_metrics(primary_ev, Evaluation(ind.result, False, ind.engine), ind.fitness)
        cross_metrics = _cell_metrics(cross_ev, c.evaluation, c.fitness)
        pf, cf = ind.fitness, c.fitness
        candidates.append({
            "dna": ind.dna.to_dict(),
            "dna_hash": ind.dna_hash,
            "effective_gene_hash": ind.effective_hash,
            "lineage": ind.lineage(),
            "metrics": {cell: primary_metrics, cross: cross_metrics},
            "sham_margin": {cell: pf.sham_margin, cross: cf.sham_margin},
            "dsr": {
                "primary_at_eval": None if pf.dsr is None else asdict(pf.dsr),
                "primary_final": asdict(result.final_dsr[ind.effective_hash]),
                "cross": None if cf.dsr is None else asdict(cf.dsr),
            },
            "cross_cell_delta": {
                "sharpe": None if cf.sharpe is None or pf.sharpe is None else cf.sharpe - pf.sharpe,
                "fitness": cf.fitness - pf.fitness,
                "alive_on_both": cf.alive,
                "killed_on_cross_by": cf.killed_by,
            },
            "degeneracy_census": census,
            "controls_id": controls,
            "audit": {k: ctxs[k]["audit"] for k in cells},
            "spread_sha256": {k: ctxs[k]["spread_artifact_sha256"] for k in cells},
            "schema_versions": schema_versions,
            "config_hash": cfg_hash,
            "run_hash": r_hash,
            "dataset_id": {k: ctxs[k]["dataset_id"] for k in cells},
            "dataset_provenance": {k: dataset_provenance_for(ctxs[k]["dataset_id"]) for k in cells},
        })
    engines_all = Counter(i.engine for i in result.archive)
    wall = time.perf_counter() - started
    run = {
        "campaign_id": campaign_id,
        "config": campaign_config,
        "config_hash": cfg_hash,
        "run_hash": r_hash,
        "git_sha": git_sha,
        "gene_pool": pool_summary(cfg.gene_pool),
        "stop_reason": result.stop_reason,
        "aborted": result.aborted,
        "generations_run": len(result.generations),
        "trials_used": result.trials_used,
        "trials_budget": cfg.trials_budget,
        "collisions": result.collisions,
        "engines": dict(sorted(engines_all.items())),
        "kernel_share": kernel_share(engines_all),
        "cross_engines": dict(sorted(Counter(c.evaluation.engine for c in crossed).items())),
        "survivors": sum(1 for i in result.archive if i.fitness.alive),
        "finalists": len(result.finalists),
        "finalists_alive_on_both": sum(1 for c in crossed if c.fitness.alive),
        "dsr_n_final": result.dsr_n_final,
        "dsr_n_cross": cross_n,
        "controls": controls,
        "schema_versions": schema_versions,
        "cross_campaign_id": cross_id,
        "cross_config": cross_config,
        "cross_config_hash": cross_cfg_hash,
        "excluded_generation": result.excluded_generation,
        "sham_errors": {
            "search": {h: e for g in result.generations for h, e in g["sham_errors"].items()},
            "cross": {c.evaluation.result.dna.dna_hash: c.sham_error for c in crossed if c.sham_error},
        },
        "sham_fills": {k: _sham_fills(ev_) for k, ev_ in ((cell, primary_ev), (cross, cross_ev))},
        "cells": {
            k: {**ctxs[k], "manifest": cell_manifest(
                ctxs[k], campaign_id=cid, git_sha=git_sha, cfg_hash=h,
                pool_summary=pool_summary(cfg.gene_pool), seed=cfg.seed,
            )}
            for k, cid, h in ((cell, campaign_id, cfg_hash), (cross, cross_id, cross_cfg_hash))
        },
        "degeneracy_census": census,
        "cost_totals": cost_totals([i.result for i in result.archive]),
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(tz=UTC).isoformat(),
        "wall_seconds": {"total": wall, "prepare": t_prep, "search": t_search, "cross": t_cross},
        # Outside the hashed config and excluded from the canonical form.
        "operational": {
            "checkpoint": None if checkpoint_path is None else str(checkpoint_path),
            "resumed_from": None if resume_path is None else str(resume_path),
            "resumed_after_generation": None if resume_state is None else resume_state["generation"],
            "hold_after_generation": hold_after_generation,
            "monitor_snapshot_every": monitor_every,
            "workers": workers,
        },
    }
    doc = mv2.build_manifest(run=run, candidates=candidates, cells=cells)
    paths = {
        "manifest": mv2.write_json(artifact_path("ga", campaign_id, "manifest_v2.json"), doc),
        "individuals": mv2.write_json(artifact_path("ga", campaign_id, "individuals.json"), table),
        "generations": mv2.write_json(artifact_path("ga", campaign_id, "generations.json"), result.generations),
    }
    echo(f"  stop={result.stop_reason} aborted={result.aborted} trials={result.trials_used} "
         f"survivors={run['survivors']} finalists={run['finalists']} alive_on_both={run['finalists_alive_on_both']}")
    echo(f"  config_hash={cfg_hash} run_hash={r_hash}")
    for k, p in paths.items():
        echo(f"  WROTE {k}: {p}")
    return {"campaign_id": campaign_id, "run_hash": r_hash, "config_hash": cfg_hash,
            "paths": {k: str(p) for k, p in paths.items()}, "stop_reason": result.stop_reason,
            "aborted": result.aborted, "wall_seconds": wall}


def main(argv: list[str] | None = None) -> int:
    import argparse

    from engine.core.paths import get_arqvio_home
    from engine.research.ga.config import PINNED_CELLS
    from engine.research.search import gene_pool

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cell", required=True, choices=sorted(PINNED_CELLS))  # VALIDATED and QUICK
    parser.add_argument("--pop", type=int, default=200, help="population per generation")
    parser.add_argument("--gen", type=int, default=25, help="generations, generation 0 included")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--budget", type=int, default=8000, help="trials budget (new backtests)")
    parser.add_argument("--final-n", type=int, default=20)
    # Kernel-first by construction: "slow" can never meet kernel_share_min.
    parser.add_argument("--engine", default="auto", choices=("auto", "fast"))
    parser.add_argument("--islands", type=int, default=None,
                        help="enable the island model with K islands (documented defaults for its other knobs)")
    # Operational, outside the hashed config (recorded in run.operational).
    parser.add_argument("--checkpoint", type=Path, default=None, help="write the loop state after every generation")
    parser.add_argument("--resume", type=Path, default=None, help="continue a checkpointed run of the same config")
    parser.add_argument("--hold-after-generation", type=int, default=None,
                        help="after that generation's checkpoint, wait to be stopped (kill/resume proof)")
    parser.add_argument("--monitor-snapshot-every", type=float, default=None,
                        help="write monitor_snapshot.json at most every N seconds")
    power = parser.add_mutually_exclusive_group()
    power.add_argument("--power", choices=POWER_MODES, default=None,
                       help="eco 1 process, medium half the cores, high all but one, extreme every core")
    power.add_argument("--workers", type=int, default=None, help="an explicit number of worker processes")
    args = parser.parse_args(argv)
    workers = args.workers if args.workers is not None else workers_for(args.power or "eco")
    print(f"ARQVIO_HOME={get_arqvio_home()}")
    try:
        cfg = ga_config(seed=args.seed, gates=CAMPAIGN_GATES, gene_pool=gene_pool.load(),
                        population=args.pop, generations=args.gen, trials_budget=args.budget,
                        final_n=args.final_n,
                        islands=None if args.islands is None else IslandConfig(k=args.islands))
    except ConfigError as exc:
        print(f"REFUSED: {exc}")
        return 2
    summary = run_pinned(cfg, args.cell, engine_mode=args.engine, checkpoint_path=args.checkpoint,
                         resume_path=args.resume, hold_after_generation=args.hold_after_generation,
                         monitor_every=args.monitor_snapshot_every, workers=workers)
    return 1 if summary["aborted"] else 0


__all__ = [
    "CellEvaluator",
    "CrossResult",
    "Evaluation",
    "Evaluator",
    "GARunError",
    "MONITOR_FILE",
    "SearchResult",
    "breed",
    "cache_key",
    "campaign_id_for",
    "cell_manifest",
    "cross_check",
    "kernel_share",
    "main",
    "prepare_cell",
    "prepare_worker_cell",
    "run_pinned",
    "search",
    "searcher_for",
    "with_power",
]


if __name__ == "__main__":
    sys.exit(main())
