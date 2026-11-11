"""The V1 breeding loop: proving grounds -> TRIBE A / TRIBE B x generations -> Final War (research APP-2, BP03 §1).

Order of one run (each step logs a feed line; research PART 4.8 vocabulary):
1. Proving grounds: the 16 seeds + fresh draws up to ``proving_grounds`` strategies, each run on the in-sample bars.
2. Tribes: the best ``tribe_size`` alive by TRIBE A's fitness form TRIBE A · CONSISTENCY, the best of the rest by
   TRIBE B's form TRIBE B · GROWTH.
3. Each generation, per tribe: elite kept, children bred (tournament on family-adjusted fitness -> crossover ->
   mutation), evaluated; then culled -- DRAWDOWN_BREACH (never back: research patch note), LOW_FITNESS (bottom
   share), and after ``stagnation_after`` generations without a better best, STAGNATION: a revolution replaces the
   weaker half with fresh draws.
4. A skip request finishes the current generation and goes straight to the Final War (research: SKIP never aborts
   mid-generation).
5. Final War: every survivor of both tribes in one pool, one last generation bred from it, all ranked on both
   tribes' measures; the best ``finalists`` distinct strategies are the run's result.

Grade locking: a strategy is evaluated once (the run registry admits each effective hash once) and keeps the
fitness it earned -- the research fix "kept strategies judged the same as the first time".
Checkpoint after the proving grounds and every generation: tribes, archive, registry, rng bit state. A resumed run
continues from it and ends where the uninterrupted run would have (tests/research/v1/test_engine.py).
"""
from __future__ import annotations

import dataclasses
import json
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from engine.core import atomic
from engine.research.ga.operators import crossover, mutate
from engine.research.ga.population import Individual, Registry, checked
from engine.research.search import live_view
from engine.research.search.dna import DNA
from engine.research.search.sampler import random_dna
from engine.research.v1.config import V1Config, v1_gene_pool
from engine.research.v1.score import (
    DRAWDOWN_BREACH,
    FINAL_WAR,
    LOW_FITNESS,
    STAGNATION,
    TRIBE_A,
    TRIBE_B,
    Metrics,
    fitness,
    metrics,
    war_fitness,
)
from engine.research.v1.seeds import seeds

CHECKPOINT_VERSION = 1
MAX_RESAMPLES = 2000


@dataclasses.dataclass
class Member:
    ind: Individual
    origin: str                      # tribe_origin: Wild / TRIBE A / TRIBE B / FINAL WAR / "" (a fresh draw)
    source: str                      # pg_source: seed / draw / child / revolution / war
    metrics: Metrics | None = None
    alive: bool = False
    killed_by: str | None = None
    engine: str = ""
    fit_a: float = 0.0
    fit_b: float = 0.0
    engine_sharpe: float | None = None  # the backtester's own Sharpe: the trial distribution CERTIFY's DSR uses
    # the CAMPAIGN tab's equity curve (live_view.curve_of); display only, never checkpointed or compared
    curve: list[float] | None = dataclasses.field(default=None, compare=False, repr=False)

    @property
    def key(self) -> str:
        return self.ind.effective_hash

    def fit(self, tribe: str) -> float:
        if tribe == TRIBE_A:
            return self.fit_a
        if tribe == TRIBE_B:
            return self.fit_b
        return 0.5 * (self.fit_a + self.fit_b)

    def to_dict(self) -> dict[str, Any]:
        return {"dna": self.ind.dna.to_dict(), "effective_hash": self.key, "dna_hash": self.ind.dna_hash,
                "generation": self.ind.generation, "parents": list(self.ind.parents),
                "operators": list(self.ind.operators), "origin": self.origin, "source": self.source,
                "metrics": None if self.metrics is None else self.metrics.to_dict(), "alive": self.alive,
                "killed_by": self.killed_by, "engine": self.engine, "fit_a": self.fit_a, "fit_b": self.fit_b,
                "engine_sharpe": self.engine_sharpe}

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Member:
        ind = Individual(dna=DNA(**{k: tuple(v) if isinstance(v, list) else v for k, v in d["dna"].items()}),
                         effective_hash=d["effective_hash"], generation=int(d["generation"]),
                         parents=tuple(d["parents"]), operators=tuple(d["operators"]))
        m = d["metrics"]
        return Member(ind=ind, origin=d["origin"], source=d["source"],
                      metrics=None if m is None else Metrics(**m), alive=bool(d["alive"]),
                      killed_by=d["killed_by"], engine=d["engine"], fit_a=float(d["fit_a"]),
                      fit_b=float(d["fit_b"]), engine_sharpe=d.get("engine_sharpe"))


class V1Run:
    """One V1 campaign on one prepared cell. ``evaluator`` is a CellEvaluator (or its PoolEvaluator)."""

    def __init__(self, cfg: V1Config, evaluator: Any, timeframe: str, *, out_dir: Path,
                 log: Callable[[str], None] = print, skip_path: Path | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.cfg = cfg
        self.ev = evaluator
        self.timeframe = timeframe
        self.out_dir = Path(out_dir)
        self.log = log
        self.skip_path = skip_path
        self.clock = clock
        self.pool = v1_gene_pool()
        self.rng = np.random.default_rng(cfg.seed)
        self.registry = Registry()
        self.archive: dict[str, Member] = {}
        self.tribes: dict[str, list[str]] = {TRIBE_A: [], TRIBE_B: []}
        self.best: dict[str, float] = {TRIBE_A: 0.0, TRIBE_B: 0.0}
        self.flat: dict[str, int] = {TRIBE_A: 0, TRIBE_B: 0}
        self.generation = 0
        self.phase = "PROVING GROUNDS"
        self.generations_log: list[dict[str, Any]] = []
        self.engines: Counter = Counter()
        self.started = clock()
        self.skipped = False

    # -- evaluation -------------------------------------------------------------------------------------------
    def _pin(self, dna: DNA) -> DNA:
        """risk % is the campaign's, never bred (research: a campaign setting)."""
        return dna if dna.risk_pct == self.cfg.risk_pct else dataclasses.replace(dna, risk_pct=self.cfg.risk_pct)

    def _evaluate(self, members: list[Member]) -> None:
        todo = [m for m in members if m.metrics is None and m.killed_by is None]
        if not todo:
            return
        batch = getattr(self.ev, "evaluate_batch", None)
        evals = batch([m.ind.dna for m in todo]) if batch else [self.ev.evaluate(m.ind.dna) for m in todo]
        for m, e in zip(todo, evals, strict=True):
            self.engines[e.engine] += 1
            m.engine = e.engine
            r = e.result
            n = int(r.n_trades or 0)
            m.engine_sharpe = None if r.sharpe is None else float(r.sharpe)
            if e.crashed or r.per_period_returns is None:
                m.killed_by, m.metrics = "CRASH", Metrics(0.0, 0.0, 1.0, n, 0.0, 0.0)
            else:
                m.metrics = metrics(np.asarray(r.per_period_returns), self.timeframe, n)
                m.curve = live_view.curve_of(r.per_period_returns)
                if n < self.cfg.min_trades:
                    m.killed_by = "MIN_TRADES"
                elif m.metrics.sharpe <= 0.0:
                    m.killed_by = "LOSING"
                elif m.metrics.max_dd > self.cfg.drawdown_limit:
                    m.killed_by = DRAWDOWN_BREACH
            m.alive = m.killed_by is None
            if m.alive:
                m.fit_a, m.fit_b = fitness(m.metrics, TRIBE_A), fitness(m.metrics, TRIBE_B)
            self.archive[m.key] = m

    def _fresh(self, generation: int, source: str) -> Member:
        for _ in range(MAX_RESAMPLES):
            dna = self._pin(random_dna(self.rng, gene_pool=self.pool))
            eh = checked(dna)
            if self.registry.admit(eh):
                return Member(Individual(dna=dna, effective_hash=eh, generation=generation, operators=(source,)),
                              origin="", source=source)
        raise RuntimeError(f"{MAX_RESAMPLES} draws in a row were strategies this run already tested")

    # -- breeding ---------------------------------------------------------------------------------------------
    def _adjusted(self, members: list[Member], tribe: str) -> list[float]:
        """Family penalty / reward (research meta genes _family_penalty / _family_reward) on selection only."""
        fams = Counter(m.ind.dna.signal for m in members)
        n, k = len(members), max(len(fams), 1)
        out = []
        for m in members:
            share = fams[m.ind.dna.signal] / n
            f = m.fit(tribe)
            if share > self.cfg.family_cap:
                f *= self.cfg.family_penalty
            elif share < 1.0 / (2 * k):
                f *= self.cfg.family_reward
            out.append(f)
        return out

    def _pick(self, members: list[Member], adj: list[float]) -> Member:
        idx = self.rng.integers(0, len(members), size=self.cfg.tournament_k)
        best = max(idx, key=lambda i: (adj[i], members[i].key))
        return members[int(best)]

    def _children(self, parents: list[Member], n: int, generation: int, tribe: str, source: str) -> list[Member]:
        adj = self._adjusted(parents, tribe)
        out: list[Member] = []
        while len(out) < n:
            a, b = self._pick(parents, adj), self._pick(parents, adj)
            if a is not b and self.rng.random() < self.cfg.crossover_p:
                dna, ops, lin = crossover(self.rng, a.ind.dna, b.ind.dna), ["crossover"], (a.key, b.key)
            else:
                dna, ops, lin = a.ind.dna, ["clone"], (a.key,)
            dna, changed = mutate(self.rng, dna, self.pool, p=self.cfg.mutation_p, scale=self.cfg.mutation_scale)
            if changed:
                ops.append("mutate:" + ",".join(changed))
            dna = self._pin(dna)
            eh = checked(dna)
            if self.registry.admit(eh):
                out.append(Member(Individual(dna=dna, effective_hash=eh, generation=generation, parents=lin,
                                             operators=tuple(ops)), origin=tribe, source=source))
            else:
                out.append(self._fresh(generation, source))
        return out

    # -- phases -----------------------------------------------------------------------------------------------
    # -- the CAMPAIGN tab's sidecar (engine/research/search/live_view.py) ---------------------------------------
    def _live(self, phase: str, members: list[Member] | None = None) -> None:
        def write() -> None:
            if members is None:
                pool = [(self.archive[k], t) for t in (TRIBE_A, TRIBE_B) for k in self.tribes[t]]
            else:
                pool = [(m, FINAL_WAR) for m in members]
            pool.sort(key=lambda x: (-x[0].fit(x[1]), x[0].key))
            bots = [{"id": m.key[:6].upper(), "tribe": t, "ret": round(m.metrics.total_return * 100, 2),
                     "dd": round(m.metrics.max_dd * 100, 2), "fit": round(m.fit(t), 3), "curve": m.curve}
                    for m, t in pool if m.curve and m.metrics is not None]
            local = getattr(self.ev, "local", self.ev)
            top = [(m.key[:6].upper(), m.ind.dna) for m, _ in pool if m.engine == "kernel"][: live_view.TAPE_BOTS]
            tape = live_view.merged_tape(getattr(local, "backtest_fn", None), top)
            live_view.write(self.out_dir, generation=self.generation, generations=self.cfg.generations, phase=phase,
                            survivors=len(pool), bots=bots, tape=tape)
        live_view.safe(write)

    def proving_grounds(self) -> None:
        pg: list[Member] = []
        for origin, dna in seeds(self.pool, self.cfg.risk_pct):
            eh = checked(dna)
            if self.registry.admit(eh):
                pg.append(Member(Individual(dna=dna, effective_hash=eh, generation=0, operators=("seed",)),
                                 origin=origin, source="seed"))
        while len(pg) < self.cfg.proving_grounds:
            pg.append(self._fresh(0, "draw"))
        self._evaluate(pg)
        alive = [m for m in pg if m.alive]
        a = sorted(alive, key=lambda m: (-m.fit_a, m.key))[: self.cfg.tribe_size]
        taken = {m.key for m in a}
        b = sorted((m for m in alive if m.key not in taken), key=lambda m: (-m.fit_b, m.key))[: self.cfg.tribe_size]
        self.tribes = {TRIBE_A: [m.key for m in a], TRIBE_B: [m.key for m in b]}
        for t in (TRIBE_A, TRIBE_B):
            self.best[t] = max((self.archive[k].fit(t) for k in self.tribes[t]), default=0.0)
        self.log(f"PROVING GROUNDS · {len(pg)} strategies tested · {len(alive)} alive · "
                 f"TRIBE A {len(a)} · TRIBE B {len(b)} · ELAPSED {self.clock() - self.started:.0f}s")
        self.generations_log.append({"generation": 0, "phase": "PROVING GROUNDS", "tested": len(pg),
                                     "alive": len(alive), "tribe_a": len(a), "tribe_b": len(b)})
        self._live("PROVING GROUNDS")

    def _generation(self, tribe: str, g: int) -> dict[str, Any]:
        cfg = self.cfg
        members = [self.archive[k] for k in self.tribes[tribe]]
        if not members:
            return {"tribe": tribe, "size": 0, "culled": {}}
        ranked = sorted(members, key=lambda m: (-m.fit(tribe), m.key))
        elite = ranked[: cfg.elite]
        kids = self._children(ranked, max(cfg.tribe_size - len(elite), 0), g, tribe, "child")
        self._evaluate(kids)
        culled: Counter = Counter()
        for k in kids:
            if not k.alive:
                culled[DRAWDOWN_BREACH if k.killed_by == DRAWDOWN_BREACH else LOW_FITNESS] += 1
        pool = [m for m in ranked[cfg.elite:] + kids if m.alive]
        pool.sort(key=lambda m: (-m.fit(tribe), m.key))
        n_low = int(len(pool) * cfg.low_fitness_share)
        if n_low:
            culled[LOW_FITNESS] += n_low
            pool = pool[: len(pool) - n_low]
        nxt = elite + pool[: max(cfg.tribe_size - len(elite), 0)]
        top = max((m.fit(tribe) for m in nxt), default=0.0)
        revolution = 0
        if top > self.best[tribe] + 1e-12:
            self.best[tribe], self.flat[tribe] = top, 0
        else:
            self.flat[tribe] += 1
        if self.flat[tribe] >= cfg.stagnation_after and len(nxt) > cfg.elite:
            keep = max(cfg.elite, len(nxt) - int(len(nxt) * cfg.revolution_share))
            revolution = len(nxt) - keep
            fresh = [self._fresh(g, "revolution") for _ in range(revolution)]
            self._evaluate(fresh)
            nxt = nxt[:keep] + [m for m in fresh if m.alive]
            culled[STAGNATION] += revolution
            self.flat[tribe] = 0
        self.tribes[tribe] = [m.key for m in nxt]
        return {"tribe": tribe, "size": len(nxt), "top_fitness": round(top, 4),
                "culled": dict(culled), "revolution": revolution}

    def generations(self, start: int = 1) -> None:
        for g in range(start, self.cfg.generations + 1):
            self.generation = g
            self.phase = f"GENERATION {g}/{self.cfg.generations}"
            rows = [self._generation(TRIBE_A, g), self._generation(TRIBE_B, g)]
            survivors = sum(r["size"] for r in rows)
            top = max((r.get("top_fitness", 0.0) for r in rows), default=0.0)
            self.generations_log.append({"generation": g, "phase": "TRIBES", "tribes": rows,
                                         "survivors": survivors, "tested": len(self.archive)})
            culls = "; ".join(f"{r['tribe'][-1]}: " + " ".join(f"{k} {v}" for k, v in sorted(r["culled"].items()))
                              for r in rows if r["culled"])
            self.log(f"GEN {g}/{self.cfg.generations} · SURVIVORS {survivors} · TOP FITNESS {top:.2f} · "
                     f"tested {len(self.archive)} · ELAPSED {self.clock() - self.started:.0f}s"
                     + (f" · culled {culls}" if culls else ""))
            self.checkpoint()
            self._live("TRIBES")
            if self.skip_path is not None and self.skip_path.exists():
                self.skipped = True
                self.log(f"SKIP: generation {g} finished; straight to the Final War")
                break

    def final_war(self) -> list[Member]:
        self.phase = FINAL_WAR
        survivors = [self.archive[k] for t in (TRIBE_A, TRIBE_B) for k in self.tribes[t]]
        if not survivors:
            self.log("FINAL WAR · no survivor reached it")
            return []
        kids = self._children(survivors, len(survivors), self.generation + 1, FINAL_WAR, "war")
        self._evaluate(kids)
        field = {m.key: m for m in survivors + [k for k in kids if k.alive]}
        ranked = sorted(field.values(), key=lambda m: (-war_fitness(m.metrics), m.key))
        finalists = ranked[: self.cfg.finalists]
        self._live(FINAL_WAR, ranked)  # the whole field, finalists first
        self.generations_log.append({"generation": self.generation + 1, "phase": FINAL_WAR,
                                     "field": len(field), "finalists": len(finalists)})
        self.log(f"FINAL WAR · {len(field)} in the field · {len(finalists)} finalists · "
                 f"TOP FITNESS {war_fitness(finalists[0].metrics):.2f}" if finalists else "FINAL WAR · no finalist")
        return finalists

    # -- checkpoint -------------------------------------------------------------------------------------------
    @property
    def checkpoint_path(self) -> Path:
        return self.out_dir / "v1_checkpoint.json"

    def checkpoint(self) -> None:
        doc = {"version": CHECKPOINT_VERSION, "config": self.cfg.block(), "generation": self.generation,
               "rng": self.rng.bit_generator.state, "registry": sorted(self.registry.seen),
               "archive": [m.to_dict() for m in self.archive.values()], "tribes": self.tribes,
               "best": self.best, "flat": self.flat, "log": self.generations_log,
               "engines": dict(self.engines)}
        self.out_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.checkpoint_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(doc, default=_json_default), encoding="utf-8")
        atomic.replace(tmp, self.checkpoint_path)

    def restore(self) -> bool:
        p = self.checkpoint_path
        if not p.is_file():
            return False
        doc = json.loads(p.read_text(encoding="utf-8"))
        if doc.get("version") != CHECKPOINT_VERSION or doc.get("config") != self.cfg.block():
            raise RuntimeError("the checkpoint belongs to a different V1 configuration; it cannot be resumed")
        self.rng.bit_generator.state = doc["rng"]
        self.registry = Registry()
        for eh in doc["registry"]:
            self.registry.admit(eh)
        self.archive = {d["effective_hash"]: Member.from_dict(d) for d in doc["archive"]}
        self.tribes, self.best, self.flat = doc["tribes"], doc["best"], doc["flat"]
        self.generation, self.generations_log = int(doc["generation"]), doc["log"]
        self.engines = Counter(doc.get("engines") or {})
        return True

    # -- the whole run ----------------------------------------------------------------------------------------
    def run(self, *, resume: bool = False) -> dict[str, Any]:
        resumed = resume and self.restore()
        if resumed:
            self.log(f"resumed at generation {self.generation}")
        else:
            self.proving_grounds()
            self.checkpoint()
        if not (self.skip_path is not None and self.skip_path.exists() and resumed):
            self.generations(start=self.generation + 1)
        finalists = self.final_war()
        kernel = self.engines.get("kernel", 0)
        total = sum(self.engines.values())
        return {"engine": "v1", "config": self.cfg.block(), "gene_pool": self.pool.hash,
                "tested": len(self.archive), "generations": self.generations_log, "skipped": self.skipped,
                "kernel_share": (kernel / total) if total else None, "engines": dict(self.engines),
                "finalists": [m.to_dict() for m in finalists],
                "elapsed_s": round(self.clock() - self.started, 1)}


def _json_default(x: Any) -> Any:
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, tuple):
        return list(x)
    raise TypeError(f"not JSON: {type(x).__name__}")


__all__ = ["CHECKPOINT_VERSION", "Member", "V1Run"]
