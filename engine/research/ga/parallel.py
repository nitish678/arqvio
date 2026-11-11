"""Power modes: one campaign on several CPU cores, with results identical to one core.

Each worker process prepares the SAME pinned cell once (its own backtest and sham, compiled in that
process) and then runs pure functions of the DNA: ``evaluate(dna)`` and the sham for
``(dna, entries)``. The parent places every result back at its DNA's position, so the loop, the RNG,
the archive and every artifact are byte-identical whatever the worker count. The shams computed
ahead are exactly the ones ``score`` will ask for (``fitness.early_kill``), so the sham cache -- which
the manifest records -- is identical too. A sham that raises in a worker is not cached; the parent
re-runs it during scoring and records the crash exactly as one core would.

Speed comes from cache affinity. The fast backtest keeps its indicator arrays per distinct
(family, genes) key; a DNA is routed to the worker that already holds its key (sticky across
generations), and a key seen for the first time goes to the least-loaded worker of the batch.
Routing changes where work runs, never what it returns. Each worker runs its DNAs in input order, so
the fast path's fallback record (a count, and reasons kept once each in first-seen order) merges into
the parent's in input order exactly as one process would have written it.

Power modes are operational (outside the hashed config): eco 1 worker, medium half the cores,
high all cores but one, extreme every core.
"""
from __future__ import annotations

import os
from collections.abc import Callable, Hashable, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from typing import Any

from engine.research.ga.config import Gates
from engine.research.ga.fitness import early_kill
from engine.research.search.dna import DNA
from engine.research.search.runner import TrialResult

POWER_MODES = ("eco", "medium", "high", "extreme")
_JOB = 8  # DNAs per submitted job: small enough to stream, large enough to amortise the IPC


def cpu_count() -> int:
    return os.cpu_count() or 1


def workers_for(power: str, cores: int | None = None) -> int:
    """Worker processes for a power mode on a machine with ``cores`` logical cores."""
    n = cores if cores is not None else cpu_count()
    table = {"eco": 1, "medium": max(1, n // 2), "high": max(1, n - 1), "extreme": max(1, n)}
    if power not in table:
        raise ValueError(f"unknown power mode {power!r}; one of {POWER_MODES}")
    return table[power]


def windowless_workers() -> str | None:
    """In the installed app (CPython shipped as arqvio-engine.exe beside its windowless twin arqvio-enginew.exe), start
    worker processes with the twin, so a GUI-launched campaign opens no console window per worker. Returns the
    interpreter set, or None where nothing changes (a source checkout, the tests)."""
    import multiprocessing
    import sys
    from pathlib import Path

    exe = Path(sys.executable)
    twin = exe.with_name("arqvio-enginew.exe")
    if exe.name.lower() != "arqvio-engine.exe" or not twin.is_file():
        return None
    multiprocessing.set_executable(str(twin))
    return str(twin)


_CELL: Any = None  # the evaluator this worker process prepared


def _init(factory: Callable[[], Any]) -> None:
    global _CELL
    _CELL = factory()


def _ready(_: int) -> bool:
    """A no-op whose only effect is that the worker has run its initializer (prepared its cell)."""
    return _CELL is not None


def _fallback_state(ev: Any) -> dict[str, Any] | None:
    return getattr(getattr(ev, "backtest_fn", None), "fallback_state", None)


def _evaluate_many(dnas: list[DNA]) -> list[tuple[Any, tuple[int, list[Any]] | None]]:
    """Each evaluation with the fallbacks it caused in this worker (count, newly seen reasons)."""
    out = []
    for dna in dnas:
        state = _fallback_state(_CELL)
        n0, r0 = (0, 0) if state is None else (int(state["fallbacks"]), len(state["reasons"]))
        e = _CELL.evaluate(dna)
        delta = None if state is None else (int(state["fallbacks"]) - n0, list(state["reasons"][r0:]))
        out.append((e, delta))
    return out


def _sham_many(jobs: list[tuple[DNA, int]]) -> list[dict[str, Any]]:
    out = []
    for dna, entries in jobs:
        try:
            out.append(_CELL.sham_hit(dna, entries))
        except Exception as exc:  # re-run in the parent, which records it as the candidate's crash
            out.append({"error": f"{type(exc).__name__}: {exc}"})
    return out


class PoolEvaluator:
    """An evaluator whose batch work runs on worker processes; single calls and state stay local.

    ``local`` is the parent's own prepared evaluator (attributes, sham cache, single evaluations);
    ``factory`` is a picklable callable that prepares the same evaluator inside each worker;
    ``affinity`` maps a DNA to its cache key (None: every DNA is its own key, plain load balancing).
    """

    def __init__(self, local: Any, factory: Callable[[], Any], workers: int,
                 affinity: Callable[[DNA], Hashable] | None = None) -> None:
        if workers < 2:
            raise ValueError("a pool needs at least 2 workers; use the evaluator itself for 1")
        self.local = local
        self.workers = workers
        self.affinity = affinity
        self._home: dict[Hashable, int] = {}
        windowless_workers()
        # One single-process executor per worker, so a DNA can be sent to the worker holding its cache.
        self._execs = [ProcessPoolExecutor(max_workers=1, initializer=_init, initargs=(factory,))
                       for _ in range(workers)]
        # Start every worker now, so preparing their cells overlaps the parent's own preparation.
        self._warm = [ex.submit(_ready, i) for i, ex in enumerate(self._execs)]

    def __getattr__(self, name: str) -> Any:
        if name in ("local", "workers", "affinity", "_home", "_execs", "_warm"):
            raise AttributeError(name)
        return getattr(self.local, name)

    def _route(self, keys: Sequence[Hashable]) -> list[list[int]]:
        """Indices per worker: a known key goes home; a new key to the batch's least-loaded worker."""
        load = [0] * self.workers
        lanes: list[list[int]] = [[] for _ in range(self.workers)]
        for i, key in enumerate(keys):
            w = self._home.get(key)
            if w is None:
                w = min(range(self.workers), key=lambda k: (load[k], k))
                self._home[key] = w
            load[w] += 1
            lanes[w].append(i)
        return lanes

    def _run(self, fn: Callable[[list[Any]], list[Any]], items: Sequence[Any], lanes: list[list[int]]) -> list[Any]:
        out: list[Any] = [None] * len(items)
        pending: list[tuple[list[int], Future]] = []
        for w, lane in enumerate(lanes):
            for s in range(0, len(lane), _JOB):
                idx = lane[s:s + _JOB]
                pending.append((idx, self._execs[w].submit(fn, [items[i] for i in idx])))
        for idx, fut in pending:
            for i, r in zip(idx, fut.result(), strict=True):
                out[i] = r
        return out

    def evaluate(self, dna: DNA) -> Any:
        return self.local.evaluate(dna)

    def evaluate_batch(self, dnas: Sequence[DNA]) -> list[Any]:
        """Evaluations in input order; worker fallbacks merged into the parent's state in that order."""
        keys = [self.affinity(d) if self.affinity is not None else d.dna_hash for d in dnas]
        results = self._run(_evaluate_many, dnas, self._route(keys))
        state = _fallback_state(self.local)
        out = []
        for e, delta in results:
            if delta is not None and state is not None:
                state["fallbacks"] = int(state["fallbacks"]) + delta[0]
                state["reasons"].extend(r for r in delta[1] if r not in state["reasons"])
            out.append(e)
        return out

    def sham_sharpe(self, result: TrialResult) -> float:
        return self.local.sham_sharpe(result)

    def prefetch_shams(self, scored: Sequence[tuple[TrialResult, bool]], gates: Gates) -> int:
        """Run, in parallel, every sham ``score`` will request for these (result, crashed) pairs."""
        from engine.research.search.sham_control import attempted_entries

        need: list[tuple[DNA, tuple[str, int]]] = []
        seen: set[tuple[str, int]] = set()
        for result, crashed in scored:
            if early_kill(result, crashed=crashed, gates=gates) is not None:
                continue
            key = (result.dna.dna_hash, attempted_entries(result))
            if key in self.local.shams or key in seen:
                continue
            seen.add(key)
            need.append((result.dna, key))
        jobs = [(dna, key[1]) for dna, key in need]
        lanes = [list(range(w, len(jobs), self.workers)) for w in range(self.workers)]  # shams share no cache
        for (_, key), hit in zip(need, self._run(_sham_many, jobs, lanes), strict=True):
            if "error" not in hit:
                self.local.shams[key] = hit
        return len(need)

    def close(self) -> None:
        for ex in self._execs:
            ex.shutdown(wait=True, cancel_futures=True)


__all__ = ["POWER_MODES", "PoolEvaluator", "cpu_count", "windowless_workers", "workers_for"]
