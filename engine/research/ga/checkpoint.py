"""GA checkpoint / resume: the full loop state after a generation, written atomically.

A checkpoint holds everything the core loop (runner.search) needs to continue
EXACTLY: the generator's bit state, the run registry (admitted effective
hashes, collision count), every evaluated individual (DNA, lineage, engine,
backtest result, fitness record), the current population, the per-period
returns still kept for finalists, the generation logs (census included),
the trial budget used, and the stop state. run_pinned adds what the run
around the loop needs: the sham memo, the fast factory's fallback counter,
the registry row count and the run's start stamp.

Format: one .npz -- "state" is canonical JSON bytes, "sha256" its digest,
"r_<effective_hash>" a kept returns array. It is written to a temporary file
and renamed over the target (os.replace), so a reader sees either the old
checkpoint or the new one, never half of one. load() refuses with
CheckpointError on anything it cannot verify: an unreadable file, a missing
member, a digest mismatch, a schema version or config hash it does not
match.
"""
from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from engine.research.ga.fitness import DSRReading, FitnessRecord
from engine.research.ga.population import Individual
from engine.research.search.dna import DNA
from engine.research.search.runner import TrialResult

CHECKPOINT_SCHEMA = 1


class CheckpointError(RuntimeError):
    """A checkpoint that cannot be trusted to resume from."""


_RESULT_FIELDS = tuple(f.name for f in dataclasses.fields(TrialResult) if f.name not in ("dna", "per_period_returns"))


def _individual(ind: Individual) -> dict[str, Any]:
    r = ind.result
    return {
        "dna": ind.dna.to_dict(), "effective_hash": ind.effective_hash, "generation": ind.generation,
        "parents": list(ind.parents), "operators": list(ind.operators), "engine": ind.engine,
        "result": None if r is None else {name: getattr(r, name) for name in _RESULT_FIELDS},
        "fitness": None if ind.fitness is None else ind.fitness.to_dict(),
    }


def _fitness(d: dict[str, Any] | None) -> FitnessRecord | None:
    if d is None:
        return None
    dsr = d.get("dsr")
    return FitnessRecord(**{**d, "dsr": None if dsr is None else DSRReading(**dsr)})


def _restore(d: dict[str, Any], returns: dict[str, np.ndarray]) -> Individual:
    dna = DNA(**d["dna"])
    result = None
    if d["result"] is not None:
        result = TrialResult(dna=dna, **d["result"], per_period_returns=returns.get(d["effective_hash"]))
    return Individual(dna=dna, effective_hash=d["effective_hash"], generation=d["generation"],
                      parents=tuple(d["parents"]), operators=tuple(d["operators"]), result=result,
                      fitness=_fitness(d["fitness"]), engine=d["engine"])


def save(path: Path, *, loop: dict[str, Any], extra: dict[str, Any], config_hash: str) -> Path:
    """Write the checkpoint atomically: temp file in the same directory, then os.replace."""
    archive: list[Individual] = loop["archive"]
    state = {
        "schema": CHECKPOINT_SCHEMA,
        "config_hash": config_hash,
        "generation": loop["generation"],
        "rng": loop["rng"].bit_generator.state,
        "registry": {"seen": sorted(loop["registry"].seen), "collisions": loop["registry"].collisions},
        "archive": [_individual(i) for i in archive],
        "population": [i.effective_hash for i in loop["population"]],
        "keep": sorted(loop["keep"]),
        "generations": loop["generations"],
        "trials_used": loop["trials_used"],
        "stop": loop["stop"],
        "aborted": loop["aborted"],
        "extra": extra,
    }
    payload = json.dumps(state, sort_keys=True, separators=(",", ":"), default=_json_default).encode("utf-8")
    arrays = {f"r_{i.effective_hash}": np.asarray(i.result.per_period_returns, dtype=np.float64)
              for i in archive if i.result is not None and i.result.per_period_returns is not None}
    buf = io.BytesIO()
    np.savez(buf, state=np.frombuffer(payload, dtype=np.uint8),
             sha256=np.frombuffer(hashlib.sha256(payload).hexdigest().encode(), dtype=np.uint8), **arrays)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        handle.write(buf.getvalue())
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return path


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer | np.floating):
        return value.item()
    raise TypeError(f"checkpoint: cannot serialise {type(value).__name__}")


def load(path: Path, *, config_hash: str) -> dict[str, Any]:
    """Read and verify a checkpoint; CheckpointError on anything unverifiable."""
    try:
        with np.load(path, allow_pickle=False) as z:
            files = set(z.files)
            if not {"state", "sha256"} <= files:
                raise CheckpointError(f"{path}: not a GA checkpoint (members {sorted(files)})")
            payload = z["state"].tobytes()
            digest = z["sha256"].tobytes().decode()
            returns = {name[2:]: np.array(z[name]) for name in files if name.startswith("r_")}
    except CheckpointError:
        raise
    except Exception as exc:
        raise CheckpointError(f"{path}: unreadable checkpoint ({type(exc).__name__}: {exc})") from exc
    if hashlib.sha256(payload).hexdigest() != digest:
        raise CheckpointError(f"{path}: checkpoint digest mismatch; the file is corrupted")
    state = json.loads(payload.decode("utf-8"))
    if state.get("schema") != CHECKPOINT_SCHEMA:
        raise CheckpointError(f"{path}: checkpoint schema {state.get('schema')!r}, expected {CHECKPOINT_SCHEMA}")
    if state.get("config_hash") != config_hash:
        raise CheckpointError(
            f"{path}: checkpoint is for config {state.get('config_hash')!r}, not {config_hash!r}; "
            "a resume must run the same configuration"
        )
    archive = [_restore(d, returns) for d in state["archive"]]
    by_hash = {i.effective_hash: i for i in archive}
    rng = np.random.default_rng()
    rng.bit_generator.state = state["rng"]
    return {
        "generation": state["generation"], "rng": rng, "seen": set(state["registry"]["seen"]),
        "collisions": state["registry"]["collisions"], "archive": archive,
        "population": [by_hash[h] for h in state["population"]], "keep": set(state["keep"]),
        "generations": state["generations"], "trials_used": state["trials_used"], "stop": state["stop"],
        "aborted": state["aborted"], "extra": state["extra"],
    }


__all__ = ["CHECKPOINT_SCHEMA", "CheckpointError", "load", "save"]
