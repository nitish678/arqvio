"""The run hash: same config + seed => byte-identical individuals table.

The table is every individual the run evaluated, in admission order, with
its DNA, lineage, engine, backtest metrics and fitness record -- everything
the search decided on, and nothing that depends on the clock. canonical()
is sorted-key compact JSON; wall times and timestamps are removed by key and
non-finite floats become explicit strings, so the text is real JSON and the
same run always produces the same bytes. run_hash is its sha256.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from engine.research.ga.population import Individual

# Keys that carry a clock reading anywhere in a GA document, and the
# "operational" block (checkpoint / resume / monitor arguments): how a run was
# OPERATED, never what it computed, so neither may change its identity.
TIME_KEYS = frozenset({
    "completed_at_utc", "started_at_utc", "as_of_utc", "wall_seconds", "elapsed_seconds",
    "operational",
})


def plain_json(value: Any, *, drop_time: bool) -> Any:
    """``value`` as plain JSON types: str keys, lists for tuples, numpy scalars
    unwrapped, non-finite floats as "nan" / "inf" / "-inf"; clock readings
    (TIME_KEYS) dropped when ``drop_time``."""
    if isinstance(value, dict):
        return {
            str(k): plain_json(v, drop_time=drop_time)
            for k, v in value.items() if not (drop_time and k in TIME_KEYS)
        }
    if isinstance(value, list | tuple):
        return [plain_json(v, drop_time=drop_time) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")
    if hasattr(value, "item") and not isinstance(value, str | bytes):  # numpy scalar
        return plain_json(value.item(), drop_time=drop_time)
    return value


def canonical(obj: Any) -> str:
    """Sorted-key compact JSON of ``obj`` with every clock reading removed."""
    return json.dumps(plain_json(obj, drop_time=True), sort_keys=True, separators=(",", ":"), allow_nan=False)


def individual_row(ind: Individual) -> dict[str, Any]:
    r = ind.result
    return {
        "dna_hash": ind.dna_hash,
        "effective_gene_hash": ind.effective_hash,
        "dna": ind.dna.to_dict(),
        "lineage": ind.lineage(),
        "engine": ind.engine,
        "metrics": None if r is None else {
            "n_trades": r.n_trades,
            "sharpe": r.sharpe,
            "total_return": r.total_return,
            "max_dd": r.max_dd,
            "blocked_max_spread": r.blocked_max_spread,
            "costs": dict(r.costs),
            "exit_reasons": dict(r.exit_reasons),
        },
        "fitness": None if ind.fitness is None else ind.fitness.to_dict(),
    }


def individuals_table(individuals: Iterable[Individual]) -> list[dict[str, Any]]:
    return [individual_row(ind) for ind in individuals]


def run_hash(table: list[dict[str, Any]]) -> str:
    return hashlib.sha256(canonical(table).encode("utf-8")).hexdigest()


__all__ = ["TIME_KEYS", "canonical", "individual_row", "individuals_table", "plain_json", "run_hash"]
