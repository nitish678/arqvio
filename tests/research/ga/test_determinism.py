"""The run hash: canonical, clock-free, content-sensitive."""
from __future__ import annotations

import json

import numpy as np

from engine.research.ga import determinism as det
from engine.research.ga.population import Individual
from engine.research.search.effective_hash import effective_gene_hash
from engine.research.search.runner import TrialResult
from engine.research.search.sampler import random_dna


def test_canonical_ignores_key_order_and_clock_readings() -> None:
    a = {"b": 1, "a": [1, 2], "completed_at_utc": "x", "nested": {"wall_seconds": 3.0, "k": 1}}
    b = {"nested": {"k": 1, "wall_seconds": 99.0}, "a": [1, 2], "b": 1, "completed_at_utc": "y"}
    assert det.canonical(a) == det.canonical(b) == '{"a":[1,2],"b":1,"nested":{"k":1}}'


def test_non_finite_floats_and_numpy_scalars_become_plain_json() -> None:
    text = det.canonical({"n": float("nan"), "i": float("inf"), "m": -float("inf"), "s": np.float64(1.5),
                          "k": np.int64(3), "t": (1, 2)})
    assert json.loads(text) == {"n": "nan", "i": "inf", "m": "-inf", "s": 1.5, "k": 3, "t": [1, 2]}


def _ind(seed: int, sharpe: float) -> Individual:
    dna = random_dna(np.random.default_rng(seed))
    ind = Individual(dna=dna, effective_hash=effective_gene_hash(dna), generation=1,
                     parents=("p" * 16,), operators=("crossover", "mutate:sl_atr"), engine="kernel")
    ind.result = TrialResult(dna=dna, sharpe=sharpe, n_trades=40, costs={"spread": 1.0})
    return ind


def test_the_row_carries_what_the_search_decided_on() -> None:
    row = det.individual_row(_ind(1, 1.25))
    assert set(row) == {"dna_hash", "effective_gene_hash", "dna", "lineage", "engine", "metrics", "fitness"}
    assert row["lineage"] == {"generation_found": 1, "parents": ["p" * 16], "operators": ["crossover", "mutate:sl_atr"]}
    assert row["metrics"]["sharpe"] == 1.25 and row["metrics"]["costs"] == {"spread": 1.0}


def test_the_run_hash_is_stable_and_moves_with_any_content() -> None:
    table = det.individuals_table([_ind(1, 1.0), _ind(2, 2.0)])
    same = det.individuals_table([_ind(1, 1.0), _ind(2, 2.0)])
    assert det.run_hash(table) == det.run_hash(same)
    assert det.run_hash(det.individuals_table([_ind(1, 1.0), _ind(2, 2.0000001)])) != det.run_hash(table)
    assert det.run_hash(det.individuals_table([_ind(2, 2.0), _ind(1, 1.0)])) != det.run_hash(table)  # order is content
