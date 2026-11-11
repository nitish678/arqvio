"""S2-D: the batch path is deterministic and the dedupe cache is honest.

Parallelism is over CANDIDATES, not over bars. Each candidate is a
wholly independent sequential run writing only its own row, so the
thread count cannot change a result -- there is no cross-candidate
reduction whose order could vary. Parallelising over bars would have
been the opposite: equity accumulation is inherently sequential and
reassociating it changes the sum.

These tests are what makes that claim checkable rather than asserted.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from engine.research.backtest.fast_path import (
    DedupeCache,
    record_to_dict,
    run_batch,
    run_fast,
)
from engine.research.backtest.prep import prep_cell
from engine.research.data.mt5_source import BarData, Timeframe
from engine.research.search.effective_hash import effective_gene_hash
from engine.research.search.real_backtest import dna_to_ma_config, spec_for
from engine.research.search.sampler import random_dna

SYMBOL = "USDCADm"
N_BARS = 600
N_CANDIDATES = 40


def _bars() -> BarData:
    rng = np.random.default_rng(21)
    walk = np.cumsum(rng.normal(0.0, 0.0018, size=N_BARS)) + 1.10
    return BarData(
        time=np.array([86400 * 3 + i * 3600 for i in range(N_BARS)], dtype=np.int64),
        open=walk.copy(), high=walk + 0.0012, low=walk - 0.0012,
        close=walk.copy(), tick_volume=np.ones(N_BARS),
        spread=np.full(N_BARS, 14, dtype=np.int32), real_volume=np.zeros(N_BARS),
        symbol=SYMBOL, timeframe=Timeframe.H1,
    )


_CFG = {
    "charge_swap": True,
    "charge_commission": True,
    "slippage_mean_points": 0.3,
    "slippage_std_points": 0.2,
}


@pytest.fixture(scope="module")
def prepped():
    return prep_cell(_bars(), spec_for(SYMBOL))


@pytest.fixture(scope="module")
def genes():
    rng = np.random.default_rng(99)
    out = []
    for _ in range(N_CANDIDATES):
        ma = dna_to_ma_config(random_dna(rng))
        out.append((ma.fast, ma.slow, ma.volume, ma.sl_atr, ma.tp_atr))
    return out


# ---------------------------------------------------------------------------
# Thread-count determinism
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("threads", [1, 2, 4])
def test_batch_results_are_identical_across_thread_counts(
    prepped, genes, threads,
) -> None:
    """N threads must equal 1 thread, bit for bit.

    Not approx: if the thread count could move a result, the campaign
    would be unreproducible on a different machine and no parity claim
    would survive.
    """
    one = run_batch(prepped, genes, n_threads=1, want_equity=False, **_CFG)
    many = run_batch(prepped, genes, n_threads=threads, want_equity=False, **_CFG)

    assert np.array_equal(one.status, many.status)
    assert np.array_equal(one.records, many.records), (
        f"{threads} threads produced different records from 1 thread; "
        f"first differing candidate: "
        f"{int(np.flatnonzero((one.records != many.records).any(axis=1))[0])}"
    )


def test_batch_is_repeatable_within_a_thread_count(prepped, genes) -> None:
    a = run_batch(prepped, genes, n_threads=4, want_equity=False, **_CFG)
    b = run_batch(prepped, genes, n_threads=4, want_equity=False, **_CFG)
    assert np.array_equal(a.records, b.records)


def test_batch_matches_single_candidate_runs(prepped, genes) -> None:
    """The batch is not a different engine from the single path."""
    batch = run_batch(prepped, genes, n_threads=4, want_equity=False, **_CFG)
    for idx, (fast, slow, volume, sl_atr, tp_atr) in enumerate(genes):
        record, _, status = run_fast(
            prepped, fast=fast, slow=slow, volume=volume,
            sl_atr=sl_atr, tp_atr=tp_atr, want_equity=False,
            rng_seed=0, **_CFG,
        )
        assert status == 0
        assert np.array_equal(batch.records[idx], record), (
            f"candidate {idx} differs between batch and single-run paths"
        )


def test_every_candidate_in_the_batch_actually_ran(prepped, genes) -> None:
    """A batch of zero-trade candidates would prove nothing."""
    batch = run_batch(prepped, genes, n_threads=4, want_equity=False, **_CFG)
    assert np.all(batch.status == 0)
    traded = sum(
        1 for r in batch.records if record_to_dict(r)["n_trades"] > 0
    )
    assert traded > N_CANDIDATES // 2, f"only {traded} candidates traded"


def test_equity_rows_are_also_thread_independent(prepped, genes) -> None:
    small = genes[:8]
    one = run_batch(prepped, small, n_threads=1, want_equity=True, **_CFG)
    four = run_batch(prepped, small, n_threads=4, want_equity=True, **_CFG)
    assert np.array_equal(one.equity, four.equity)


# ---------------------------------------------------------------------------
# Dedupe cache
# ---------------------------------------------------------------------------

def test_effective_hash_cache_reuses_result() -> None:
    """Two DNAs differing ONLY in engine-invisible genes.

    One backtest is executed; two rows exist; both are present. W2
    established these are the same backtest with two names -- running
    both wastes work, reporting one loses a trial the luck bar counted.
    """
    base = random_dna(np.random.default_rng(7))
    # friday_close is sampled, stored, hashed -- and never reaches the
    # engine, so it cannot change a result.
    twin = dataclasses.replace(base, friday_close=not base.friday_close)

    assert base.dna_hash != twin.dna_hash, "fixture DNAs are not distinct"
    assert effective_gene_hash(base) == effective_gene_hash(twin), (
        "fixture genes are not engine-invisible; pick another"
    )

    cache = DedupeCache()
    rows = []
    computed = 0
    for dna in (base, twin):
        idx, reused = cache.index_for(effective_gene_hash(dna), computed)
        if not reused:
            computed += 1
        rows.append((dna.dna_hash, idx, reused))

    assert computed == 1, f"{computed} backtests executed, expected 1"
    assert cache.hits == 1
    assert len(rows) == 2, "both DNAs must still produce a row"
    assert rows[0][1] == rows[1][1], "the two rows must share one result"
    assert rows[0][2] is False
    assert rows[1][2] is True


def test_distinct_projections_are_not_deduped() -> None:
    """A gene that DOES reach the engine must not be collapsed."""
    base = random_dna(np.random.default_rng(8))
    other = dataclasses.replace(base, sl_atr=base.sl_atr + 1.0)
    cache = DedupeCache()
    computed = 0
    for dna in (base, other):
        _, reused = cache.index_for(effective_gene_hash(dna), computed)
        if not reused:
            computed += 1
    assert computed == 2, "two different backtests were deduped into one"
    assert cache.hits == 0


def test_dedupe_hits_counts_reuses_not_entries() -> None:
    base = random_dna(np.random.default_rng(9))
    twins = [
        dataclasses.replace(base, friday_close=not base.friday_close),
        dataclasses.replace(base, news_filter=not base.news_filter),
    ]
    cache = DedupeCache()
    computed = 0
    for dna in (base, *twins):
        _, reused = cache.index_for(effective_gene_hash(dna), computed)
        if not reused:
            computed += 1
    assert computed == 1
    assert cache.hits == 2, "three DNAs, one backtest, two reuses"
    assert len(cache) == 1
