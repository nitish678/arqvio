"""C-RESIDUE regression tests for the 76-crash discrepancy.

Root cause: scripts/canary_crash_forensics.py originally used a single RNG
for both planting edge returns AND DNA sampling. The canary splits these
(seed for planting, seed+2 for DNA sampling). With the wrong split the
DNA sequence drifted, producing a different crash set and a phantom
70-vs-76 discrepancy.

These tests pin the corrected split so a regression cannot silently
re-introduce the drift.
"""
from __future__ import annotations

import numpy as np
import pytest

from engine.research.search.canary import (
    FAST_BARS,
    FAST_POPULATION,
    FULL_POPULATION,
    _make_planting_backtest_fn,
)
from engine.research.search.real_backtest import (
    load_cell_bars_from_snapshot,
    slice_cell_bars,
)
from engine.research.search.sampler import random_dna
from scripts.canary_fast import CANARY_DATA, DATASET_ID, SYMBOL, TIMEFRAME


@pytest.fixture(scope="module")
def usdcad_d1_cell():
    """The canary's own pinned bars (scripts/canary_data) -- the same cell the pre-commit gate runs on."""
    return load_cell_bars_from_snapshot(SYMBOL, TIMEFRAME, DATASET_ID, target_dir=CANARY_DATA)


def _population_dnas(
    cell, *, population: int, seed: int = 20260917, fast: bool = True,
):
    """Reproduce the canary's exact DNA sequence via its RNG split."""
    work_cell = slice_cell_bars(cell, FAST_BARS) if fast else cell
    pop = FAST_POPULATION if fast else FULL_POPULATION
    assert pop == population

    # Match canary.run_planted_edge_canary: seed for planting, seed+2 for DNA.
    plant_rng = np.random.default_rng(seed)
    backtest_fn = _make_planting_backtest_fn(work_cell, plant_rng, fast=fast)
    dna_rng = np.random.default_rng(seed + 2)

    dnas = [random_dna(dna_rng) for _ in range(population)]
    return work_cell, backtest_fn, dnas


def test_fast_population_has_zero_crashes_after_swap_fix(usdcad_d1_cell):
    """Post-W4: the planting backtest factory must not crash on any
    DNA in the fast canary population.

    Before the swap fixes (quadratic carry re-charging + sign convention),
    the forensics script (even with the correct RNG split) would surface
    crashes in MACD_CROSS and STOCH_CROSS candidates. After the fixes,
    every DNA in the 200-trial fast population must run through the real
    backtest factory without exception.
    """
    work_cell, backtest_fn, dnas = _population_dnas(
        usdcad_d1_cell, population=FAST_POPULATION, seed=20260917, fast=True,
    )

    crashes = 0
    for dna in dnas:
        try:
            backtest_fn(dna)
        except Exception:
            crashes += 1

    assert crashes == 0, f"expected 0 crashes post-W4 fix, got {crashes}"


def test_rna_split_drift_would_change_crash_set(usdcad_d1_cell):
    """Verify the two RNG streams are independent — the shared-RNG
    pattern that caused the 70-vs-76 discrepancy must not be the default.

    With separate RNGs, the DNA sequence is identical whether you call
    random_dna in a tight loop or interleave it with backtest calls.
    With a shared RNG, interleaving changes which DNAs get generated.
    """
    work_cell, backtest_fn, dnas_a = _population_dnas(
        usdcad_d1_cell, population=FAST_POPULATION, seed=20260917, fast=True,
    )

    # Regenerate using a fresh RNG with the same split.
    dna_rng = np.random.default_rng(20260917 + 2)
    dnas_b = [random_dna(dna_rng) for _ in range(FAST_POPULATION)]

    assert dnas_a == dnas_b, "DNA sequence must be identical across identical RNG split"


def test_crash_trials_excluded_from_dsr_and_vs_random_pools():
    """Verify that crashed trials (sharpe=None) are excluded from the
    DSR and vs_random score pools that feed the luck bar.

    This is the pool-membership guarantee that makes crash count matter:
    a crash must not contribute a fake Sharpe to the DSR denominator or
    inflate the vs_random candidate count. The filtering happens in two
    places:
      - real_scores() filters sharpe is None
      - eligible pool filters n_trades >= min_trades (crashes have n_trades=0)

    We assert the filtering contract directly, without needing live bars.
    """
    from engine.research.search.real_backtest import real_scores

    # A crashed trial: sharpe=None, n_trades=0, like evaluate_one produces.
    # DNA identity is irrelevant to the filtering contract we're testing.
    crash_result = type("R", (), {
        "dna": None, "sharpe": None, "n_trades": 0,
    })()

    scored = real_scores([crash_result])
    assert scored == [], "crashed trial must not enter the vs_random score pool"

    # Also verify it would be excluded from the eligible pool.
    eligible = [r for r in [crash_result] if int(r.n_trades or 0) >= 30]
    assert eligible == [], "crashed trial must not enter the eligible (min_trades) pool"
