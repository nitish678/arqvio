"""Canary tests: the full S5 pipeline must find an obvious planted edge.

The reusable canary lives in engine/research/search/canary.py so campaign
preflight can call the same code. These tests exercise both profiles:
    - fast: sliced bars + small population, asserted < 60s (preflight budget)
    - full: real cell bars + full population (weekly)
"""
from __future__ import annotations

import time

import pytest

from engine.research.search.canary import run_planted_edge_canary
from engine.research.search.real_backtest import load_cell_bars_from_snapshot
from scripts.canary_fast import CANARY_DATA, DATASET_ID, SYMBOL, TIMEFRAME


@pytest.fixture(scope="module")
def usdcad_d1_cell():
    """The canary's own pinned bars (scripts/canary_data) -- the same cell the pre-commit gate runs on."""
    return load_cell_bars_from_snapshot(SYMBOL, TIMEFRAME, DATASET_ID, target_dir=CANARY_DATA)


PREFLIGHT_BUDGET_S = 60.0


def test_fast_canary_finds_planted_edge_within_preflight_budget(
    usdcad_d1_cell, tmp_path
) -> None:
    """The gate is the planted edge; the clock is a guardrail, not the point.

    Budget raised from 30s to 60s on 2026-09-18: PBO/CSCV now actually runs
    (it used to raise TypeError inside a bare except and return None), which
    costs ~13s at any realistic matrix size. Shrinking FAST_BARS or
    FAST_POPULATION to fit 30s would weaken the preflight it exists to
    provide - fewer bars means fewer CSCV blocks, fewer candidates means a
    weaker ranking. See docs/DECISIONS.md.
    """
    started = time.time()
    result = run_planted_edge_canary(
        usdcad_d1_cell, fast=True, tmp_dir=str(tmp_path)
    )
    elapsed = time.time() - started
    assert result.passed, f"fast canary failed: {result.reason} ({result})"
    # UPDATED IN S1/W4, same replaced policy as the canary's own fifth
    # check. STRUCTURE FOUND additionally requires the whole eligible
    # POPULATION to beat the control pool, which is a property of the
    # cell and the strategy family rather than of planted-edge
    # detection. Fixing the two swap bugs (quadratic re-charging, and a
    # pay rate treated as a credit) made it false here: the population
    # stopped being paid to hold positions. The plant is still found.
    assert result.exit_verdict, "a verdict must still be produced"
    assert result.dsr_within is not None and result.dsr_within > 0.5
    assert result.vs_random_p_max is not None and result.vs_random_p_max < 0.05
    assert result.crashes == 0
    assert result.git_sha != "unknown", (
        f"canary must report the commit it ran on, got git_sha={result.git_sha!r}"
    )
    assert elapsed < PREFLIGHT_BUDGET_S, (
        f"fast canary took {elapsed:.1f}s (budget {PREFLIGHT_BUDGET_S:.0f}s)"
    )


def test_full_canary_finds_planted_edge(usdcad_d1_cell, tmp_path) -> None:
    result = run_planted_edge_canary(
        usdcad_d1_cell, fast=False, tmp_dir=str(tmp_path)
    )
    assert result.passed, f"full canary failed: {result.reason} ({result})"
    # UPDATED IN S1/W4, same replaced policy as the canary's own fifth
    # check. STRUCTURE FOUND additionally requires the whole eligible
    # POPULATION to beat the control pool, which is a property of the
    # cell and the strategy family rather than of planted-edge
    # detection. Fixing the two swap bugs (quadratic re-charging, and a
    # pay rate treated as a credit) made it false here: the population
    # stopped being paid to hold positions. The plant is still found.
    assert result.exit_verdict, "a verdict must still be produced"
    assert result.dsr_within is not None and result.dsr_within > 0.5
    assert result.vs_random_p_max is not None and result.vs_random_p_max < 0.05
    assert result.crashes == 0
    assert result.git_sha != "unknown"
