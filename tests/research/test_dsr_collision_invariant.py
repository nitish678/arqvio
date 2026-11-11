"""The DSR-provenance invariant must catch defect #6 and pass a healthy run.

Ported 2026-09-18 from the pairwise form to the population form. The old
check compared candidates pairwise on ``round(dsr, 12)`` and raised when two
with materially different Sharpe shared a key. That key is an ABSOLUTE 1e-12
grid applied to a quantity spanning twelve orders of magnitude: at DSR ~1e-9
genuine candidate spacing is ~1e-11, so the check ran below its own
resolution and aborted GBPJPYm D1 at DSR=1.677e-09.

Defect #6 - DSR computed from one shared market-return series - makes nearly
EVERY candidate share a value, not two of them:

    USDCADm D1, healthy      1827/5000 distinct = 0.365
    USDCADm H4, healthy      1637/5000 distinct = 0.327
    one shared series           1/5000 distinct = 0.000

so the invariant is now a population ratio with a 1% floor.
"""
from __future__ import annotations

import numpy as np
import pytest

from engine.research.search.dna import DNA, FailureTaxonomy
from engine.research.search.runner import (
    MIN_DSR_POPULATION,
    TrialResult,
    _assert_dsr_is_candidate_specific,
)

POPULATION = 200  # above MIN_DSR_POPULATION, so the ratio carries signal


def _dna(seed: int) -> DNA:
    return DNA(
        bias="NONE", filter="NONE", signal="RSI_REVERSION", signal2="",
        exec_mode="MARKET2", risk="FIXED_ATR", mgmt="NONE",
        sl_atr=1.0, tp_atr=2.0, risk_pct=0.01,
        friday_close=False, news_filter=False,
        daily_dd_limit=0.03, max_agg_risk=0.02,
        max_feature_lookback=40 + seed, max_hold_time=20,
        window_start_hour_utc=0, window_end_hour_utc=23,
    )


def _population(
    n: int, *, dsr_of, sharpe_of=lambda i: 0.5 + 0.001 * i, seed: int = 0
) -> list[TrialResult]:
    rng = np.random.default_rng(seed)
    return [
        TrialResult(
            dna=_dna(i), outcome=FailureTaxonomy.EVALUATED.value,
            sharpe=sharpe_of(i),
            n_trades=50, total_return=0.1, max_dd=0.1,
            post_dsr=dsr_of(i),
            per_period_returns=rng.normal(0, 0.01, 500),
        )
        for i in range(n)
    ]


def test_invariant_fires_on_defect6_shared_dsr() -> None:
    """One DSR value across the whole population is defect #6."""
    results = _population(POPULATION, dsr_of=lambda i: 0.42)
    with pytest.raises(AssertionError, match="not candidate-specific"):
        _assert_dsr_is_candidate_specific(results)


def test_invariant_passes_when_dsr_tracks_sharpe() -> None:
    results = _population(POPULATION, dsr_of=lambda i: 0.30 + 0.001 * i)
    _assert_dsr_is_candidate_specific(results)  # no raise


def test_invariant_passes_at_the_measured_healthy_ratio() -> None:
    """0.327 distinct - the worst of the two 2026-09-18 cells - must pass."""
    n_distinct = int(POPULATION * 0.327)
    results = _population(
        POPULATION, dsr_of=lambda i: 0.20 + (i % n_distinct) * 1e-4
    )
    _assert_dsr_is_candidate_specific(results)  # no raise


def test_invariant_fires_on_a_fully_saturated_population() -> None:
    """A population pinned to one boundary value now fires, deliberately.

    The pairwise form exempted DSR == 0.0 and == 1.0 on the grounds that a
    cluster of clearly unskilled or clearly skilled candidates legitimately
    pins there. Under a population check that exemption would blank the
    defect's own signature, and it is not needed: reaching exactly 0.0
    requires norm.cdf to underflow at z < -38, which at n ~ 4000 bars needs
    a per-bar Sharpe ~0.6 below SR*, and exactly 1.0 requires the mirror of
    that for EVERY candidate at once. Values merely near the boundary stay
    distinct floats and pass.
    """
    results = _population(
        POPULATION, dsr_of=lambda i: 1.0, sharpe_of=lambda i: 3.0 + i
    )
    with pytest.raises(AssertionError, match="not candidate-specific"):
        _assert_dsr_is_candidate_specific(results)

    # Near-boundary but distinct: passes.
    near = _population(POPULATION, dsr_of=lambda i: 1.0 - i * 1e-12)
    _assert_dsr_is_candidate_specific(near)


def test_invariant_declines_to_judge_small_populations() -> None:
    """Below MIN_DSR_POPULATION a ratio carries no signal."""
    tiny = _population(MIN_DSR_POPULATION - 1, dsr_of=lambda i: 0.42)
    _assert_dsr_is_candidate_specific(tiny)  # no raise


def test_invariant_ignores_candidates_without_a_dsr() -> None:
    """Unreliable candidates carry post_dsr=None and are not counted."""
    results = _population(POPULATION, dsr_of=lambda i: 0.30 + 0.001 * i)
    for r in results[:150]:
        r.post_dsr = None
        r.unreliable_dsr = True
    # 50 finite DSRs left, below the floor: declines to judge rather than
    # firing on a thin sample.
    _assert_dsr_is_candidate_specific(results)
