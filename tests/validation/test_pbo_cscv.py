"""PBO / CSCV tests, including the noise gate.

Ported 2026-09-18 to the rewritten ``pbo_cscv``. Three things changed in the
implementation and nothing had followed them:

  - the matrix is ``(T, N)``: T observations down the ROWS, one COLUMN per
    trial. It used to be ``(S, T)``. Every fixture here is transposed
    accordingly. Both orientations satisfy the shape guards, so a fixture
    left the old way round does not fail - it silently blocks over
    strategies and ranks over time.
  - ``n_combinations`` and ``seed`` are gone. CSCV is now exhaustive over
    C(n_blocks, n_blocks//2) balanced splits, so it is deterministic and
    there is nothing to sample or seed.
  - PBO is the rank-logit definition (fraction of splits where the IS
    winner lands below the OOS median), not the old "fraction of splits
    where the IS winner had a negative OOS mean".

``PBOResult.test_returns`` no longer exists; the per-split record is
``logits``, with ``is_perf_selected`` / ``oos_perf_selected`` alongside it.
"""
# ruff: noqa: N806  # matrix dim names M/S/T are conventional in numerical code
from __future__ import annotations

from math import comb

import numpy as np
import pytest

from engine.core.errors import DataError
from engine.validation.pbo_cscv import pbo_cscv


def test_pbo_cscv_shape_validation() -> None:
    with pytest.raises(DataError):
        pbo_cscv(np.zeros(10))  # 1-D
    with pytest.raises(DataError):
        pbo_cscv(np.zeros((100, 1)))  # N < 2: nothing to rank against
    with pytest.raises(DataError):
        pbo_cscv(np.zeros((100, 2)), n_blocks=1)  # n_blocks < 4
    with pytest.raises(DataError):
        pbo_cscv(np.zeros((100, 2)), n_blocks=9)  # n_blocks odd
    with pytest.raises(DataError):
        pbo_cscv(np.zeros((30, 2)), n_blocks=16)  # T < n_blocks * 2


def test_pbo_cscv_orientation_is_observations_by_trials() -> None:
    """(T, N), not (N, T). The guards accept both, so this is the only check.

    A 40-column, 500-row matrix must report 500 observations and 40 trials.
    Feeding the transpose reports the numbers swapped, which is exactly how
    the defect in ``_compute_pbo`` stayed invisible.
    """
    M = np.random.default_rng(0).normal(0.0, 0.01, size=(500, 40))
    out = pbo_cscv(M, n_blocks=16)
    assert out.n_obs == 500
    assert out.n_trials == 40

    flipped = pbo_cscv(M.T, n_blocks=16)
    assert flipped.n_obs == 40
    assert flipped.n_trials == 500


def test_pbo_cscv_pure_noise_around_half() -> None:
    """GATE: 200 periods x 50 strategies of pure noise, 16 blocks
    -> PBO should be in [0.3, 0.7].

    This is the property the spec calls out as the S1 hard gate
    (overfitting detection on noise).
    """
    rng = np.random.default_rng(2024)
    M = rng.normal(loc=0.0, scale=0.01, size=(200, 50))
    out = pbo_cscv(M, n_blocks=16)
    assert 0.3 <= out.pbo <= 0.7, f"PBO={out.pbo} (expected ~0.5 on noise)"
    assert out.n_splits == comb(16, 8)
    assert out.n_blocks == 16
    assert out.logits.size == comb(16, 8)


def test_pbo_cscv_real_edge_low_pbo() -> None:
    """A matrix with one truly superior strategy and the rest noise should
    produce a PBO well below 0.5."""
    rng = np.random.default_rng(7)
    T = 400
    S = 30
    M = rng.normal(loc=0.0, scale=0.01, size=(T, S))
    # Inject a real edge in the first strategy: a COLUMN, not a row.
    M[:, 0] += 0.005
    out = pbo_cscv(M, n_blocks=16)
    assert out.pbo < 0.4, f"PBO={out.pbo} (expected low for real edge)"


def test_pbo_cscv_perfect_overfit_high_pbo() -> None:
    """A matrix where the in-sample winner reliably underperforms out-of-sample
    should yield PBO well above 0.5.

    Construction: strategy k returns +0.05 on its own block of ROWS and
    -0.05 on every other row. Whenever block k is in-sample, strategy k is
    the IS winner; all of its OOS rows are -0.05, so it ranks bottom OOS.
    """
    n_blocks = 8
    T, S = 400, n_blocks
    block_size = T // n_blocks
    M = np.full((T, S), -0.05)
    for k in range(S):
        rows = np.arange(k * block_size, (k + 1) * block_size)
        M[rows, k] = 0.05
    out = pbo_cscv(M, n_blocks=n_blocks)
    assert out.pbo > 0.8, f"PBO={out.pbo} (expected > 0.8 for disjoint-block overfit)"


def test_pbo_cscv_is_deterministic() -> None:
    """Exhaustive CSCV has nothing to sample, so repeat calls agree exactly.

    This replaces test_pbo_cscv_reproducible_with_seed: the rewritten
    implementation enumerates every balanced split, so there is no seed and
    no sampling error to reproduce.
    """
    rng = np.random.default_rng(0)
    M = rng.normal(loc=0.0, scale=0.01, size=(200, 40))
    a = pbo_cscv(M, n_blocks=8)
    b = pbo_cscv(M, n_blocks=8)
    assert a.pbo == b.pbo
    assert np.array_equal(a.logits, b.logits)


# ---------------------------------------------------------------------------
# Spec-pinned tests (Part 0 of the master prompt, 2026-09-17)
# ---------------------------------------------------------------------------

def test_pbo_split_count_is_combinatorial() -> None:
    """n_splits must be C(n_blocks, n_blocks//2), not n_blocks.

    For n_blocks=8: C(8,4)=70.  For n_blocks=16 (exhaustive): 12,870.
    """
    M8 = np.random.default_rng(0).normal(0, 0.01, (100, 100))
    r8 = pbo_cscv(M8, n_blocks=8)
    assert r8.n_splits == comb(8, 4), f"n_splits={r8.n_splits}, expected {comb(8,4)}"

    M16 = np.random.default_rng(0).normal(0, 0.01, (200, 100))
    r16 = pbo_cscv(M16, n_blocks=16)
    assert r16.n_splits == comb(16, 8), f"n_splits={r16.n_splits}, expected {comb(16,8)}"


def test_pbo_on_zero_edge_is_near_half() -> None:
    """CSCV on iid zero-edge strategies has E[PBO] near 0.5.

    We verify this in expectation over 20 seeds rather than per-seed, because
    any individual seed can produce PBO far from 0.5 when all strategy means
    happen to be the same sign (a valid noise outcome).  The mean over 20
    seeds must be in [0.35, 0.65].

    We also verify the two degenerate sentinel values (exactly 0.0 or 1.0
    on every seed) are absent - those indicate a structural bug.
    """
    pbos = []
    for seed in range(20):
        M = np.random.default_rng(seed).normal(0, 0.011, (3915, 10))
        r = pbo_cscv(M, n_blocks=16)
        assert r.pbo > 0.0, f"seed={seed}: PBO=0.0 exactly (structural bug)"
        assert r.pbo < 1.0, f"seed={seed}: PBO=1.0 exactly (structural bug)"
        pbos.append(r.pbo)
    avg = sum(pbos) / len(pbos)
    assert 0.35 <= avg <= 0.65, (
        f"Mean PBO over 20 seeds = {avg:.4f}; expected near 0.5 on zero-edge input. "
        f"Per-seed: {[f'{p:.3f}' for p in pbos]}"
    )


def test_pbo_near_zero_only_with_dominant_strategy() -> None:
    """With one genuinely dominant strategy, PBO must be meaningfully lower
    than the zero-edge level.  This verifies ranking is not inverted: if it
    were, the dominant strategy would *raise* PBO rather than lower it.

    Uses 10 strategies (not 100) so the dominant column wins IS reliably.
    A 2.5-sigma edge (mean = 2.5 * std) gives SR ~ 159 per period, so it
    wins IS in essentially every split.
    """
    noise_pbo_vals = []
    dominant_pbo_vals = []
    for seed in range(3):
        rng2 = np.random.default_rng(seed)
        std = 0.011
        M_noise = rng2.normal(0, std, (3915, 10))
        M_dom = rng2.normal(0, std, (3915, 10))
        M_dom[:, 3] = rng2.normal(2.5 * std, std, 3915)  # very strong IS winner
        noise_pbo_vals.append(pbo_cscv(M_noise, n_blocks=16).pbo)
        dominant_pbo_vals.append(pbo_cscv(M_dom, n_blocks=16).pbo)

    avg_noise = sum(noise_pbo_vals) / len(noise_pbo_vals)
    avg_dominant = sum(dominant_pbo_vals) / len(dominant_pbo_vals)
    assert avg_dominant < avg_noise, (
        f"Dominant avg PBO={avg_dominant:.4f} should be < noise avg PBO={avg_noise:.4f}; "
        f"ranking may be inverted (dominant: {dominant_pbo_vals}, noise: {noise_pbo_vals})"
    )


def test_pbo_degradation_slope_separates_noise_from_dominance() -> None:
    """The level-based diagnostic must agree with the rank-based one.

    ``degradation_slope`` regresses the winner's OOS performance on its IS
    performance across splits. On noise the winner is a different strategy
    each split and the slope is shallow; with one dominant column the same
    strategy wins throughout and its IS and OOS levels move together, which
    drives the slope toward -1. A near-zero PBO alongside a slope near -1 is
    therefore agreement, not contradiction.
    """
    rng = np.random.default_rng(0)
    std = 0.011
    M_noise = rng.normal(0, std, (3915, 10))
    M_dom = rng.normal(0, std, (3915, 10))
    M_dom[:, 3] = rng.normal(2.5 * std, std, 3915)

    noise = pbo_cscv(M_noise, n_blocks=16)
    dominant = pbo_cscv(M_dom, n_blocks=16)
    assert dominant.degradation_slope < noise.degradation_slope
    assert dominant.pbo < noise.pbo
