"""The S1 HARD GATE: synthetic pure noise must yield PBO ~= 0.5 and DSR ~ 0.

This is the *integration* test of the validation core. It does not test
individual functions; it asserts the system as a whole rejects noise.

If this test fails, the validation math is broken and the entire downstream
pipeline would be decoration. Per the master spec, "if this gate fails,
everything downstream is decoration. Fix it before writing the GA."

Note on DSR's numeric gate: DSR is Phi((SR_hat - E[max SR]) * sqrt(n-1) /
sqrt(V[SR_hat])). Phi is a continuous CDF bounded in [0, 1]. For finite N
the gate is therefore not "DSR == 0" (which is unreachable) but
"DSR is very small, many orders of magnitude smaller than on a real signal".
We use 0.01 as the operational ceiling for noise; 0.5 as the floor for a
real edge; the ratio of the two is what the gate is really testing.
"""
# ruff: noqa: N806  # matrix dim names M/S/T are conventional in numerical code
from __future__ import annotations

from functools import lru_cache

import numpy as np

from engine.validation.dsr import dsr
from engine.validation.pbo_cscv import pbo_cscv
from engine.validation.vs_random import vs_random

# Operational ceilings for the noise gate.
NOISE_DSR_CEILING = 0.01     # DSR on pure noise must be below this.
EDGE_DSR_FLOOR = 0.5        # DSR on a clear edge must be above this.
PBO_NOISE_LO = 0.30         # PBO on pure noise must be in this band.
PBO_NOISE_HI = 0.70
VS_RANDOM_NOISE_LO = 0.10   # p(real > random) on noise must be in this band.
VS_RANDOM_NOISE_HI = 0.90


NOISE_SEEDS = (11, 3, 17, 23, 29, 31, 37, 41, 43, 47)


@lru_cache(maxsize=1)
def _noise_pbo() -> tuple[float, tuple[float, ...]]:
    """Mean PBO over NOISE_SEEDS on pure noise, plus the per-seed values.

    Asserted in aggregate, not per-seed. A single CSCV draw on 50 iid
    zero-edge columns ranges from 0.28 to 0.88 across these seeds: when the
    column means happen to share a sign, the IS winner's OOS rank is a
    coin flip amplified by the rank granularity of 50 columns. That is a
    valid noise outcome, not a gate failure. Same reasoning, and the same
    aggregate form, as test_vs_random_gate_pure_noise_does_not_look_good
    below and test_pbo_on_zero_edge_is_near_half in test_pbo_cscv.py.
    """
    vals = []
    for seed in NOISE_SEEDS:
        rng = np.random.default_rng(seed)
        M = rng.normal(loc=0.0, scale=0.01, size=(200, 50))  # (T, N)
        vals.append(float(pbo_cscv(M, n_blocks=16).pbo))
    return sum(vals) / len(vals), tuple(vals)


def _noise_returns(n: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=0.01, size=n)


def test_dsr_gate_pure_noise_dsr_is_small() -> None:
    """Pure noise + a six-figure trial count -> DSR is essentially zero."""
    r = _noise_returns(50_000, seed=2024)
    out = dsr(r, n_trials=100_000, periods_per_year=252)
    assert out.dsr <= NOISE_DSR_CEILING, (
        f"GATE FAIL: DSR={out.dsr:.6f} on pure noise (expected <= {NOISE_DSR_CEILING}). "
        f"selected SR={out.sr_hat:.4f}, expected max SR={out.expected_max_sr:.4f}"
    )


def test_dsr_gate_pure_noise_vs_real_signal_separation() -> None:
    """The signal we want: DSR(noise) << DSR(edge).

    The math is doing real work only if the noise and edge DSR differ by
    several orders of magnitude. A ratio of at least 100 is the spec's
    bar (edge DSR > 0.5, noise DSR < 0.005; 100x separation).
    """
    rng = np.random.default_rng(2025)
    noise = rng.normal(loc=0.0, scale=0.01, size=50_000)
    edge = rng.normal(loc=0.005, scale=0.01, size=10_000)
    dsr_noise = dsr(noise, n_trials=100_000, periods_per_year=252)
    dsr_edge = dsr(edge, n_trials=100_000, periods_per_year=252)
    assert dsr_edge.dsr > EDGE_DSR_FLOOR, (
        f"DSR(edge)={dsr_edge.dsr:.4f} (expected > {EDGE_DSR_FLOOR})"
    )
    assert dsr_noise.dsr < NOISE_DSR_CEILING, (
        f"DSR(noise)={dsr_noise.dsr:.6f} (expected < {NOISE_DSR_CEILING})"
    )
    # Separation check: edge should be at least 100x noise.
    assert dsr_edge.dsr > 100 * max(dsr_noise.dsr, 1e-12), (
        f"Separation too small: DSR(edge)={dsr_edge.dsr:.4f}, "
        f"DSR(noise)={dsr_noise.dsr:.6f}"
    )


def test_pbo_gate_pure_noise_yields_pbo_near_half() -> None:
    """Pure noise across many fake strategies -> PBO in the noise band."""
    mean_pbo, per_seed = _noise_pbo()
    assert PBO_NOISE_LO <= mean_pbo <= PBO_NOISE_HI, (
        f"GATE FAIL: mean PBO={mean_pbo:.3f} over {len(per_seed)} seeds on pure "
        f"noise (expected ~0.5 +/- 0.2). Per-seed: "
        f"{[f'{v:.3f}' for v in per_seed]}"
    )


def test_vs_random_gate_pure_noise_does_not_look_good() -> None:
    """Pure noise real vs pure noise control -> verdict NOT "STRUCTURE FOUND".

    On small samples the max of a noise population has noticeable sampling
    noise (max-of-2000 from N(0, 0.01) is ~0.039 ± 0.003), so a single
    comparison can give a small p_max by chance. We aggregate over 10
    seeds and assert the aggregate behaviour.
    """
    not_structure = 0
    for seed in (3, 11, 17, 23, 29, 31, 37, 41, 43, 47):
        rng = np.random.default_rng(seed)
        real = rng.normal(loc=0.0, scale=0.01, size=5000)
        control = rng.normal(loc=0.0, scale=0.01, size=5000)
        out = vs_random(real, control, n_bootstrap=5_000, n_permutations=5_000, seed=seed)
        if "STRUCTURE FOUND" not in out.verdict:
            not_structure += 1
    # The hard gate: at least 8/10 seeds must NOT find structure on pure noise.
    # This is the integration test for the S1 spec: "pure noise must yield
    # vs-random p > 0.05" - interpreted as the AGGREGATE behaviour, not a
    # single-seed p value (which is uniform on [0, 1] by definition).
    assert not_structure >= 8, (
        f"GATE FAIL: vs_random found spurious structure on noise in "
        f"{10 - not_structure}/10 seeds"
    )


def test_combined_gate_pure_noise_passes_every_subgate() -> None:
    """The combined hard gate: DSR small AND PBO ~ 0.5 on pure noise."""
    r = _noise_returns(50_000, seed=2024)
    dsr_out = dsr(r, n_trials=100_000, periods_per_year=252)

    mean_pbo, per_seed = _noise_pbo()

    assert dsr_out.dsr <= NOISE_DSR_CEILING, (
        f"DSR={dsr_out.dsr} (expected < {NOISE_DSR_CEILING} on noise)"
    )
    assert PBO_NOISE_LO <= mean_pbo <= PBO_NOISE_HI, (
        f"mean PBO={mean_pbo:.3f} over {len(per_seed)} seeds "
        f"(expected ~0.5 on noise). Per-seed: {[f'{v:.3f}' for v in per_seed]}"
    )
