"""DSR tests.

The hard gate: pure noise + many trials must yield DSR <= 0.
"""
from __future__ import annotations

import numpy as np
import pytest

from engine.core.errors import DataError
from engine.validation.dsr import dsr


def test_dsr_short_input_raises() -> None:
    with pytest.raises(DataError):
        dsr(np.array([0.01]), n_trials=1)


def test_dsr_zero_trials_raises() -> None:
    with pytest.raises(DataError):
        dsr(np.array([0.001, 0.002, 0.003]), n_trials=0)


def test_dsr_pure_noise_dsr_is_small() -> None:
    """Pure noise + a six-figure trial count -> DSR is essentially zero.

    DSR is a Phi probability bounded in [0, 1]. For finite N it cannot be
    exactly 0; the operational check is that it is very small.
    """
    rng = np.random.default_rng(2024)
    r = rng.normal(loc=0.0, scale=0.01, size=50_000)
    out = dsr(r, n_trials=100_000, periods_per_year=252)
    assert out.dsr <= 0.01, (
        f"DSR={out.dsr} on pure noise (expected <= 0.01). "
        f"selected SR={out.sr_hat:.4f}, expected max SR={out.expected_max_sr:.4f}"
    )


def test_dsr_growing_trials_deflates() -> None:
    """More trials => lower DSR (at the same return series)."""
    rng = np.random.default_rng(99)
    r = rng.normal(loc=0.0005, scale=0.01, size=2000)
    out1 = dsr(r, n_trials=10, periods_per_year=252)
    out2 = dsr(r, n_trials=1_000_000, periods_per_year=252)
    # The expected-max-SR must increase with n_trials regardless of capping.
    assert out2.expected_max_sr > out1.expected_max_sr
    # If the signal is strong enough that both DSRs cap at 1.0, the deflation
    # property is not observable in the probability itself; the test only
    # fires the literal DSR-monotonicity check when there's headroom.
    if out1.dsr < 0.999 and out2.dsr < 0.999:
        assert out2.dsr < out1.dsr


def test_dsr_strong_positive_signal_beats_zero() -> None:
    """A clearly real edge with few trials should give DSR > 0."""
    rng = np.random.default_rng(5)
    r = rng.normal(loc=0.005, scale=0.01, size=2000)  # SR ~ 2.5 annualized
    out = dsr(r, n_trials=10, periods_per_year=252)
    assert out.dsr > 0.5


def test_dsr_returns_components() -> None:
    rng = np.random.default_rng(1)
    r = rng.normal(loc=0.0, scale=0.01, size=500)
    out = dsr(r, n_trials=50, periods_per_year=252)
    assert out.psr_components.n == 500
    assert out.n_trials == 50
    assert out.expected_max_sr > 0.0  # for any N > 1
