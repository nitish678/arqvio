"""PSR tests.

Known answers and monotonicity in N.
"""
from __future__ import annotations

import numpy as np
import pytest

from engine.core.errors import DataError
from engine.validation.psr import psr


def test_psr_short_input_raises() -> None:
    with pytest.raises(DataError):
        psr(np.array([0.01]))
    with pytest.raises(DataError):
        psr(np.array([], dtype=np.float64))


def test_psr_positive_returns_have_high_psr() -> None:
    rng = np.random.default_rng(42)
    r = rng.normal(loc=0.001, scale=0.01, size=2000)
    out = psr(r)
    assert 0.0 < out.psr < 1.0
    assert out.n == 2000
    assert out.sr_star == 0.0
    # A clearly positive mean should give a PSR comfortably above 0.5.
    assert out.psr > 0.9


def test_psr_negative_returns_have_low_psr() -> None:
    rng = np.random.default_rng(0)
    r = rng.normal(loc=-0.001, scale=0.01, size=2000)
    out = psr(r)
    assert out.psr < 0.1


def test_psr_pure_noise_around_half() -> None:
    rng = np.random.default_rng(7)
    # Use a large sample so the sample mean is close to 0 (a small sample
    # can produce a sample mean with high sampling noise).
    r = rng.normal(loc=0.0, scale=0.01, size=20_000)
    out = psr(r)
    # Pure noise should give a PSR near 0.5 (allow a wide band).
    assert 0.1 < out.psr < 0.9, f"PSR={out.psr} on pure noise (expected ~0.5)"


def test_psr_grows_with_n_at_fixed_sharpe() -> None:
    """Larger sample at the same true Sharpe should give a larger PSR."""
    rng = np.random.default_rng(11)
    out_small = psr(rng.normal(loc=0.001, scale=0.01, size=200))
    out_large = psr(rng.normal(loc=0.001, scale=0.01, size=2000))
    assert out_large.psr > out_small.psr


def test_psr_zero_variance_returns_half_when_equal() -> None:
    out = psr(np.zeros(100))
    assert out.psr == 0.5


def test_psr_zero_variance_above_benchmark_returns_one() -> None:
    out = psr(np.full(100, 0.001), sr_star=0.0)
    assert out.psr == 1.0


def test_psr_annualization_changes_sr_by_sqrt_time() -> None:
    """Annualizing multiplies sr_hat by sqrt(periods_per_year)."""
    rng = np.random.default_rng(3)
    r = rng.normal(loc=0.001, scale=0.01, size=2000)
    raw = psr(r)
    ann = psr(r, periods_per_year=252)
    # sr_hat annualised = sr_hat per-obs * sqrt(252).
    expected = raw.sr_hat * np.sqrt(252.0)
    assert abs(ann.sr_hat - expected) < 1e-9, (
        f"annualised sr_hat={ann.sr_hat}, expected={expected}"
    )
