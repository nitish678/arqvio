"""MinBTL tests."""
from __future__ import annotations

import numpy as np
import pytest

from engine.core.errors import DataError
from engine.validation.minbtl import minbtl, minbtl_from_returns


def test_minbtl_grows_with_trials() -> None:
    base = minbtl(target_sharpe=1.0, n_trials=1)
    many = minbtl(target_sharpe=1.0, n_trials=10_000)
    assert many.min_years > base.min_years


def test_minbtl_shrinks_with_target_sharpe() -> None:
    """Higher target Sharpe -> smaller MinBTL. Use values that keep the
    variance term (1 - skew*SR + ((kurt-1)/4)*SR^2) positive: with default
    skew=0, kurt=0 this means SR^2 < 4, i.e. SR < 2.0."""
    small = minbtl(target_sharpe=0.5, n_trials=1000)
    big = minbtl(target_sharpe=1.5, n_trials=1000)
    assert small.min_years > big.min_years


def test_minbtl_invalid_target_raises() -> None:
    with pytest.raises(DataError):
        minbtl(target_sharpe=0.0, n_trials=100)
    with pytest.raises(DataError):
        minbtl(target_sharpe=-0.5, n_trials=100)


def test_minbtl_history_satisfaction() -> None:
    out = minbtl(target_sharpe=1.0, n_trials=10_000, history_years=5.0)
    assert out.satisfied is True or out.satisfied is False
    # Manually recompute to confirm.
    assert out.satisfied == (out.history_years >= out.min_years)


def test_minbtl_six_figure_trials_needs_many_years() -> None:
    """Concrete claim from the spec: 100k+ trials against 4 years of H2
    history almost certainly fails MinBTL. Make this explicit."""
    out = minbtl(target_sharpe=1.0, n_trials=200_000, history_years=4.0)
    # MinBTL ~= 1 + 1 * (200000-1) / 1 = 200000 years. History fails.
    assert out.satisfied is False
    assert out.min_years > 4.0


def test_minbtl_from_returns_uses_observed_moments() -> None:
    rng = np.random.default_rng(0)
    r = rng.normal(loc=0.0, scale=0.01, size=2000)
    # Pass an explicit target (the observed Sharpe on noise is near 0 and
    # would fail the > 0 check; this is the realistic use case where the
    # operator hypothesizes a small edge).
    out = minbtl_from_returns(r, n_trials=100, target_sharpe=0.8, periods_per_year=252)
    # No history supplied.
    assert out.satisfied is None
    assert out.history_years is None
    # MinBTL is positive.
    assert out.min_years > 0


def test_minbtl_from_returns_raises_on_zero_variance() -> None:
    with pytest.raises(DataError):
        minbtl_from_returns(np.zeros(100), n_trials=10)
