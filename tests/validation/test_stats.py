"""Tests for the shared statistical primitives."""
from __future__ import annotations

import numpy as np

from engine.validation.stats import (
    PERIODS_PER_YEAR,
    annualise_sharpe,
    as_returns,
    deannualise_sharpe,
    equity_from_returns,
    max_drawdown,
    moments,
    sharpe,
)


def test_as_returns_drops_non_finite() -> None:
    a = as_returns([1.0, np.nan, np.inf, 2.0, -np.inf, 3.0])
    assert a.size == 3
    assert list(a) == [1.0, 2.0, 3.0]


def test_sharpe_constant_returns_is_nan() -> None:
    # np.std of a float64 constant is not exactly 0 (~1e-19); sharpe() must
    # return nan rather than blow up.
    out = sharpe(np.full(100, 0.001))
    assert np.isnan(out)


def test_sharpe_known_value_gaussian() -> None:
    rng = np.random.default_rng(0)
    r = rng.normal(loc=0.001, scale=0.01, size=10_000)
    s = sharpe(r)
    # Mean / std ~ 0.1 for these params.
    assert 0.05 < s < 0.2


def test_annualise_deannualise_round_trip() -> None:
    s = 0.5
    out = deannualise_sharpe(annualise_sharpe(s, 252.0), 252.0)
    assert abs(out - s) < 1e-12


def test_annualise_sharpe_uses_sqrt_time() -> None:
    # 0.1 per-bar * sqrt(252) = 1.587.
    assert abs(annualise_sharpe(0.1, 252.0) - 0.1 * np.sqrt(252.0)) < 1e-12


def test_moments_gaussian_near_zero_three() -> None:
    rng = np.random.default_rng(0)
    r = rng.normal(loc=0.0, scale=1.0, size=10_000)
    n, s, g3, g4 = moments(r)
    assert n == 10_000
    assert abs(g3) < 0.1  # roughly 0
    assert abs(g4 - 3.0) < 0.3  # roughly 3 (Pearson)


def test_moments_too_small_falls_back_to_gaussian() -> None:
    n, s, g3, g4 = moments(np.array([0.001, 0.002, 0.003]))
    assert n == 3
    assert g3 == 0.0
    assert g4 == 3.0


def test_max_drawdown_flat_curve_is_zero() -> None:
    e = np.ones(100)
    assert max_drawdown(e) == 0.0


def test_max_drawdown_50_percent_drop() -> None:
    # 1.0 -> 1.0 -> 1.0 -> 0.5 -> 0.5; max DD from peak 1.0 to 0.5 = 50%
    e = np.array([1.0, 1.0, 1.0, 0.5, 0.5])
    assert abs(max_drawdown(e) - 0.5) < 1e-9


def test_max_drawdown_25_percent_drop_after_recovery() -> None:
    # 1.0 -> 2.0 -> 1.5 -> 2.0; max DD from 2.0 peak to 1.5 = 25%
    e = np.array([1.0, 2.0, 1.5, 2.0])
    assert abs(max_drawdown(e) - 0.25) < 1e-9


def test_equity_from_returns_compounds() -> None:
    r = np.array([0.10, -0.10, 0.10])
    e = equity_from_returns(r, start=100.0)
    # 100 * 1.10 * 0.90 * 1.10 = 108.9
    assert abs(e[-1] - 108.9) < 1e-9


def test_periods_per_year_has_expected_keys() -> None:
    assert PERIODS_PER_YEAR["H2"] == 252 * 12
    assert PERIODS_PER_YEAR["D1"] == 252
    assert PERIODS_PER_YEAR["H1"] == 252 * 24
    assert PERIODS_PER_YEAR["M5"] == 252 * 24 * 12
    assert PERIODS_PER_YEAR["trades"] is None
