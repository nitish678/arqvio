"""A2: the DSR luck bar must scale with the trial-Sharpe dispersion.

Dropping sqrt(var_sr_trials) fixes the expected-max-Sharpe scale to the
theoretical 1/(n_obs-1) regardless of how dispersed the search population
actually was. Two searches with identical n_trials but very different
Sharpe variance must earn materially different luck bars.
"""
from __future__ import annotations

import numpy as np

from engine.validation.dsr import _expected_max_sharpe, dsr


def test_expected_max_scales_with_var_sr_trials() -> None:
    n_obs = 2000
    n_trials = 5000
    tight = _expected_max_sharpe(
        n_obs, n_trials, skew=0.0, excess_kurt=0.0, var_sr_trials=0.01
    )
    wide = _expected_max_sharpe(
        n_obs, n_trials, skew=0.0, excess_kurt=0.0, var_sr_trials=1.0
    )
    # sqrt(1.0)/sqrt(0.01) = 10x scale difference.
    assert wide > tight
    assert wide / tight > 5.0


def test_dsr_luck_bar_differs_for_same_n_trials_different_variance() -> None:
    rng = np.random.default_rng(7)
    r = rng.normal(loc=0.001, scale=0.01, size=2000)
    low_var = dsr(r, n_trials=5000, periods_per_year=252, var_sr_trials=0.02)
    high_var = dsr(r, n_trials=5000, periods_per_year=252, var_sr_trials=2.0)
    assert high_var.expected_max_sr > low_var.expected_max_sr
    # A higher luck bar cannot yield a higher DSR for the same candidate.
    assert high_var.dsr <= low_var.dsr


def test_dsr_defaults_to_theoretical_scale_when_var_absent() -> None:
    rng = np.random.default_rng(11)
    r = rng.normal(loc=0.0, scale=0.01, size=2000)
    out = dsr(r, n_trials=5000, periods_per_year=252)
    assert out.expected_max_sr > 0.0
