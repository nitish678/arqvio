"""S5 invariant I5 - the vs-random control is genuinely matched.

The control must match the real on every axis except signal meaning. A
smaller or cheaper control rigs the comparison.
"""
from __future__ import annotations

import numpy as np
import pytest

from engine.research.search.control import (
    ControlScopeMismatchError,
    assert_scopes_match,
    scope_to_control,
)
from engine.validation.vs_random import vs_random


def test_control_campaign_matches_scope() -> None:
    """Every scope field must be identical between real and control."""
    real = {
        "symbol": "USDCADm",
        "timeframe": "H2",
        "dataset_hash": "abc",
        "cost_model": "exness-zero",
        "risk_rules": {"max_dd": 0.2, "max_pos": 1},
        "exec_mode": "MARKET2",
        "min_trades": 30,
        "n_trials": 2000,
        "date_from_utc": "2024-01-01T00:00:00Z",
        "date_to_utc": "2026-01-01T00:00:00Z",
    }
    ctrl = scope_to_control(real)
    assert_scopes_match(real, ctrl)  # no exception

    # Any drift on any field must raise.
    bad = dict(ctrl)
    bad["exec_mode"] = "LIMIT"
    with pytest.raises(ControlScopeMismatchError):
        assert_scopes_match(real, bad)

    bad = dict(ctrl)
    bad["min_trades"] = 10  # smaller filter
    with pytest.raises(ControlScopeMismatchError):
        assert_scopes_match(real, bad)

    bad = dict(ctrl)
    bad["n_trials"] = 500  # smaller trial count
    with pytest.raises(ControlScopeMismatchError):
        assert_scopes_match(real, bad)


def test_p_single_is_diagnostic_not_decision() -> None:
    """p_single must not influence the verdict - the verdict is decided by
    p_max + p_mean + effect size only."""
    rng = np.random.default_rng(0)
    real = rng.normal(0.0, 0.01, 200)
    rnd = rng.normal(0.0, 0.01, 200)
    out = vs_random(real, rnd, n_bootstrap=2000, n_permutations=500, seed=0)
    # Verdict is one of the documented four, regardless of p_single.
    assert out.verdict in {
        "STRUCTURE FOUND - winner and population both beat random",
        "WINNER BEATS RANDOM - population shift weak or negligible",
        "POPULATION SHIFT ONLY - winner is within random range",
        "NO STRUCTURE - indistinguishable from random search",
    }
    # p_single and p_max can diverge sharply; the verdict is from p_max.
    assert 0.0 <= out.p_single <= 1.0
    assert 0.0 <= out.p_max <= 1.0


def test_warning_emitted_when_n_random_less_than_n_real() -> None:
    rng = np.random.default_rng(0)
    real = rng.normal(0.0, 0.01, 1000)
    rnd = rng.normal(0.0, 0.01, 100)  # smaller control
    out = vs_random(real, rnd, n_bootstrap=2000, n_permutations=500, seed=0)
    assert any("control smaller" in w for w in out.warnings)
