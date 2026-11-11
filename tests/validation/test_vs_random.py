"""Vs-Random tests, including the max-vs-max bug-catcher.

Reference: arqvio v1 README documents that a naive max-vs-single
implementation returns p=0.001 on pure noise (declaring "structure
found"), while the correct max-vs-max gives p=0.63. That test must stay.
"""
from __future__ import annotations

import numpy as np
import pytest

from engine.validation.vs_random import vs_random


def test_vs_random_empty_input_raises() -> None:
    with pytest.raises(ValueError, match="both score arrays"):
        vs_random([], np.array([1.0]))


def test_vs_random_warns_when_control_smaller() -> None:
    rng = np.random.default_rng(0)
    real = rng.normal(loc=0.0, scale=0.5, size=2000)
    control = rng.normal(loc=0.0, scale=0.5, size=500)  # smaller
    out = vs_random(real, control, n_bootstrap=500, n_permutations=500, seed=0)
    assert any("smaller" in w for w in out.warnings)


def test_vs_random_warns_when_control_coarse() -> None:
    rng = np.random.default_rng(0)
    real = rng.normal(loc=0.0, scale=0.5, size=50)
    control = rng.normal(loc=0.0, scale=0.5, size=50)  # < 100
    out = vs_random(real, control, n_bootstrap=200, n_permutations=200, seed=0)
    assert any("coarse" in w for w in out.warnings)


def test_vs_random_no_structure_on_pure_noise() -> None:
    """Sweep 10 seeds: pure noise vs pure noise must NOT declare STRUCTURE FOUND.

    On small samples the max of a noise population has noticeable sampling
    noise, so a single comparison can give a small p_max by chance. The
    aggregate behaviour over 10 seeds is the right test: in expectation,
    no seed should yield STRUCTURE FOUND.
    """
    not_structure = 0
    for seed in (3, 11, 17, 23, 29, 31, 37, 41, 43, 47):
        rng = np.random.default_rng(seed)
        real = rng.normal(loc=0.0, scale=1.0, size=2_000)
        control = rng.normal(loc=0.0, scale=1.0, size=2_000)
        out = vs_random(real, control, n_bootstrap=2_000, n_permutations=2_000, seed=seed)
        if "STRUCTURE FOUND" not in out.verdict:
            not_structure += 1
    # At least 9/10 seeds must NOT find structure on pure noise.
    assert not_structure >= 9, (
        f"vs_random found spurious structure on noise in "
        f"{10 - not_structure}/10 seeds - the max-vs-max test is broken"
    )


def test_vs_random_finds_structure_with_real_edge() -> None:
    """Real population shifted + winner above random -> STRUCTURE FOUND."""
    rng = np.random.default_rng(0)
    real = rng.normal(loc=1.0, scale=0.5, size=500)
    control = rng.normal(loc=0.0, scale=0.5, size=500)
    out = vs_random(real, control, n_bootstrap=2_000, n_permutations=2_000, seed=0)
    assert out.p_max < 0.05
    assert out.p_mean < 0.05
    assert out.effect_size > 0.5
    assert "STRUCTURE" in out.verdict


def test_vs_random_compares_max_to_max_not_max_to_single() -> None:
    """The documented trap: a naive max-vs-single would say p_single ~ 0
    on noise (because the max of 2000 noise draws is many sigma above the
    population mean). The correct p_max should be unremarkable.
    """
    rng = np.random.default_rng(7)
    real = rng.normal(loc=0.0, scale=1.0, size=2_000)
    control = rng.normal(loc=0.0, scale=1.0, size=2_000)
    out = vs_random(real, control, n_bootstrap=2_000, n_permutations=2_000, seed=7)
    # p_max must be > 0.05 (the "no structure" threshold) and
    # p_single (the diagnostic for the trap) is allowed to be tiny -
    # that is precisely what makes the naive implementation declare
    # false discoveries.
    assert out.p_max > 0.05
    assert out.p_single < out.p_max, (
        "p_single should be smaller than p_max on noise - if it isn't, "
        "we have the naive max-vs-single bug back."
    )


def test_vs_random_deterministic_with_seed() -> None:
    rng = np.random.default_rng(0)
    real = rng.normal(loc=0.0, scale=0.5, size=500)
    control = rng.normal(loc=0.0, scale=0.5, size=500)
    a = vs_random(real, control, n_bootstrap=1_000, n_permutations=1_000, seed=123)
    b = vs_random(real, control, n_bootstrap=1_000, n_permutations=1_000, seed=123)
    assert a.p_max == b.p_max
    assert a.p_mean == b.p_mean
    assert a.real_max == b.real_max


def test_vs_random_to_dict_round_trip() -> None:
    rng = np.random.default_rng(0)
    real = rng.normal(loc=0.0, scale=0.5, size=100)
    control = rng.normal(loc=0.0, scale=0.5, size=100)
    out = vs_random(real, control, n_bootstrap=200, n_permutations=200, seed=0)
    d = out.to_dict()
    assert d["p_max"] == out.p_max
    assert d["verdict"] == out.verdict
    assert d["n_real"] == 100
    assert d["n_random"] == 100
