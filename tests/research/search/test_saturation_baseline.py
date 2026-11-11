"""Boundary saturation read against its uniform-draw baseline (S4 stage 1, W3).

The S3 W4 census flagged STOCH_CROSS.stoch_d_period at 0.722 and
stoch_smoothing at 0.556 -- above the 0.5 flag, but a gene with k legal values
sits on one of its 2 ends with probability 2/k under a uniform draw: 0.667 for
k = 3, 0.400 for k = 5. The baseline fields make that readable beside the raw
share; the flag and every original field are unchanged.
"""
from __future__ import annotations

import dataclasses
from fractions import Fraction

import numpy as np
import pytest

from engine.research.search.degeneracy import (
    FLOAT_EDGE_FRACTION,
    SATURATION_FLAG,
    boundary_saturation,
    saturation_context,
)
from engine.research.search.dna import FailureTaxonomy
from engine.research.search.runner import TrialResult
from engine.research.search.sampler import STOCH_D_PERIOD_VALUES, random_dna

TOL = 1e-12


def test_k3_stoch_d_period_case() -> None:
    """(a) the S3 stoch_d_period reading: 39 of 54 at a boundary, k = 3."""
    share = 39 / 54
    ctx = saturation_context(share, distinct_values=3, value_set_size=3)
    assert ctx["raw_share"] == share
    assert ctx["uniform_baseline"] == float(Fraction(2, 3))
    assert ctx["adjusted_margin"] == pytest.approx(float(Fraction(39, 54) - Fraction(2, 3)), abs=TOL)
    assert ctx["adjusted_margin"] == pytest.approx(0.0556, abs=5e-5)
    assert ctx["small_value_set_caveat"] is True


def test_k5_stoch_smoothing_case() -> None:
    """(b) the S3 stoch_smoothing reading: 30 of 54, k = 5."""
    share = 30 / 54
    ctx = saturation_context(share, distinct_values=5, value_set_size=5)
    assert ctx["uniform_baseline"] == float(Fraction(2, 5))
    assert ctx["adjusted_margin"] == pytest.approx(float(Fraction(30, 54) - Fraction(2, 5)), abs=TOL)
    assert ctx["adjusted_margin"] == pytest.approx(0.1556, abs=5e-5)
    assert ctx["small_value_set_caveat"] is True


def test_k50_clean_case() -> None:
    """(c) a wide range: the baseline is small and the caveat is off."""
    ctx = saturation_context(0.3, distinct_values=37, value_set_size=50)
    assert ctx["uniform_baseline"] == float(Fraction(2, 50))
    assert ctx["adjusted_margin"] == pytest.approx(0.3 - 0.04, abs=TOL)
    assert ctx["distinct_values"] == 37
    assert ctx["small_value_set_caveat"] is False


def test_k1_degenerate_case() -> None:
    """(d) a one-value gene: no baseline, no margin, no crash."""
    ctx = saturation_context(1.0, distinct_values=1, value_set_size=1)
    assert ctx["uniform_baseline"] is None
    assert ctx["adjusted_margin"] is None
    assert ctx["small_value_set_caveat"] is True


def test_float_gene_baseline_is_the_two_edge_bands() -> None:
    ctx = saturation_context(0.1, distinct_values=12, value_set_size=None)
    assert ctx["uniform_baseline"] == 2 * FLOAT_EDGE_FRACTION
    assert ctx["adjusted_margin"] == pytest.approx(0.1 - 2 * FLOAT_EDGE_FRACTION, abs=TOL)
    assert ctx["small_value_set_caveat"] is False


def test_census_entry_for_stoch_d_period_carries_the_baseline() -> None:
    """Through boundary_saturation itself: 54 STOCH survivors, 39 at d = 3 or 5
    -- the S3 numbers -- flag as before and read against 2/3."""
    rng = np.random.default_rng(3)
    trials = []
    for i in range(54):
        d = (3, 5)[i % 2] if i < 39 else 4
        dna = dataclasses.replace(random_dna(rng), signal="STOCH_CROSS", stoch_d_period=d)
        trials.append(TrialResult(dna=dna, sharpe=1.0, n_trades=50))
    sat = boundary_saturation(trials, [FailureTaxonomy.VAULTED.value] * 54, min_trades=30)
    entry = sat["per_gene"]["STOCH_CROSS.stoch_d_period"]
    assert (entry["n"], entry["at_boundary"]) == (54, 39)
    assert entry["share"] == entry["raw_share"] == 39 / 54
    assert entry["value_set_size"] == len(STOCH_D_PERIOD_VALUES) == 3
    assert entry["distinct_values"] == 3
    assert entry["uniform_baseline"] == 2 / 3
    assert entry["adjusted_margin"] == 39 / 54 - 2 / 3
    assert entry["small_value_set_caveat"] is True
    assert entry["share"] > SATURATION_FLAG
    assert "STOCH_CROSS.stoch_d_period" in sat["flagged"]  # the flag itself is unchanged
