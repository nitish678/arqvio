"""S2-C step 0: the kernel's summation must be numpy's, bit-for-bit.

This is the foundation every later parity test rests on. np.add.reduce
on contiguous float64 does not sum left to right -- it uses eight
accumulators in blocks of eight, combined in a fixed tree, and splits
recursively above 128 elements. A sequential sum or a prefix-sum
difference agrees to about 1e-16 and disagrees in the last bit.

One ulp is enough to matter. The MA-cross decision is
``fast_prev <= slow_prev and fast_now > slow_now``; on a bar where the
two means are within an ulp, a different rounding flips the comparison
and the candidate takes a different trade. That surfaces later as an
unexplained parity failure that looks like a logic bug.
"""
from __future__ import annotations

import numpy as np
import pytest

from engine.research.backtest.kernel import _pairwise_sum, _slice_mean

# Sizes chosen to hit every branch: sequential (<8), the eight-accumulator
# block (8..128) including non-multiples of 8, and the recursive split
# (>128). 37 and 9 are real MA windows the sampler produces.
SIZES = (1, 2, 7, 8, 9, 15, 16, 37, 63, 64, 100, 127, 128, 129, 150, 199, 256, 1000)
STARTS = (0, 1, 3, 7, 13, 100, 2000)


@pytest.fixture(scope="module")
def series() -> np.ndarray:
    rng = np.random.default_rng(0)
    # Prices, not standard normals: the magnitudes here are what the
    # cancellation behaviour actually depends on.
    return (1.10 + rng.normal(0.0, 0.01, size=5000)).astype(np.float64)


def test_pairwise_sum_is_bit_exact_against_numpy(series) -> None:
    checked = 0
    for n in SIZES:
        for start in STARTS:
            if start + n > series.size:
                continue
            got = _pairwise_sum(series, start, n)
            expected = float(np.sum(series[start : start + n]))
            assert got == expected, (
                f"n={n} start={start}: {got!r} != {expected!r} "
                f"(delta {abs(got - expected):.3e})"
            )
            checked += 1
    assert checked >= 80, f"only {checked} shapes exercised"


def test_slice_mean_is_bit_exact_against_numpy(series) -> None:
    for n in SIZES:
        for start in STARTS:
            if start + n > series.size:
                continue
            got = _slice_mean(series, start, n)
            expected = float(np.mean(series[start : start + n]))
            assert got == expected, f"n={n} start={start}: {got!r} != {expected!r}"


def test_the_recursive_branch_is_actually_exercised(series) -> None:
    """Sizes above 128 take a different code path; prove it runs."""
    n = 1000
    assert n > 128
    assert _pairwise_sum(series, 0, n) == float(np.sum(series[:n]))


def test_a_naive_sequential_sum_would_not_have_passed(series) -> None:
    """The control: shows this test can fail.

    If a left-to-right sum agreed with numpy everywhere, none of the
    above would be evidence of anything.
    """
    n, start = 1000, 0
    naive = 0.0
    for i in range(n):
        naive += float(series[start + i])
    expected = float(np.sum(series[start : start + n]))
    assert naive != expected, (
        "a sequential sum matched numpy exactly on this fixture; the "
        "fixture is too benign to detect a summation-order error"
    )
    # And the kernel's version does match.
    assert _pairwise_sum(series, start, n) == expected


def test_fastmath_is_not_enabled_on_the_summation() -> None:
    """fastmath would license reassociation and break all of the above."""
    for fn in (_pairwise_sum, _slice_mean):
        for sig, flags in fn.targetoptions.items() if hasattr(fn, "targetoptions") else []:
            del sig, flags
        assert not fn.targetoptions.get("fastmath", False), (
            f"{fn.__name__} has fastmath enabled"
        )
