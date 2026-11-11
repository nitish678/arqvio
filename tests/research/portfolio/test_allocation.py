"""HRP / ERC (DR §3): known answers, properties, determinism, refusals. No optimality is asserted or claimed."""
from __future__ import annotations

import numpy as np
import pytest

from engine.research.portfolio.allocation import (AllocationError, allocate, covariance, erc, hrp, panel_digest,
                                                  risk_contributions)

DIAG = np.diag([0.04, 0.01, 0.0025, 0.09])


def test_a_diagonal_covariance_gives_inverse_vol_erc_and_inverse_variance_hrp() -> None:
    v = np.diag(DIAG)
    iv = (1 / np.sqrt(v)) / (1 / np.sqrt(v)).sum()
    ivar = (1 / v) / (1 / v).sum()
    np.testing.assert_allclose(erc(DIAG).weights, iv, rtol=1e-12, atol=0)
    np.testing.assert_allclose(hrp(DIAG).weights, ivar, rtol=1e-12, atol=0)


def _equicorr(n: int, rho: float = 0.3, var: float = 0.02) -> np.ndarray:
    return np.full((n, n), rho * var) + np.eye(n) * (var - rho * var)


@pytest.mark.parametrize("n", [2, 3, 5, 8])
def test_equal_correlation_and_variance_give_equal_erc_weights(n: int) -> None:
    # ERC is iterated to its pre-registered stop (risk contributions within ERC_TOL = 1e-10, DR §3); off the
    # diagonal that bounds the weights to ~1e-11 relative, so the symmetric answer is asserted at 1e-9.
    np.testing.assert_allclose(erc(_equicorr(n)).weights, [1 / n] * n, rtol=1e-9, atol=0)


@pytest.mark.parametrize("n", [2, 4, 8])
def test_equal_correlation_gives_equal_hrp_weights_when_the_bisection_is_balanced(n: int) -> None:
    np.testing.assert_allclose(hrp(_equicorr(n)).weights, [1 / n] * n, rtol=1e-12, atol=0)


def test_hrp_on_an_unbalanced_bisection_is_symmetric_not_equal() -> None:
    """S7 DEV-2: HRP bisects by COUNT, so 5 equicorrelated members split 2 / 3 and a 2-member inverse-variance
    cluster is riskier than a 3-member one -- the weights are equal WITHIN each balanced sub-cluster and invariant
    under relabelling, but not 1/5. The DR's 'equal correlation => equal weights' holds for ERC and for HRP only
    when every bisection is balanced."""
    a = hrp(_equicorr(5))
    w = np.asarray(a.weights)
    order = a.detail["order"]
    left, right = order[:2], order[2:]
    assert len(set(np.round(w[left], 15))) == 1 and abs(w.sum() - 1.0) < 1e-12 and not np.allclose(w, 0.2)
    perm = [4, 2, 0, 3, 1]
    c = _equicorr(5)[np.ix_(perm, perm)]
    np.testing.assert_allclose(sorted(hrp(c).weights), sorted(a.weights), rtol=1e-12, atol=0)
    assert len(right) == 3 and np.isclose(w[right].sum() + w[left].sum(), 1.0, rtol=0, atol=1e-12)


def test_a_single_asset_is_all_in() -> None:
    assert erc(np.array([[0.01]])).weights == (1.0,) and hrp(np.array([[0.01]])).weights == (1.0,)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_properties_on_seeded_correlated_panels(seed: int) -> None:
    rng = np.random.default_rng(seed)
    f = rng.normal(size=(500, 2))
    panel = f @ rng.uniform(-1, 1, (2, 6)) * 0.01 + rng.normal(scale=[0.002, 0.005, 0.01, 0.003, 0.02, 0.007],
                                                              size=(500, 6))
    cov = covariance(panel)
    for a in (hrp(cov), erc(cov)):
        w = np.asarray(a.weights)
        assert np.all(w >= 0) and np.all(w <= 1) and abs(w.sum() - 1.0) < 1e-12
    rc = risk_contributions(np.asarray(erc(cov).weights), cov)
    np.testing.assert_allclose(rc, np.full(6, rc.mean()), rtol=1e-8, atol=0)
    assert allocate(panel, list("abcdef")) == allocate(panel, list("abcdef"))  # determinism, bitwise


def test_degenerate_panels_refuse() -> None:
    good = np.random.default_rng(9).normal(size=(10, 3))
    with pytest.raises(AllocationError, match="need >= 2 rows"):
        covariance(good[:1])
    bad = good.copy()
    bad[3, 1] = np.nan
    with pytest.raises(AllocationError, match="non-finite"):
        covariance(bad)
    flat = good.copy()
    flat[:, 2] = 0.5
    with pytest.raises(AllocationError, match=r"zero-variance member column\(s\) \[2\]"):
        covariance(flat)
    with pytest.raises(AllocationError, match="not symmetric"):
        erc(np.array([[1.0, 0.1], [0.2, 1.0]]))
    with pytest.raises(AllocationError, match="member ids"):
        allocate(good, ["a", "b"])


def test_the_artifact_carries_the_method_digest_and_ids() -> None:
    panel = np.random.default_rng(3).normal(size=(50, 2)) * 0.01
    art = allocate(panel, ["x", "y"])
    assert art["panel_sha256"] == panel_digest(panel) and art["members"] == ["x", "y"]
    assert set(art["HRP"]["weights"]) == {"x", "y"} and art["ERC"]["max_rel_dev"] < 1e-10
