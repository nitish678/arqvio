"""HRP and ERC weights from the member return panel (DR §3). Deterministic allocation rules; NO optimality claim.

Input: a (rows x members) return panel; covariance = the sample covariance (``numpy.cov``, ddof = 1), no shrinkage.
A panel with < 2 rows, a non-finite value or a zero-variance column REFUSES (``AllocationError``).

* HRP (Lopez de Prado): correlation distance sqrt(0.5 (1 - rho)); single linkage (``scipy`` ``linkage``,
  method="single") on the condensed distance; the leaves' quasi-diagonal order; recursive bisection, each half
  weighted by the inverse of its inverse-variance portfolio's variance. On a diagonal covariance this is the
  inverse-VARIANCE portfolio.
* ERC: weights with equal risk contributions w_i (S w)_i. Solved for x > 0 with x_i (S x)_i = 1/n by cyclic
  coordinate updates -- each coordinate's equation is a quadratic whose positive root is closed-form -- from the
  inverse-volatility start; w = x / sum(x). Stops when the risk contributions' max relative deviation < ERC_TOL, or
  refuses after ERC_MAX_SWEEPS. On a diagonal covariance this is the inverse-VOLATILITY portfolio.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

ERC_TOL = 1e-10
ERC_MAX_SWEEPS = 10_000


class AllocationError(ValueError):
    """A panel or covariance the allocators refuse (named in the message)."""


@dataclass(frozen=True)
class Allocation:
    method: str
    weights: tuple[float, ...]
    detail: dict


def panel_digest(panel: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(panel, dtype=np.float64).tobytes()).hexdigest()


def covariance(panel: np.ndarray) -> np.ndarray:
    p = np.asarray(panel, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] < 2 or p.shape[1] < 1:
        raise AllocationError(f"panel shape {p.shape}: need >= 2 rows and >= 1 member")
    if not np.all(np.isfinite(p)):
        raise AllocationError("panel has a non-finite value")
    cov = np.atleast_2d(np.cov(p, rowvar=False, ddof=1))
    zero = [j for j in range(cov.shape[0]) if not cov[j, j] > 0.0]
    if zero:
        raise AllocationError(f"zero-variance member column(s) {zero}")
    return cov


def _check_cov(cov: np.ndarray) -> np.ndarray:
    c = np.asarray(cov, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] != c.shape[1] or c.shape[0] < 1 or not np.all(np.isfinite(c)):
        raise AllocationError(f"covariance must be a finite square matrix; got shape {c.shape}")
    if not np.allclose(c, c.T, rtol=0.0, atol=0.0):
        raise AllocationError("covariance is not symmetric")
    if not np.all(np.diag(c) > 0.0):
        raise AllocationError("covariance has a non-positive variance")
    return c


def _ivp_var(cov: np.ndarray, idx: list[int]) -> float:
    sub = cov[np.ix_(idx, idx)]
    ivp = 1.0 / np.diag(sub)
    ivp = ivp / ivp.sum()
    return float(ivp @ sub @ ivp)


def hrp(cov: np.ndarray) -> Allocation:
    c = _check_cov(cov)
    n = c.shape[0]
    if n == 1:
        return Allocation("HRP", (1.0,), {"order": [0]})
    sd = np.sqrt(np.diag(c))
    corr = np.clip(c / np.outer(sd, sd), -1.0, 1.0)
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))
    np.fill_diagonal(dist, 0.0)
    order = [int(k) for k in leaves_list(linkage(squareform(dist, checks=False), method="single"))]
    w = np.ones(n)
    clusters = [order]
    while clusters:
        nxt = []
        for cl in clusters:
            if len(cl) < 2:
                continue
            left, right = cl[: len(cl) // 2], cl[len(cl) // 2:]
            vl, vr = _ivp_var(c, left), _ivp_var(c, right)
            alpha = 1.0 - vl / (vl + vr)
            w[left] *= alpha
            w[right] *= 1.0 - alpha
            nxt += [left, right]
        clusters = nxt
    w = w / w.sum()
    return Allocation("HRP", tuple(float(x) for x in w), {"order": order})


def risk_contributions(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64)
    return w * (np.asarray(cov) @ w)


def erc(cov: np.ndarray) -> Allocation:
    c = _check_cov(cov)
    n = c.shape[0]
    if n == 1:
        return Allocation("ERC", (1.0,), {"sweeps": 0, "max_rel_dev": 0.0})
    diag = np.diag(c)
    x = 1.0 / np.sqrt(n * diag)  # the inverse-volatility start: the exact solution when cov is diagonal
    for sweep in range(1, ERC_MAX_SWEEPS + 1):
        for i in range(n):
            ci = float(c[i] @ x) - diag[i] * x[i]
            x[i] = (-ci + np.sqrt(ci * ci + 4.0 * diag[i] / n)) / (2.0 * diag[i])
        rc = risk_contributions(x, c)
        dev = float(np.max(np.abs(rc - rc.mean())) / rc.mean())
        if dev < ERC_TOL:
            w = x / x.sum()
            return Allocation("ERC", tuple(float(v) for v in w), {"sweeps": sweep, "max_rel_dev": dev})
    raise AllocationError(f"ERC did not converge in {ERC_MAX_SWEEPS} sweeps (max rel dev {dev:.3e})")


def allocate(panel: np.ndarray, member_ids: list[str]) -> dict:
    """The allocation artifact: both methods, the input digest, the member ids (DR §3)."""
    cov = covariance(panel)
    if len(member_ids) != cov.shape[0]:
        raise AllocationError(f"{len(member_ids)} member ids for {cov.shape[0]} panel columns")
    out = {"schema": "arqvio.portfolio.allocation", "version": 1, "members": list(member_ids),
           "panel_rows": int(np.asarray(panel).shape[0]), "panel_sha256": panel_digest(panel),
           "covariance": "sample, numpy.cov ddof=1, no shrinkage"}
    for a in (hrp(cov), erc(cov)):
        rc = risk_contributions(np.asarray(a.weights), cov)
        out[a.method] = {"weights": dict(zip(member_ids, a.weights)), "risk_contributions": dict(
            zip(member_ids, (float(v) for v in rc))), **a.detail}
    return out


__all__ = ["Allocation", "AllocationError", "ERC_MAX_SWEEPS", "ERC_TOL", "allocate", "covariance", "erc", "hrp",
           "panel_digest", "risk_contributions"]
