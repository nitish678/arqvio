"""The diversity census (DR §4): named axes from the DNA config, |corr| statistics, per-axis shares, ENB.

``AXES`` is the registry: each axis maps a member to a label read from its DNA config or its cell symbol -- never
free text. ``census`` reports, for the member return panel:

* pairwise |corr| statistics over the off-diagonal (min, median, max, mean);
* per-axis shares: member counts and, per weighting, the weight on each label;
* the effective number of bets per weighting, ENB = 1 / sum_c W_c^2, where members whose return series have
  |rho| >= 1 - DUP_TOL are one bet (a duplicate cluster) and W_c is the cluster's summed weight.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from engine.research.portfolio.allocation import covariance

DUP_TOL = 1e-12


def _session(m: Any) -> str:
    return f"{int(m.dna.window_start_hour_utc):02d}-{int(m.dna.window_end_hour_utc):02d}"


AXES: dict[str, Callable[[Any], str]] = {
    "family": lambda m: str(m.dna.signal),
    "symbol": lambda m: str(m.symbol),
    "session_window": _session,
}


def axis_labels(member: Any) -> dict[str, str]:
    return {name: fn(member) for name, fn in AXES.items()}


def correlation(panel: np.ndarray) -> np.ndarray:
    cov = covariance(panel)
    sd = np.sqrt(np.diag(cov))
    return np.clip(cov / np.outer(sd, sd), -1.0, 1.0)


def duplicate_clusters(corr: np.ndarray) -> list[list[int]]:
    """Connected components of |rho| >= 1 - DUP_TOL (each component is one bet)."""
    n = corr.shape[0]
    seen, out = set(), []
    for i in range(n):
        if i in seen:
            continue
        comp, stack = [], [i]
        while stack:
            j = stack.pop()
            if j in seen:
                continue
            seen.add(j)
            comp.append(j)
            stack += [k for k in range(n) if k not in seen and abs(corr[j, k]) >= 1.0 - DUP_TOL]
        out.append(sorted(comp))
    return out


def enb(weights: list[float], clusters: list[list[int]]) -> float:
    w = np.asarray(weights, dtype=np.float64)
    cw = np.array([w[c].sum() for c in clusters])
    return float(1.0 / np.sum(cw * cw))


def census(panel: np.ndarray, members: list[Any], weightings: dict[str, list[float]]) -> dict:
    corr = correlation(panel)
    n = corr.shape[0]
    if len(members) != n:
        raise ValueError(f"{len(members)} members for {n} panel columns")
    off = np.abs(corr[~np.eye(n, dtype=bool)]) if n > 1 else np.array([])
    stats = {} if off.size == 0 else {"min": float(off.min()), "median": float(np.median(off)),
                                      "max": float(off.max()), "mean": float(off.mean())}
    labels = [axis_labels(m) for m in members]
    clusters = duplicate_clusters(corr)
    shares: dict[str, dict] = {}
    for axis in AXES:
        names = sorted({lab[axis] for lab in labels})
        shares[axis] = {"counts": {v: sum(1 for lab in labels if lab[axis] == v) for v in names},
                        "weights": {wn: {v: float(sum(w for w, lab in zip(ws, labels) if lab[axis] == v))
                                         for v in names} for wn, ws in weightings.items()}}
    return {"schema": "arqvio.portfolio.diversity", "version": 1, "members": [m.mid for m in members],
            "labels": {m.mid: lab for m, lab in zip(members, labels)}, "abs_corr_offdiag": stats,
            "duplicate_clusters": [[members[k].mid for k in c] for c in clusters],
            "enb": {wn: enb(ws, clusters) for wn, ws in weightings.items()}, "axis_shares": shares}


__all__ = ["AXES", "DUP_TOL", "axis_labels", "census", "correlation", "duplicate_clusters", "enb"]
