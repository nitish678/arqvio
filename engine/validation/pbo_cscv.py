"""
Probability of Backtest Overfitting (PBO) via Combinatorially Symmetric
Cross-Validation (CSCV).

Reference
---------
Bailey, D., Borwein, J., Lopez de Prado, M. and Zhu, Q. (2017),
"The Probability of Backtest Overfitting", Journal of Computational Finance
20(4), 39-69.

What it measures
----------------
DSR judges a single winning strategy. PBO judges the SELECTION PROCEDURE that
produced it. It answers:

    "When I pick the in-sample winner, how often does that winner land below
     the median of all candidates out-of-sample?"

If your search has no skill, the in-sample winner is essentially random
out-of-sample, so it lands below the OOS median about half the time and
PBO converges to 0.5. If the search is finding something real, the in-sample
winner keeps winning out-of-sample and PBO approaches 0.

PBO > 0.5 is worse than a coin flip: your selection rule is actively
anti-predictive, which happens when the search is optimising noise.

Algorithm
---------
1. Build matrix M of shape (T, N): T time observations by N trial return
   series. Every trial must be evaluated on the SAME observations.
2. Partition the T rows into S disjoint contiguous blocks (S even).
   Contiguous, not random, so that serial dependence stays inside blocks.
3. For each of the C(S, S/2) ways to split the blocks into an in-sample half
   and an out-of-sample half:
       a. rank all N trials in-sample, take the argmax -> n*
       b. rank all N trials out-of-sample
       c. take the relative rank w of n* out-of-sample, in (0, 1)
       d. logit lambda = ln(w / (1 - w))
4. PBO = fraction of splits with lambda < 0.

C(16, 8) = 12,870 splits, which is the usual choice: enough resolution without
combinatorial blow-up. S=10 gives 252 splits and is noisy; S=20 gives 184,756
and is slow for little gain.

Implementation note
-------------------
The Sharpe of a union of blocks is computed from precomputed per-block sums
and sums of squares, so each split costs O(S*N) additions rather than a full
pass over the data. That makes the default configuration run in well under a
second for realistic N.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from itertools import combinations

import numpy as np
from scipy.stats import rankdata

from engine.core.errors import DataError


@dataclass
class PBOResult:
    pbo: float                       # probability of backtest overfitting
    n_splits: int
    n_trials: int
    n_obs: int
    n_blocks: int
    logits: np.ndarray = field(repr=False)
    is_perf_selected: np.ndarray = field(repr=False)   # IS perf of the winner
    oos_perf_selected: np.ndarray = field(repr=False)  # OOS perf of the winner
    degradation_slope: float = np.nan   # OLS slope of OOS on IS, across splits
    prob_oos_loss: float = np.nan       # P(winner's OOS perf < 0)
    verdict: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("logits", "is_perf_selected", "oos_perf_selected"):
            d.pop(k, None)
        return d


def _block_bounds(n_obs: int, n_blocks: int) -> list[tuple[int, int]]:
    """Contiguous, as-equal-as-possible block boundaries covering [0, n_obs)."""
    edges = np.linspace(0, n_obs, n_blocks + 1).astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(n_blocks)]


def _sharpe_from_moments(s1: np.ndarray, s2: np.ndarray, n: int) -> np.ndarray:
    """
    Vectorised per-observation Sharpe from aggregated first and second moments.

    s1, s2 have shape (N,) - per-trial sum and sum of squares over n rows.
    Returns shape (N,). Zero-variance trials yield -inf so they can never be
    selected as the in-sample winner and always rank worst out-of-sample.
    """
    if n < 2:
        return np.full(s1.shape, -np.inf)
    mean = s1 / n
    var = (s2 - n * mean**2) / (n - 1)
    var = np.where(var <= 0, np.nan, var)
    with np.errstate(invalid="ignore", divide="ignore"):
        sr = mean / np.sqrt(var)
    return np.where(np.isfinite(sr), sr, -np.inf)


def pbo_cscv(perf_matrix, n_blocks: int = 16,
             higher_is_better: bool = True) -> PBOResult:
    """
    Compute PBO by CSCV on a (T, N) matrix of per-observation returns.

    Parameters
    ----------
    perf_matrix : array-like, shape (T, N)
        Column j is the return series of trial j, aligned on the same T
        observations. Per-bar returns or per-trade returns both work, as long
        as every column uses the same index.
    n_blocks : int
        Must be even. 16 is the reference default (12,870 splits).
    higher_is_better : bool
        Set False if you feed a loss-like statistic.

    Returns
    -------
    PBOResult
    """
    M = np.asarray(perf_matrix, dtype=np.float64)
    if M.ndim != 2:
        raise DataError("perf_matrix must be 2-D, shape (T, N)")
    T, N = M.shape
    if N < 2:
        raise DataError("need at least 2 trials to rank")
    if n_blocks % 2 != 0:
        raise DataError("n_blocks must be even")
    if n_blocks < 4:
        raise DataError("n_blocks must be >= 4")
    if T < n_blocks * 2:
        raise DataError(
            f"need at least {n_blocks * 2} observations for {n_blocks} blocks, got {T}")

    M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)

    bounds = _block_bounds(T, n_blocks)
    counts = np.array([hi - lo for lo, hi in bounds], dtype=np.int64)
    S1 = np.empty((n_blocks, N), dtype=np.float64)
    S2 = np.empty((n_blocks, N), dtype=np.float64)
    for b, (lo, hi) in enumerate(bounds):
        chunk = M[lo:hi]
        S1[b] = chunk.sum(axis=0)
        S2[b] = (chunk**2).sum(axis=0)

    half = n_blocks // 2
    all_blocks = set(range(n_blocks))

    logits: list[float] = []
    is_sel: list[float] = []
    oos_sel: list[float] = []

    for combo in combinations(range(n_blocks), half):
        idx_is = list(combo)
        idx_oos = sorted(all_blocks - set(combo))

        n_is = int(counts[idx_is].sum())
        n_oos = int(counts[idx_oos].sum())

        sr_is = _sharpe_from_moments(S1[idx_is].sum(axis=0),
                                     S2[idx_is].sum(axis=0), n_is)
        sr_oos = _sharpe_from_moments(S1[idx_oos].sum(axis=0),
                                      S2[idx_oos].sum(axis=0), n_oos)

        if not higher_is_better:
            sr_is, sr_oos = -sr_is, -sr_oos

        if not np.any(np.isfinite(sr_is)):
            continue

        n_star = int(np.argmax(sr_is))

        # Average ranks so ties do not bias the logit in either direction.
        # rank 1 == worst, rank N == best.
        ranks = rankdata(sr_oos, method="average")
        w = ranks[n_star] / (N + 1.0)
        w = min(max(w, 1e-12), 1.0 - 1e-12)
        logits.append(float(np.log(w / (1.0 - w))))

        is_sel.append(float(sr_is[n_star]))
        oos_sel.append(float(sr_oos[n_star]))

    lam = np.asarray(logits, dtype=np.float64)
    is_arr = np.asarray(is_sel, dtype=np.float64)
    oos_arr = np.asarray(oos_sel, dtype=np.float64)

    pbo = float(np.mean(lam < 0.0)) if lam.size else np.nan

    # Performance degradation: regress the winner's OOS performance on its IS
    # performance across splits. A negative slope means that the better a
    # strategy looks in-sample, the worse it does out-of-sample - the
    # signature of an overfit search.
    slope = np.nan
    finite = np.isfinite(is_arr) & np.isfinite(oos_arr)
    if finite.sum() >= 2 and np.std(is_arr[finite]) > 0:
        slope = float(np.polyfit(is_arr[finite], oos_arr[finite], 1)[0])

    prob_loss = float(np.mean(oos_arr[finite] < 0.0)) if finite.any() else np.nan

    if not np.isfinite(pbo):
        verdict = "INDETERMINATE"
    elif pbo < 0.10:
        verdict = "STRONG - selection generalises"
    elif pbo < 0.30:
        verdict = "ACCEPTABLE"
    elif pbo <= 0.60:
        verdict = "OVERFIT - selection is near-random out-of-sample"
    else:
        verdict = "SEVERELY OVERFIT - selection is anti-predictive"

    return PBOResult(
        pbo=pbo,
        n_splits=int(lam.size),
        n_trials=N,
        n_obs=T,
        n_blocks=n_blocks,
        logits=lam,
        is_perf_selected=is_arr,
        oos_perf_selected=oos_arr,
        degradation_slope=slope,
        prob_oos_loss=prob_loss,
        verdict=verdict,
    )
