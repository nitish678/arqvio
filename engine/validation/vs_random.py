"""Vs-Random benchmark.

Idea borrowed from Build Alpha's "Vs Random" test, formalised.

The question
------------
"Could a search over random, meaningless signals - run under IDENTICAL
constraints, on the SAME data, with the SAME trial count - have produced a
winner as good as mine?"

If yes, the campaign found nothing. This is a direct, assumption-light control
that catches what DSR can miss: DSR assumes a Gaussian null, whereas this test
builds the null empirically from the actual strategy space, cost model and
data.

THE TRAP THIS MODULE MUST NOT FALL INTO
---------------------------------------
The naive implementation compares the real winner against the distribution of
INDIVIDUAL random trials. That is the multiple-testing error itself. With 2,000
zero-edge draws the maximum is ~3.5 sigma, so P(one random trial >= 3.5) looks
like 0.001 and pure noise is declared a discovery.

The correct null is the distribution of the MAXIMUM of a random search of the
same size. Under that null the same case gives p ~ 0.63 - correctly
unremarkable. We bootstrap that distribution from the control pool.

`p_single` is retained as a diagnostic only. It must never be used as a
decision rule, and it is deliberately not part of the verdict.

How to use this correctly
-------------------------
The control campaign must match the real one on EVERY axis except signal
meaningfulness: same symbol, same period, same cost model, same risk rules,
same exec mode, same trial count, same minimum trade filter. If the control is
smaller or cheaper than the real campaign, the comparison is rigged and the
p-value is meaningless. A warning is emitted when n_random < n_real.

Cross-checked against `arqvio v1` reference: the bootstrap logic, the
p_max / p_mean / p_single / effect_size outputs, and the verdict rules
match the reference. Naive max-vs-single is exposed as `p_single` so the
trap stays visible in reports rather than invisible.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

# A mean shift can be statistically significant and practically irrelevant at
# large n. The verdict requires an effect size floor as well as a p-value.
MIN_EFFECT_SIZE = 0.20


@dataclass(frozen=True)
class VsRandomResult:
    p_max: float                # P(random search of same size >= real winner)
    p_mean: float               # permutation test on the mean shift
    p_single: float             # DIAGNOSTIC ONLY - do not use as a decision rule
    real_max: float
    real_mean: float
    random_max: float
    random_mean: float
    random_max_q95: float       # 95th pct of the bootstrapped random-search max
    n_real: int
    n_random: int
    n_bootstrap: int
    effect_size: float          # Cohen's d, real vs random populations
    verdict: str
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _clean(x: object) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64).ravel()
    return a[np.isfinite(a)]


def vs_random(
    real_scores: object,
    random_scores: object,
    n_bootstrap: int = 10_000,
    n_permutations: int = 10_000,
    seed: int | None = 0,
) -> VsRandomResult:
    """Compare a real campaign's trial scores against a control campaign's.

    Parameters
    ----------
    real_scores, random_scores : array-like
        One score per trial (typically per-observation or annualised Sharpe).
    n_bootstrap : int
        Resamples used to build the null distribution of the random search
        maximum. The smallest reportable p_max is 1/(n_bootstrap+1).
    n_permutations : int
        Resamples for the mean-shift permutation test.
    seed : int | None
        RNG seed. Record it in the reproducibility manifest.
    """
    real = _clean(real_scores)
    rnd = _clean(random_scores)
    if real.size == 0 or rnd.size == 0:
        raise ValueError("both score arrays must contain finite values")

    warns: list[str] = []
    if rnd.size < real.size:
        warns.append(
            f"control smaller than real campaign ({rnd.size} vs {real.size}); "
            "the bootstrapped maximum understates the random tail and p_max is "
            "optimistic"
        )
    if rnd.size < 100:
        warns.append(f"control has only {rnd.size} trials; p_max is coarse")

    real_max = float(np.max(real))
    real_mean = float(np.mean(real))
    rnd_max = float(np.max(rnd))
    rnd_mean = float(np.mean(rnd))

    rng = np.random.default_rng(seed)

    # --- p_max: max-vs-max, the only valid winner comparison ----------------
    # Bootstrap a random search of the SAME SIZE as the real campaign and take
    # its maximum. Vectorised: (n_bootstrap, n_real) draws with replacement.
    draws = rng.choice(rnd, size=(n_bootstrap, real.size), replace=True)
    boot_max = draws.max(axis=1)
    p_max = float((np.sum(boot_max >= real_max) + 1) / (n_bootstrap + 1))
    random_max_q95 = float(np.quantile(boot_max, 0.95))

    # --- p_single: kept only to make the trap visible in reports ------------
    p_single = float((np.sum(rnd >= real_max) + 1) / (rnd.size + 1))

    # --- p_mean: is the whole population shifted, not just the winner -------
    pooled = np.concatenate([real, rnd])
    n_r = real.size
    observed = real_mean - rnd_mean
    count = 0
    for _ in range(n_permutations):
        rng.shuffle(pooled)
        if (pooled[:n_r].mean() - pooled[n_r:].mean()) >= observed:
            count += 1
    p_mean = float((count + 1) / (n_permutations + 1))

    # --- effect size --------------------------------------------------------
    sd_pool = np.sqrt(
        ((real.size - 1) * np.var(real, ddof=1)
         + (rnd.size - 1) * np.var(rnd, ddof=1))
        / max(1, real.size + rnd.size - 2)
    )
    d = float(observed / sd_pool) if sd_pool > 0 else float("nan")

    # --- verdict: p_max decides the winner, p_mean AND effect size decide
    #     whether the search space itself has exploitable structure ----------
    winner_ok = p_max < 0.05
    pop_ok = (p_mean < 0.05) and np.isfinite(d) and (d >= MIN_EFFECT_SIZE)

    if winner_ok and pop_ok:
        verdict = "STRUCTURE FOUND - winner and population both beat random"
    elif winner_ok:
        verdict = "WINNER BEATS RANDOM - population shift weak or negligible"
    elif pop_ok:
        verdict = "POPULATION SHIFT ONLY - winner is within random range"
    else:
        verdict = "NO STRUCTURE - indistinguishable from random search"

    return VsRandomResult(
        p_max=p_max,
        p_mean=p_mean,
        p_single=p_single,
        real_max=real_max,
        real_mean=real_mean,
        random_max=rnd_max,
        random_mean=rnd_mean,
        random_max_q95=random_max_q95,
        n_real=int(real.size),
        n_random=int(rnd.size),
        n_bootstrap=int(n_bootstrap),
        effect_size=d,
        verdict=verdict,
        warnings=tuple(warns),
    )
