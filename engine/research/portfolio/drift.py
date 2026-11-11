"""The decay / drift monitor (DR §5): rolling two-sample KS per member, Benjamini-Yekutieli across members x checks.

Geometry (pre-registered): reference = bars [REF_START, REF_END) = [0, 300), fixed; recent = the RECENT = 100 bars
[t - 100, t); checks at t = 400, 500, ... while t <= the series length (recent windows never overlap). Each check is
``scipy.stats.ks_2samp(reference, recent, method="exact")`` (two-sided). All members x checks of ONE monitoring run
are corrected jointly with Benjamini-Yekutieli at q = Q = 0.05 -- BH at q / c(m), c(m) = sum 1/i -- which bounds the
probability of any rejection under the complete null by q under ARBITRARY dependence (the checks share their
reference window). A FIRE is any BY rejection.

``degraded_fixture`` / ``healthy_draw`` are the DR's seeded series; ``healthy_control`` measures the family
false-alarm rate over CONTROL_DRAWS draws.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import ks_2samp

REF_START, REF_END = 0, 300
RECENT = 100
STEP = 100
Q = 0.05
FIXTURE_SEED = 20261002
FIXTURE_MEMBERS = ("m0", "m1", "m2", "m3", "m4")
FIXTURE_N = 1000
FIXTURE_SIGMA = 1e-3
DEGRADED_MEMBER = "m2"
DEGRADE_FROM = 600
DEGRADE_MEAN_SIGMAS = 0.5
DEGRADE_VOL_FACTOR = 2.0
CONTROL_DRAWS = 200


class DriftError(ValueError):
    """A series the monitor refuses (too short, non-finite, mismatched)."""


@dataclass(frozen=True)
class Check:
    member: str
    t: int
    d: float
    p: float


def check_times(n: int) -> list[int]:
    if n < REF_END + RECENT:
        raise DriftError(f"series of {n} bars is shorter than reference {REF_END} + recent {RECENT}")
    return list(range(REF_END + RECENT, n + 1, STEP))


def ks_checks(member: str, series: np.ndarray) -> list[Check]:
    x = np.asarray(series, dtype=np.float64)
    if x.ndim != 1 or not np.all(np.isfinite(x)):
        raise DriftError(f"{member}: series must be 1-D and finite")
    ref = x[REF_START:REF_END]
    out = []
    for t in check_times(x.size):
        r = ks_2samp(ref, x[t - RECENT:t], alternative="two-sided", method="exact")
        out.append(Check(member, t, float(r.statistic), float(r.pvalue)))
    return out


def by_threshold(pvalues: list[float], q: float = Q) -> float:
    """The BY step-up cut: reject every p <= the returned value (0.0 when nothing is rejected)."""
    m = len(pvalues)
    if m == 0:
        return 0.0
    cm = float(np.sum(1.0 / np.arange(1, m + 1)))
    ps = np.sort(np.asarray(pvalues, dtype=np.float64))
    ok = np.nonzero(ps <= np.arange(1, m + 1) * q / (m * cm))[0]
    return float(ps[ok[-1]]) if ok.size else 0.0


@dataclass(frozen=True)
class MonitorResult:
    checks: tuple[Check, ...]
    cut: float
    rejected: tuple[Check, ...]

    @property
    def fired(self) -> bool:
        return bool(self.rejected)

    def first_fire(self, member: str) -> Check | None:
        hits = [c for c in self.rejected if c.member == member]
        return min(hits, key=lambda c: c.t) if hits else None


def monitor(series: dict[str, np.ndarray], q: float = Q) -> MonitorResult:
    lengths = {k: np.asarray(v).size for k, v in series.items()}
    if len(set(lengths.values())) > 1:
        raise DriftError(f"members have different lengths {lengths}")
    checks = [c for name in series for c in ks_checks(name, series[name])]
    cut = by_threshold([c.p for c in checks], q)
    rejected = tuple(c for c in checks if cut > 0.0 and c.p <= cut)
    return MonitorResult(tuple(checks), cut, rejected)


def healthy_draw(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {m: rng.normal(0.0, FIXTURE_SIGMA, FIXTURE_N) for m in FIXTURE_MEMBERS}


def degraded_fixture(seed: int = FIXTURE_SEED) -> dict[str, np.ndarray]:
    """Healthy members, one stream each in member order; DEGRADED_MEMBER from DEGRADE_FROM on: mean + 0.5 sigma,
    volatility x 2 (the planted mean shift plus vol change)."""
    s = healthy_draw(seed)
    x = s[DEGRADED_MEMBER].copy()
    x[DEGRADE_FROM:] = DEGRADE_MEAN_SIGMAS * FIXTURE_SIGMA + DEGRADE_VOL_FACTOR * x[DEGRADE_FROM:]
    s[DEGRADED_MEMBER] = x
    return s


def healthy_control(draws: int = CONTROL_DRAWS) -> dict:
    alarms = []
    for seed in range(draws):
        r = monitor(healthy_draw(seed))
        if r.fired:
            alarms.append({"seed": seed, "rejected": [(c.member, c.t, c.p) for c in r.rejected]})
    return {"draws": draws, "false_alarm_draws": len(alarms), "rate": len(alarms) / draws, "budget": Q,
            "alarms": alarms}


__all__ = ["CONTROL_DRAWS", "Check", "DEGRADED_MEMBER", "DEGRADE_FROM", "DriftError", "MonitorResult", "Q",
           "RECENT", "REF_END", "STEP", "by_threshold", "check_times", "degraded_fixture", "healthy_control",
           "healthy_draw", "ks_checks", "monitor"]
