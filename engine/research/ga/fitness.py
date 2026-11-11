"""GA fitness: gated, every component recorded, dead anywhere means 0.

Gate order (S4 stage 2 spec):

    crash                       -> killed, fitness 0 (the run's crash rate must stay 0)
    n_trades < min_trades       -> killed, fitness 0
    sharpe <= sharpe_min        -> killed, fitness 0
    base = sharpe
    dsr >= dsr_min              -> + 0.25 * sharpe
    sham_margin >= sham_margin_min -> + 0.25 * sham_margin

sham_margin is the candidate's Sharpe minus its own sham's (W3 machinery:
sham_backtest_factory, salt astra-s3-sham-v1, seeded by sha256(dna_hash:salt),
the slow engine under the identical costs and gates). A killed individual is
not scored further: its DSR and sham are never computed, and the record says
which gate killed it. Nothing is folded into an opaque number -- the record
carries every component the fitness was built from.

DSR is the per-candidate Deflated Sharpe exactly as runner._populate_post_dsr
computes it (engine/validation/cell_dsr.compute_cell_dsr on the candidate's
own per-period returns, READY required). Its luck bar depends on how many
distinct backtests the search has run, so a DSR is relative to the moment it
was computed; n_trials travels with it.

DSR recency policy (S4 stage 3): breeding uses the EVAL-TIME reading. An
individual's fitness is fixed when it is scored and never re-scored, so a
run's history -- every tournament, every elite, the run hash -- stays
reproducible. Reporting uses the reading at the run's own luck bar
(manifest v2 "primary_final"), and both readings are recorded. In the S4
Stage 2 full run the policy was moot (no individual ever earned the bonus);
it is written down before a run where it matters.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from engine.research.ga.config import Gates
from engine.research.search.runner import TrialResult
from engine.validation.cell_dsr import CellDSRStatus, compute_cell_dsr

KILLED_CRASH = "crash"
KILLED_MIN_TRADES = "min_trades"
KILLED_SHARPE_MIN = "sharpe_min"
DSR_BONUS_WEIGHT = 0.25
SHAM_BONUS_WEIGHT = 0.25
MIN_OBS_FOR_DSR = 100  # run_campaign's min_obs_for_dsr


@dataclass(frozen=True)
class DSRReading:
    """One candidate's DSR and the numbers it cannot be read without."""

    dsr: float | None
    status: str
    sr_hat_per_bar: float | None
    sr_star_per_bar: float | None
    n_trials: int


@dataclass(frozen=True)
class FitnessRecord:
    fitness: float
    killed_by: str | None
    n_trades: int
    sharpe: float | None
    base: float | None = None
    dsr: DSRReading | None = None
    dsr_passed: bool | None = None
    dsr_bonus: float = 0.0
    sham_sharpe: float | None = None
    sham_margin: float | None = None
    sham_passed: bool | None = None
    sham_bonus: float = 0.0

    @property
    def alive(self) -> bool:
        return self.killed_by is None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def dsr_trials(sharpes: Iterable[float | None], n_effective: int) -> tuple[int, list[float]]:
    """(N, trial_sharpes) for the luck bar, as run_campaign computes them.

    N is the count of distinct backtests, never above the scored count and
    never below 1 (runner._populate_post_dsr). The Sharpes are sorted so the
    variance -- and so every DSR -- does not depend on evaluation order.
    """
    finite = sorted(float(s) for s in sharpes if s is not None and math.isfinite(s))
    n = max(1, min(int(n_effective), max(1, len(finite))))
    return n, finite


def candidate_dsr(
    result: TrialResult,
    *,
    n_trials: int,
    trial_sharpes: list[float],
    symbol: str,
    timeframe: str,
) -> DSRReading:
    """The candidate's own DSR (runner._populate_post_dsr, one candidate)."""
    if result.sharpe is None or not np.isfinite(result.sharpe) or result.per_period_returns is None:
        return DSRReading(None, CellDSRStatus.UNRELIABLE.value, None, None, int(n_trials))
    try:
        cell = compute_cell_dsr(
            result.per_period_returns,
            n_trials=int(n_trials),
            trial_sharpes=trial_sharpes,
            symbol=symbol,
            timeframe=timeframe,
            min_obs_for_dsr=MIN_OBS_FOR_DSR,
        )
    except Exception:
        return DSRReading(None, CellDSRStatus.UNRELIABLE.value, None, None, int(n_trials))
    value = cell.dsr
    ready = cell.status is CellDSRStatus.READY and value is not None and np.isfinite(value)
    return DSRReading(
        dsr=float(value) if ready else None,
        status=cell.status.value,
        sr_hat_per_bar=cell.sr_hat_per_bar,
        sr_star_per_bar=cell.expected_max_sr,
        n_trials=int(n_trials),
    )


def early_kill(result: TrialResult, *, crashed: bool, gates: Gates) -> FitnessRecord | None:
    """The gates before DSR and the sham, in order; None means the candidate survives them.

    ``score`` asks for a sham exactly when this returns None, so a parallel evaluator can
    compute precisely the shams the sequential loop would have computed, no more.
    """
    n_trades = int(result.n_trades or 0)
    sharpe = None if result.sharpe is None else float(result.sharpe)
    if crashed:
        return FitnessRecord(0.0, KILLED_CRASH, n_trades, sharpe)
    if n_trades < gates.min_trades:
        return FitnessRecord(0.0, KILLED_MIN_TRADES, n_trades, sharpe)
    if sharpe is None or not math.isfinite(sharpe) or sharpe <= gates.sharpe_min:
        return FitnessRecord(0.0, KILLED_SHARPE_MIN, n_trades, sharpe)
    return None


def score(
    result: TrialResult,
    *,
    crashed: bool,
    gates: Gates,
    dsr_of: Callable[[TrialResult], DSRReading],
    sham_of: Callable[[TrialResult], float],
) -> FitnessRecord:
    """Apply the gates in order; compute DSR and the sham only for survivors."""
    killed = early_kill(result, crashed=crashed, gates=gates)
    if killed is not None:
        return killed
    n_trades = int(result.n_trades or 0)
    sharpe = float(result.sharpe)
    dsr = dsr_of(result)
    dsr_passed = dsr.dsr is not None and dsr.dsr >= gates.dsr_min
    dsr_bonus = DSR_BONUS_WEIGHT * sharpe if dsr_passed else 0.0
    sham_sharpe = float(sham_of(result))
    margin = sharpe - sham_sharpe
    sham_passed = margin >= gates.sham_margin_min
    sham_bonus = SHAM_BONUS_WEIGHT * margin if sham_passed else 0.0
    return FitnessRecord(
        fitness=sharpe + dsr_bonus + sham_bonus,
        killed_by=None,
        n_trades=n_trades,
        sharpe=sharpe,
        base=sharpe,
        dsr=dsr,
        dsr_passed=dsr_passed,
        dsr_bonus=dsr_bonus,
        sham_sharpe=sham_sharpe,
        sham_margin=margin,
        sham_passed=sham_passed,
        sham_bonus=sham_bonus,
    )


__all__ = [
    "DSR_BONUS_WEIGHT",
    "DSRReading",
    "FitnessRecord",
    "KILLED_CRASH",
    "KILLED_MIN_TRADES",
    "KILLED_SHARPE_MIN",
    "MIN_OBS_FOR_DSR",
    "SHAM_BONUS_WEIGHT",
    "candidate_dsr",
    "dsr_trials",
    "early_kill",
    "score",
]
