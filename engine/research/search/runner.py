"""Search runner - S5 rerun version.

Defect fixes (Maalik 2026-09-16 review):
    D1: bars_loaded must match the audit's bar_count for the chosen cell;
        runner fails loudly otherwise.
    D2: FailedDSR, HighPBO, DsrUnreliable are part of the funnel and
        apply to every candidate in the order
        LowTrades -> HighDD -> NegReturn -> DsrUnreliable -> FailedDSR
        Campaign-level PBO is computed over the trial matrix.
        dsr_within and dsr_across are exposed on CampaignResult.
    D3: MinBTL uses the S5-calibrated formula
        (engine.validation.minbtl.min_backtest_length).
    D4: zero crashes. The synthetic backtest factory has no exception
        path; the backtest_fn wrapper captures every exception with the
        full traceback for the report.
"""
from __future__ import annotations

import multiprocessing
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from engine.research.campaign.budget import CampaignScope
from engine.research.campaign.preflight import Verdict, ensure_verdict_complete
from engine.research.provenance import Provenance
from engine.research.search.control import (
    assert_scopes_match,
    scope_from_campaign,
    scope_to_control,
)
from engine.research.search.degeneracy import cost_totals, degeneracy_report
from engine.research.search.dna import DNA, FailureTaxonomy
from engine.research.search.effective_hash import count_effective_trials
from engine.research.search.funnel import FunnelCounts, summarise
from engine.research.search.gene_pool import GenePool
from engine.research.search.lookahead_guard import is_lookahead_dna
from engine.research.search.sampler import random_dna
from engine.validation.cell_dsr import (
    DEFAULT_PERIODS_PER_YEAR,
    PERIODS_PER_YEAR,
    CellDSRResult,
    CellDSRStatus,
    compute_cell_dsr,
    compute_cross_cell_dsr,
)
from engine.validation.op_types import OpType
from engine.validation.pbo_cscv import pbo_cscv
from engine.validation.purged_cv import PurgedCVConfig, purged_cv
from engine.validation.trial_registry import TrialRegistry
from engine.validation.vs_random import VsRandomResult, vs_random


@dataclass
class CrashRecord:
    """One captured exception from a trial evaluation."""

    dna_hash: str
    exception_type: str
    message: str
    traceback: str


@dataclass
class TrialResult:
    """A single evaluated candidate. Backtester-supplied + post-hoc flags."""

    dna: DNA
    outcome: str = FailureTaxonomy.EVALUATED.value
    sharpe: float | None = None
    n_trades: int | None = 0
    total_return: float | None = 0.0
    max_dd: float | None = 1.0
    spread_uncalibrated: bool = False
    post_dsr: float | None = None
    post_pbo: float | None = None
    unreliable_dsr: bool = False
    # Real-backtest path: per-period return series (length = bars-1).
    # PBO/CSCV consumes this. Synthetic backtesters leave it None.
    per_period_returns: np.ndarray | None = None
    # Entries refused and exits deferred inside a zero-spread window.
    # Aggregated onto the campaign so the cost of the window is reported
    # rather than silently reshaping the equity curve.
    blocked_zero_spread_events: int = 0
    # Of those, the ones that filled after the window closed (R1).
    deferred_zero_spread_fills: int = 0
    # SL/TP that came due inside a no-trade hour and was taken at the
    # first tradeable open instead of at its own level.
    sltp_deferred_zero_spread: int = 0
    # Entries refused because the bar's spread exceeded the A5 gate.
    blocked_max_spread: int = 0
    # Account-currency cost lines charged to this candidate: spread,
    # commission, slippage, swap. Empty when the backtester has no costs.
    costs: dict[str, float] = field(default_factory=dict)
    # Exit-reason distribution for this candidate. A partition over its
    # trades: SIGNAL / SL / TP / EOD / SLTP_DEFERRED_ZERO_SPREAD.
    exit_reasons: dict[str, int] = field(default_factory=dict)
    # "TICK" when this candidate's exits can trigger INTRABAR (it has an
    # SL or a TP), else "M1_OHLC". A TICK candidate's fill prices depend
    # on within-bar path, which OHLC bars do not record, so S6 parity
    # against MT5 must be checked at tick resolution for these and may
    # be checked on M1 bars for the rest. Stored here and in the report,
    # NOT in the trial registry -- that schema is frozen.
    parity_class: str = "M1_OHLC"
    # The two numbers that make post_dsr readable: this candidate's own
    # per-bar Sharpe, and the luck bar it had to clear at this n_trials.
    post_dsr_sr_per_bar: float | None = None
    post_dsr_sr_star: float | None = None


@dataclass
class CampaignResult:
    campaign_id: str
    n_trials: int
    funnel: FunnelCounts
    best_dna: DNA | None
    best_sharpe: float | None
    vs_random: VsRandomResult | None
    provenance: Provenance
    as_of: str
    verdict: Verdict | None = None
    # S5 rerun fields
    n_obs_real: int = 0
    bars_loaded: int = 0
    audit_bars_expected: int = 0
    dsr_within: float | None = None
    dsr_within_status: str | None = None
    # A cross-cell DSR over ONE cell is a no-op by construction: the second
    # selection step has a population of one and returns the within value
    # unchanged. dsr_across_n_cells is what tells a reader whether any
    # second correction was actually applied.
    dsr_across: float | None = None
    dsr_across_status: str | None = None
    dsr_across_n_cells: int = 0
    # dsr_within is uninterpretable on its own. These are the DSR winner's
    # own Sharpe and the luck bar it cleared, per-bar and annualised. The
    # luck bar scales with n_trials: a candidate that clears it at 200
    # trials can fail the same test at 5000.
    dsr_winner_sr_per_bar: float | None = None
    dsr_winner_sr_annualised: float | None = None
    dsr_sr_star_per_bar: float | None = None
    dsr_sr_star_annualised: float | None = None
    # The same three headline statistics over ELIGIBLE candidates only -
    # those that cleared min_trades. dsr_within, best_sharpe and vs_random
    # above are computed over EVERY candidate, funnel rejects included, and
    # on 2026-09-18 all three of cell 1's belonged to one 23-trade candidate
    # the funnel had bucketed LowTrades. Report both or neither.
    n_eligible: int = 0
    dsr_within_eligible: float | None = None
    dsr_within_eligible_status: str | None = None
    dsr_eligible_winner_sr_per_bar: float | None = None
    dsr_eligible_winner_sr_annualised: float | None = None
    dsr_eligible_sr_star_per_bar: float | None = None
    dsr_eligible_sr_star_annualised: float | None = None
    vs_random_eligible: VsRandomResult | None = None
    best_sharpe_eligible: float | None = None
    best_dna_eligible: DNA | None = None
    pbo: float | None = None
    pbo_n_splits: int = 0
    crashes: tuple[CrashRecord, ...] = field(default_factory=tuple)
    # Report diagnostics (Stage B): trade counts per funnel bucket and the
    # number of DISTINCT candidate DSR values (defect #6 sentinel — a single
    # distinct value across thousands of candidates means DSR is not
    # candidate-specific).
    trades_by_bucket: dict[str, int] = field(default_factory=dict)
    distinct_dsr_count: int = 0
    # Total entries refused and exits deferred across the cell because the
    # bar fell in a zero-spread hour. Reported so the window's cost on the
    # campaign is visible rather than inferred.
    blocked_zero_spread_events: int = 0
    # Of those, the ones that filled on the first tradeable bar after
    # the window (R1). blocked minus deferred is work the window
    # absorbed rather than merely delayed.
    deferred_zero_spread_fills: int = 0
    # Cell totals for the SL/TP window interaction and the exit mix.
    sltp_deferred_zero_spread: int = 0
    exit_reasons: dict[str, int] = field(default_factory=dict)
    parity_classes: dict[str, int] = field(default_factory=dict)
    # W2/G3. trials_raw is how many genomes were evaluated;
    # trials_effective is how many DISTINCT backtests that amounted to;
    # dsr_N_used is the N the luck bar actually scaled with. Reporting
    # all three means a reader can see the deduplication rather than
    # having to trust it.
    trials_raw: int = 0
    trials_effective: int = 0
    dsr_N_used: int = 0  # noqa: N815 - field name fixed by the S1 spec
    # degeneracy.degeneracy_report / cost_totals over this campaign's trials.
    degeneracy: dict[str, Any] = field(default_factory=dict)
    cost_totals: dict[str, float] = field(default_factory=dict)

    def exit_verdict(self) -> str:
        """The spec's exit criterion, stated explicitly."""
        if self.funnel.survivors == 0:
            return (
                "NO SURVIVORS - no accessible edge in this configuration. "
                "Change symbol, timeframe, or gene families; do NOT build a GA."
            )
        # The verdict is about candidates that could actually be traded, so
        # it reads the ELIGIBLE statistics. The all-candidate versions stay on
        # the result for the report and for the canary, which plants an edge
        # deliberately and wants the whole population.
        v = self.vs_random_eligible
        if v is None:
            return "INCONCLUSIVE - vs-random could not be computed."
        winner = v.p_max < 0.05
        population = (v.p_mean < 0.05) and (v.effect_size >= 0.20)
        dsr_e = self.dsr_within_eligible
        if winner and population and dsr_e and dsr_e > 0.5:
            return (
                "STRUCTURE FOUND - winner and population both beat random; "
                "per-cell DSR > 0.5. Build the GA in a follow-up pass."
            )
        if winner:
            return (
                "WINNER BEATS RANDOM - population shift weak or negligible. "
                "Effect-size limited; not a discovery."
            )
        if population:
            return (
                "POPULATION SHIFT ONLY - winner is within random range. "
                "No clear edge in this cell."
            )
        return "NO STRUCTURE - indistinguishable from random search."


def evaluate_one(
    dna: DNA, *, backtest_fn: Callable[[DNA], TrialResult]
) -> tuple[TrialResult, CrashRecord | None]:
    """Evaluate one DNA. Captures every exception as a CrashRecord."""
    if is_lookahead_dna(dna):
        return (
            TrialResult(
                dna=dna, outcome=FailureTaxonomy.Crash.value,
                sharpe=None, n_trades=0, total_return=0.0, max_dd=1.0,
            ),
            CrashRecord(
                dna_hash=dna.dna_hash,
                exception_type="LookaheadDetected",
                message="DNA uses_future_bar=True; refused.",
                traceback="",
            ),
        )
    try:
        return backtest_fn(dna), None
    except Exception as exc:
        return (
            TrialResult(
                dna=dna, outcome=FailureTaxonomy.Crash.value,
                sharpe=None, n_trades=0, total_return=0.0, max_dd=1.0,
            ),
            CrashRecord(
                dna_hash=dna.dna_hash,
                exception_type=type(exc).__name__,
                message=str(exc),
                traceback=traceback.format_exc(),
            ),
        )


def _apply_funnel(
    result: TrialResult,
    *,
    min_trades: int,
    max_dd_limit: float,
    return_threshold: float,
    d_sharpe_threshold: float,
    pbo_threshold: float = 0.5,
) -> str:
    """Apply the funnel in the S5 spec order.

    Crash -> SpreadUncalibrated -> LowTrades -> HighDD -> NegReturn
    -> DsrUnreliable -> FailedDSR -> HighPBO -> VAULTED
    """
    if result.outcome == FailureTaxonomy.Crash.value:
        return result.outcome
    if result.spread_uncalibrated:
        return FailureTaxonomy.SpreadUncalibrated.value
    n_trades = result.n_trades or 0
    if n_trades < min_trades:
        return FailureTaxonomy.LowTrades.value
    max_dd = result.max_dd if result.max_dd is not None else 1.0
    if max_dd > max_dd_limit:
        return FailureTaxonomy.HighDD.value
    total_return = result.total_return if result.total_return is not None else -1.0
    if total_return < return_threshold:
        return FailureTaxonomy.NegReturn.value
    if result.unreliable_dsr:
        return FailureTaxonomy.DsrUnreliable.value
    if (
        result.post_dsr is not None
        and result.post_dsr < d_sharpe_threshold
    ):
        return FailureTaxonomy.FailedDSR.value
    if (result.post_pbo is not None) and (result.post_pbo > pbo_threshold):
        return FailureTaxonomy.HighPBO.value
    return FailureTaxonomy.VAULTED.value


def _populate_post_dsr(
    results: list[TrialResult],
    *,
    symbol: str,
    timeframe: str,
    registry: TrialRegistry,
    campaign_id: str,
    min_obs_for_dsr: int = 100,
    n_effective_trials: int | None = None,
) -> int:
    """Compute candidate DSR and tag post_dsr + unreliable_dsr.

    DSR must evaluate each candidate's own per-period return series. Using
    the raw cell market returns assigns the same near-zero DSR to every trial
    and hides an actually planted edge. There is therefore no cell-level
    return series parameter here at all.

    Returns the N it actually handed the luck bar, so the campaign can
    report it instead of the caller having to infer it.

    W2/G3. This used to pass ``len(cell_sharpes)`` -- one per registry
    row, i.e. the raw genome count. That defends against a search that
    never happened: DNAs differing only in genes the engine ignores are
    one backtest with several names, and each extra name raises the bar
    a real edge must clear without a second real test behind it.
    ``n_effective_trials`` is the count of DISTINCT projections and is
    what the luck bar now scales with.

    engine/validation/dsr.py is frozen, but it never needed editing:
    ``dsr(returns, n_trials, ...)`` takes N as a caller argument, so the
    fix belongs here, at the call site, which is where the knowledge of
    what counts as a trial actually lives.
    """
    cell_sharpes = registry.search_sharpes(campaign_id, symbol, timeframe)
    n_raw = max(1, len(cell_sharpes))
    if n_effective_trials is None:
        n_cell_trials = n_raw
    else:
        # Never above the raw count, never below 1. Effective trials are
        # a DEDUPLICATION of the raw ones; a larger number would mean the
        # two were counting different things.
        n_cell_trials = max(1, min(int(n_effective_trials), n_raw))
    # (dna_hash, dsr, dsr_status, sr_star, n_obs) per candidate, written to
    # the registry in ONE batched update at the end so a finished campaign
    # can be re-analysed without re-running it.
    dsr_rows: list[tuple[str, float | None, str | None, float | None, int | None]] = []
    for r in results:
        if r.sharpe is None or not np.isfinite(r.sharpe):
            continue
        candidate_returns = r.per_period_returns
        if candidate_returns is None:
            r.post_dsr = None
            r.unreliable_dsr = True
            dsr_rows.append(
                (r.dna.dna_hash, None, CellDSRStatus.UNRELIABLE.value, None, 0)
            )
            continue
        try:
            cell = compute_cell_dsr(
                candidate_returns,
                n_trials=n_cell_trials,
                trial_sharpes=cell_sharpes,
                symbol=symbol,
                timeframe=timeframe,
                min_obs_for_dsr=min_obs_for_dsr,
            )
        except Exception:
            r.post_dsr = None
            r.unreliable_dsr = True
            dsr_rows.append(
                (
                    r.dna.dna_hash, None, CellDSRStatus.UNRELIABLE.value, None,
                    int(np.asarray(candidate_returns).size),
                )
            )
            continue
        r.post_dsr_sr_per_bar = cell.sr_hat_per_bar
        r.post_dsr_sr_star = cell.expected_max_sr
        if cell.status is not CellDSRStatus.READY:
            # UNRELIABLE (too few observations) and NO_TRIALS (nothing to
            # correct for) both mean "no usable DSR". Either way the
            # candidate must reach the DsrUnreliable bucket rather than
            # sail past a gate that has no number to test.
            r.unreliable_dsr = True
            r.post_dsr = None
        else:
            r.post_dsr = float(cell.dsr) if cell.dsr is not None else None
        dsr_rows.append(
            (
                r.dna.dna_hash,
                r.post_dsr,
                cell.status.value,
                cell.expected_max_sr,
                int(cell.n_obs),
            )
        )
    registry.update_dsr_many(campaign_id, dsr_rows)
    return n_cell_trials


MIN_DSR_POPULATION = 100
MIN_DISTINCT_DSR_RATIO = 0.01


def _assert_dsr_is_candidate_specific(results: list[TrialResult]) -> None:
    """Invariant: DSR must vary across the candidate population.

    Defect #6 signature: DSR was computed from ONE shared market-return
    series, so every candidate got the same value regardless of its own
    Sharpe. That is a POPULATION property - nearly every candidate shares a
    value - and it is checked as one.

    This replaces a pairwise check keyed on ``round(dsr, 12)``. DSR spans
    twelve orders of magnitude and that key is an ABSOLUTE grid: at
    DSR ~1e-9 it retains about three significant digits, while genuine
    candidate-to-candidate spacing down there is ~1e-11. The check fired
    below its own resolution and aborted GBPJPYm D1 on 2026-09-18 at
    DSR=1.677e-09, for two candidates whose Sharpes differed by 3.5e-4.

    Measured distinct-value ratios:

        USDCADm D1, healthy      1827/5000 = 0.365
        USDCADm H4, healthy      1637/5000 = 0.327
        one shared series           1/5000 = 0.000

    A 1% floor sits two orders of magnitude below the healthy cells and two
    above the defect, so it does not need fine tuning.

    Saturation is NOT excluded here, unlike the pairwise version. Reaching
    DSR exactly 0.0 needs scipy's norm.cdf to underflow at z < -38, which at
    n ~ 4000 bars requires the candidate's per-bar Sharpe to sit ~0.6 below
    SR* - far outside anything a real candidate produces. Values merely
    close to zero stay distinct floats and count normally.

    Populations under MIN_DSR_POPULATION are skipped: a ratio over a handful
    of candidates carries no signal. ``CampaignResult.distinct_dsr_count``
    reports the raw number for those cases.
    """
    finite_dsrs = [
        float(r.post_dsr)
        for r in results
        if r.post_dsr is not None and np.isfinite(r.post_dsr)
    ]
    if len(finite_dsrs) < MIN_DSR_POPULATION:
        return
    n_distinct = len(set(finite_dsrs))
    distinct_ratio = n_distinct / len(finite_dsrs)
    if distinct_ratio < MIN_DISTINCT_DSR_RATIO:
        raise AssertionError(
            f"DSR is not candidate-specific: only {n_distinct} distinct "
            f"values across {len(finite_dsrs)} candidates "
            f"(ratio {distinct_ratio:.4f} < {MIN_DISTINCT_DSR_RATIO}). "
            "DSR is probably being computed from one shared return series "
            "instead of each candidate's per_period_returns."
        )


@dataclass
class _DSRView:
    """``dsr_within`` plus the two numbers it cannot be read without.

    Computed twice per cell: once over every candidate carrying a finite
    DSR, once over only those that cleared ``min_trades``. Cell 1 of the
    2026-09-18 campaign is why: its dsr_within of 0.916 belonged to a
    23-trade candidate the funnel had already bucketed LowTrades.
    """

    dsr_within: float | None = None
    status: str | None = None
    winner_sr_per_bar: float | None = None
    winner_sr_annualised: float | None = None
    sr_star_per_bar: float | None = None
    sr_star_annualised: float | None = None
    n_candidates: int = 0


def _dsr_view(
    results: list[TrialResult],
    *,
    ppy: float,
    n_real_obs: int,
) -> _DSRView:
    """Best DSR over ``results``, with its owner's Sharpe and the luck bar."""
    view = _DSRView(n_candidates=len(results))
    scored = [
        r for r in results
        if r.post_dsr is not None and np.isfinite(r.post_dsr)
    ]
    if not scored:
        if any(r.unreliable_dsr for r in results):
            view.status = CellDSRStatus.UNRELIABLE.value
        elif n_real_obs > 0:
            view.status = CellDSRStatus.REJECTED.value
        return view
    winner = max(scored, key=lambda r: float(r.post_dsr))
    view.dsr_within = float(winner.post_dsr)
    view.status = CellDSRStatus.READY.value
    view.winner_sr_per_bar = winner.post_dsr_sr_per_bar
    view.sr_star_per_bar = winner.post_dsr_sr_star
    if view.winner_sr_per_bar is not None:
        view.winner_sr_annualised = float(view.winner_sr_per_bar * np.sqrt(ppy))
    if view.sr_star_per_bar is not None:
        view.sr_star_annualised = float(view.sr_star_per_bar * np.sqrt(ppy))
    return view


def _compute_pbo(
    results: list[TrialResult],
    *,
    symbol: str,
    timeframe: str,
    registry: TrialRegistry,
    campaign_id: str,
    n_blocks: int = 16,
) -> tuple[float | None, int]:
    """Run PBO/CSCV on the per-cell trial matrix of per-period returns.

    ``pbo_cscv`` takes a (T, N) matrix: T observations down the ROWS, one
    COLUMN per trial. The columns are the candidates' own
    per_period_returns; trials that carry no series are skipped.

    Orientation is load-bearing and fails silently if it is wrong: a
    (N, T) matrix validates just as happily, then blocks over candidates
    and ranks over bars. It is therefore asserted before the call.

    Returns (None, 0) when fewer than 2 candidates carry a series, or when
    the shared series length is under ``n_blocks * 2`` observations.
    """
    series: list[np.ndarray] = []
    for r in results:
        if r.per_period_returns is None:
            continue
        arr = np.asarray(r.per_period_returns, dtype=np.float64)
        if arr.size < n_blocks * 2:
            continue
        series.append(arr)
    # CSCV ranks candidates against each other: two is the floor.
    if len(series) < 2:
        return None, 0
    # All columns must share one index; truncate to the shortest series.
    min_len = min(s.size for s in series)
    # The TIME axis carries the block partition, so it is the one that has
    # to be long enough.
    if min_len < n_blocks * 2:
        return None, 0
    # axis=1 -> (T, N). axis=0 would give (N, T) and quietly invert the test.
    matrix = np.stack([s[:min_len] for s in series], axis=1)
    if matrix.shape != (min_len, len(series)):
        raise AssertionError(
            f"PBO matrix must be (n_obs, n_candidates) = "
            f"({min_len}, {len(series)}), got {matrix.shape}"
        )
    try:
        result = pbo_cscv(matrix, n_blocks=n_blocks)
    except (TypeError, AttributeError):
        # A wrong call signature is a programming error, not a data
        # condition. It was swallowed here before, which made PBO silently
        # None on every campaign; it must be loud.
        raise
    except Exception:
        # Degenerate matrix (constant columns, too few blocks): no PBO.
        return None, 0
    return float(result.pbo), result.n_splits


def run_campaign(
    *,
    campaign_id: str,
    scope: CampaignScope,
    registry: TrialRegistry,
    backtest_fn: Callable[[DNA], TrialResult],
    control_fn: Callable[[list[TrialResult]], list[float]] | None,
    real_scores_factory: Callable[[list[TrialResult]], list[float]],
    provenance: Provenance,
    target_is_annual_sharpe: float = 1.5,
    seed: int = 0,
    min_trades: int = 30,
    max_dd_limit: float = 0.30,
    return_threshold: float = -0.10,
    d_sharpe_threshold: float = 0.5,
    pbo_threshold: float = 0.5,
    dataset_hash: str = "synthetic",
    git_sha: str = "unknown",
    config_hash: str = "synthetic",
    n_purged_splits: int = 5,
    n_obs: int,
    min_obs_for_dsr: int = 100,
    bars_loaded: int = 0,
    audit_bars_expected: int = 0,
    on_progress: Callable[[int, int], None] | None = None,
    gene_pool: GenePool | None = None,
) -> CampaignResult:
    """Run a single-cell random-search campaign.

    S5 rerun additions:
        - bars_loaded vs audit_bars_expected is reported on the result.
        - Per-cell DSR is computed and feeds FailedDSR/DsrUnreliable.
        - Campaign-level PBO is computed and feeds HighPBO.
        - All exceptions are captured as CrashRecord; nothing is swallowed.
    """
    rng = np.random.default_rng(seed)
    symbol = scope.symbols[0]
    timeframe = scope.timeframes[0]
    registry.open_campaign(
        campaign_id=campaign_id,
        symbol=symbol,
        timeframe=timeframe,
        dataset_hash=dataset_hash,
        git_sha=git_sha,
        config_hash=config_hash,
        seed=int(seed),
    )
    started_at = datetime.now(tz=UTC).isoformat()

    # n_obs is the caller's declared observation count for this cell. It is
    # reported, never used to fabricate data: a missing value raises rather
    # than silently becoming a made-up default.
    if n_obs is None:
        raise ValueError(
            "run_campaign: n_obs is required (the cell's observation count); "
            "there is no default"
        )
    n_obs = int(n_obs)
    if n_obs < 0:
        raise ValueError(f"run_campaign: n_obs must be >= 0, got {n_obs}")

    outcomes: list[str] = []
    trial_results: list[TrialResult] = []
    crashes: list[CrashRecord] = []
    best_dna: DNA | None = None
    best_sharpe = float("-inf")
    n = max(0, int(scope.population_per_cell))
    for i in range(n):
        dna = random_dna(rng, gene_pool=gene_pool)
        result, crash = evaluate_one(dna, backtest_fn=backtest_fn)
        trial_results.append(result)
        if crash is not None:
            crashes.append(crash)
        initial_outcome = FailureTaxonomy.Crash.value if crash else result.outcome
        outcomes.append(initial_outcome)
        registry.record(
            campaign_id,
            dna_hash=dna.dna_hash,
            outcome=initial_outcome,
            sharpe=result.sharpe,
            n_trades=result.n_trades,
            total_return=result.total_return,
            max_dd=result.max_dd,
            symbol=symbol,
            timeframe=timeframe,
            op_type=OpType.SEARCH_TRIAL.value,
            dataset_hash=dataset_hash,
            source=provenance.source.value,
            as_of=provenance.as_of.isoformat(),
        )
        if result.sharpe is not None and result.sharpe > best_sharpe:
            best_sharpe = float(result.sharpe)
            best_dna = dna
        if on_progress is not None:
            on_progress(i + 1, n)

    # The only return series in this function are the candidates' own
    # per_period_returns, handed over by the backtester. Nothing here is
    # generated: no rng feeds any DSR, PBO or vs-random input.
    observed_obs = [
        int(np.asarray(r.per_period_returns).size)
        for r in trial_results
        if r.per_period_returns is not None
    ]
    n_real_obs = max(observed_obs) if observed_obs else 0

    purged = purged_cv(
        n_samples=n_real_obs if n_real_obs > 0 else 1000,
        config=PurgedCVConfig(
            n_splits=int(n_purged_splits),
            horizon_bars=50,
            pct_embargo=0.01,
        ),
    )
    _ = purged

    # Defect 2 fix: per-cell DSR feeds the funnel.
    # W2: the luck bar scales with DISTINCT backtests, not genome count.
    trials_raw = len(trial_results)
    trials_effective = count_effective_trials(
        r.dna for r in trial_results if r.dna is not None
    )
    dsr_N_used = _populate_post_dsr(  # noqa: N806 - matches the field name
        trial_results,
        symbol=symbol,
        timeframe=timeframe,
        registry=registry,
        campaign_id=campaign_id,
        min_obs_for_dsr=min_obs_for_dsr,
        n_effective_trials=trials_effective,
    )
    _assert_dsr_is_candidate_specific(trial_results)
    # Re-apply funnel now that post_dsr is set. Crash rows stay Crash.
    final_outcomes: list[str] = []
    for idx, r in enumerate(trial_results):
        if outcomes[idx] == FailureTaxonomy.Crash.value:
            final_outcomes.append(FailureTaxonomy.Crash.value)
            continue
        new_outcome = _apply_funnel(
            r, min_trades=min_trades, max_dd_limit=max_dd_limit,
            return_threshold=return_threshold,
            d_sharpe_threshold=d_sharpe_threshold, pbo_threshold=pbo_threshold,
        )
        final_outcomes.append(new_outcome)
    outcomes = final_outcomes

    # Defect 2 fix: campaign-level PBO.
    pbo, pbo_n = _compute_pbo(
        trial_results,
        symbol=symbol, timeframe=timeframe,
        registry=registry, campaign_id=campaign_id,
    )
    # The documented order (module docstring D2, _apply_funnel) puts the
    # early gates FIRST: a candidate rejected there keeps that label. Only
    # candidates that reached the DSR stage are relabelled here. Labels only:
    # the survivor set, the registry rows (initial outcomes) and every count
    # DSR or the verdict reads are the same either way.
    early_rejects = {
        FailureTaxonomy.Crash.value, FailureTaxonomy.SpreadUncalibrated.value,
        FailureTaxonomy.LowTrades.value, FailureTaxonomy.HighDD.value,
        FailureTaxonomy.NegReturn.value,
    }
    for idx, r in enumerate(trial_results):
        if r.outcome == FailureTaxonomy.Crash.value or outcomes[idx] in early_rejects:
            continue
        if r.post_dsr is not None and r.post_dsr < d_sharpe_threshold:
            outcomes[idx] = FailureTaxonomy.FailedDSR.value
            continue
        if r.unreliable_dsr and outcomes[idx] != FailureTaxonomy.FailedDSR.value:
            outcomes[idx] = FailureTaxonomy.DsrUnreliable.value
            continue
        if pbo is not None and pbo > pbo_threshold and outcomes[idx] not in {
            FailureTaxonomy.FailedDSR.value, FailureTaxonomy.DsrUnreliable.value,
        }:
            outcomes[idx] = FailureTaxonomy.HighPBO.value

    funnel = summarise(outcomes)

    # Stage B diagnostics: trade counts per funnel bucket and distinct DSR count.
    trades_by_bucket: dict[str, int] = {}
    for idx, r in enumerate(trial_results):
        bucket = outcomes[idx]
        trades_by_bucket[bucket] = trades_by_bucket.get(bucket, 0) + int(r.n_trades or 0)
    blocked_zero_spread_events = sum(
        int(getattr(r, "blocked_zero_spread_events", 0) or 0)
        for r in trial_results
    )
    deferred_zero_spread_fills = sum(
        int(getattr(r, "deferred_zero_spread_fills", 0) or 0)
        for r in trial_results
    )
    sltp_deferred_zero_spread = sum(
        int(getattr(r, "sltp_deferred_zero_spread", 0) or 0)
        for r in trial_results
    )
    exit_reasons_total: dict[str, int] = {}
    for r in trial_results:
        for reason, count in (getattr(r, "exit_reasons", None) or {}).items():
            exit_reasons_total[reason] = exit_reasons_total.get(reason, 0) + int(count)
    parity_classes: dict[str, int] = {}
    for r in trial_results:
        pc = str(getattr(r, "parity_class", "M1_OHLC") or "M1_OHLC")
        parity_classes[pc] = parity_classes.get(pc, 0) + 1
    distinct_dsr_count = len({
        round(float(r.post_dsr), 9)
        for r in trial_results
        if r.post_dsr is not None and np.isfinite(r.post_dsr)
    })

    # Per-cell DSR, reported over BOTH populations. A headline statistic
    # taken over every candidate describes the search; taken over those that
    # cleared min_trades it describes something tradeable. They are not the
    # same number and a report must never carry only one of them.
    ppy = float(PERIODS_PER_YEAR.get(timeframe, DEFAULT_PERIODS_PER_YEAR))
    eligible = [r for r in trial_results if int(r.n_trades or 0) >= min_trades]
    dsr_all = _dsr_view(trial_results, ppy=ppy, n_real_obs=n_real_obs)
    dsr_eligible = _dsr_view(eligible, ppy=ppy, n_real_obs=n_real_obs)

    # Cross-cell DSR. This used to be `dsr_across = dsr_all.dsr_within`,
    # with the real call sitting dead behind `if False`. The copy matched
    # what compute_cross_cell_dsr returns for a one-cell input, which is
    # exactly why it went unnoticed - the value looked right while nothing
    # computed it. It now goes through the real function, and the result
    # carries the cell count so a single-cell number is never mistaken for
    # a second selection correction that actually corrected something.
    cell_sharpes_ann = [
        float(r.sharpe)
        for r in trial_results
        if r.sharpe is not None and np.isfinite(r.sharpe)
    ]
    this_cell = CellDSRResult(
        symbol=symbol,
        timeframe=timeframe,
        status=(
            CellDSRStatus.READY
            if dsr_all.status == CellDSRStatus.READY.value
            else CellDSRStatus(dsr_all.status or CellDSRStatus.REJECTED.value)
        ),
        dsr=dsr_all.dsr_within,
        n_obs=int(n_real_obs),
        n_trials=len(cell_sharpes_ann),
        var_sr_trials=(
            float(np.var(cell_sharpes_ann, ddof=1))
            if len(cell_sharpes_ann) >= 2
            else float("nan")
        ),
        winner_sharpe=dsr_all.winner_sr_annualised,
        min_obs_threshold=int(min_obs_for_dsr),
        sr_hat_per_bar=dsr_all.winner_sr_per_bar,
        expected_max_sr=dsr_all.sr_star_per_bar,
    )
    cross = compute_cross_cell_dsr((this_cell,))
    dsr_across = cross.dsr_across_cells
    dsr_across_status = cross.status.value
    dsr_across_n_cells = cross.n_cells

    def _vs_random_for(trials: list[TrialResult]) -> VsRandomResult | None:
        scores = real_scores_factory(trials)
        if control_fn is None or not scores:
            return None
        # The control sees the trials it is scored against, so it can match
        # each one (a sham keeps the candidate's trade count and exit genes).
        scored = [
            t for t in trials if t.sharpe is not None and np.isfinite(t.sharpe)
        ]
        control_scores = control_fn(scored)
        scope_real = scope_from_campaign(
            symbol=symbol,
            timeframe=timeframe,
            dataset_hash=dataset_hash,
            cost_model="synthetic",
            risk_rules={"max_dd": max_dd_limit},
            exec_mode="MARKET2",
            min_trades=min_trades,
            n_trials=len(scores),
            date_from_utc=started_at,
            date_to_utc=started_at,
        )
        assert_scopes_match(scope_real, scope_to_control(scope_real))
        return vs_random(
            scores, control_scores,
            n_bootstrap=2000, n_permutations=500, seed=seed,
        )

    vsr = _vs_random_for(trial_results)
    vsr_eligible = _vs_random_for(eligible)

    # Same split for the raw winner.
    best_sharpe_eligible: float | None = None
    best_dna_eligible: DNA | None = None
    for r in eligible:
        if r.sharpe is None or not np.isfinite(r.sharpe):
            continue
        if best_sharpe_eligible is None or float(r.sharpe) > best_sharpe_eligible:
            best_sharpe_eligible = float(r.sharpe)
            best_dna_eligible = r.dna

    return CampaignResult(
        campaign_id=campaign_id,
        n_trials=n,
        funnel=funnel,
        best_dna=best_dna,
        best_sharpe=best_sharpe if best_sharpe != float("-inf") else None,
        vs_random=vsr,
        provenance=provenance,
        as_of=started_at,
        n_obs_real=int(n_obs) if n_obs else 0,
        bars_loaded=int(bars_loaded),
        audit_bars_expected=int(audit_bars_expected),
        dsr_within=dsr_all.dsr_within,
        dsr_within_status=dsr_all.status,
        dsr_across=dsr_across,
        dsr_across_status=dsr_across_status,
        dsr_across_n_cells=dsr_across_n_cells,
        dsr_winner_sr_per_bar=dsr_all.winner_sr_per_bar,
        dsr_winner_sr_annualised=dsr_all.winner_sr_annualised,
        dsr_sr_star_per_bar=dsr_all.sr_star_per_bar,
        dsr_sr_star_annualised=dsr_all.sr_star_annualised,
        n_eligible=len(eligible),
        dsr_within_eligible=dsr_eligible.dsr_within,
        dsr_within_eligible_status=dsr_eligible.status,
        dsr_eligible_winner_sr_per_bar=dsr_eligible.winner_sr_per_bar,
        dsr_eligible_winner_sr_annualised=dsr_eligible.winner_sr_annualised,
        dsr_eligible_sr_star_per_bar=dsr_eligible.sr_star_per_bar,
        dsr_eligible_sr_star_annualised=dsr_eligible.sr_star_annualised,
        vs_random_eligible=vsr_eligible,
        best_sharpe_eligible=best_sharpe_eligible,
        best_dna_eligible=best_dna_eligible,
        pbo=pbo,
        pbo_n_splits=pbo_n,
        crashes=tuple(crashes),
        trades_by_bucket=trades_by_bucket,
        distinct_dsr_count=distinct_dsr_count,
        blocked_zero_spread_events=blocked_zero_spread_events,
        deferred_zero_spread_fills=deferred_zero_spread_fills,
        sltp_deferred_zero_spread=sltp_deferred_zero_spread,
        exit_reasons=exit_reasons_total,
        parity_classes=parity_classes,
        trials_raw=trials_raw,
        trials_effective=trials_effective,
        dsr_N_used=dsr_N_used,
        degeneracy=degeneracy_report(
            trial_results, outcomes, min_trades=min_trades,
            trials_raw=trials_raw, trials_effective=trials_effective,
        ),
        cost_totals=cost_totals(trial_results),
    )


def runner_main(
    *,
    out_path: Path,
    payload: dict[str, object],
) -> int:
    """Entry point for a separate OS process. ``payload`` is JSON-serialisable."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        out_path.write_text("OK", encoding="utf-8")
        return 0
    except Exception as exc:
        out_path.write_text(
            f"CRASH {type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            encoding="utf-8",
        )
        return 1


def spawn_runner_subprocess(payload: dict[str, object], out_path: Path) -> multiprocessing.Process:
    """Spawn a separate OS process. The UI reattaches by reading the result."""
    proc = multiprocessing.Process(
        target=runner_main,
        kwargs={"out_path": out_path, "payload": payload},
        daemon=False,
    )
    proc.start()
    return proc


__all__ = [
    "CampaignResult",
    "CrashRecord",
    "TrialResult",
    "evaluate_one",
    "run_campaign",
    "runner_main",
    "spawn_runner_subprocess",
]


# Avoid "unused import" complaints for symbols that are part of the public
# surface referenced by tests/UI.
_ = field
_ensure_verdict_complete: Any = ensure_verdict_complete
_Verdict: Any = Verdict
