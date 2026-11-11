"""Planted-edge canary: exercise the FULL search pipeline end-to-end.

Six invalid verdicts were shipped because every component was individually
green while the whole path was broken (synthetic backtester, DSR not wired,
DSR from market returns, ...). This canary plants a deliberately obvious edge
into one candidate and asserts the pipeline SURVIVES it and returns
STRUCTURE FOUND. If it fails, any campaign verdict from the same code is void.

Two profiles:
    fast (default): sliced bars + small population, target < 30s, run before
        every campaign as a preflight.
    full: the real cell bars + full population, run weekly.

The planting is done by wrapping ``backtest_fn`` so the FIRST evaluated
candidate returns a genuine edged series (per-bar Sharpe ~0.27). No global
monkeypatching — the injection is local and deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np

from engine.research.campaign.budget import CampaignScope
from engine.research.provenance import DataSource, Provenance
from engine.research.registry.annualisation import bars_per_year
from engine.research.search.dna import DNA, FailureTaxonomy
from engine.core.git_sha import resolve_git_sha
from engine.research.search.real_backtest import (
    GAUSSIAN_CONTROL_KIND,
    CellBars,
    fast_backtest_factory,
    gaussian_control_factory,
    real_backtest_factory,
    real_scores,
    slice_cell_bars,
)
from engine.research.search.runner import CampaignResult, TrialResult, run_campaign
from engine.validation.trial_registry import TrialRegistry

FAST_BARS = 1200
FAST_POPULATION = 200
FULL_POPULATION = 500

# Planted edge parameters. loc/scale give a per-bar Sharpe ~0.27, deliberately
# obvious so a working pipeline cannot miss it.
PLANTED_LOC = 0.003
PLANTED_SCALE = 0.011


@dataclass(frozen=True)
class CanaryResult:
    passed: bool
    survivors: int
    dsr_within: float | None
    vs_random_p_max: float | None
    exit_verdict: str
    crashes: int
    git_sha: str
    reason: str = ""
    # The canary's vs-random control is iid N(0, 0.5) Sharpes, not the
    # sham real campaigns use: it has to stay inside the preflight budget,
    # and its question is whether a planted edge surfaces.
    control: str = GAUSSIAN_CONTROL_KIND


def _make_planting_backtest_fn(
    cell: CellBars, rng: np.random.Generator, *, fast: bool = True
):
    base = fast_backtest_factory(cell) if fast else real_backtest_factory(cell)
    planted_returns = rng.normal(
        loc=PLANTED_LOC, scale=PLANTED_SCALE, size=cell.n_returns
    )
    ppy = bars_per_year(cell.timeframe)
    planted_sharpe = float(
        np.mean(planted_returns) / np.std(planted_returns, ddof=1) * np.sqrt(ppy)
    )
    state = {"planted": False}

    def _bt(dna: DNA) -> TrialResult:
        result = base(dna)
        if state["planted"]:
            return result
        state["planted"] = True
        return TrialResult(
            dna=dna,
            outcome=FailureTaxonomy.EVALUATED.value,
            sharpe=planted_sharpe,
            n_trades=max(60, int(result.n_trades or 0)),
            total_return=1.0,
            max_dd=min(float(result.max_dd or 0.05), 0.05),
            spread_uncalibrated=False,
            per_period_returns=planted_returns,
        )

    _bt.fallback_state = getattr(base, "fallback_state", {"fallbacks": 0, "reasons": []})
    return _bt


def run_planted_edge_canary(
    cell: CellBars,
    *,
    fast: bool = True,
    seed: int = 20260917,
    tmp_dir=None,
) -> CanaryResult:
    """Run the full pipeline with one planted edge; assert it survives."""
    import tempfile
    from pathlib import Path

    work_cell = slice_cell_bars(cell, FAST_BARS) if fast else cell
    population = FAST_POPULATION if fast else FULL_POPULATION
    rng = np.random.default_rng(seed)
    backtest_fn = _make_planting_backtest_fn(work_cell, rng, fast=fast)

    if tmp_dir is None:
        tmp_dir = tempfile.mkdtemp(prefix="canary-")
    reg_path = Path(tmp_dir) / "canary.sqlite"
    registry = TrialRegistry(reg_path)
    scope = CampaignScope(
        symbols=(work_cell.symbol,),
        timeframes=(work_cell.timeframe,),
        population_per_cell=population,
    )
    provenance = Provenance(
        # A pinned snapshot (the canary's own, since 2026-10-26) is bundled data, not a live read.
        source=DataSource.MT5_LIVE if work_cell.source == "MT5_LIVE" else DataSource.BUNDLED,
        dataset_hash=work_cell.dataset_hash,
        as_of=datetime.now(tz=UTC),
        mixed_source=False,
    )
    git_sha = resolve_git_sha()
    result: CampaignResult = run_campaign(
        campaign_id="planted-edge-canary",
        scope=scope,
        registry=registry,
        backtest_fn=backtest_fn,
        control_fn=gaussian_control_factory(np.random.default_rng(seed + 1)),
        real_scores_factory=real_scores,
        provenance=provenance,
        target_is_annual_sharpe=1.5,
        seed=seed + 2,
        n_obs=work_cell.n_returns,
        bars_loaded=work_cell.n_bars,
        audit_bars_expected=work_cell.n_bars,
        git_sha=git_sha,
    )

    verdict = result.exit_verdict()
    p_max = result.vs_random.p_max if result.vs_random else None
    survivors = result.funnel.survivors
    # WHAT THIS CANARY ASSERTS, and what it deliberately no longer does.
    #
    # Its job, per its own docstring and scripts/canary_fast.py, is
    # "the pipeline can still detect an edge it was handed deliberately".
    # It plants ONE candidate with known-good synthetic returns and asks
    # whether the pipeline surfaces it. The four checks below are that
    # question, stated directly.
    #
    # WEAKENED IN S1/W4, with spec-owner sign-off, and this is the one
    # gate relaxation in the stage. A fifth check required exit_verdict()
    # to start with STRUCTURE FOUND. That verdict is a conjunction: the
    # winner beats random AND the whole eligible POPULATION shifts away
    # from the control pool. The population half is a property of the
    # cell and the strategy family. It is not a property of the
    # pipeline's ability to detect a plant, which is what this gate
    # exists to protect.
    #
    # W4 made the population half false on this cell, for a good reason.
    # Two pre-existing swap bugs were fixed: the engine re-charged the
    # whole holding period at every day-roll (quadratic), and it treated
    # MT5's signed rate as though a pay rate were a credit. Together
    # they had been PAYING each candidate to hold positions. With carry
    # costed correctly the eligible population's mean Sharpe falls to
    # where it no longer separates from an unskilled N(0, 0.5) control:
    #
    #     before W4   p_mean 0.002   effect 1.153   STRUCTURE FOUND
    #     after  W4   p_mean 0.299   effect 0.154   WINNER BEATS RANDOM
    #
    # The planted edge is detected exactly as well as before
    # (survivors=1, dsr_within=0.99999, p_max=0.0005). What changed is a
    # research finding about MA-cross on USDCADm D1 once it pays its
    # carry -- and a research finding must not masquerade as a pipeline
    # regression that blocks every commit.
    #
    # The verdict is still computed, still returned on CanaryResult and
    # still printed by canary_fast.py, so the finding stays visible. It
    # just no longer gates.
    checks = [
        (survivors >= 1, f"survivors={survivors} < 1"),
        (
            result.dsr_within is not None and result.dsr_within > 0.5,
            f"dsr_within={result.dsr_within} not > 0.5",
        ),
        (p_max is not None and p_max < 0.05, f"vs_random p_max={p_max} not < 0.05"),
        (len(result.crashes) == 0, f"crashes={len(result.crashes)}"),
    ]
    failed = [msg for ok, msg in checks if not ok]
    return CanaryResult(
        passed=not failed,
        survivors=survivors,
        dsr_within=result.dsr_within,
        vs_random_p_max=p_max,
        exit_verdict=verdict,
        crashes=len(result.crashes),
        git_sha=git_sha,
        reason="; ".join(failed),
    )


__all__ = [
    "CanaryResult",
    "run_planted_edge_canary",
    "FAST_BARS",
    "FAST_POPULATION",
    "FULL_POPULATION",
]
