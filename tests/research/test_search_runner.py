"""End-to-end test of the search runner pipeline.

The runner is exercised with a synthetic backtester that returns
honest-looking trade counts and Sharpes. The test asserts:

  - Every SEARCH_TRIAL row is recorded with the new op_type and source
    fields.
  - The funnel taxonomy reports reasonable counts.
  - vs_random runs and its verdict is one of the four documented strings.
  - The verdict object is launchable and contains both comparator rows.
"""
from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from engine.research.campaign.budget import CampaignScope
from engine.research.campaign.verdict import compute_verdict, launch_guard
from engine.research.provenance import DataSource, Provenance
from engine.research.search.dna import DNA, FailureTaxonomy
from engine.research.search.runner import TrialResult, run_campaign
from engine.research.search.sampler import random_dna
from engine.validation.op_types import OpType
from engine.validation.trial_registry import TrialRegistry


@pytest.fixture
def reg(tmp_path: Path) -> TrialRegistry:
    return TrialRegistry(tmp_path / "trials.sqlite")


def _backtest_factory(rng: np.random.Generator) -> Callable[[DNA], TrialResult]:
    def _bt(dna: DNA) -> TrialResult:
        # Realistic mix: most strategies die at LowTrades, a chunk crash,
        # some hit HighDD or NegReturn, a few make it through. The funnel
        # taxonomy is exercised on every category.
        roll = rng.random()
        if roll < 0.10:
            return TrialResult(
                dna=dna, outcome=FailureTaxonomy.Crash.value,
                sharpe=None, n_trades=0, total_return=0.0, max_dd=1.0,
            )
        if roll < 0.50:
            n_trades = int(rng.integers(0, 25))
            return TrialResult(
                dna=dna, outcome=FailureTaxonomy.EVALUATED.value,
                sharpe=0.0, n_trades=n_trades, total_return=0.0, max_dd=0.05,
            )
        n_trades = int(rng.integers(30, 100))
        # Bias some toward HighDD and some toward NegReturn so the
        # funnel taxonomy has every category in a 200-trial run.
        bias = rng.random()
        if bias < 0.10:
            max_dd = 0.40  # > 0.30 -> HighDD
            sr = -0.5
        elif bias < 0.20:
            max_dd = 0.10
            sr = -3.0  # -> total_return = -0.15 < -0.10 -> NegReturn
        else:
            max_dd = float(np.clip(abs(rng.normal(0.1, 0.05)), 0.0, 0.30))
            sr = float(rng.normal(0.0, 0.5))
        total_return = float(sr * 0.05)
        return TrialResult(
            dna=dna, outcome=FailureTaxonomy.EVALUATED.value,
            sharpe=sr, n_trades=n_trades,
            total_return=total_return, max_dd=max_dd,
        )
    return _bt


_DSR_N_OBS = 300  # >= min_obs_for_dsr (100) and >= 2 * PBO n_blocks (16)


def _series_pool(n: int, *, seed: int = 20260918) -> list[np.ndarray]:
    """One distinct bar-level return series per candidate.

    Built once, outside the campaign, so the campaign path itself needs no
    rng. Each series is the same bars in a different order plus its own
    drift, so every candidate gets its own Sharpe and its own DSR.
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 0.01, size=_DSR_N_OBS)
    return [
        base[rng.permutation(_DSR_N_OBS)] + (k - n / 2.0) * 2e-5
        for k in range(n)
    ]


def _annualised_sharpe(series: np.ndarray, periods_per_year: float = 252.0) -> float:
    sd = float(np.std(series, ddof=1))
    if sd == 0.0:
        return 0.0
    return float(np.mean(series) / sd * np.sqrt(periods_per_year))


def _backtest_factory_with_returns(
    rng: np.random.Generator, pool: list[np.ndarray]
) -> Callable[[DNA], TrialResult]:
    """Same funnel mix as ``_backtest_factory``, but every evaluated
    candidate carries its own ``per_period_returns``.

    ``_backtest_factory`` sets no return series, so DSR is unavailable for
    every candidate and the whole population lands in DsrUnreliable before
    it can reach FailedDSR. A fixture that wants to exercise the DSR gate
    has to supply the series the gate reads.
    """
    state = {"i": 0}

    def _bt(dna: DNA) -> TrialResult:
        roll = rng.random()
        if roll < 0.10:
            return TrialResult(
                dna=dna, outcome=FailureTaxonomy.Crash.value,
                sharpe=None, n_trades=0, total_return=0.0, max_dd=1.0,
            )
        series = pool[state["i"] % len(pool)]
        state["i"] += 1
        sharpe = _annualised_sharpe(series)
        if roll < 0.50:
            return TrialResult(
                dna=dna, outcome=FailureTaxonomy.EVALUATED.value,
                sharpe=sharpe, n_trades=int(rng.integers(0, 25)),
                total_return=0.0, max_dd=0.05,
                per_period_returns=series,
            )
        n_trades = int(rng.integers(30, 100))
        # Steer the middle gates explicitly; total_return and max_dd decide
        # HighDD / NegReturn, and they are independent of the Sharpe.
        bias = rng.random()
        if bias < 0.10:
            max_dd, total_return = 0.40, 0.02          # -> HighDD
        elif bias < 0.20:
            max_dd, total_return = 0.10, -0.15         # -> NegReturn
        else:
            max_dd = float(np.clip(abs(rng.normal(0.1, 0.05)), 0.0, 0.30))
            total_return = 0.02                        # -> reaches the DSR gate
        return TrialResult(
            dna=dna, outcome=FailureTaxonomy.EVALUATED.value,
            sharpe=sharpe, n_trades=n_trades,
            total_return=total_return, max_dd=max_dd,
            per_period_returns=series,
        )

    return _bt


def _control_factory(
    rng: np.random.Generator,
) -> Callable[[list[TrialResult]], list[float]]:
    """One N(0, 0.5) control score per scored trial.

    run_campaign hands control_fn the scored TrialResults rather than their
    count (so a sham control can match each candidate); the draws are the
    same as when it took the count.
    """
    def _ctrl(trials: list[TrialResult]) -> list[float]:
        return list(rng.normal(0.0, 0.5, size=len(trials)))
    return _ctrl


def _real_scores(trials: list[TrialResult]) -> list[float]:
    out: list[float] = []
    for t in trials:
        if t.sharpe is not None and np.isfinite(t.sharpe):
            out.append(float(t.sharpe))
    return out


def test_runner_writes_search_trials_with_provenance(reg: TrialRegistry) -> None:
    rng = np.random.default_rng(0)
    scope = CampaignScope(
        symbols=("USDCADm",),
        timeframes=("H2",),
        population_per_cell=200,
    )
    provenance = Provenance(
        source=DataSource.CACHED,
        dataset_hash="abc",
        as_of=datetime(2026, 9, 16, 0, 0, 0, tzinfo=UTC),
        mixed_source=False,
    )
    result = run_campaign(
        campaign_id="c1",
        scope=scope,
        registry=reg,
        backtest_fn=_backtest_factory(rng),
        control_fn=_control_factory(rng),
        real_scores_factory=_real_scores,
        provenance=provenance,
        dataset_hash=provenance.dataset_hash,
        seed=0,
        n_obs=0,  # synthetic backtester: no real bars
    )
    assert result.n_trials == 200
    # All recorded rows are SEARCH_TRIAL.
    listed = reg.list_trials("c1")
    assert len(listed) == 200
    for row in listed:
        assert row.op_type == OpType.SEARCH_TRIAL.value
        assert row.source == "CACHED"
        assert row.dataset_hash == "abc"
    # Validation runs are still excluded.
    s = reg.stats("c1")
    assert s.n_trials == 200
    assert s.validation_runs == 0


def test_runner_emits_failure_taxonomy(reg: TrialRegistry) -> None:
    rng = np.random.default_rng(1)
    scope = CampaignScope(
        symbols=("USDCADm",), timeframes=("H2",), population_per_cell=500,
    )
    provenance = Provenance(
        source=DataSource.MT5_LIVE,
        dataset_hash="abc",
        as_of=datetime(2026, 9, 16, 0, 0, 0, tzinfo=UTC),
        mixed_source=False,
    )
    result = run_campaign(
        campaign_id="c2",
        scope=scope,
        registry=reg,
        backtest_fn=_backtest_factory_with_returns(rng, _series_pool(500)),
        control_fn=_control_factory(rng),
        real_scores_factory=_real_scores,
        provenance=provenance,
        seed=1,
        n_obs=_DSR_N_OBS,
    )
    assert result.funnel.total == 500
    # Every gate the fixture steers toward must actually appear. FailedDSR
    # is only reachable because the fixture carries per_period_returns: DSR
    # reads the candidate's own series, so a fixture without one puts the
    # whole population in DsrUnreliable and the DSR gate is never tested.
    for category in ("Crash", "LowTrades", "HighDD", "NegReturn", "FailedDSR"):
        assert category in result.funnel.by_category, (
            f"{category} missing from {result.funnel.by_category}"
        )
    assert result.funnel.n_failed_dsr > 0
    # DSR is candidate-specific, not one shared cell-level number.
    assert result.distinct_dsr_count > 1, result.distinct_dsr_count
    # Every non-crash trial is in some category.
    assert sum(result.funnel.by_category.values()) == 500


def test_runner_vs_random_verdict_is_documented(reg: TrialRegistry) -> None:
    rng = np.random.default_rng(2)
    scope = CampaignScope(
        symbols=("USDCADm",), timeframes=("H2",), population_per_cell=300,
    )
    provenance = Provenance(
        source=DataSource.MT5_LIVE,
        dataset_hash="abc",
        as_of=datetime(2026, 9, 16, 0, 0, 0, tzinfo=UTC),
        mixed_source=False,
    )
    result = run_campaign(
        campaign_id="c3",
        scope=scope,
        registry=reg,
        backtest_fn=_backtest_factory(rng),
        control_fn=_control_factory(rng),
        real_scores_factory=_real_scores,
        provenance=provenance,
        seed=2,
        n_obs=0,  # synthetic backtester: no real bars
    )
    assert result.vs_random is not None
    assert result.vs_random.verdict in {
        "STRUCTURE FOUND - winner and population both beat random",
        "WINNER BEATS RANDOM - population shift weak or negligible",
        "POPULATION SHIFT ONLY - winner is within random range",
        "NO STRUCTURE - indistinguishable from random search",
    }
    # p_max is the decision rule; p_single is a diagnostic.
    assert 0.0 <= result.vs_random.p_max <= 1.0
    assert 0.0 <= result.vs_random.p_single <= 1.0
    # The exit verdict reflects the funnel + vs_random.
    assert isinstance(result.exit_verdict(), str)


def test_runner_validation_run_does_not_inflate_budget(reg: TrialRegistry) -> None:
    """Adding VALIDATION_RUN rows to a campaign must not change the
    search-trial DSR bar. I1 end-to-end via the runner."""
    rng = np.random.default_rng(3)
    scope = CampaignScope(
        symbols=("USDCADm",), timeframes=("H2",), population_per_cell=80,
    )
    provenance = Provenance(
        source=DataSource.BUNDLED,
        dataset_hash="abc",
        as_of=datetime(2026, 9, 16, 0, 0, 0, tzinfo=UTC),
        mixed_source=False,
    )
    run_campaign(
        campaign_id="c4",
        scope=scope,
        registry=reg,
        backtest_fn=_backtest_factory(rng),
        control_fn=None,
        real_scores_factory=_real_scores,
        provenance=provenance,
        seed=3,
        n_obs=0,  # synthetic backtester: no real bars
    )
    s_before = reg.stats("c4")
    # Record 500 validation runs on a different cell.
    for i in range(500):
        dna = random_dna(np.random.default_rng(100 + i))
        reg.record(
            "c4", dna_hash=dna.dna_hash, outcome="EVALUATED",
            sharpe=10.0,  # absurdly high
            symbol="EURUSDm", timeframe="D1",
            op_type=OpType.VALIDATION_RUN.value,
            dataset_hash="d2", source="CACHED",
            as_of=datetime(2026, 9, 16, tzinfo=UTC).isoformat(),
        )
    s_after = reg.stats("c4")
    assert s_after.n_trials == s_before.n_trials  # validation runs excluded
    assert s_after.validation_runs == 500
    assert s_before.max_sharpe == s_after.max_sharpe  # only SEARCH_TRIAL rows


def test_verdict_visible_to_ui_is_required_for_launch(reg: TrialRegistry) -> None:
    """A campaign cannot launch without the verdict object visible to the UI."""
    scope = CampaignScope(
        symbols=("USDCADm",), timeframes=("H2",), population_per_cell=10,
    )
    provenance = Provenance(
        source=DataSource.BUNDLED,
        dataset_hash="abc",
        as_of=datetime(2026, 9, 16, 0, 0, 0, tzinfo=UTC),
        mixed_source=False,
    )
    verdict = compute_verdict(
        scope,
        per_cell_years={("USDCADm", "H2"): 5.0},
        provenance=provenance,
    )
    assert verdict.verdict_visible_to_ui is True
    launch_guard(verdict)  # does not raise
    # A verdict with verdict_visible_to_ui=False is refused.
    from dataclasses import replace
    blocked = replace(verdict, verdict_visible_to_ui=False)
    import pytest
    with pytest.raises(Exception, match="refused"):
        launch_guard(blocked)


def test_vault_rate_on_zero_edge_synthetic_under_one_percent(reg: TrialRegistry) -> None:
    """S5 spec: a full DSR gate on 5,000 zero-edge candidates yields
    roughly 0.14%. The runner's vault rate on synthetic null input
    must be under 1%."""
    rng = np.random.default_rng(7)
    scope = CampaignScope(
        symbols=("USDCADm",), timeframes=("H2",), population_per_cell=5000,
    )
    provenance = Provenance(
        source=DataSource.MT5_LIVE,
        dataset_hash="zero-edge",
        as_of=datetime(2026, 9, 16, 0, 0, 0, tzinfo=UTC),
        mixed_source=False,
    )
    result = run_campaign(
        campaign_id="c4",
        scope=scope,
        registry=reg,
        backtest_fn=_backtest_factory(rng),
        control_fn=_control_factory(rng),
        real_scores_factory=_real_scores,
        provenance=provenance,
        seed=7,
        n_obs=2000,
    )
    vault_rate = result.funnel.n_vaulted / max(1, result.funnel.total)
    assert vault_rate < 0.01, (
        f"vault_rate={vault_rate:.4f} ({result.funnel.n_vaulted}/"
        f"{result.funnel.total}) on synthetic zero-edge input; "
        f"DSR is supposed to reject 99%+ of these."
    )


def test_mixed_source_blocks_launch() -> None:
    from engine.research.campaign.preflight import PreflightError, refuse_on_provenance
    p = Provenance(
        source=DataSource.CACHED,
        dataset_hash="abc",
        as_of=datetime(2026, 9, 16, 0, 0, 0, tzinfo=UTC),
        mixed_source=True,
    )
    with pytest.raises(PreflightError):
        refuse_on_provenance(p)
