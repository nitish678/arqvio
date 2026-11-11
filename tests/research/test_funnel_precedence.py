"""The funnel's documented order holds: an early-gate reject keeps its label.

runner.py's contract (module docstring D2, _apply_funnel) is Crash ->
SpreadUncalibrated -> LowTrades -> HighDD -> NegReturn -> DsrUnreliable ->
FailedDSR -> HighPBO -> VAULTED. The post-DSR relabelling used to overwrite
every non-crash candidate, so a 5-trade candidate with a low DSR read FailedDSR.
It now relabels only candidates that reached the DSR stage. Labels only: the
evidence (artifacts/reports/s4_stage3/funnel_evidence.log) shows every leading
count, DSR number and registry row unchanged.
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from engine.research.campaign.budget import CampaignScope
from engine.research.provenance import DataSource, Provenance
from engine.research.search.runner import TrialResult, run_campaign
from engine.validation.trial_registry import TrialRegistry


def _campaign(tmp_path: Path, backtest) -> object:
    rng = np.random.default_rng(1)
    return run_campaign(
        campaign_id="precedence",
        scope=CampaignScope(symbols=("USDCADm",), timeframes=("H1",), population_per_cell=150),
        registry=TrialRegistry(tmp_path / "t.sqlite"), backtest_fn=backtest,
        control_fn=lambda trials: list(rng.normal(0.0, 0.5, size=len(trials))),
        real_scores_factory=lambda ts: [float(t.sharpe) for t in ts if t.sharpe is not None],
        provenance=Provenance(source=DataSource.MT5_LIVE, dataset_hash="synthetic",
                              as_of=datetime(2026, 10, 1, tzinfo=UTC), mixed_source=False),
        seed=3, n_obs=500, dataset_hash="synthetic-precedence", git_sha="test", config_hash="test",
    )


def _returns(dna) -> np.ndarray:
    return np.random.default_rng(int(dna.dna_hash, 16) % 2**32).normal(0.0, 0.01, size=500)


def test_too_few_trades_stays_low_trades_even_with_a_low_dsr(tmp_path: Path) -> None:
    def backtest(dna):
        r = _returns(dna)
        return TrialResult(dna=dna, sharpe=float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(6048)),
                           n_trades=5, total_return=0.1, max_dd=0.05, per_period_returns=r)

    result = _campaign(tmp_path, backtest)
    assert dict(result.funnel.by_category) == {"LowTrades": 150}  # used to read FailedDSR: 150


def test_each_early_gate_keeps_its_own_label(tmp_path: Path) -> None:
    def backtest(dna):
        r = _returns(dna)
        k = int(dna.dna_hash[:2], 16) % 3
        return TrialResult(dna=dna, sharpe=float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(6048)),
                           n_trades=5 if k == 0 else 60, total_return=-0.5 if k == 2 else 0.1,
                           max_dd=0.9 if k == 1 else 0.05, per_period_returns=r)

    result = _campaign(tmp_path, backtest)
    cats = dict(result.funnel.by_category)
    assert set(cats) == {"LowTrades", "HighDD", "NegReturn"} and sum(cats.values()) == 150
    assert "FailedDSR" not in cats and result.funnel.survivors == 0
