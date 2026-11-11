"""NO_TRIALS and REJECTED are different facts and must stay different.

"No trials" means the cell had nothing to correct for, so a multiple-testing
correction is undefined. "Rejected" means a correction was computed and the
result did not clear its bar. Collapsing the two loses the distinction in
any stored record, and the trial registry is a permanent one.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from engine.validation.cell_dsr import CellDSRStatus, compute_cell_dsr
from engine.validation.op_types import OpType
from engine.validation.trial_registry import TrialRegistry


def _returns(n: int = 500, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, 0.01, n)


def test_no_trials_is_its_own_status() -> None:
    assert CellDSRStatus.NO_TRIALS is not CellDSRStatus.REJECTED
    assert CellDSRStatus.NO_TRIALS.value == "NO_TRIALS"


def test_zero_trials_reports_no_trials_not_rejected() -> None:
    """The n_trials < 1 branch used to return REJECTED."""
    result = compute_cell_dsr(
        _returns(),
        n_trials=0,
        trial_sharpes=[],
        symbol="USDCADm",
        timeframe="D1",
    )
    assert result.status is CellDSRStatus.NO_TRIALS
    assert result.dsr is None
    assert "NO_TRIALS" in result.display()
    assert "REJECTED" not in result.display()


def test_too_few_observations_still_reports_unreliable() -> None:
    """NO_TRIALS must not swallow the sample-size gate."""
    result = compute_cell_dsr(
        _returns(10),
        n_trials=100,
        trial_sharpes=[0.1, 0.2, 0.3],
        symbol="USDCADm",
        timeframe="D1",
        min_obs_for_dsr=100,
    )
    assert result.status is CellDSRStatus.UNRELIABLE
    assert "UNRELIABLE" in result.display()


def test_a_real_cell_still_reports_ready() -> None:
    result = compute_cell_dsr(
        _returns(3915),
        n_trials=5000,
        trial_sharpes=list(np.random.default_rng(1).normal(0.5, 0.4, 5000)),
        symbol="USDCADm",
        timeframe="D1",
    )
    assert result.status is CellDSRStatus.READY
    assert result.dsr is not None


def test_no_trials_round_trips_through_the_registry(tmp_path: Path) -> None:
    """The stored dsr_status must carry the fact, not a near-enough one."""
    db = tmp_path / "trials.sqlite"
    reg = TrialRegistry(db)
    reg.open_campaign(
        campaign_id="c1", symbol="USDCADm", timeframe="D1",
        dataset_hash="h", git_sha="g", config_hash="cfg", seed=0,
    )
    for dna_hash in ("a", "b", "c"):
        reg.record(
            "c1", dna_hash=dna_hash, outcome="EVALUATED", sharpe=1.0,
            n_trades=50, symbol="USDCADm", timeframe="D1",
            op_type=OpType.SEARCH_TRIAL.value,
        )
    reg.update_dsr_many(
        "c1",
        [
            ("a", 0.72, CellDSRStatus.READY.value, 0.11, 3915),
            ("b", None, CellDSRStatus.UNRELIABLE.value, None, 40),
            ("c", None, CellDSRStatus.NO_TRIALS.value, None, 3915),
        ],
    )

    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = dict(
            conn.execute(
                "SELECT dna_hash, dsr_status FROM trials WHERE campaign_id='c1'"
            ).fetchall()
        )
        n_obs_c = conn.execute(
            "SELECT n_obs FROM trials WHERE dna_hash='c'"
        ).fetchone()[0]
    finally:
        conn.close()

    assert rows == {"a": "READY", "b": "UNRELIABLE", "c": "NO_TRIALS"}
    # n_obs is what separates UNRELIABLE from NO_TRIALS: candidate c has a
    # full series and still no correction to apply.
    assert n_obs_c == 3915


def test_a_candidate_without_a_usable_dsr_cannot_pass_the_gate() -> None:
    """Any non-READY status marks the trial unreliable in the runner.

    Before, only UNRELIABLE did. A NO_TRIALS candidate would have carried
    post_dsr=None with unreliable_dsr=False, skipping both DSR gates in
    _apply_funnel and reaching VAULTED with no number ever tested.
    """
    from engine.research.search.dna import FailureTaxonomy
    from engine.research.search.runner import TrialResult, _apply_funnel

    unreliable = TrialResult(
        dna=None,  # type: ignore[arg-type]
        outcome=FailureTaxonomy.EVALUATED.value,
        sharpe=1.0, n_trades=60, total_return=0.2, max_dd=0.05,
        post_dsr=None, unreliable_dsr=True,
    )
    assert _apply_funnel(
        unreliable, min_trades=30, max_dd_limit=0.30,
        return_threshold=-0.10, d_sharpe_threshold=0.5,
    ) == FailureTaxonomy.DsrUnreliable.value


def test_empty_series_is_unreliable_not_no_trials() -> None:
    """An empty return series is a sample-size problem, not a trial-count one.

    It reaches the min_obs gate first and reports UNRELIABLE. NO_TRIALS is
    reserved for the case where the CELL has no trials to correct for.
    """
    result = compute_cell_dsr(
        np.zeros(0),
        n_trials=10,
        trial_sharpes=[0.1],
        symbol="USDCADm",
        timeframe="D1",
    )
    assert result.status is CellDSRStatus.UNRELIABLE
    assert result.n_obs == 0
