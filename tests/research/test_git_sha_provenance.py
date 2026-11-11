"""F7: trial rows carry the real commit, not "unknown".

run_campaign_cli never passed git_sha, so every row it wrote was stamped
"unknown". A provenance column that is present and always the same value
is worse than an absent one: it looks filled in, so nobody checks it, and
a result cannot be tied back to the code that produced it.
"""
from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from engine.core.git_sha import ENV_VAR, UNKNOWN, resolve_git_sha
from engine.research.campaign.budget import CampaignScope
from engine.research.provenance import DataSource, Provenance
from engine.research.search.dna import DNA, FailureTaxonomy
from engine.research.search.runner import TrialResult, run_campaign
from engine.validation.trial_registry import TrialRegistry

REPO_ROOT = Path(__file__).resolve().parents[2]


def _real_head_sha() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sha = proc.stdout.strip()
    return sha if proc.returncode == 0 and sha else None


def test_resolve_git_sha_matches_git() -> None:
    head = _real_head_sha()
    if head is None:
        pytest.skip("no git available")
    assert resolve_git_sha() == head
    assert resolve_git_sha() != UNKNOWN


def test_resolve_git_sha_falls_back_to_env(monkeypatch, tmp_path) -> None:
    """Outside a repo, the env override is used before giving up."""
    monkeypatch.setenv(ENV_VAR, "deadbee")
    assert resolve_git_sha(repo_root=tmp_path) == "deadbee"


def test_resolve_git_sha_never_raises_without_git(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert resolve_git_sha(repo_root=tmp_path) == UNKNOWN


def _series(n_obs: int = 300, population: int = 5) -> list[np.ndarray]:
    rng = np.random.default_rng(11)
    base = rng.normal(0.0, 0.01, size=n_obs)
    return [base[rng.permutation(n_obs)] + k * 1e-4 for k in range(population)]


def test_registry_rows_carry_the_real_sha(tmp_path) -> None:
    """A 5-trial campaign stamps the actual SHA on every row."""
    head = _real_head_sha()
    if head is None:
        pytest.skip("no git available")

    series = _series()
    state = {"i": 0}

    def _bt(dna: DNA) -> TrialResult:
        arr = series[state["i"] % len(series)]
        state["i"] += 1
        sd = float(np.std(arr, ddof=1))
        return TrialResult(
            dna=dna,
            outcome=FailureTaxonomy.EVALUATED.value,
            sharpe=float(np.mean(arr) / sd * np.sqrt(252.0)),
            n_trades=60,
            total_return=float(np.sum(arr)),
            max_dd=0.05,
            per_period_returns=arr,
        )

    db = tmp_path / "trials.sqlite"
    registry = TrialRegistry(db)
    run_campaign(
        campaign_id="sha-smoke",
        scope=CampaignScope(
            symbols=("USDCADm",), timeframes=("D1",), population_per_cell=5
        ),
        registry=registry,
        backtest_fn=_bt,
        control_fn=None,
        real_scores_factory=lambda trials: [],
        provenance=Provenance(
            source=DataSource.MT5_LIVE,
            dataset_hash="h",
            as_of=datetime.now(tz=UTC),
            mixed_source=False,
        ),
        dataset_hash="h",
        git_sha=resolve_git_sha(),
        seed=0,
        n_obs=300,
        bars_loaded=301,
        audit_bars_expected=301,
    )

    import sqlite3

    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        shas = {
            row[0]
            for row in conn.execute("SELECT git_sha FROM campaigns")
        }
    finally:
        conn.close()

    assert shas == {head}, f"campaign rows carry {shas}, expected {{{head}}}"
    assert UNKNOWN not in shas


def test_cli_passes_git_sha_to_run_campaign() -> None:
    """The call site must actually pass it.

    Asserted on the source because running the CLI needs live MT5; a
    campaign that silently stops passing git_sha would otherwise only be
    caught by inspecting a registry after the fact.
    """
    src = (
        REPO_ROOT / "engine" / "research" / "search" / "run_campaign_cli.py"
    ).read_text(encoding="utf-8")
    assert "git_sha=resolve_git_sha()" in src
    assert "from engine.core.git_sha import resolve_git_sha" in src


def test_resolve_git_sha_falls_back_to_the_build_stamp(monkeypatch, tmp_path) -> None:
    """The installed app has no git: its campaigns are stamped with the build's commit, not refused as 'unknown'."""
    from engine.core import build_stamp

    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.setattr(build_stamp, "BUILD_COMMIT", "fc0ab890248362c119c353edc82d4452795f9ed9")
    assert resolve_git_sha(repo_root=tmp_path) == "fc0ab89"
