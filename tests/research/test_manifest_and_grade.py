"""B3 manifest completeness and C4 the WEAK cap."""
from __future__ import annotations

import pytest

from engine.research.campaign.manifest import (
    GRADE_UNGRADED,
    GRADE_WEAK,
    Manifest,
    config_hash,
    grade_for,
)


def _manifest(**overrides) -> Manifest:
    data = {
        "campaign_id": "rerun-usdcadm-h1",
        "symbol": "USDCADm",
        "timeframe": "H1",
        "dataset_hash": "real-mt5-USDCADm-H1-60345",
        "dataset_source": "MT5_LIVE",
        # Required since S4 stage 1: live vs pinned cannot default silently.
        "dataset_provenance": "live",
        "bars_loaded": 60345,
        "date_range_utc": ("2014-01-14T00:00:00+00:00", "2026-09-23T04:00:00+00:00"),
        "git_sha": "4dab669",
        "config_hash": "abcdef0123456789",
        "gene_pool": "none (uniform over every family; pre-W2 draw)",
        "seed": 0,
        "completed_at_utc": "2026-09-23T13:03:04+00:00",
    }
    data.update(overrides)
    return Manifest(**data)


# ---------------------------------------------------------------------------
# config_hash
# ---------------------------------------------------------------------------

def test_config_hash_is_stable_and_order_independent() -> None:
    a = config_hash({"symbol": "USDCADm", "population": 500, "seed": 0})
    b = config_hash({"seed": 0, "population": 500, "symbol": "USDCADm"})
    assert a == b, "key order changed the hash"
    assert a == config_hash({"symbol": "USDCADm", "population": 500, "seed": 0})
    assert len(a) == 16


def test_config_hash_separates_different_configurations() -> None:
    base = {"symbol": "USDCADm", "population": 500, "seed": 0}
    assert config_hash(base) != config_hash({**base, "population": 5000})
    assert config_hash(base) != config_hash({**base, "seed": 1})


# ---------------------------------------------------------------------------
# Manifest completeness
# ---------------------------------------------------------------------------

def test_a_complete_manifest_reports_nothing_missing() -> None:
    m = _manifest()
    assert m.missing_fields() == ()
    assert "complete" in m.render()
    assert "MANIFEST" in m.render()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("git_sha", "unknown"),
        ("config_hash", "synthetic"),
        ("dataset_hash", ""),
        ("bars_loaded", 0),
    ],
)
def test_placeholders_count_as_missing_not_present(field, value) -> None:
    """A column filled with "unknown" is not provenance.

    git_sha="unknown" and config_hash="synthetic" were both real
    defaults that made the field present and useless.
    """
    m = _manifest(**{field: value})
    assert field in m.missing_fields()
    assert "NO - missing" in m.render()


def test_manifest_round_trips_to_a_dict() -> None:
    d = _manifest().to_dict()
    assert d["git_sha"] == "4dab669"
    assert d["date_range_utc"][0].startswith("2014-01-14")
    assert d["seed"] == 0


# ---------------------------------------------------------------------------
# C4: the WEAK cap, and only that
# ---------------------------------------------------------------------------

def test_winner_within_random_range_is_capped_at_weak() -> None:
    """p_max at or above alpha means the winner was not shown to be
    anything, whatever its Sharpe says."""
    assert grade_for(vs_random_p_max=0.20) == GRADE_WEAK
    assert grade_for(vs_random_p_max=0.05) == GRADE_WEAK
    # The cap overrides a grade someone else assigned.
    assert grade_for(vs_random_p_max=0.20, current_grade="STRONG") == GRADE_WEAK


def test_an_untestable_winner_is_also_capped() -> None:
    """"We could not test it" is not a reason to grade it higher."""
    assert grade_for(vs_random_p_max=None) == GRADE_WEAK
    assert grade_for(vs_random_p_max=None, current_grade="STRONG") == GRADE_WEAK


def test_a_winner_outside_the_random_range_is_not_capped() -> None:
    assert grade_for(vs_random_p_max=0.0005) != GRADE_WEAK
    assert grade_for(vs_random_p_max=0.0005) == GRADE_UNGRADED
    assert grade_for(vs_random_p_max=0.0005, current_grade="STRONG") == "STRONG"


def test_no_other_grading_rule_is_implemented() -> None:
    """Only the sanctioned cap exists.

    A fuller ladder would put an authoritative-looking letter on
    evidence that does not support it. Clearing the cap yields
    UNGRADED, not a flattering default.
    """
    assert grade_for(vs_random_p_max=1e-12) == GRADE_UNGRADED
