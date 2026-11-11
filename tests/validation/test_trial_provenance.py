from pathlib import Path

import pytest

from engine.validation.trial_registry import TrialRegistry


def registry(tmp_path: Path) -> TrialRegistry:
    r = TrialRegistry(tmp_path / "trials.sqlite")
    r.open_campaign("c1", "XAUUSD", "H1", "d" * 16, "g" * 40, "c" * 64, 7)
    return r


def test_new_trial_requires_campaign_provenance(tmp_path):
    r = registry(tmp_path)
    with pytest.raises(ValueError, match="required"):
        r.record("c1", "a" * 16, "EVALUATED", sharpe=1.0)
    r.record("c1", "a" * 16, "EVALUATED", sharpe=1.0, dataset_hash="d" * 16,
             source="BUNDLED", as_of="2026-11-11T00:00:00+00:00")
    assert r.stats("c1").n_trials == 1


def test_trial_dataset_cannot_change_campaign_provenance(tmp_path):
    r = registry(tmp_path)
    with pytest.raises(ValueError, match="must match"):
        r.record("c1", "a" * 16, "EVALUATED", dataset_hash="e" * 16,
                 source="BUNDLED", as_of="2026-11-11T00:00:00+00:00")
