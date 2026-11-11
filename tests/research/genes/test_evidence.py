from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from engine.research.genes.evidence import EvaluationEvidence


def record(**changes):
    return {"experiment_id": "experiment", "run_id": "run", "candidate_hash": "a" * 16,
            "dataset_sha256": "b" * 64, "source_sha256": "c" * 64, "config_sha256": "d" * 64,
            "artifact_sha256": "e" * 64, "cost_model_version": "cost-v1", "validation_version": "checks-v1",
            "target": "kernel", "execution_path": "kernel", "stage": "parity", "status": "PASS",
            "covered_genes": ("signal.macd",), "completed_utc": datetime(2026, 11, 11, tzinfo=UTC), **changes}


def test_evidence_roundtrips_without_altering_identity():
    evidence = EvaluationEvidence(**record())
    assert EvaluationEvidence.model_validate_json(evidence.model_dump_json()) == evidence
    assert evidence.run_id == "run" and evidence.covered_genes == ("signal.macd",)


@pytest.mark.parametrize("changes", [
    {"source_sha256": "unknown"}, {"dataset_sha256": "b" * 16}, {"artifact_sha256": ""},
    {"status": "SKIP"}, {"status": "PASS", "execution_path": "not_executed"},
    {"execution_path": "slow_fallback"}, {"covered_genes": ("fake.gene",)},
    {"covered_genes": ("signal.macd", "signal.macd")},
    {"completed_utc": datetime(2026, 11, 11)}, {"surprise": True},
])
def test_evidence_rejects_missing_or_misleading_provenance(changes):
    with pytest.raises(ValidationError):
        EvaluationEvidence(**record(**changes))


def test_skip_has_reason_and_is_not_pass():
    result = EvaluationEvidence(**record(status="SKIP", execution_path="not_executed", reason="missing pinned dataset"))
    assert result.status == "SKIP"
