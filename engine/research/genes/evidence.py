"""Strict, additive run-evidence contract; never changes legacy DNA or registry hashes.

A valid record is a claim with provenance, not independent proof it executed.
ponytail: records are not auto-promoted; add a trusted runner/artifact verifier
before using them to enable deployment or display VERIFIED capabilities.
"""
from datetime import datetime, timedelta
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator

Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Identity = Annotated[str, StringConstraints(min_length=1, max_length=256)]


class EvaluationEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal[1] = 1
    experiment_id: Identity
    run_id: Identity
    candidate_hash: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{16}$")]
    dataset_sha256: Digest
    source_sha256: Digest
    config_sha256: Digest
    artifact_sha256: Digest
    cost_model_version: Identity
    validation_version: Identity
    target: Literal["oracle", "kernel", "live_ea", "tester_ea"]
    execution_path: Literal["slow", "kernel", "slow_fallback", "live_ea", "tester_ea", "not_executed"]
    stage: Literal["discovery", "validation", "parity"]
    status: Literal["PASS", "FAIL", "SKIP", "ERROR"]
    covered_genes: tuple[Identity, ...] = Field(min_length=1)
    completed_utc: datetime
    reason: str = ""

    @field_validator("completed_utc")
    @classmethod
    def utc_only(cls, value: datetime) -> datetime:
        if value.utcoffset() != timedelta(0):
            raise ValueError("completed_utc must be timezone-aware UTC")
        return value

    @field_validator("covered_genes")
    @classmethod
    def known_unique_genes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        from engine.research.genes.registry import BY_NAME

        if len(set(value)) != len(value) or any(name not in BY_NAME for name in value):
            raise ValueError("covered_genes must be unique registry gene names")
        return value

    @model_validator(mode="after")
    def truthful_outcome(self):
        if self.status != "PASS" and not self.reason.strip():
            raise ValueError("non-passing evidence needs an explicit reason")
        if self.status == "PASS" and self.execution_path == "not_executed":
            raise ValueError("unexecuted evaluation cannot pass")
        if self.status == "PASS" and self.target == "kernel" and self.execution_path != "kernel":
            raise ValueError("slow fallback cannot prove kernel execution")
        if self.target == "oracle" and self.execution_path not in ("slow", "slow_fallback", "not_executed"):
            raise ValueError("oracle evidence requires a slow execution path")
        if self.target in ("live_ea", "tester_ea") and self.execution_path not in (self.target, "not_executed"):
            raise ValueError("EA evidence requires its own runtime path, not a Python backtest")
        return self
