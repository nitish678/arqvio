"""C0 gate-artifact alignment fixture.

Pins the contract between the audit CSV's TIME_BASIS and the spread
matrix's time_basis. The gate (refuse_on_basis_mismatch) must pass when
the two agree and refuse when they diverge — this is the invariant that
keeps spread-cost hours and engine-bar hours in the same reference.
"""
from __future__ import annotations

import pytest

from dataclasses import replace

from engine.data.spread_calibration import SpreadMatrix
from engine.research.campaign.preflight import (
    PreflightError,
    refuse_on_basis_mismatch,
)

_FLAT_MATRIX = SpreadMatrix(
    symbol="EURUSDm",
    buckets=tuple(0.8 for _ in range(24)),
    mean_buckets=tuple(1.0 for _ in range(24)),
    sample_count=1000,
    source="BUNDLED",
    calibrator_version=2,
    time_basis="utc",
)


def _matrix(basis: str) -> SpreadMatrix:
    return replace(_FLAT_MATRIX, time_basis=basis)


@pytest.mark.parametrize(
    ("matrix_basis", "audit_basis"),
    [
        ("utc", "utc"),
        ("utc", ""),  # blank audit defaults to utc
        ("server_equals_utc", "utc"),
        ("utc", "server_equals_utc"),
    ],
)
def test_basis_passes_on_equivalent_classes(matrix_basis: str, audit_basis: str) -> None:
    """utc and server_equals_utc are in the same equivalence class."""
    refuse_on_basis_mismatch(
        _matrix(matrix_basis), audit_basis, "EURUSDm:D1"
    )


@pytest.mark.parametrize(
    ("matrix_basis", "audit_basis"),
    [
        ("broker_offset", "utc"),
        ("utc", "broker_offset"),
        ("broker_offset", "server_equals_utc"),
        ("unresolved", "utc"),
    ],
)
def test_basis_refused_on_mismatch(matrix_basis: str, audit_basis: str) -> None:
    """Mismatched basis classes must raise PreflightError."""
    with pytest.raises(PreflightError, match="BASIS_MISMATCH"):
        refuse_on_basis_mismatch(
            _matrix(matrix_basis), audit_basis, "EURUSDm:D1"
        )
