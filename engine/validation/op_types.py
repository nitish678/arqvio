"""Operation-type vocabulary for the trial registry.

Search trials inflate the multiple-testing budget (they are candidates the
research is *choosing between*). Validation runs are robustness re-evaluations
of an already-selected strategy on a different cell; they do NOT inflate the
budget and they do NOT feed DSR.

The legacy value ``UNKNOWN`` is intentional: pre-S5 rows whose origin we
cannot prove are NOT silently treated as search trials. They are excluded
from DSR stats; a migration step may later promote them to ``SEARCH_TRIAL``
or ``VALIDATION_RUN`` with explicit operator consent.
"""
from __future__ import annotations

from enum import Enum


class OpType(str, Enum):
    """How a trials row was produced."""

    SEARCH_TRIAL = "SEARCH_TRIAL"          # a candidate evaluated to decide what to keep
    VALIDATION_RUN = "VALIDATION_RUN"      # an already-selected strategy re-run on a new cell
    UNKNOWN = "UNKNOWN"                    # legacy row, NOT counted by DSR until migrated


# Values that count toward the DSR / MinBTL trial budget.
COUNTED_BY_DSR: frozenset[str] = frozenset({OpType.SEARCH_TRIAL.value})


def counts_toward_dsr(op_type: str) -> bool:
    """True iff rows of this op_type should be included in DSR / MinBTL stats."""
    return op_type in COUNTED_BY_DSR


def is_valid_op_type(op_type: str) -> bool:
    """True iff the value is one of the recognised op_type strings."""
    return op_type in {member.value for member in OpType}
