"""UUIDv4 helpers.

WHAT: Generate opaque string IDs (account_id, strategy_id, trial_id, request_id).
WHY: We need stable, unguessable, collision-resistant identifiers from day one.
HOW: uuid.uuid4() as canonical str. Single helper, no extras.
TRADE-OFF: We do NOT use broker login as ID (Rule: account_id is opaque, not derived).
TEST: tests/core/test_ids.py asserts format and uniqueness.
FAILURE: If uuid4 is broken, raise ConfigError; the system is unusable without IDs.
"""
from __future__ import annotations

import uuid


def new_id() -> str:
    """Return a fresh UUIDv4 as a lowercase string."""
    return str(uuid.uuid4())


def is_valid_id(value: str) -> bool:
    """Return True if value is a well-formed UUID4 string."""
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError, TypeError):
        return False
    return parsed.version == 4
