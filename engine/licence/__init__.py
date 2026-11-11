"""ARQVIO licensing library: keys, signed plan tokens, the free caps and the plan state machine.

WHY a library with nothing wired to it: the owner decided the app stays FREE with no lock until a later release.
This package is built and tested now so that the later release only has to wire it in (docs/LICENCE.md). Nothing in
the app or engine imports it today; ``state.effective(..., licensing_enabled=False)`` returns ``NO_LIMITS``, which is
exactly today's behaviour.

The modules use the stdlib only so the separate licence server (``server/licence``) can share the one definition of
the key format, the token and the caps.
"""
from __future__ import annotations

from engine.licence.keys import KeyFormatError, LicenceKey, make_key, parse, verify_key
from engine.licence.limits import FREE_LIMITS, NO_LIMITS, PAID_LIMITS, Limits, check_launch
from engine.licence.state import Effective, PlanState, PlanStatus, Tier, effective
from engine.licence.token import TokenError, hash_hwid, issue, verify

__all__ = [
    "FREE_LIMITS",
    "NO_LIMITS",
    "PAID_LIMITS",
    "Effective",
    "KeyFormatError",
    "LicenceKey",
    "Limits",
    "PlanState",
    "PlanStatus",
    "Tier",
    "TokenError",
    "check_launch",
    "effective",
    "hash_hwid",
    "issue",
    "make_key",
    "parse",
    "verify",
    "verify_key",
]
