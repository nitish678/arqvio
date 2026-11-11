"""The plan state machine: PENDING -> UNRESOLVED / RESOLVED, and what each state lets the user do.

Rules, each one learned from how licence checks go wrong for paying users:

- Unresolved is not a tier. Until a verified token says otherwise the effective tier is CHECKING, never FREE. Showing
  a paying user the free caps because the server was slow to answer reads as "you lost your plan".
- ``licence_ended`` becomes true only on a positive server verdict (``on_server_verdict(ended=True)`` or a signed
  token that says so). A timeout, a refused connection or an expired token is not a verdict.
- Offline never demotes a resolved tier while its token is valid. The caller passes the payload of a token it has
  just verified (token.verify with the current time) or None; with a valid payload a RESOLVED state keeps its tier.
  Once the token is no longer valid the tier is unconfirmed, which is CHECKING again, still not FREE.
- An ended subscription stops NEW trades and lets open trades run to their own exit: closing them would turn a
  billing event into a trading loss. Hence ``allow_new_trades=False, close_open_trades=False``.
- Licensing disabled (the app today, by the owner's decision) always gives ``NO_LIMITS``.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

from engine.licence.limits import FREE_LIMITS, NO_LIMITS, PAID_LIMITS, Limits

__all__ = [
    "Effective",
    "PlanState",
    "PlanStatus",
    "Tier",
    "effective",
    "initial",
    "on_offline",
    "on_server_verdict",
    "on_token",
]


class PlanStatus(str, Enum):
    """Where the plan lookup stands."""

    PENDING = "PENDING"  # not asked yet this session
    UNRESOLVED = "UNRESOLVED"  # asked, no verified answer (offline, server error, no token)
    RESOLVED = "RESOLVED"  # a verified token or a server verdict arrived


class Tier(str, Enum):
    """What the user effectively has right now."""

    UNLOCKED = "UNLOCKED"  # licensing disabled: today's free, unlocked app
    CHECKING = "CHECKING"  # plan not confirmed; never treated as FREE
    FREE = "FREE"
    PAID = "PAID"
    ENDED = "ENDED"  # positive server verdict that the subscription ended


@dataclass(frozen=True, slots=True)
class PlanState:
    """The machine's memory: its status and whether the server has said the licence ended."""

    status: PlanStatus = PlanStatus.PENDING
    licence_ended: bool = False


@dataclass(frozen=True, slots=True)
class Effective:
    """The decision the engine and UI read. ``limits`` is None while the plan is being checked."""

    tier: Tier
    limits: Limits | None
    launch_allowed: bool
    allow_new_trades: bool
    close_open_trades: bool = False


def initial() -> PlanState:
    """A fresh session: nothing asked yet."""
    return PlanState()


def on_token(state: PlanState, payload: dict[str, Any]) -> PlanState:
    """A token from the server was verified: the plan is resolved. Its ``licence_ended`` is a signed verdict."""
    return replace(state, status=PlanStatus.RESOLVED, licence_ended=bool(payload.get("licence_ended", False)))


def on_server_verdict(state: PlanState, *, ended: bool) -> PlanState:
    """The server answered about the plan itself (e.g. 403 PLAN ENDED is ``ended=True``)."""
    return replace(state, status=PlanStatus.RESOLVED, licence_ended=ended)


def on_offline(state: PlanState) -> PlanState:
    """The server could not be reached. A resolved plan stays resolved; nothing is ever ended by silence."""
    if state.status is PlanStatus.RESOLVED:
        return state
    return replace(state, status=PlanStatus.UNRESOLVED)


_CHECKING = Effective(Tier.CHECKING, None, launch_allowed=False, allow_new_trades=False)
_UNLOCKED = Effective(Tier.UNLOCKED, NO_LIMITS, launch_allowed=True, allow_new_trades=True)
_ENDED = Effective(Tier.ENDED, FREE_LIMITS, launch_allowed=True, allow_new_trades=False, close_open_trades=False)


def _from_payload(payload: dict[str, Any]) -> Effective:
    if payload.get("is_free", True):
        return Effective(Tier.FREE, FREE_LIMITS, launch_allowed=True, allow_new_trades=True)
    limits = payload.get("limits")
    paid = Limits.from_dict(limits) if isinstance(limits, dict) else PAID_LIMITS
    return Effective(Tier.PAID, paid, launch_allowed=True, allow_new_trades=True)


def effective(state: PlanState, token_payload: dict[str, Any] | None, *, licensing_enabled: bool) -> Effective:
    """What the user has now, given the machine state and a just-verified token payload (or None)."""
    if not licensing_enabled:
        return _UNLOCKED
    if state.status is not PlanStatus.RESOLVED:
        return _CHECKING
    if state.licence_ended or (token_payload is not None and token_payload.get("licence_ended")):
        return _ENDED
    if token_payload is None:
        return _CHECKING
    return _from_payload(token_payload)
