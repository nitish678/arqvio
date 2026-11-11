"""The plan state machine: unresolved is never free, offline keeps a resolved tier, licensing off means NO_LIMITS.

WHY: these are the rules that keep a licence check from punishing a paying user for a network problem.
"""
from __future__ import annotations

import pytest

from engine.licence.limits import FREE_LIMITS, NO_LIMITS, PAID_LIMITS
from engine.licence.state import (
    PlanState,
    PlanStatus,
    Tier,
    effective,
    initial,
    on_offline,
    on_server_verdict,
    on_token,
)

PAID = {"plan": "pro", "is_free": False, "limits": PAID_LIMITS.to_dict(), "licence_ended": False}
FREE = {"plan": "free", "is_free": True, "limits": FREE_LIMITS.to_dict(), "licence_ended": False}


@pytest.mark.parametrize("status", [PlanStatus.PENDING, PlanStatus.UNRESOLVED])
@pytest.mark.parametrize("payload", [None, FREE, PAID])
def test_unresolved_is_never_free(status: PlanStatus, payload: dict | None) -> None:
    result = effective(PlanState(status=status), payload, licensing_enabled=True)
    assert result.tier is Tier.CHECKING
    assert result.limits is None and not result.launch_allowed


def test_pending_goes_unresolved_offline_and_stays_checking() -> None:
    state = on_offline(initial())
    assert state.status is PlanStatus.UNRESOLVED and not state.licence_ended
    assert effective(state, None, licensing_enabled=True).tier is Tier.CHECKING


def test_offline_keeps_resolved_tier_while_token_valid() -> None:
    state = on_offline(on_offline(on_token(initial(), PAID)))
    assert state.status is PlanStatus.RESOLVED
    result = effective(state, PAID, licensing_enabled=True)
    assert result.tier is Tier.PAID and result.limits == PAID_LIMITS and result.allow_new_trades


def test_expired_token_offline_is_checking_not_free() -> None:
    state = on_offline(on_token(initial(), PAID))
    assert effective(state, None, licensing_enabled=True).tier is Tier.CHECKING


def test_free_plan_resolves_to_free_limits() -> None:
    result = effective(on_token(initial(), FREE), FREE, licensing_enabled=True)
    assert result.tier is Tier.FREE and result.limits == FREE_LIMITS


def test_ended_only_on_a_positive_verdict() -> None:
    assert not on_offline(on_token(initial(), PAID)).licence_ended
    ended = on_server_verdict(on_token(initial(), PAID), ended=True)
    result = effective(ended, PAID, licensing_enabled=True)
    assert result.tier is Tier.ENDED
    assert result.allow_new_trades is False and result.close_open_trades is False


def test_signed_ended_token_ends_the_plan() -> None:
    ended_payload = {**PAID, "licence_ended": True}
    state = on_token(initial(), ended_payload)
    assert state.licence_ended
    result = effective(state, ended_payload, licensing_enabled=True)
    assert result.tier is Tier.ENDED and not result.allow_new_trades and not result.close_open_trades


@pytest.mark.parametrize(
    "state", [initial(), PlanState(PlanStatus.UNRESOLVED), PlanState(PlanStatus.RESOLVED, licence_ended=True)],
)
@pytest.mark.parametrize("payload", [None, FREE, PAID])
def test_licensing_disabled_is_no_limits(state: PlanState, payload: dict | None) -> None:
    result = effective(state, payload, licensing_enabled=False)
    assert result.tier is Tier.UNLOCKED and result.limits == NO_LIMITS
    assert result.launch_allowed and result.allow_new_trades and not result.close_open_trades
