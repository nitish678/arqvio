"""Signed plan tokens: round trip, every refusal reason, and the one-hour clock skew.

WHY: each TokenError reason drives a different message to the user, so each must be reachable and exact.
"""
from __future__ import annotations

import base64
import json

import pytest

from engine.licence.limits import FREE_LIMITS
from engine.licence.token import (
    REASON_BAD_SIGNATURE,
    REASON_EXPIRED,
    REASON_HWID_MISMATCH,
    REASON_MALFORMED,
    REASON_NOT_YET_VALID,
    TokenError,
    hash_hwid,
    issue,
    verify,
)

SECRET = b"unit-test-token-secret"
HWID = hash_hwid("test-machine")
T0 = 1_800_000_000
TTL = 86_400


def _payload() -> dict:
    return {
        "key_serial": "0000000001", "hwid_hash": HWID, "plan": "free", "is_free": True, "seats": 1,
        "limits": FREE_LIMITS.to_dict(), "licence_ended": False,
    }


def _token() -> str:
    return issue(_payload(), SECRET, TTL, now=T0)


def _reason(token: str, *, now: int = T0, hwid: str = HWID, secret: bytes = SECRET) -> str:
    with pytest.raises(TokenError) as info:
        verify(token, secret, hwid_hash=hwid, now=now)
    return info.value.reason


def test_round_trip() -> None:
    body = verify(_token(), SECRET, hwid_hash=HWID, now=T0 + 10)
    assert body["iat"] == T0 and body["exp"] == T0 + TTL
    assert body["limits"] == FREE_LIMITS.to_dict()
    assert body["is_free"] is True and body["licence_ended"] is False


def test_hwid_is_hashed_not_raw() -> None:
    assert HWID != "test-machine" and len(HWID) == 64


def test_bad_signature() -> None:
    segment, _sig = _token().split(".")
    edited = json.loads(base64.urlsafe_b64decode(segment + "=" * (-len(segment) % 4)))
    edited["is_free"] = False
    new_segment = base64.urlsafe_b64encode(json.dumps(edited).encode()).rstrip(b"=").decode()
    assert _reason(f"{new_segment}.{_sig}") == REASON_BAD_SIGNATURE
    assert _reason(_token(), secret=b"wrong-secret") == REASON_BAD_SIGNATURE


def test_expired() -> None:
    assert _reason(_token(), now=T0 + TTL + 3600 + 1) == REASON_EXPIRED


def test_not_yet_valid() -> None:
    assert _reason(_token(), now=T0 - 3600 - 1) == REASON_NOT_YET_VALID


def test_hwid_mismatch() -> None:
    assert _reason(_token(), hwid=hash_hwid("another-machine")) == REASON_HWID_MISMATCH


@pytest.mark.parametrize("token", ["", "abc", "a.b.c", ".sig", "payload.", "!!!.???", "e30.a=b"])
def test_malformed(token: str) -> None:
    assert _reason(token) == REASON_MALFORMED


def _signed(body_bytes: bytes) -> str:
    import hashlib
    import hmac

    segment = base64.urlsafe_b64encode(body_bytes).rstrip(b"=").decode()
    sig = hmac.new(SECRET, segment.encode(), hashlib.sha256).digest()
    return f"{segment}.{base64.urlsafe_b64encode(sig).rstrip(b'=').decode()}"


@pytest.mark.parametrize(
    "body",
    [b"not json", b"[1, 2]", b'{"plan": "free"}', json.dumps({**_payload(), "iat": "x", "exp": 1}).encode()],
)
def test_malformed_but_correctly_signed(body: bytes) -> None:
    """A signed body that is not a valid payload is still malformed, not trusted."""
    assert _reason(_signed(body)) == REASON_MALFORMED


def test_skew_one_hour_accepted_both_ways() -> None:
    token = _token()
    assert verify(token, SECRET, hwid_hash=HWID, now=T0 + TTL + 3600)["plan"] == "free"
    assert verify(token, SECRET, hwid_hash=HWID, now=T0 - 3600)["plan"] == "free"


def test_skew_one_hour_and_one_second_rejected() -> None:
    token = _token()
    assert _reason(token, now=T0 + TTL + 3601) == REASON_EXPIRED
    assert _reason(token, now=T0 - 3601) == REASON_NOT_YET_VALID


def test_issue_refuses_bad_input() -> None:
    with pytest.raises(ValueError):
        issue(_payload(), b"", TTL)
    with pytest.raises(ValueError):
        issue(_payload(), SECRET, 0)
    with pytest.raises(ValueError):
        issue({"plan": "free"}, SECRET, TTL)
