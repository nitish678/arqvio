"""Signed plan tokens: ``base64url(JSON payload) + "." + base64url(HMAC-SHA256(secret, payload segment))``.

WHY a token at all: the app must know its plan while offline. The server signs what it decided (plan, seats, caps,
whether the licence ended) and the app keeps that verdict until it expires, so a flaky connection never demotes a
paying user mid-session (state.py).

WHY ``verify`` names an exact reason: support needs to tell "your clock is wrong" (not yet valid / expired) from
"this token was copied from another PC" (hwid mismatch) from "someone edited it" (bad signature). ``TokenError.reason``
is one of the ``REASON_*`` constants.

Clock skew: home PCs drift, so ``iat`` and ``exp`` are each given ``skew_s`` (one hour by default) of grace. Every
comparison of secret-derived bytes goes through ``hmac.compare_digest``.

Signing is symmetric (HMAC, stdlib only). An app that verifies tokens itself would have to hold this secret, so the
token secret must be separate from the key-minting secret (server/licence/app.py derives the two apart).
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import re
import time
from typing import Any

__all__ = [
    "PAYLOAD_FIELDS",
    "REASON_BAD_SIGNATURE",
    "REASON_EXPIRED",
    "REASON_HWID_MISMATCH",
    "REASON_MALFORMED",
    "REASON_NOT_YET_VALID",
    "TokenError",
    "hash_hwid",
    "issue",
    "verify",
]

REASON_BAD_SIGNATURE = "bad signature"
REASON_EXPIRED = "expired"
REASON_NOT_YET_VALID = "not yet valid"
REASON_HWID_MISMATCH = "hwid mismatch"
REASON_MALFORMED = "malformed"

DEFAULT_SKEW_S = 3600
PAYLOAD_FIELDS = (
    "key_serial", "hwid_hash", "plan", "is_free", "seats", "limits", "iat", "exp", "licence_ended",
)
_B64URL = re.compile(r"^[A-Za-z0-9_-]+$")


class TokenError(ValueError):
    """A token was refused; ``reason`` says exactly why."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


def hash_hwid(hwid: str) -> str:
    """The only form a hardware id is ever stored or signed in: its SHA-256 hex digest."""
    return hashlib.sha256(hwid.encode("utf-8")).hexdigest()


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64d(text: str) -> bytes:
    if not _B64URL.match(text):
        raise TokenError(REASON_MALFORMED, "not base64url")
    try:
        return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
    except (binascii.Error, ValueError) as exc:
        raise TokenError(REASON_MALFORMED, "not base64url") from exc


def _sign(segment: str, secret: bytes) -> bytes:
    return hmac.new(secret, segment.encode("ascii"), hashlib.sha256).digest()


def issue(payload: dict[str, Any], secret: bytes, ttl_s: int, *, now: int | None = None) -> str:
    """Sign ``payload`` (``iat``/``exp`` are set here from ``now`` and ``ttl_s``) and return the token text."""
    if not secret:
        raise ValueError("secret must not be empty")
    if ttl_s <= 0:
        raise ValueError("ttl_s must be positive")
    issued_at = int(time.time()) if now is None else int(now)
    body = {**payload, "iat": issued_at, "exp": issued_at + int(ttl_s)}
    missing = [name for name in PAYLOAD_FIELDS if name not in body]
    if missing:
        raise ValueError(f"payload is missing {missing}")
    segment = _b64e(json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return f"{segment}.{_b64e(_sign(segment, secret))}"


def _decode_payload(segment: str) -> dict[str, Any]:
    try:
        body = json.loads(_b64d(segment).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TokenError(REASON_MALFORMED, "payload is not JSON") from exc
    if not isinstance(body, dict):
        raise TokenError(REASON_MALFORMED, "payload is not an object")
    missing = [name for name in PAYLOAD_FIELDS if name not in body]
    if missing:
        raise TokenError(REASON_MALFORMED, f"payload is missing {missing}")
    for name in ("iat", "exp"):
        if not isinstance(body[name], int) or isinstance(body[name], bool):
            raise TokenError(REASON_MALFORMED, f"{name} is not an integer")
    if not isinstance(body["hwid_hash"], str) or not isinstance(body["limits"], dict):
        raise TokenError(REASON_MALFORMED, "hwid_hash or limits has the wrong type")
    return body


def verify(token: str, secret: bytes, *, hwid_hash: str, now: int, skew_s: int = DEFAULT_SKEW_S) -> dict[str, Any]:
    """Return the payload of a genuine, current token for this machine, or raise ``TokenError``."""
    if not isinstance(token, str) or token.count(".") != 1:
        raise TokenError(REASON_MALFORMED, "expected <payload>.<signature>")
    segment, signature_text = token.split(".")
    if not segment or not signature_text:
        raise TokenError(REASON_MALFORMED, "empty segment")
    signature = _b64d(signature_text)
    if not hmac.compare_digest(signature, _sign(segment, secret)):
        raise TokenError(REASON_BAD_SIGNATURE)
    body = _decode_payload(segment)
    if now < body["iat"] - skew_s:
        raise TokenError(REASON_NOT_YET_VALID, "check the PC clock")
    if now > body["exp"] + skew_s:
        raise TokenError(REASON_EXPIRED)
    if not hmac.compare_digest(body["hwid_hash"].encode("utf-8"), hwid_hash.encode("utf-8")):
        raise TokenError(REASON_HWID_MISMATCH)
    return body
