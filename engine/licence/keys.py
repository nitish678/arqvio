"""Licence keys: ``XXXX-DDDDDDDDDD-XXXXXXXXXXXXXXXX`` (4-hex check, 10-digit serial, 16-hex secret).

WHY this shape:
- The serial is the key's public identity (the server's row id, what support asks for). It carries no secret.
- The 16-hex secret is HMAC-SHA256(server_secret, serial), truncated. The server stores no per-key secret at all: it
  re-derives the secret from the serial, so a stolen database alone mints no keys.
- The 4-hex prefix is a plain check value over serial+secret. The app can catch a typo offline, before any network
  call, without holding the server secret (``checksum_ok``). It is NOT a security boundary; ``verify_key`` is.

People paste keys from e-mails, PDFs and chat apps, with spaces, line breaks, odd dashes and any case, so ``parse`` is
aggressive: it keeps only letters and digits, casefolds, and then reads the three groups by position.
"""
from __future__ import annotations

import hashlib
import hmac
import re
from dataclasses import dataclass

__all__ = ["KeyFormatError", "LicenceKey", "checksum_ok", "make_key", "parse", "verify_key"]

PREFIX_LEN = 4
SERIAL_LEN = 10
SECRET_LEN = 16
_KEY_RE = re.compile(rf"^([0-9a-f]{{{PREFIX_LEN}}})([0-9]{{{SERIAL_LEN}}})([0-9a-f]{{{SECRET_LEN}}})$")
_NON_ALNUM = re.compile(r"[^0-9A-Za-z]+")


class KeyFormatError(ValueError):
    """The text is not a licence key in the ARQVIO format."""


@dataclass(frozen=True, slots=True)
class LicenceKey:
    """A parsed key. Fields are canonical: lowercase hex, a zero-padded 10-digit serial."""

    prefix: str
    serial: str
    secret: str

    def __str__(self) -> str:
        return f"{self.prefix.upper()}-{self.serial}-{self.secret.upper()}"


def parse(text: str) -> LicenceKey:
    """Read a key from whatever the user pasted; raise ``KeyFormatError`` when it cannot be one."""
    if not isinstance(text, str):
        raise KeyFormatError("a licence key must be text")
    compact = _NON_ALNUM.sub("", text).casefold()
    match = _KEY_RE.match(compact)
    if match is None:
        raise KeyFormatError("not a licence key (expected XXXX-DDDDDDDDDD-XXXXXXXXXXXXXXXX)")
    return LicenceKey(prefix=match.group(1), serial=match.group(2), secret=match.group(3))


def _serial_text(serial: int | str) -> str:
    text = str(serial)
    if not text.isdigit() or len(text) > SERIAL_LEN:
        raise KeyFormatError(f"serial must be at most {SERIAL_LEN} digits")
    return text.zfill(SERIAL_LEN)


def _derive_secret(serial: str, server_secret: bytes) -> str:
    return hmac.new(server_secret, serial.encode("ascii"), hashlib.sha256).hexdigest()[:SECRET_LEN]


def _check_value(serial: str, secret: str) -> str:
    return hashlib.sha256(f"{serial}:{secret}".encode("ascii")).hexdigest()[:PREFIX_LEN]


def make_key(serial: int | str, server_secret: bytes) -> LicenceKey:
    """Build the key for ``serial``; the same serial and server secret always give the same key."""
    if not server_secret:
        raise ValueError("server_secret must not be empty")
    serial_text = _serial_text(serial)
    secret = _derive_secret(serial_text, server_secret)
    return LicenceKey(prefix=_check_value(serial_text, secret), serial=serial_text, secret=secret)


def checksum_ok(key: LicenceKey | str) -> bool:
    """Offline typo check: does the prefix match the serial and secret? Needs no server secret."""
    parsed = parse(key) if isinstance(key, str) else key
    return hmac.compare_digest(parsed.prefix, _check_value(parsed.serial, parsed.secret))


def verify_key(key: LicenceKey | str, server_secret: bytes) -> bool:
    """True when the key's check value and its secret are both genuine for this server secret."""
    try:
        parsed = parse(key) if isinstance(key, str) else key
    except KeyFormatError:
        return False
    if not server_secret or not checksum_ok(parsed):
        return False
    return hmac.compare_digest(parsed.secret, _derive_secret(parsed.serial, server_secret))
