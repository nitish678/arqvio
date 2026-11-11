"""Licence keys: the format, the aggressive parser, the round trip and tampering.

WHY: users paste keys from anywhere; the parser must forgive formatting and nothing else.
"""
from __future__ import annotations

import pytest

from engine.licence.keys import KeyFormatError, LicenceKey, checksum_ok, make_key, parse, verify_key

SECRET = b"unit-test-key-secret"


def test_make_key_format_and_round_trip() -> None:
    key = make_key(42, SECRET)
    text = str(key)
    prefix, serial, secret = text.split("-")
    assert (len(prefix), serial, len(secret)) == (4, "0000000042", 16)
    assert parse(text) == key
    assert verify_key(text, SECRET)
    assert make_key("0000000042", SECRET) == key  # deterministic


@pytest.mark.parametrize(
    "mangle",
    [
        lambda t: t.lower(),
        lambda t: t.replace("-", " "),
        lambda t: " ".join(t.replace("-", "")),
        lambda t: f"  \t{t}.\r\n",
        lambda t: t.replace("-", "–"),  # en dash from a word processor
        lambda t: t.replace("-", "--"),
    ],
)
def test_parser_is_aggressive(mangle) -> None:  # noqa: ANN001
    key = make_key(1234567890, SECRET)
    assert parse(mangle(str(key))) == key


@pytest.mark.parametrize(
    "text",
    ["", "hello", "ABCD-12345-0123456789ABCDEF", "GHIJ-1234567890-0123456789ABCDEF", "ABCD-1234567890-0123"],
)
def test_parser_rejects_non_keys(text: str) -> None:
    with pytest.raises(KeyFormatError):
        parse(text)


def test_parser_rejects_non_text() -> None:
    with pytest.raises(KeyFormatError):
        parse(None)  # type: ignore[arg-type]


def _flip(ch: str) -> str:
    return "1" if ch != "1" else "2"


def test_tampered_keys_rejected() -> None:
    key = make_key(7, SECRET)
    bad_secret = LicenceKey(key.prefix, key.serial, key.secret[:-1] + _flip(key.secret[-1]))
    bad_serial = LicenceKey(key.prefix, "0000000008", key.secret)
    bad_prefix = LicenceKey(key.prefix[:-1] + _flip(key.prefix[-1]), key.serial, key.secret)
    for tampered in (bad_secret, bad_serial, bad_prefix):
        assert not verify_key(tampered, SECRET)
    assert not verify_key(key, b"another-server-secret")
    assert not verify_key("not a key", SECRET)


def test_forged_check_value_still_needs_the_server_secret() -> None:
    """The prefix is a typo check anyone can compute; it must not stand in for the secret."""
    from engine.licence.keys import _check_value

    serial, candidate = "0000000099", "0123456789abcdef"
    forged = LicenceKey(_check_value(serial, candidate), serial, secret)
    assert checksum_ok(forged)
    assert not verify_key(forged, SECRET)


def test_serial_bounds() -> None:
    with pytest.raises(KeyFormatError):
        make_key(10**10, SECRET)
    with pytest.raises(ValueError):
        make_key(1, b"")
