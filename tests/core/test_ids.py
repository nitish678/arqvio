"""UUID helpers."""
from __future__ import annotations

import re

from engine.core.ids import is_valid_id, new_id

_UUID4_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")


def test_new_id_is_uuid4() -> None:
    nid = new_id()
    assert _UUID4_RE.match(nid), f"not uuid4: {nid}"
    assert is_valid_id(nid)


def test_new_id_is_unique() -> None:
    ids = {new_id() for _ in range(1000)}
    assert len(ids) == 1000


def test_is_valid_id_rejects_garbage() -> None:
    assert not is_valid_id("")
    assert not is_valid_id("not-a-uuid")
    assert not is_valid_id("12345678-1234-1234-1234-123456789012")  # not v4
    assert is_valid_id("12345678-1234-4234-8234-123456789012")  # v4
