"""Schema names in ARQVIO's JSON artifacts, and the names they carried before the ASTRA -> ARQVIO rename.

The rename (a7127b9, 2026-10-23) turned every ``astra.<name>`` schema into ``arqvio.<name>``. A file written before
it keeps the old name -- the banked GA manifest the MT5 parity battery reads was refused for exactly that
(2026-10-26). A reader accepts both names; a writer writes only the new one. No schema name is part of a hash, so
nothing else changes. The frozen hash/seed strings (``ASTRA-DATASET-v1``, ``astra-s3-sham-v1``, ...) are not schema
names and are untouched.
"""
from __future__ import annotations

from typing import Any

NEW_PREFIX = "arqvio."
OLD_PREFIX = "astra."


def same_schema(found: Any, expected: str) -> bool:
    """True when ``found`` is ``expected`` or its pre-rename ``astra.`` spelling."""
    if found == expected:
        return True
    return (isinstance(found, str) and expected.startswith(NEW_PREFIX)
            and found == OLD_PREFIX + expected[len(NEW_PREFIX):])


__all__ = ["same_schema"]
