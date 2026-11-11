"""power_status(cell): the registry's power-check record for a cell, the read API U4 / S5 call (DR-2026-10-08-u2 §2).

It reads the U1 registry's ``power_check`` {status, ref} through the verifying loader; an unknown cell or an
untrusted registry is refused by the registry itself.
"""
from __future__ import annotations

from engine.research.registry.cells import get_cell


def power_status(cell_id: str) -> dict[str, str]:
    """{"status", "ref"} of ``cell_id``'s power check, as sealed in the registry."""
    return dict(get_cell(cell_id)["power_check"])
