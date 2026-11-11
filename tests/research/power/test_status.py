"""power_status: the registry's power-check record, read through the verifying loader."""
from __future__ import annotations

import pytest

from engine.research.power.status import power_status
from engine.research.registry.cells import UnknownCellError, get_cell


def test_power_status_is_the_registry_record_for_every_cell() -> None:
    for cell in ("usdcadm_h1", "xauusdm_h1"):
        got = power_status(cell)
        assert set(got) == {"status", "ref"}
        assert got == get_cell(cell)["power_check"]
    assert power_status("usdcadm_h1")["status"] == "COMPUTED"


def test_power_status_is_a_copy() -> None:
    got = power_status("usdcadm_h1")
    got["status"] = "EDITED"
    assert power_status("usdcadm_h1")["status"] == "COMPUTED"


def test_an_unknown_cell_is_refused() -> None:
    with pytest.raises(UnknownCellError):
        power_status("eurusdm_h1")
