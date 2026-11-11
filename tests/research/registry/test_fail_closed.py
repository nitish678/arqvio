"""The GA resolves cells only through the registry, and stays fail-closed: an unknown cell or an untrusted registry
is refused at every entry point (the runner CLI, prepare_cell, cross_cell_for)."""
from __future__ import annotations

import json

import pytest

from engine.core import paths
from engine.research.ga.config import PINNED_CELLS, GAConfigError, cross_cell_for
from engine.research.ga.runner import GARunError, main, prepare_cell
from engine.research.registry.annualisation import TF_MINUTES, bars_per_year
from engine.research.registry.cells import PinnedCellsView, RegistryError, RegistryIntegrityError, RegistryMissing


@pytest.fixture
def tampered_root(monkeypatch, tmp_path, registry_doc):
    """PROJECT_ROOT pointed at a tree whose configs/cells_registry.json had a dataset hash swapped, unsealed."""
    registry_doc["entries"][0]["dataset_hash16"] = "0123456789abcdef"
    target = tmp_path / "configs" / "cells_registry.json"
    target.parent.mkdir()
    target.write_text(json.dumps(registry_doc), encoding="utf-8")
    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path)
    return tmp_path


def test_the_ga_table_is_the_registry_view() -> None:
    assert isinstance(PINNED_CELLS, PinnedCellsView)
    assert dict(PINNED_CELLS) == {
        "usdcadm_h1": ("USDCADm", "H1", "aed3bf314539ec5a"),
        "xauusdm_h1": ("XAUUSDm", "H1", "447b843f87438a9a"),
        "gbpjpym_h1": ("GBPJPYm", "H1", "5979a019aa6b3e0c"),  # U4 W4: selectable once VALIDATED (DR §4)
    }


def test_the_cli_offers_exactly_the_validated_cells(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "--cell {gbpjpym_h1,usdcadm_h1,xauusdm_h1}" in capsys.readouterr().out


def test_the_cli_refuses_an_unknown_cell(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--cell", "eurusdm_h1", "--seed", "7"])
    assert exc.value.code == 2
    assert "invalid choice: 'eurusdm_h1'" in capsys.readouterr().err


def test_prepare_cell_refuses_an_unknown_cell() -> None:
    with pytest.raises(GARunError, match="unknown cell 'eurusdm_h1'"):
        prepare_cell("eurusdm_h1")


def test_cross_cell_for_refuses_an_unknown_cell() -> None:
    with pytest.raises(GAConfigError, match="unknown cell 'eurusdm_h1'"):
        cross_cell_for("eurusdm_h1")


def test_a_tampered_registry_stops_every_ga_entry_point(tampered_root) -> None:
    with pytest.raises(RegistryIntegrityError):
        "usdcadm_h1" in PINNED_CELLS  # noqa: B015 -- the membership test itself must refuse
    with pytest.raises(RegistryIntegrityError):
        cross_cell_for("usdcadm_h1")
    with pytest.raises(RegistryIntegrityError):
        prepare_cell("usdcadm_h1")
    with pytest.raises(RegistryIntegrityError):
        main(["--cell", "usdcadm_h1", "--seed", "7"])


def test_a_missing_registry_stops_the_ga(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path)
    with pytest.raises(RegistryMissing):
        prepare_cell("usdcadm_h1")
    assert issubclass(RegistryMissing, RegistryError)


def test_the_dsr_table_is_the_law() -> None:
    from engine.validation.cell_dsr import DEFAULT_PERIODS_PER_YEAR, PERIODS_PER_YEAR

    assert PERIODS_PER_YEAR == {tf: bars_per_year(tf) for tf in TF_MINUTES}
    assert PERIODS_PER_YEAR["M5"] == 72576 and PERIODS_PER_YEAR["H1"] == 6048
    assert DEFAULT_PERIODS_PER_YEAR == 252
