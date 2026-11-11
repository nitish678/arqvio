"""The cell-generic power-check trigger: clause (b) is mechanical on every cell, clause (a) only as a named judgment."""
from __future__ import annotations

import pytest

from engine.core.paths import PROJECT_ROOT
from engine.research.power.trigger import (
    CELL_GENERIC_READING,
    OWNER_TRIGGER,
    S4_FINDING,
    power_check_required,
)
from engine.research.registry.cells import UnknownCellError


@pytest.mark.parametrize("cell", ["usdcadm_h1", "xauusdm_h1"])
def test_a_final_negative_verdict_requires_the_power_check_on_any_cell(cell) -> None:
    d = power_check_required(cell, final_negative=True)
    assert d.required and d.clauses == ("(b) final NEGATIVE verdict",) and d.cell_id == cell
    assert set(d.status) == {"status", "ref"}


def test_borderline_is_a_named_judgment_never_a_bare_flag() -> None:
    with pytest.raises(ValueError, match="judgment"):
        power_check_required("xauusdm_h1", final_negative=False, borderline=True)
    d = power_check_required("xauusdm_h1", final_negative=False, borderline=True, borderline_source="owner")
    assert d.required and d.clauses == ("(a) borderline weak-edge candidate",) and "owner" in d.note
    both = power_check_required("xauusdm_h1", final_negative=True, borderline=True, borderline_source="owner")
    assert both.clauses == ("(a) borderline weak-edge candidate", "(b) final NEGATIVE verdict")


def test_neither_clause_means_not_required() -> None:
    d = power_check_required("usdcadm_h1", final_negative=False, borderline=False,
                             borderline_source="XAU-H1 Q7: planner-answered NO")
    assert not d.required and d.clauses == () and "planner-answered NO" in d.note


def test_an_unknown_cell_is_refused() -> None:
    with pytest.raises(UnknownCellError):
        power_check_required("eurusdm_h1", final_negative=True)


def _source(rel: str) -> str:
    return " ".join((PROJECT_ROOT / rel).read_text(encoding="utf-8").split())


def test_the_quotes_are_verbatim_from_their_sources() -> None:
    dr = "docs/decisions/DR-2026-09-30-sham-fill-v2-unchanged.md"
    assert OWNER_TRIGGER in _source(dr)
    report = PROJECT_ROOT / "artifacts" / "reports" / "s4_stage3_report.md"
    if not report.is_file():
        pytest.skip("the S4 Stage 3 report (untracked artifacts) is not in this tree")
    assert S4_FINDING in " ".join(report.read_text(encoding="utf-8").split())
    assert "61%" in CELL_GENERIC_READING
