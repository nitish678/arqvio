"""The power-check trigger, cell-generic (DR-2026-10-08-u2 §7).

The owner widened the trigger for the pinned USDCADm H1 cell; it now applies to every registered cell. Clause (b),
a final NEGATIVE verdict, is mechanical. Clause (a), a borderline weak-edge candidate, is a JUDGMENT the owner never
reduced to a threshold, so it enters only as an explicit flag with its source named -- never computed.
"""
from __future__ import annotations

from dataclasses import dataclass

from engine.research.power.status import power_status

# docs/decisions/DR-2026-09-30-sham-fill-v2-unchanged.md:75-77, verbatim.
OWNER_TRIGGER = (
    "fills-matched sham control trigger WIDENED (2026-09-30): (a) a borderline weak-edge candidate, OR (b) ANY final "
    "NEGATIVE verdict — before publishing, run a bounded power-check (planted + no-plant on the pinned cell, 61%-fill "
    "regime). v2 registry/salt unchanged (comparability)."
)
# artifacts/reports/s4_stage3_report.md:284-287, verbatim: the standing answer to "should fills be matched?".
S4_FINDING = (
    "The two regimes detect identically at every strength and every N: the DSR gate at N = 4,760 binds, and the sham "
    "margin does not. The 61% fill asymmetry is therefore not what limits power, and PART 1 does not motivate a "
    "fills-matched or multi-draw control (a multi-draw sham is a regime change and forbidden this stage in any case)."
)
# How the USDCADm-specific words of OWNER_TRIGGER read on any cell (DR §3 note, §7).
CELL_GENERIC_READING = (
    "'the pinned cell, 61%-fill regime' reads 'the campaign's search cell, its own v2 sham regime': 61% was USDCADm "
    "H1's v2 fill. The protocol runs regime (a) v2 exactly and regime (b) the fills-matched reference on every cell; "
    "per S4_FINDING, a fills-matched or multi-draw CONTROL is not motivated and is not re-litigated."
)


@dataclass(frozen=True)
class TriggerDecision:
    cell_id: str
    required: bool
    clauses: tuple[str, ...]
    status: dict[str, str]
    note: str


def power_check_required(cell_id: str, *, final_negative: bool, borderline: bool = False,
                         borderline_source: str | None = None) -> TriggerDecision:
    """Whether a campaign verdict on ``cell_id`` requires the power check before publishing.

    ``borderline`` is a judgment, so it must name who judged it (e.g. "XAU-H1 Q7: planner-answered NO" is
    borderline=False). An unknown cell is refused by the registry.
    """
    status = power_status(cell_id)
    if borderline and not borderline_source:
        raise ValueError("clause (a) is a judgment: borderline=True needs borderline_source naming who judged it")
    clauses = tuple(c for c, met in (("(a) borderline weak-edge candidate", borderline),
                                     ("(b) final NEGATIVE verdict", final_negative)) if met)
    note = f"borderline judged by: {borderline_source}" if borderline_source else "borderline not judged"
    return TriggerDecision(cell_id=cell_id, required=bool(clauses), clauses=clauses, status=status, note=note)
