"""Verdict composition for the campaign builder.

Given a scope, a per-cell history depth, and provenance + calibration
state, produce a :class:`Verdict` ready to render in the UI.

The verdict is mandatory: a launch without one is refused. The UI is
not allowed to hide any field.
"""
from __future__ import annotations

from collections.abc import Iterable

from engine.research.campaign.budget import (
    CampaignScope,
    legacy_comparator_at_sharpe_1,
    required_min_years,
)
from engine.research.campaign.preflight import (
    PreflightError,
    Verdict,
    VerdictStatus,
    ensure_verdict_complete,
)
from engine.research.provenance import Provenance


def compute_verdict(
    scope: CampaignScope,
    *,
    per_cell_years: dict[tuple[str, str], float | None],
    provenance: Provenance | None,
    missing_calibration: Iterable[str] = (),
    target_sharpe: float | None = None,
) -> Verdict:
    target = target_sharpe if target_sharpe is not None else scope.target_is_annual_sharpe
    missing_calibration = tuple(missing_calibration)
    status = VerdictStatus.OK
    search_trials = scope.search_trials

    if scope.is_empty:
        status = VerdictStatus.EMPTY_SCOPE
    elif search_trials == 0:
        status = VerdictStatus.SEARCH_TRIALS_TOO_LARGE
    elif provenance is not None and provenance.mixed_source:
        status = VerdictStatus.MIXED_SOURCE
    elif missing_calibration:
        status = VerdictStatus.MISSING_CALIBRATION
    elif any(v is None for v in per_cell_years.values()) and per_cell_years:
        status = VerdictStatus.UNAVAILABLE_DATA
    else:
        years = min((v for v in per_cell_years.values() if v is not None), default=None)
        if years is None or years <= 0:
            status = VerdictStatus.UNAVAILABLE_DATA
        else:
            required = required_min_years(scope, target, history_years=years)
            if required > years:
                status = VerdictStatus.INSUFFICIENT_HISTORY
            min_years = required
        if status is VerdictStatus.OK:
            return Verdict(
                status=VerdictStatus.OK,
                n_symbols=scope.n_symbols,
                n_timeframes=scope.n_timeframes,
                population_per_cell=scope.population_per_cell,
                search_trials=search_trials,
                min_years_required=required,
                years_available=years,
                target_is_annual_sharpe=target,
                comparison_at_sharpe_1=legacy_comparator_at_sharpe_1(
                    scope, history_years=float(years) if years is not None else 0.0
                ),
                missing_calibration=missing_calibration,
                mixed_source=bool(provenance.mixed_source) if provenance else False,
            )

    # Default / refused verdicts.
    years_raw = [v for v in per_cell_years.values() if v is not None]
    years = float(min(years_raw)) if years_raw else 0.0
    years_for_legacy = 1.0 if years <= 0 else years
    return Verdict(
        status=status,
        n_symbols=scope.n_symbols,
        n_timeframes=scope.n_timeframes,
        population_per_cell=scope.population_per_cell,
        search_trials=search_trials,
        min_years_required=0.0,
        years_available=float(years),
        target_is_annual_sharpe=target,
        comparison_at_sharpe_1=legacy_comparator_at_sharpe_1(
            scope, history_years=float(years_for_legacy)
        ),
        missing_calibration=missing_calibration,
        mixed_source=bool(provenance.mixed_source) if provenance else False,
    )


def launch_guard(verdict: Verdict) -> None:
    ensure_verdict_complete(verdict)
    if not verdict.is_launchable():
        raise PreflightError(
            f"campaign launch refused: verdict={verdict.status.value}, "
            f"verdict_visible_to_ui={verdict.verdict_visible_to_ui}, "
            f"search_trials={verdict.search_trials}"
        )


__all__ = [
    "compute_verdict",
    "launch_guard",
    "Verdict",
]
