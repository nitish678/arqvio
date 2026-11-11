"""Campaign preflight and mandatory verdict object.

A campaign launch must present a :class:`Verdict` to the UI before it
can run. The UI cannot hide the verdict: a hidden verdict is a launch
refusal.

Launch is REFUSED when any of the following holds:

  - The verdict object is missing or its mandatory fields are None.
  - The scope is empty (no symbols, no timeframes, no trials).
  - The history for the chosen cell is UNAVAILABLE.
  - MinBTL is INSUFFICIENT.
  - Required spread calibration is missing.
  - Provenance is mixed (mixed_source=True).
  - The UI was not given the verdict to render.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from engine.data.spread_calibration import (
    SpreadCheck,
    SpreadMatrix,
    evaluate,
    load,
)
from engine.research.provenance import DataSource, Provenance


class VerdictStatus(str, enum.Enum):
    OK = "OK"
    INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"
    MISSING_CALIBRATION = "MISSING_CALIBRATION"
    MIXED_SOURCE = "MIXED_SOURCE"
    EMPTY_SCOPE = "EMPTY_SCOPE"
    NO_VERDICT = "NO_VERDICT"
    UNAVAILABLE_DATA = "UNAVAILABLE_DATA"
    SEARCH_TRIALS_TOO_LARGE = "SEARCH_TRIALS_TOO_LARGE"


@dataclass(frozen=True)
class Verdict:
    """Mandatory verdict that the UI must render before the campaign launches."""

    status: VerdictStatus
    n_symbols: int
    n_timeframes: int
    population_per_cell: int
    search_trials: int
    min_years_required: float
    years_available: float | None
    target_is_annual_sharpe: float
    comparison_at_sharpe_1: int      # legacy comparator (IS bar at SR=1.0)
    missing_calibration: tuple[str, ...] = ()
    mixed_source: bool = False
    verdict_visible_to_ui: bool = True
    as_of: str = field(default_factory=lambda: datetime.now(tz=UTC).isoformat())

    def line(self) -> str:
        if self.status is not VerdictStatus.OK:
            return (
                f"{self.n_symbols} symbols × {self.n_timeframes} timeframes × "
                f"{self.population_per_cell:,} pop = {self.search_trials:,} search trials "
                f"→ {self.status.value}"
            )
        years = (
            f"{self.years_available:.2f}y available"
            if self.years_available is not None
            else "no history"
        )
        return (
            f"{self.n_symbols} symbols × {self.n_timeframes} timeframes × "
            f"{self.population_per_cell:,} pop = {self.search_trials:,} search trials "
            f"→ MinBTL {self.min_years_required:.2f}y required, "
            f"{years} — {self.status.value}"
        )

    def is_launchable(self) -> bool:
        return (
            self.verdict_visible_to_ui
            and self.status is VerdictStatus.OK
            and self.n_symbols >= 1
            and self.n_timeframes >= 1
            and self.search_trials >= 1
        )


def _mandatory_fields() -> tuple[str, ...]:
    return (
        "status",
        "n_symbols",
        "n_timeframes",
        "population_per_cell",
        "search_trials",
        "min_years_required",
        "target_is_annual_sharpe",
        "verdict_visible_to_ui",
    )


def ensure_verdict_complete(v: Verdict | None) -> None:
    """Raise if the verdict is missing or any mandatory field is None."""
    if v is None:
        raise PreflightError("verdict is None; the campaign builder must call compute_verdict()")
    for field_name in _mandatory_fields():
        if getattr(v, field_name) is None:
            raise PreflightError(f"verdict field {field_name!r} is None")


def refuse_on_provenance(p: Provenance) -> None:
    if p.mixed_source:
        raise PreflightError(
            "provenance.mixed_source=True; refusing to launch. A campaign must "
            "be scoped to a single (BUNDLED, CACHED, or MT5_LIVE) source."
        )
    if p.source is DataSource.BUNDLED and not p.dataset_hash:
        raise PreflightError("provenance.dataset_hash is empty on a BUNDLED source")


class PreflightError(RuntimeError):
    """Raised when the campaign launch is refused for any verifiable reason."""


def spread_preflight(symbol: str, spread_dir: Path) -> dict[str, object]:
    """Grade a symbol's persisted spread row and refuse if it cannot be used.

    Returns the fragment the campaign report carries, so a block reason ends
    up in the REPORT and not only in a log line. It GRADES rather than
    raising, so the caller can record the fragment and then refuse -- a
    reason that only exists inside an exception message is a log line.

    A symbol with no persisted matrix is reported UNAVAILABLE and does NOT
    block. "Not calibrated yet" and "calibrated and unusable" are different
    facts, and refusing every uncalibrated symbol would stop every campaign
    rather than protect one.
    """
    matrix = load(symbol, spread_dir)
    if matrix is None:
        return {
            "status": "UNAVAILABLE",
            "blocks": False,
            "reason": (
                f"no spread matrix persisted for {symbol} in {spread_dir}; "
                "costs are uncalibrated and this cell's costs are unverified"
            ),
            "ratio": None,
        }
    check = evaluate(matrix)
    return {
        "status": check.status.value,
        "blocks": check.blocks,
        "reason": check.reason,
        "ratio": check.ratio,
    }


def refuse_on_spread(matrix: SpreadMatrix) -> SpreadCheck:
    """Refuse to launch on a symbol whose spread row cannot be trusted.

    Until this existed, ``is_calibration_failure`` was called by nothing but
    its own unit test: a symbol with a flat row was detected and then used
    anyway. Returns the check so a SUSPICIOUS row can be logged by the
    caller rather than silently passed.
    """
    check = evaluate(matrix)
    if check.blocks:
        raise PreflightError(
            f"SPREAD_{check.status.value}: {check.symbol} - {check.reason}. "
            "Re-run spread calibration before any campaign uses this symbol."
        )
    return check


def refuse_on_basis_mismatch(
    matrix: SpreadMatrix, audit_time_basis: str, cell_label: str
) -> None:
    """Refuse when the spread artifact's time_basis disagrees with the audit.

    The audit CSV stamps a per-cell ``time_basis`` (R-C0); the spread
    calibrator stamps ``probe.basis`` on each SpreadMatrix. They must agree,
    otherwise spread-cost hours and engine bar hours are bucketed in
    different references and the cost model is silently wrong.

    Equivalence classes:
      - {"utc", "server_equals_utc"}: the engine treats UTC-encoded timestamps
        the same; a matrix stamped "server_equals_utc" is compatible with an
        audit stamped "utc".
      - {"broker_offset"}: must match exactly; an old artifact lacking the
        field defaults to "utc" via SpreadMatrix.from_json, which is compatible
        with a "utc" audit.
    """
    matrix_basis = matrix.time_basis
    audit_basis = audit_time_basis or "utc"
    equivalent = {
        "utc",
        "server_equals_utc",
    }
    if matrix_basis in equivalent and audit_basis in equivalent:
        return
    if matrix_basis == audit_basis and matrix_basis not in ("unresolved",):
        return
    raise PreflightError(
        f"BASIS_MISMATCH on cell {cell_label}: spread artifact "
        f"time_basis={matrix_basis!r} but audit time_basis={audit_basis!r}. "
        "The two references must agree so cost hours and bar hours bucket identically."
    )


def write_verdict_manifest(v: Verdict, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    import json

    target.write_text(json.dumps(v.__dict__, indent=2, default=str), encoding="utf-8")
    return target


__all__ = [
    "PreflightError",
    "Verdict",
    "VerdictStatus",
    "ensure_verdict_complete",
    "refuse_on_basis_mismatch",
    "refuse_on_provenance",
    "write_verdict_manifest",
]
