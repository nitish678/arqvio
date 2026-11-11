"""S5 API surface for the engine.

The Stage 1 server only exposes /healthz and /api/v1/ping. The S5 pass
adds the minimum endpoints the UI needs to render real numbers, not
frozen JSON:

    GET  /api/v1/s5/audit        -> history-depth audit summary
    GET  /api/v1/s5/verdict      -> live campaign verdict (MinBTL)
    GET  /api/v1/s5/runner       -> last search-runner result
    POST /api/v1/s5/launch       -> run a search; writes artifacts/

Anything else renders NOT IMPLEMENTED on the UI side.
"""
from __future__ import annotations

import csv
import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from engine.core.logging import get_logger
from engine.core.paths import get_arqvio_home, get_artifacts_dir

log = get_logger()
router = APIRouter(prefix="/api/v1/s5", tags=["s5"])

# The campaign runs exactly the cells in the target audit. The live audit
# (history_depth_audit_live.csv) covers every MT5-discoverable symbol but is
# NOT what /launch validates against — only the target audit is. The /audit
# endpoint serves the catalog FROM the target audit so the UI's chip list
# matches what the engine can actually run. A cell absent from the target
# audit cannot be selected or launched.
AUDIT_CSV_NAME = "history_depth_audit_live.csv"
TARGET_AUDIT_CSV_NAME = "history_depth_audit_target.csv"
RUNNER_JSON_NAME = "search_runner_real.json"
CAMPAIGN_TARGET_SHARPE = 1.5


def artifact(name: str) -> Path:
    """ARQVIO_HOME/artifacts/<name>, resolved at CALL time (CLAUDE.md s7): the
    file run_campaign_cli.artifact_path(name) writes under the same home."""
    return get_artifacts_dir() / name


def audit_csv() -> Path:
    return artifact(AUDIT_CSV_NAME)


def target_audit_csv() -> Path:
    return artifact(TARGET_AUDIT_CSV_NAME)


def runner_json() -> Path:
    return artifact(RUNNER_JSON_NAME)


# Log the resolved path so a misconfigured ARQVIO_HOME is visible at import,
# not discovered later as a missing-file error. This is a LOG line, kept on
# purpose for startup visibility; it is not a path constant. Nothing reads
# these values back: every reader above resolves its path again at call time
# (CLAUDE.md s7), so a home changed after import is still the one used.
log.info(
    "ARQVIO_HOME=%s  AUDIT_CSV=%s  TARGET_AUDIT_CSV=%s",
    get_arqvio_home(), audit_csv(), target_audit_csv(),
)


def _parse_audit_header(handle) -> dict[str, str]:
    """Parse '# KEY VALUE' lines from the start of the audit CSV handle."""
    header: dict[str, str] = {}
    for line in handle:
        stripped = line.lstrip()
        if stripped.startswith("#"):
            parts = stripped.lstrip("# ").split(" ", 1)
            if len(parts) == 2:
                # The value ends where the line does: drop the terminator.
                header[parts[0]] = parts[1].rstrip("\r\n")
        else:
            break
    return header


def _audit_rows(handle) -> list[dict[str, str]]:
    """Yield data rows from a CSV that may have '#' comment header lines."""
    reader = csv.DictReader(
        (line for line in handle if not line.lstrip().startswith("#"))
    )
    return list(reader)


class LaunchRequest(BaseModel):
    population: int = Field(default=5000, ge=1, le=100_000)
    selected: list[str] | None = None


class LaunchResponse(BaseModel):
    job_id: str
    status: str


class RunnerResult(BaseModel):
    # search_runner_real.json is a LIST of per-cell reports (one dict per
    # symbol/timeframe), not an object. Typing this as `dict` made the route
    # raise a pydantic ValidationError -- a 500 -- for every artifact a real
    # campaign had ever written, which is to say always.
    exists: bool
    data: dict | list | None = None
    path: str | None = None
    cells: int = 0


class JobStatus(BaseModel):
    job_id: str
    status: str
    completed: int = 0
    total: int = 0
    progress: float = 0.0
    error: str | None = None
    result: RunnerResult | None = None


class SurvivorRecord(BaseModel):
    symbol: str
    timeframe: str
    dna_hash: str
    sharpe: float | None = None
    trades: int | None = None
    total_return: float | None = None
    max_dd: float | None = None
    dsr: float | None = None
    outcome: str
    created_at: str


class _Job:
    def __init__(self, job_id: str, population: int, selected: set[tuple[str, str]] | None) -> None:
        self.job_id = job_id
        self.population = population
        self.selected = selected
        self.status = "QUEUED"
        self.completed = 0
        self.total = 0
        self.error: str | None = None
        self.result: RunnerResult | None = None
        self.reports: list[dict[str, Any]] | None = None
        self.lock = threading.Lock()


_JOBS: dict[str, _Job] = {}
_JOBS_LOCK = threading.Lock()
_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="arqvio-s5")


def _get_job(job_id: str) -> _Job | None:
    with _JOBS_LOCK:
        return _JOBS.get(job_id)


def _most_recent_finished_job() -> _Job | None:
    """The latest job that produced reports. Falls back to the newest job."""
    with _JOBS_LOCK:
        jobs = list(_JOBS.values())
    for job in reversed(jobs):
        if job.reports:
            return job
    return jobs[-1] if jobs else None


def _feed_is_current(job: _Job | None) -> bool:
    """True only when this job actually produced the reports being served."""
    if job is None:
        return False
    with job.lock:
        return bool(job.reports)


def _job_status(job: _Job) -> JobStatus:
    with job.lock:
        progress = (job.completed / job.total * 100.0) if job.total else 0.0
        return JobStatus(
            job_id=job.job_id,
            status=job.status,
            completed=job.completed,
            total=job.total,
            progress=progress,
            error=job.error,
            result=job.result,
        )


def _run_job(job: _Job) -> None:
    from engine.research.search import gene_pool
    from engine.research.search.run_campaign_cli import run_campaign_sync

    def on_progress(completed: int, total: int) -> None:
        with job.lock:
            job.completed = completed
            job.total = total
            if job.status == "QUEUED":
                job.status = "RUNNING"

    with job.lock:
        job.status = "RUNNING"
    try:
        # Read once, here: an edit made while the job runs must not change
        # the pool halfway through a campaign.
        pool = gene_pool.load()
        reports = run_campaign_sync(
            population=job.population,
            selected=job.selected,
            on_progress=on_progress,
            gene_pool=pool,
        )
        with job.lock:
            job.completed = job.total
            job.status = "COMPLETE"
            # The job's own result, kept in memory. /gate serves from this so a
            # stale file left by an earlier run cannot present as the current
            # campaign's verdict.
            job.reports = reports
            job.result = RunnerResult(
                exists=True,
                data={"reports": reports},
                path=str(runner_json()),
            )
    except Exception as exc:
        log.exception("S5 job %s failed", job.job_id)
        with job.lock:
            job.status = "FAILED"
            job.error = f"{type(exc).__name__}: {exc}"


def _parse_selected(selected: list[str] | None) -> set[tuple[str, str]] | None:
    if not selected:
        return None
    cells: set[tuple[str, str]] = set()
    for value in selected:
        if ":" not in value:
            raise ValueError(f"invalid cell selector {value!r}; expected SYMBOL:TIMEFRAME")
        symbol, timeframe = value.split(":", 1)
        if not symbol or not timeframe:
            raise ValueError(f"invalid cell selector {value!r}")
        cells.add((symbol, timeframe))
    return cells


def _as_runner_result(reports: Any) -> RunnerResult:
    return RunnerResult(exists=True, data={"reports": reports}, path=str(runner_json()))


def _selected(argv: list[str]) -> set[tuple[str, str]] | None:
    return _parse_selected(argv)


class AuditCell(BaseModel):
    symbol: str
    timeframe: str
    years_available: str
    bars_returned: int
    first_bar_utc: str
    last_bar_utc: str
    source: str


class AuditCatalogEntry(BaseModel):
    """One cell from the audit's full catalog — the universe of cells the
    campaign can actually run. The UI uses this to build its chip list
    instead of a hardcoded cell list. """
    symbol: str
    timeframe: str
    years_available: float | None
    source: str


class AuditSummary(BaseModel):
    cells: int
    symbols: int
    timeframes: list[str]
    unavailable: int
    top_15: list[AuditCell]
    per_timeframe: dict[str, dict[str, float | int]]
    as_of: str
    csv_path: str
    rendered_cells: int  # how many rows we actually returned
    cell_catalog: list[AuditCatalogEntry]  # every audited cell; UI builds chips from this
    # G14: audit provenance surfaced to the UI so a stale or mixed-basis
    # audit CSV is visible, not inferred from a missing dataset_id.
    audit_source: str = ""
    audit_time_basis: str = ""


class PreflightCell(BaseModel):
    symbol: str
    timeframe: str
    available: bool
    bars_returned: int | None = None
    years_available: float | None = None
    minbtl_required_years: float | None = None
    minbtl_satisfied: bool | None = None
    reason: str | None = None
    # F9: a cell that runs but with something the operator must know, e.g.
    # "untradeable hour(s) in sim: [21]". Distinct from `reason`, which is
    # only ever why a cell is NOT available.
    info: str | None = None


class PreflightResponse(BaseModel):
    """A MinBTL preflight that is actually computed.

    min_backtest_length() is called here with the same population and target
    Sharpe the campaign will use, against the cell history years in
    history_depth_audit_target.csv — the same file run_campaign_sync loads
    before it starts. Before this endpoint existed the launch gate demanded a
    preflight no code path could produce, so the builder could only refuse.
    """
    population: int
    target_sharpe: float
    cells: list[PreflightCell]
    # A preflight can only pass when every cell the launch covers passes.
    all_satisfied: bool
    source: str
    note: str


SPREAD_NOT_CALIBRATED = "spread not calibrated"


def spread_blocked_reason(symbol: str) -> str | None:
    """Why this symbol cannot be traded, or None if its spread is usable.

    A campaign costed from a missing or unreadable spread matrix is costed
    from nothing. The backtester falls back to the symbol spec's single
    spread_points value, which has no hour-of-day shape at all, so the
    rollover hours come out as cheap as London noon and every candidate
    looks better than it is. That is not a degraded result, it is a wrong
    one, and it is refused rather than run.
    """
    from engine.data.spread_calibration import SpreadMatrix, spread_dir

    path = spread_dir() / f"{symbol}.json"
    if not path.is_file():
        return f"{SPREAD_NOT_CALIBRATED}: no artifact at {path.name}"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return f"{SPREAD_NOT_CALIBRATED}: {path.name} unreadable ({exc})"
    from engine.data.spread_calibration import DataError
    try:
        matrix = SpreadMatrix.from_json(text)
    except DataError as exc:
        return (
            f"{SPREAD_NOT_CALIBRATED}: {path.name} is present but incomplete "
            f"({exc})"
        )
    except Exception as exc:
        return (
            f"{SPREAD_NOT_CALIBRATED}: {path.name} does not parse as a spread "
            f"matrix ({type(exc).__name__})"
        )
    if len(matrix.buckets) != 24:
        return (
            f"{SPREAD_NOT_CALIBRATED}: {path.name} has "
            f"{len(matrix.buckets)} buckets, expected 24"
        )
    return None


def spread_info_line(symbol: str) -> str | None:
    """What the operator must know about a cell that still runs (F9).

    A calibrated matrix with a zero-spread hour is no longer a refusal:
    F1 shuts those hours inside the backtester. It is still not something
    to discover afterwards from a funnel, because the cell is trading
    fewer hours than its timeframe suggests, so it is stated up front.
    """
    from engine.data.spread_calibration import load, spread_dir

    try:
        matrix = load(symbol, spread_dir())
    except Exception:
        # Unreadable is spread_blocked_reason's call, not this one's.
        return None
    if matrix is None:
        return None
    hours = sorted(matrix.no_trade_hours)
    if not hours:
        return None
    return f"untradeable hour(s) in sim: {hours}"


@router.get("/preflight", response_model=PreflightResponse)
async def preflight(population: int = 5000, selected: str | None = None) -> PreflightResponse:
    """Compute MinBTL for the cells this launch would run."""
    from engine.validation.minbtl import min_backtest_length

    # The audit CSV is the universe of runnable cells. When `selected` is
    # provided, every requested cell is validated against it — nothing is
    # silently dropped. A requested cell with no audit row returns
    # available=False with a reason, never omission.
    audit: dict[tuple[str, str], dict[str, str]] = {}
    if target_audit_csv().exists():
        with target_audit_csv().open(encoding="utf-8") as handle:
            for row in _audit_rows(handle):
                audit[(row["symbol"], row["timeframe"])] = row

    # Determine which cells the operator asked for. With no `selected` query
    # param, preflight covers every cell the audit knows about — not a static
    # hardcoded list. This makes the verdict honest: it reflects what the
    # campaign would actually run.
    if selected:
        try:
            wanted = _parse_selected(selected.split(","))
        except ValueError as exc:
            return PreflightResponse(
                population=population,
                target_sharpe=CAMPAIGN_TARGET_SHARPE,
                cells=[],
                all_satisfied=False,
                source="none",
                note=str(exc),
            )
        cells = sorted(wanted)
    else:
        cells = sorted(audit.keys())

    if not cells:
        return PreflightResponse(
            population=population,
            target_sharpe=CAMPAIGN_TARGET_SHARPE,
            cells=[],
            all_satisfied=False,
            source="none",
            note="No cells selected.",
        )

    out: list[PreflightCell] = []
    for symbol, timeframe in cells:
        # Same refusal class as a missing audit row: the cell cannot run.
        spread_reason = spread_blocked_reason(symbol)
        if spread_reason is not None:
            out.append(
                PreflightCell(
                    symbol=symbol,
                    timeframe=timeframe,
                    available=False,
                    reason=spread_reason,
                )
            )
            continue
        row = audit.get((symbol, timeframe))
        if row is None:
            out.append(
                PreflightCell(
                    symbol=symbol,
                    timeframe=timeframe,
                    available=False,
                    reason=(
                        f"no row for {symbol} {timeframe} in "
                        f"{target_audit_csv().name}; the campaign cannot run this cell"
                    ),
                )
            )
            continue
        raw_years = str(row.get("years_available", ""))
        if raw_years == "UNAVAILABLE":
            out.append(
                PreflightCell(
                    symbol=symbol,
                    timeframe=timeframe,
                    available=False,
                    bars_returned=_as_int(row.get("bars_returned")),
                    reason="history unavailable on this terminal; MinBTL cannot be evaluated",
                )
            )
            continue
        try:
            years = float(raw_years)
        except ValueError:
            out.append(
                PreflightCell(
                    symbol=symbol,
                    timeframe=timeframe,
                    available=False,
                    reason=f"audit years_available={raw_years!r} is not a number",
                )
            )
            continue
        if years <= 0:
            out.append(
                PreflightCell(
                    symbol=symbol,
                    timeframe=timeframe,
                    available=False,
                    years_available=years,
                    reason="audit reports zero history; MinBTL cannot be evaluated",
                )
            )
            continue
        result = min_backtest_length(
            n_trials=population,
            target_sharpe=CAMPAIGN_TARGET_SHARPE,
            history_years=years,
        )
        out.append(
            PreflightCell(
                symbol=symbol,
                timeframe=timeframe,
                available=True,
                bars_returned=_as_int(row.get("bars_returned")),
                years_available=years,
                minbtl_required_years=float(result.min_years),
                minbtl_satisfied=bool(result.satisfied),
                info=spread_info_line(symbol),
            )
        )

    computable = [c for c in out if c.available]
    all_ok = bool(computable) and all(c.minbtl_satisfied for c in computable) and len(computable) == len(out)
    note = (
        f"MinBTL at {population:,} trials and target SR {CAMPAIGN_TARGET_SHARPE}: "
        f"{sum(1 for c in computable if c.minbtl_satisfied)} of {len(out)} cell(s) satisfied."
    )
    return PreflightResponse(
        population=population,
        target_sharpe=CAMPAIGN_TARGET_SHARPE,
        cells=out,
        all_satisfied=all_ok,
        source="minbtl",
        note=note,
    )


class CellVerdict(BaseModel):
    """One cell's own numbers, exactly as the runner reported them. Nothing
    here is campaign-level: a cell verdict is per symbol+timeframe."""
    symbol: str
    timeframe: str
    minbtl_required_years: float | None
    history_years: float | None
    minbtl_satisfied: bool | None
    exit_verdict: str | None
    completed_at_utc: str | None
    stale: bool


class VerdictResponse(BaseModel):
    """There is no campaign-level verdict in this system, and this response
    does not invent one.

    The previous version read the LAST element of the per-cell report list and
    returned it as ``last_runner_status``/``last_runner_verdict``, filling in
    the literals "SUFFICIENT" and "30,000 search trials -> MinBTL SUFFICIENT"
    whenever a report happened not to carry those keys. MinBTL is evaluated per
    cell inside run_cell(), so a list's last element is one cell's result, not a
    campaign's — and a file on disk is not necessarily this job's output.
    """
    source: str
    note: str
    last_runner_status: str | None
    last_runner_path: str | None
    last_runner_verdict: str | None
    # Real, traceable fields:
    file_written_utc: str | None = None      # mtime of search_runner_real.json
    job_id: str | None = None                # set when ?job_id= names this job
    is_current_job: bool = False             # True only when job_id matches
    cells: list[CellVerdict] = []            # per-cell, never collapsed
    stale_cells: int = 0


class GateCell(BaseModel):
    """One cell of the validation gate strip. Every value traces to a named
    engine field — no fabricated defaults, no numeric literals in the UI."""
    symbol: str
    timeframe: str
    bars_loaded: int | None
    audit_bars: int | None
    bars_match: bool | None
    raw_sharpe: float | None            # best_sharpe
    dsr_within: float | None
    dsr_across: float | None
    pbo: float | None
    pbo_n_splits: int | None            # must be 12870 at n_blocks=16
    p_oos_neg: float | None             # P(OOS<0)
    vs_random_p_max: float | None
    vs_random_p_single: float | None    # diagnostic only
    vs_random_effect_size: float | None
    minbtl_required_years: float | None
    history_years: float | None
    minbtl_satisfied: bool | None
    survivors: int | None
    distinct_dsr_count: int | None
    crashes: int | None
    exit_verdict: str | None
    stale: bool = False                  # carried over from an earlier run
    error: str | None = None             # the cell failed; no metrics exist


class GateStrip(BaseModel):
    """The validation gate strip. Source of truth for 6C-bis."""
    available: bool
    cells: list[GateCell]
    source: str
    note: str = ""
    # Which run these cells came from. Without this an earlier campaign's
    # numbers display identically to the campaign the operator just launched.
    is_current_job: bool = False
    file_written_utc: str | None = None
    stale_cells: int = 0


def _gate_cell_from_report(r: dict) -> GateCell:
    funnel = r.get("funnel") or {}
    by_cat = funnel.get("by_category") or {}
    vsr = r.get("vs_random") or {}
    return GateCell(
        symbol=str(r.get("symbol", "")),
        timeframe=str(r.get("timeframe", "")),
        bars_loaded=_as_int(r.get("bars_loaded")),
        audit_bars=_as_int(r.get("audit_bars")),
        bars_match=_as_bool(r.get("bars_match_audit")),
        raw_sharpe=_as_float(r.get("best_sharpe")),
        dsr_within=_as_float(r.get("dsr_within")),
        dsr_across=_as_float(r.get("dsr_across")),
        pbo=_as_float(r.get("pbo")),
        pbo_n_splits=_as_int(r.get("pbo_n_splits")),
        p_oos_neg=None,  # not yet computed by the runner; surfaced as None, not 0
        vs_random_p_max=_as_float(vsr.get("p_max")),
        vs_random_p_single=_as_float(vsr.get("p_single")),
        vs_random_effect_size=_as_float(vsr.get("effect_size")),
        minbtl_required_years=_as_float(r.get("minbtl_required_years")),
        history_years=_as_float(r.get("audit_years")),
        minbtl_satisfied=_as_bool(r.get("minbtl_satisfied")),
        survivors=_as_int(by_cat.get("VAULTED")),
        distinct_dsr_count=_as_int(r.get("distinct_dsr_count")),
        crashes=_as_int(r.get("n_crashes")),
        exit_verdict=r.get("exit_verdict"),
        stale=bool(r.get("stale")),
        error=r.get("error"),
    )


def _as_float(v: object) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return f if f == f else None  # NaN -> None


def _as_int(v: object) -> int | None:
    if v is None:
        return None
    try:
        return int(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _as_bool(v: object) -> bool | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        if v.lower() in {"true", "1", "yes"}:
            return True
        if v.lower() in {"false", "0", "no"}:
            return False
    return None



@router.get("/audit", response_model=AuditSummary)
async def audit() -> AuditSummary:
    """Return the target audit — the cells a campaign can actually run.

    Reads target_audit_csv() (Tier 1 subset), NOT audit_csv() (the live 356-symbol
    audit). The catalog this endpoint serves is what /launch validates against,
    so the UI's chip list and the launch gate agree on which cells exist.
    Falls back to the live audit only when the target audit is absent (e.g.
    a dev machine that hasn't run the Tier 1 audit yet).
    """
    source_csv = target_audit_csv() if target_audit_csv().exists() else audit_csv()
    if not source_csv.exists():
        return AuditSummary(
            cells=0, symbols=0, timeframes=[], unavailable=0,
            top_15=[], per_timeframe={}, as_of="", csv_path=str(source_csv),
            rendered_cells=0, cell_catalog=[],
            audit_source="", audit_time_basis="",
        )
    rows: list[dict[str, str]] = []
    audit_header: dict[str, str] = {}
    with source_csv.open(encoding="utf-8") as handle:
        audit_header = _parse_audit_header(handle)
        handle.seek(0)
        rows.extend(_audit_rows(handle))

    # Full catalog: every audited (symbol, timeframe) cell. The UI uses this
    # to build its chip list — no hardcoded cell list anywhere.
    cell_catalog: list[AuditCatalogEntry] = [
        AuditCatalogEntry(
            symbol=r["symbol"],
            timeframe=r["timeframe"],
            years_available=(
                None if r["years_available"] == "UNAVAILABLE"
                else float(r["years_available"])
            ),
            source=r.get("source", "MT5_LIVE"),
        )
        for r in rows
    ]
    cells = len(rows)
    symbols = len({r["symbol"] for r in rows})
    timeframes = sorted({r["timeframe"] for r in rows})
    unavailable = sum(1 for r in rows if r["years_available"] == "UNAVAILABLE")
    avail = [r for r in rows if r["years_available"] != "UNAVAILABLE"]
    avail.sort(key=lambda r: float(r["years_available"]), reverse=True)
    top_15: list[AuditCell] = []
    for r in avail[:15]:
        top_15.append(
            AuditCell(
                symbol=r["symbol"],
                timeframe=r["timeframe"],
                years_available=r["years_available"],
                bars_returned=int(r.get("bars_returned", 0) or 0),
                first_bar_utc=r.get("first_bar_utc", ""),
                last_bar_utc=r.get("last_bar_utc", ""),
                source=r.get("source", "MT5_LIVE"),
            )
        )
    by_tf: dict[str, dict[str, float | int]] = defaultdict(lambda: {"max": 0.0, "median": 0.0, "n_avail": 0, "n_unavail": 0})
    per_tf: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r["years_available"] == "UNAVAILABLE":
            by_tf[r["timeframe"]]["n_unavail"] = int(by_tf[r["timeframe"]]["n_unavail"]) + 1
        else:
            per_tf[r["timeframe"]].append(float(r["years_available"]))
            by_tf[r["timeframe"]]["n_avail"] = int(by_tf[r["timeframe"]]["n_avail"]) + 1
    for tf, ys in per_tf.items():
        ys_sorted = sorted(ys)
        by_tf[tf]["max"] = max(ys)
        by_tf[tf]["median"] = ys_sorted[len(ys_sorted) // 2]
    return AuditSummary(
        cells=cells,
        symbols=symbols,
        timeframes=timeframes,
        unavailable=unavailable,
        top_15=top_15,
        per_timeframe=dict(by_tf),
        as_of=audit_header.get("AS_OF", ""),
        csv_path=str(source_csv),
        rendered_cells=len(top_15),
        cell_catalog=cell_catalog,
        audit_source=audit_header.get("SOURCE", ""),
        audit_time_basis=audit_header.get("TIME_BASIS", ""),
    )


@router.get("/verdict", response_model=VerdictResponse)
async def verdict(job_id: str | None = None) -> VerdictResponse:
    """Per-cell MinBTL verdicts from the runner, with their provenance.

    Every field returned here traces to a report field or to the file's own
    mtime. When ``job_id`` names a job this engine started, ``is_current_job``
    says whether the file on disk is that job's output rather than an older
    run's — the difference between "this campaign's result" and "a result".
    """
    if not runner_json().exists():
        return VerdictResponse(
            source="cached",
            note="No runner result yet. Launch a campaign to compute the verdict.",
            last_runner_status=None,
            last_runner_path=str(runner_json()),
            last_runner_verdict=None,
        )
    try:
        data = json.loads(runner_json().read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return VerdictResponse(
            source="cached",
            note="Runner JSON is corrupt; no verdict can be read from it.",
            last_runner_status=None,
            last_runner_path=str(runner_json()),
            last_runner_verdict=None,
        )
    if not isinstance(data, list):
        data = [data] if isinstance(data, dict) else []

    written = datetime.fromtimestamp(runner_json().stat().st_mtime, tz=UTC).isoformat()

    cells: list[CellVerdict] = []
    for r in data:
        if not isinstance(r, dict):
            continue
        cells.append(
            CellVerdict(
                symbol=str(r.get("symbol", "")),
                timeframe=str(r.get("timeframe", "")),
                minbtl_required_years=_as_float(r.get("minbtl_required_years")),
                history_years=_as_float(r.get("audit_years")),
                minbtl_satisfied=_as_bool(r.get("minbtl_satisfied")),
                exit_verdict=r.get("exit_verdict"),
                completed_at_utc=r.get("completed_at_utc"),
                stale=bool(r.get("stale")),
            )
        )

    is_current = False
    note = "Per-cell MinBTL verdicts from the last runner invocation."
    if job_id:
        job = _get_job(job_id)
        is_current = _feed_is_current(job)
        note = (
            "Per-cell verdicts from job %s (this campaign's output)." % job_id
            if is_current
            else (
                "Job %s has not produced search_runner_real.json; what follows "
                "is an earlier run's file, not this campaign's result." % job_id
            )
        )
    if not cells:
        return VerdictResponse(
            source="last_runner",
            note="Runner file exists but holds no cell reports.",
            last_runner_status=None,
            last_runner_path=str(runner_json()),
            last_runner_verdict=None,
            file_written_utc=written,
            job_id=job_id,
            is_current_job=is_current,
            cells=[],
            stale_cells=0,
        )
    satisfied = sum(1 for c in cells if c.minbtl_satisfied)
    stale_cells = sum(1 for c in cells if c.stale)
    return VerdictResponse(
        source="last_runner",
        note=note,
        # No campaign-level pass is claimed. These two strings summarise what
        # the cells say, and are never rendered as a verdict of SUFFICIENT.
        last_runner_status=(
            f"PER_CELL: {len(cells)} cell(s); MinBTL satisfied in {satisfied}"
        ),
        last_runner_path=str(runner_json()),
        last_runner_verdict=None,
        file_written_utc=written,
        job_id=job_id,
        is_current_job=is_current,
        cells=cells,
        stale_cells=stale_cells,
    )


@router.get("/runner", response_model=RunnerResult)
async def runner() -> RunnerResult:
    if not runner_json().exists():
        return RunnerResult(exists=False, data=None, path=str(runner_json()))
    payload = json.loads(runner_json().read_text(encoding="utf-8"))
    # A truncated or hand-edited artifact must not take the route down; the
    # caller can see `exists` and an empty `data`.
    cell_count = len(payload) if isinstance(payload, list) else 1
    return RunnerResult(
        exists=True,
        data=payload,
        path=str(runner_json()),
        cells=cell_count,
    )


@router.get("/gate", response_model=GateStrip)
async def gate(job_id: str | None = None) -> GateStrip:
    """The validation gate strip — every value traced to a named engine field.
    Missing values surface as None, never fabricated.

    Precedence: the named job's feed, else the newest job that has one, else
    the runner file on disk (marked as an earlier run's output unless no job
    has run in this session). ``is_current_job`` tells the caller which of the
    three it is holding.
    """
    reports: list[dict] = []
    source = "cached"
    is_current = False

    job = _get_job(job_id) if job_id else _most_recent_finished_job()
    if job is not None:
        with job.lock:
            job_reports = list(job.reports or [])
        if job_reports:
            reports = job_reports
            source = f"job:{job.job_id}"
            is_current = True

    if not reports:
        if not runner_json().exists():
            return GateStrip(
                available=False,
                cells=[],
                source="cached",
                note="No campaign result. Launch a campaign to populate the gate.",
            )
        try:
            loaded = json.loads(runner_json().read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return GateStrip(
                available=False, cells=[], source="cached", note="Runner JSON corrupt."
            )
        if not isinstance(loaded, list):
            loaded = [loaded] if isinstance(loaded, dict) else []
        reports = [r for r in loaded if isinstance(r, dict)]
        source = "last_runner"
        # No job in this session produced these; whatever the file's mtime is,
        # this campaign did not write it.
        uploaded_at = (
            datetime.fromtimestamp(runner_json().stat().st_mtime, tz=UTC).isoformat()
        )
        cells = [_gate_cell_from_report(r) for r in reports]
        stale = sum(1 for r in reports if r.get("stale") or "error" in r)
        return GateStrip(
            available=bool(cells),
            cells=cells,
            source=source,
            is_current_job=False,
            file_written_utc=uploaded_at,
            stale_cells=stale,
            note=(
                f"{len(cells)} cell(s) from an earlier run on disk "
                f"({uploaded_at}); no job in this session produced them."
            ),
        )

    cells = [_gate_cell_from_report(r) for r in reports]
    stale = sum(1 for r in reports if r.get("stale") or "error" in r)
    return GateStrip(
        available=True,
        cells=cells,
        source=source,
        is_current_job=is_current,
        file_written_utc=(
            datetime.fromtimestamp(runner_json().stat().st_mtime, tz=UTC).isoformat()
            if runner_json().exists()
            else None
        ),
        stale_cells=stale,
        note=f"{len(cells)} cell(s) from this campaign.",
    )


@router.post("/launch", response_model=LaunchResponse)
async def launch(request: LaunchRequest | None = None) -> LaunchResponse:
    """Submit a campaign and return immediately with a pollable job ID.

    Every cell in `selected` is validated against the history-depth audit. A
    requested cell with no audit row is rejected with 422 — the engine never
    runs a subset of the requested cells and reports success.
    """
    from fastapi import HTTPException

    request = request or LaunchRequest()
    selected = _parse_selected(request.selected)

    # Validate requested cells against the audit: reject, do not trim.
    if selected is not None and target_audit_csv().exists():
        with target_audit_csv().open(encoding="utf-8") as handle:
            audited = {
                (row["symbol"], row["timeframe"])
                for row in _audit_rows(handle)
            }
        unsupported = sorted(
            f"{symbol}:{timeframe}"
            for symbol, timeframe in sorted(selected)
            if (symbol, timeframe) not in audited
        )
        if unsupported:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "unsupported cells",
                    "unsupported": unsupported,
                    "reason": (
                        "These cells have no row in the history-depth audit. "
                        "They cannot be run."
                    ),
                },
            )

    # Refuse any cell whose symbol has no usable spread matrix. Checked
    # after the audit gate so the operator gets the more fundamental
    # complaint first.
    if selected is not None:
        uncalibrated = []
        for symbol, timeframe in sorted(selected):
            reason = spread_blocked_reason(symbol)
            if reason is not None:
                uncalibrated.append(
                    {"cell": f"{symbol}:{timeframe}", "reason": reason}
                )
        if uncalibrated:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": SPREAD_NOT_CALIBRATED,
                    "blocked": uncalibrated,
                    "reason": (
                        "These symbols have no usable calibrated spread "
                        "matrix. A campaign costed from a missing matrix is "
                        "costed from nothing."
                    ),
                },
            )

    job_id = uuid.uuid4().hex
    job = _Job(job_id, request.population, selected)
    with _JOBS_LOCK:
        _JOBS[job_id] = job
    _EXECUTOR.submit(_run_job, job)
    log.info("S5 job submitted: id={} population={}", job_id, request.population)
    return LaunchResponse(job_id=job_id, status="QUEUED")


@router.get("/jobs/{job_id}", response_model=JobStatus)
async def job_status(job_id: str) -> JobStatus:
    """Return actual campaign progress and the final report when complete."""
    job = _get_job(job_id)
    if job is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="S5 job not found")
    return _job_status(job)


@router.get("/jobs/{job_id}/survivors", response_model=list[SurvivorRecord])
async def job_survivors(job_id: str) -> list[SurvivorRecord]:
    """Per-DNA survivor records (VAULTED outcome) for a finished campaign.

    Queries the per-cell TrialRegistry SQLite stores. Only trials whose
    outcome is VAULTED are returned; the trial count, Sharpe, DSR and other
    per-candidate metrics come straight from the registry — no invented rows.
    """
    job = _get_job(job_id)
    if job is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="S5 job not found")

    with job.lock:
        reports = list(job.reports or [])
    if not reports:
        return []

    from engine.validation.trial_registry import TrialRegistry

    survivors: list[SurvivorRecord] = []
    for report in reports:
        if not isinstance(report, dict) or "campaign_id" not in report:
            continue
        campaign_id = str(report["campaign_id"])
        symbol = str(report.get("symbol", ""))
        timeframe = str(report.get("timeframe", ""))
        symbol_tf = f"{symbol}:{timeframe}" if symbol and timeframe else ""

        reg_path = artifact(f"{campaign_id}.sqlite")
        if not reg_path.exists():
            continue

        registry = TrialRegistry(reg_path)
        trials = registry.list_trials(
            campaign_id, outcome="VAULTED"
        )
        for t in trials:
            survivors.append(
                SurvivorRecord(
                    symbol=symbol,
                    timeframe=timeframe,
                    dna_hash=t.dna_hash,
                    sharpe=t.sharpe,
                    trades=t.n_trades,
                    total_return=t.total_return,
                    max_dd=t.max_dd,
                    dsr=t.dsr,
                    outcome=t.outcome,
                    created_at=t.created_at,
                )
            )

    return survivors


__all__ = ["router"]
