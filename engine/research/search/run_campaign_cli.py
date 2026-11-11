# ruff: noqa: N813
"""S5 rerun search runner - real MT5 data, full DSR gate, zero crashes.

Defect fixes vs the previous version:
    D1: bars_loaded must equal the audit's bar_count; the runner asserts
        this and fails loudly otherwise.
    D2: per-cell DSR is wired into the funnel (FailedDSR, DsrUnreliable);
        campaign-level PBO is computed (HighPBO).
    D3: MinBTL uses the S5-calibrated formula
        (engine.validation.minbtl.min_backtest_length).
    D4: zero crashes. The synthetic backtester is exception-free.

Cells (S5 rerun scope):
    USDCADm D1   3,915 bars over 12.67y
    USDCADm H4  16,281 bars over 12.67y
    GBPJPYm D1  10,122 bars over 33.41y
    5,000 trials per cell
    target_is_annual_sharpe = 1.5
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import MetaTrader5 as mt5

from engine.research.campaign.budget import CampaignScope
from engine.research.campaign.preflight import (
    refuse_on_basis_mismatch,
    spread_preflight,
)
from engine.research.provenance import DataSource, Provenance
from engine.research.search.canary import run_planted_edge_canary
from engine.data.spread_calibration import CALIBRATOR_VERSION
from engine.research.search.dna import DNA_SCHEMA_VERSION, DNA, FailureTaxonomy
from engine.research.search.effective_hash import EFFECTIVE_HASH_VERSION
from engine.research.search.gene_pool import GenePool
from engine.research.search.gene_pool import summary as gene_pool_summary
from engine.research.search.real_backtest import (
    fast_backtest_factory,
    load_cell_bars,
    load_cell_bars_from_snapshot,
    real_backtest_factory,
    real_scores,
)
from engine.research.search.sham_control import (
    CONTROLS_REGIME,
    controls_block,
    sham_control_factory,
)
from engine.core.errors import ConfigError
from engine.core.paths import get_arqvio_home, get_artifacts_dir
from engine.config.settings import get_settings
from engine.core.git_sha import resolve_git_sha
from engine.research.campaign.manifest import (
    Manifest,
    config_hash,
    dataset_provenance_for,
    grade_for,
)
from engine.research.search.runner import TrialResult, run_campaign
from engine.validation.minbtl import min_backtest_length
from engine.validation.trial_registry import TrialRegistry

RUNNER_JSON_NAME = "search_runner_real.json"
TARGET_AUDIT_CSV_NAME = "history_depth_audit_target.csv"


def artifact_path(*parts: str) -> Path:
    """ARQVIO_HOME/artifacts/<parts>, resolved at CALL time (CLAUDE.md s7).

    Never at import: a module constant freezes the home of whoever imported
    first, so an ARQVIO_HOME set later (isolated_arqvio_home) is ignored and a
    campaign run under an isolated home writes into the real one. The
    resolved path (``..`` collapsed, symlinks and junctions followed) must
    lie inside ARQVIO_HOME and inside its artifacts dir, else ConfigError.
    """
    home = get_arqvio_home()
    root = get_artifacts_dir()
    path = root.joinpath(*parts)
    resolved = path.resolve()
    if not resolved.is_relative_to(home.resolve()):
        raise ConfigError(f"artifact path {path} escapes ARQVIO_HOME {home}")
    if not resolved.is_relative_to(root.resolve()):
        raise ConfigError(f"artifact path {path} escapes the artifacts dir {root}")
    return path


def runner_json_path() -> Path:
    """Where run_campaign_sync writes (and merges) the per-cell reports."""
    return artifact_path(RUNNER_JSON_NAME)

# The target annual Sharpe ratio every cell targets. Must match
# routes_s5.CAMPAIGN_TARGET_SHARPE so the preflight verdict and the
# actual campaign agree on what "satisfied" means.
CAMPAIGN_TARGET_SHARPE = 1.5


def resolve_engine_mode(engine_mode: str, dataset_id: str | None) -> str:
    """Resolve the ``engine_mode`` argument to a concrete engine.

    ``"auto"`` engages the fast kernel only on a PINNED dataset
    (``dataset_id`` is set), because pinned data is reproducible and the
    fast kernel has parity proof against it. Live data slides with
    wall-clock time and is not reproducible, so ``"auto"`` falls back to
    the slow engine there -- the kernel itself is method-proven, but fast
    is only engaged where its output can be compared against a stable
    baseline. Explicit ``"slow"`` / ``"fast"`` values pass through
    unchanged.
    """
    if engine_mode == "auto":
        return "fast" if dataset_id else "slow"
    return engine_mode


def load_target_audit() -> tuple[dict[str, str], dict[tuple[str, str], dict[str, Any]]]:
    """Load the target audit CSV and its header metadata.

    Returns ``(header, rows)`` where ``header`` maps the ``#``-prefixed
    keys emitted by scripts/history_depth_audit_target.py (``AUDIT_CELL_SET``,
    ``CELLS``, ``AS_OF``, ``SOURCE``, ``TIME_BASIS``) to their string values,
    and ``rows`` maps ``(symbol, timeframe)`` to its DictReader row.

    The header travels with every cell report so consumers can validate that
    a requested cell was in the audited set (not silently trimmed) and detect
    a stale CSV by its AS_OF / SOURCE instead of inferring it from a missing
    dataset_id.

    A QUICK cell added in the app (engine/research/cells/add_market.py) carries
    its own row -- the pull's measurement -- and it joins ``rows`` for a
    (symbol, timeframe) the CSV does not hold; the CSV's own rows always win.
    """
    from engine.research.cells.add_market import quick_audit_rows

    audit_csv = artifact_path(TARGET_AUDIT_CSV_NAME)
    quick = quick_audit_rows()
    if not audit_csv.exists():
        if quick:
            return {}, quick
        raise FileNotFoundError(
            f"{audit_csv} not found; run scripts/history_depth_audit_target.py first"
        )
    header: dict[str, str] = {}
    data_rows: list[str] = []
    with audit_csv.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.lstrip()
            if stripped.startswith("#"):
                parts = stripped.lstrip("# ").split(" ", 1)
                if len(parts) == 2:
                    # The value ends where the line does: drop the terminator.
                    header[parts[0]] = parts[1].rstrip("\r\n")
            else:
                data_rows.append(line)
    reader = csv.DictReader(data_rows)
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in reader:
        key = (row["symbol"], row["timeframe"])
        rows[key] = row
    for key, row in quick.items():
        rows.setdefault(key, row)
    return header, rows


# The floor a cell needs to be worth running at all. MIN_OBS_FOR_DSR is
# the validation layer's own bar for a usable Deflated Sharpe, so a
# snapshot below it cannot produce a verdict whatever else is true.
MIN_SNAPSHOT_OBS = 100


def snapshot_depth_check(dataset_id: str, n_obs: int) -> tuple[bool, str]:
    """Is this pinned snapshot deep enough to run, per its own sidecar?

    Reads the snapshot's recorded window rather than the audit CSV. The
    CSV is itself a sliding-window artifact and cannot describe a fixed
    one (G14).
    """
    from engine.research.data.dataset_snapshot import load_meta

    meta = load_meta(dataset_id)
    if meta is None:
        return False, f"snapshot {dataset_id} has no sidecar metadata"
    if meta.n_bars != n_obs + 1:
        return False, (
            f"sidecar says {meta.n_bars} bars but {n_obs + 1} were loaded; "
            "the snapshot and its metadata disagree"
        )
    if n_obs < MIN_SNAPSHOT_OBS:
        return False, (
            f"{n_obs} returns is below the {MIN_SNAPSHOT_OBS} needed for a "
            "usable DSR"
        )
    return True, (
        f"PINNED {meta.n_bars} bars, {meta.start_utc[:10]} -> "
        f"{meta.end_utc[:10]} (sidecar), {n_obs} returns"
    )


class SpreadBlockedError(RuntimeError):
    """A symbol refused at the spread gate. Carries the report fragment."""

    def __init__(self, symbol: str, timeframe: str, fragment: dict) -> None:
        super().__init__(
            f"SPREAD_{fragment['status']}: {symbol} {timeframe} - "
            f"{fragment['reason']}"
        )
        self.symbol = symbol
        self.timeframe = timeframe
        self.fragment = fragment


def campaign_config_for(
    *,
    symbol: str,
    timeframe: str,
    population: int,
    target_sharpe: float,
    seed: int,
    dataset_id: str | None,
    gene_pool: GenePool | None,
    min_trades: int = 30,
    ga: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The configuration a cell run is launched with; config_hash hashes it.

    B3: the registry has had a config_hash column since S5 and nothing ever
    filled it, so two differently-configured runs were indistinguishable in
    provenance.

    ``ga`` is a GA campaign's hashed block (engine.research.ga.config
    GAConfig.block); ``population`` is then the per-generation size and
    ``min_trades`` the GA's own gate, so the config cannot contradict itself.
    """
    config: dict[str, Any] = {
        "symbol": symbol,
        "timeframe": timeframe,
        "population": population,
        "target_is_annual_sharpe": target_sharpe,
        "seed": seed,
        # IN the hash: which bars were used is part of the
        # configuration. Two runs on different data are not the same
        # experiment, and until G14 nothing recorded the difference.
        "dataset": dataset_id or "LIVE",
        "min_trades": min_trades,
        "commission_per_lot_per_side": (
            get_settings().commission_per_lot_per_side
        ),
        "max_spread_gate_percentile": (
            get_settings().max_spread_gate_percentile
        ),
        "slippage_cap_points_fx": get_settings().slippage_cap_points_fx,
        "slippage_cap_points_metals": (
            get_settings().slippage_cap_points_metals
        ),
    }
    # Absent rather than null without a pool, so configurations launched
    # before gene pools existed keep the hash they were recorded under. A
    # pool (even the default one) changes the draw, so it must change it.
    if gene_pool is not None:
        config["gene_pool"] = gene_pool.canonical()
    # Absent for a random search, by the same rule, so its hash is unchanged.
    if ga is not None:
        config["ga"] = ga
    return config


def run_cell(
    *,
    symbol: str,
    timeframe: str,
    population: int,
    target_sharpe: float,
    seed: int,
    dataset_id: str | None = None,
    engine_mode: str = "auto",
    flat_cost: bool = False,
    on_progress: Callable[[int, int], None] | None = None,
    gene_pool: GenePool | None = None,
) -> dict[str, Any]:
    print(f"\n=== {symbol} {timeframe}  pop={population}  target_SR={target_sharpe} ===")

    # Spread gate, before anything expensive. A blocked symbol cannot
    # enter a campaign, and the reason travels in the REPORT, not just
    # in this line.
    spread_dir = artifact_path("spread")
    spread = spread_preflight(symbol, spread_dir)
    print(f"  spread: {spread['status']} - {spread['reason']}")
    if spread["blocks"]:
        raise SpreadBlockedError(symbol, timeframe, spread)
    audit_header, audit = load_target_audit()
    audit_row = audit[(symbol, timeframe)]
    audit_bars = int(audit_row["bars_returned"])
    audit_years = float(audit_row["years_available"])
    print(f"  audit: bars={audit_bars}  years={audit_years:.4f}")

    # C0: the spread artifact's time_basis must match the audit cell's.
    # Mismatched bases bucket hours in different references and silently
    # corrupt the cost model. refuse_on_basis_mismatch raises PreflightError
    # on conflict; equivalent classes {"utc","server_equals_utc"} pass.
    from engine.data.spread_calibration import load as load_matrix

    audit_basis = audit_header.get("TIME_BASIS", "")
    cell_label = f"{symbol}:{timeframe}"
    audit_row_basis = audit_row.get("time_basis", "") or audit_basis
    matrix = load_matrix(symbol, spread_dir)
    spread_artifact = spread_dir / f"{symbol}.json"
    spread_artifact_sha256 = (
        hashlib.sha256(spread_artifact.read_bytes()).hexdigest()
        if matrix is not None else None
    )
    if matrix is not None:
        refuse_on_basis_mismatch(matrix, audit_row_basis, cell_label)
    else:
        # No artifact loaded; spread_preflight already reported UNAVAILABLE
        # and does not block. The basis gate cannot fire — there is no
        # artifact to compare — so we record it in the report instead.
        spread["basis_checked"] = False
        spread["basis_reason"] = "no spread artifact to compare"
    print(f"  basis: audit={audit_row_basis!r} artifact={matrix.time_basis if matrix else 'N/A'}")

    # Fix 2: use real_backtest_factory — this is the only path that sets
    # per_period_returns on TrialResult, which PBO requires.
    # honest_synthetic_backtest never sets per_period_returns, so PBO was
    # always None regardless of trial count.
    # G14. A pinned snapshot is reproducible; the live feed is not --
    # copy_rates_from_pos returns the most recent N bars, so its window
    # slides with wall-clock time and the same run tomorrow uses
    # different data. Both are allowed; only one is recorded as
    # reproducible, and the report says which.
    if dataset_id:
        cell = load_cell_bars_from_snapshot(symbol, timeframe, dataset_id)
        dataset_reproducible = True
        print(f"  dataset: PINNED {dataset_id} ({cell.n_bars} bars)")
    else:
        cell = load_cell_bars(
            symbol=symbol,
            timeframe=timeframe,
            expected_bars=audit_bars,
        )
        dataset_reproducible = False
        print(
            "  dataset: LIVE (window slides with wall-clock time; this run "
            "is NOT reproducible -- see G14)"
        )
    bars_loaded = cell.n_bars
    n_obs = cell.n_returns
    returns = cell.returns  # noqa: F841  (used via cell in real_backtest_factory)

    if n_obs < 100:
        raise RuntimeError(
            f"D1 violation: only {n_obs} returns pulled for {symbol} {timeframe}"
        )
    # G14/R1. A pinned run is checked against its OWN sidecar, not
    # against history_depth_audit_target.csv. The audit CSV was built
    # from live most-recent-N pulls, so it describes a sliding window
    # and can never agree with a fixed one -- comparing them printed
    # match_audit=NO on every pinned run, which reads as a defect and
    # is not one. The question the check actually needs to answer is
    # "are there enough bars to run this cell", and the sidecar answers
    # it for exactly the bars that will be used.
    #
    # The full audit redesign is S3-W0; this is the bounded fix.
    if dataset_id:
        depth_ok, depth_note = snapshot_depth_check(dataset_id, n_obs)
        print(f"  depth: {depth_note}")
        if not depth_ok:
            raise RuntimeError(
                f"snapshot {dataset_id} is too shallow for {symbol} "
                f"{timeframe}: {depth_note}"
            )
    else:
        print(f"  pulled: returns={n_obs}  bars_loaded={bars_loaded}  "
              f"match_audit={'YES' if bars_loaded == audit_bars else 'NO'}")

    # Preflight: run the fast planted-edge canary on THIS cell's bars before
    # trusting any verdict from it. If it fails, the cell result is void.
    canary = run_planted_edge_canary(cell, fast=True, tmp_dir=None)
    print(f"  canary: passed={canary.passed}  verdict={canary.exit_verdict[:40]!r}  "
          f"dsr_within={canary.dsr_within}  p_max={canary.vs_random_p_max}")
    if not canary.passed:
        raise RuntimeError(
            f"CANARY FAILED for {symbol} {timeframe}: {canary.reason}. "
            "Verdict VOID — pipeline cannot detect a planted edge on this cell."
        )

    min_btl = min_backtest_length(
        n_trials=population,
        target_sharpe=target_sharpe,
        history_years=audit_years,
    )
    print(f"  MinBTL @ SR={target_sharpe}: {min_btl.min_years:.2f}y required, "
          f"{audit_years:.2f}y available, satisfied={min_btl.satisfied}")

    campaign_id = f"rerun-{symbol}-{timeframe}".lower()
    reg_path = artifact_path(f"{campaign_id}.sqlite")
    # A stale registry must be GONE, not "best effort". TrialRegistry opens
    # whatever file is there, and registry.stats() would then count a
    # previous campaign's rows into this cell's funnel and its DSR n_trials.
    for suffix in ("", "-wal", "-shm"):
        stale = artifact_path(f"{campaign_id}.sqlite{suffix}")
        if not stale.exists():
            continue
        try:
            stale.unlink()
        except OSError as exc:
            raise RuntimeError(
                f"{stale} could not be removed ({exc}); refusing to run "
                f"{symbol} {timeframe} against a registry that still carries "
                "rows from an earlier campaign"
            ) from exc
        if stale.exists():
            raise RuntimeError(f"{stale} still present after unlink()")
    print(f"  registry: {reg_path.name} clean "
          f"(exists={reg_path.exists()})")
    registry = TrialRegistry(reg_path)
    scope = CampaignScope(
        symbols=(symbol,),
        timeframes=(timeframe,),
        population_per_cell=population,
    )
    # MT5_LIVE for BOTH modes, and that is not a fudge. This value ends
    # up in the trial registry, whose source vocabulary lives in the
    # frozen engine/validation/trial_registry.py (_VALID_SOURCES) and
    # cannot gain a member. It answers "where did these bars originate",
    # and a snapshot originates from MT5 exactly as a live pull does.
    #
    # WHICH bars is answered by dataset_hash, which is now the content
    # digest rather than the old symbol-timeframe-count label, so the
    # registry does record the pinning. Whether the run is reproducible
    # is answered by dataset_reproducible in the report.
    provenance = Provenance(
        source=DataSource.MT5_LIVE,
        dataset_hash=cell.dataset_hash,
        as_of=datetime.now(tz=UTC),
        mixed_source=False,
    )

    campaign_config = campaign_config_for(
        symbol=symbol,
        timeframe=timeframe,
        population=population,
        target_sharpe=target_sharpe,
        seed=seed,
        dataset_id=dataset_id,
        gene_pool=gene_pool,
    )
    cfg_hash = config_hash(campaign_config)

    # engine_mode is deliberately NOT in campaign_config above, so
    # config_hash stays comparable across modes: the same experiment
    # run two ways is the same experiment. It is recorded as
    # informational output instead.
    engine_mode_actual = resolve_engine_mode(engine_mode, dataset_id)
    if engine_mode_actual == "fast":
        backtest_fn = fast_backtest_factory(cell, flat_cost=flat_cost)
    else:
        backtest_fn = real_backtest_factory(cell, flat_cost=flat_cost)
    print(f"  engine: {engine_mode} -> {engine_mode_actual}")
    # The control is a sham per real DNA: same entry count, random bars and
    # sides, the DNA's own risk and exit genes, on the slow engine under the
    # same costs and gates. Its draws are seeded per DNA, not from `seed`.
    control_fn = sham_control_factory(cell, flat_cost=flat_cost)

    min_trades_used = 30  # run_campaign's default; the eligibility bar
    started = time.time()
    try:
        result = run_campaign(
            campaign_id=campaign_id,
            scope=scope,
            registry=registry,
            backtest_fn=backtest_fn,
            control_fn=control_fn,
            real_scores_factory=real_scores,
            provenance=provenance,
            target_is_annual_sharpe=target_sharpe,
            dataset_hash=provenance.dataset_hash,
            # Without this every registry row was stamped git_sha="unknown",
            # which makes the provenance column present and useless.
            git_sha=resolve_git_sha(),
            config_hash=cfg_hash,
            seed=seed,
            n_obs=n_obs,
            bars_loaded=bars_loaded,
            audit_bars_expected=audit_bars,
            on_progress=on_progress,
            gene_pool=gene_pool,
        )
    finally:
        import gc
        gc.collect()  # release any lingering SQLite handles
    elapsed = time.time() - started

    s = registry.stats(campaign_id)
    report: dict[str, Any] = {
        "campaign_id": campaign_id,
        "symbol": symbol,
        "timeframe": timeframe,
        "population": population,
        "target_is_annual_sharpe": target_sharpe,
        "audit_bars": audit_bars,
        "audit_years": audit_years,
        "bars_loaded": bars_loaded,
        "n_obs_real": n_obs,
        "bars_match_audit": bars_loaded == audit_bars,
        # G14: audit header metadata from history_depth_audit_target.csv.
        # Lets consumers validate that a requested cell was in the audited
        # set (not silently trimmed) and detect a stale CSV by its AS_OF
        # / SOURCE rather than inferring from a missing dataset_id.
        "audit_source": audit_header.get("SOURCE", ""),
        "audit_time_basis": audit_header.get("TIME_BASIS", ""),
        "audit_as_of": audit_header.get("AS_OF", ""),
        "audit_cell_set": audit_header.get("AUDIT_CELL_SET", ""),
        "spread_check": spread,
        "minbtl_required_years": float(min_btl.min_years),
        "minbtl_satisfied": bool(min_btl.satisfied),
        "funnel": {
            "total": result.funnel.total,
            "by_category": dict(result.funnel.by_category),
            "trades_by_bucket": dict(result.trades_by_bucket),
        },
        "distinct_dsr_count": result.distinct_dsr_count,
        "blocked_zero_spread_events": result.blocked_zero_spread_events,
        "deferred_zero_spread_fills": result.deferred_zero_spread_fills,
        "sltp_deferred_zero_spread": result.sltp_deferred_zero_spread,
        "exit_reasons": dict(result.exit_reasons),
        "parity_classes": dict(result.parity_classes),
        # W0: stated, not inferred. A reader must know which spread cost
        # model produced the fills and that it has not drifted between
        # runs. spread_cost_source is "mean_buckets" (a calibrated
        # SpreadMatrix), "flat" (spec.spread_in_price repeated), or
        # "none". cost_vector_hash is a 16-hex digest of the 24-hour cost
        # vector the kernel charged; a mismatch between two runs with
        # the same dataset_hash means the cost model changed, not the data.
        "spread_cost_source": getattr(backtest_fn, "spread_cost_source", "none"),
        "cost_vector_hash": getattr(backtest_fn, "cost_vector_hash", ""),
        "flat_cost_mode": flat_cost,
        # The exact spread artifact the cost model was built from.
        "spread_artifact_sha256": spread_artifact_sha256,
        "cost_totals": dict(result.cost_totals),
        "degeneracy": result.degeneracy,
        "schema_versions": {
            "dna": DNA_SCHEMA_VERSION,
            "effective_hash": EFFECTIVE_HASH_VERSION,
            "spread_calibrator": CALIBRATOR_VERSION,
        },
        # G14: stated, not inferred. A reader must not have to know
        # that a missing dataset id means the window moved.
        "engine_mode_configured": engine_mode,
        "engine_mode_actual": engine_mode_actual,
        "fast_fallback_count": int(
            getattr(backtest_fn, "fallback_state", {}).get("fallbacks", 0)
        ),
        "fast_fallback_reasons": list(
            getattr(backtest_fn, "fallback_state", {}).get("reasons", [])
        ),
        "dataset_id": dataset_id,
        "dataset_reproducible": dataset_reproducible,
        "dataset_warning": (
            None if dataset_reproducible else
            "LIVE DATASET: the bar window slides with wall-clock time, so "
            "this result cannot be reproduced. Pin one with "
            "scripts/snapshot_dataset.py and pass dataset=<id>."
        ),
        "manifest": Manifest(
            campaign_id=campaign_id,
            symbol=symbol,
            timeframe=timeframe,
            dataset_hash=provenance.dataset_hash,
            dataset_source=provenance.source.value,
            dataset_provenance=dataset_provenance_for(dataset_id),
            bars_loaded=int(bars_loaded),
            date_range_utc=(
                datetime.fromtimestamp(
                    int(cell.bars.time[0]), tz=UTC
                ).isoformat(),
                datetime.fromtimestamp(
                    int(cell.bars.time[-1]), tz=UTC
                ).isoformat(),
            ) if cell.bars.n_bars else None,
            git_sha=resolve_git_sha(),
            config_hash=cfg_hash,
            gene_pool=gene_pool_summary(gene_pool),
            seed=int(seed),
            completed_at_utc=datetime.now(tz=UTC).isoformat(),
        ).to_dict(),
        # C4: the one sanctioned rule -- a winner inside the random
        # range is capped at WEAK regardless of its Sharpe.
        "grade": grade_for(
            vs_random_p_max=(
                result.vs_random_eligible.p_max
                if result.vs_random_eligible else None
            ),
        ),
        "trials_raw": result.trials_raw,
        "trials_effective": result.trials_effective,
        "dsr_N_used": result.dsr_N_used,
        "canary_passed": canary.passed,
        "canary_verdict": canary.exit_verdict,
        "canary_control": canary.control,
        "controls": controls_block(
            control_fn,
            vs_all=result.vs_random,
            vs_eligible=result.vs_random_eligible,
        ),
        "controls_regime": CONTROLS_REGIME,
        # "dsr_within" / "best_sharpe" / "vs_random" are the ALL-candidate
        # values and keep their names: engine/api/routes_s5.py reads those
        # keys. The *_eligible companions are the same statistics over
        # candidates that cleared min_trades. Read them together or not at
        # all - on 2026-09-18 cell 1's all-candidate trio belonged to one
        # 23-trade candidate the funnel had already rejected.
        "n_eligible": result.n_eligible,
        "dsr_within": result.dsr_within,
        "dsr_within_all": result.dsr_within,
        "dsr_within_status": result.dsr_within_status,
        "dsr_within_eligible": result.dsr_within_eligible,
        "dsr_within_eligible_status": result.dsr_within_eligible_status,
        "dsr_eligible_winner_sr_per_bar": result.dsr_eligible_winner_sr_per_bar,
        "dsr_eligible_winner_sr_annualised": (
            result.dsr_eligible_winner_sr_annualised
        ),
        "dsr_eligible_sr_star_per_bar": result.dsr_eligible_sr_star_per_bar,
        "dsr_eligible_sr_star_annualised": result.dsr_eligible_sr_star_annualised,
        "dsr_across": result.dsr_across,
        "dsr_across_status": result.dsr_across_status,
        "dsr_across_n_cells": result.dsr_across_n_cells,
        # DSR is a probability of clearing a bar that moves with n_trials.
        # Without the candidate's own Sharpe and that bar, the probability
        # cannot be interpreted or reproduced.
        "dsr_winner_sr_per_bar": result.dsr_winner_sr_per_bar,
        "dsr_winner_sr_annualised": result.dsr_winner_sr_annualised,
        "dsr_sr_star_per_bar": result.dsr_sr_star_per_bar,
        "dsr_sr_star_annualised": result.dsr_sr_star_annualised,
        "pbo": result.pbo,
        "pbo_n_splits": result.pbo_n_splits,
        "best_sharpe": result.best_sharpe,
        "best_dna_hash": result.best_dna.dna_hash if result.best_dna else None,
        "best_sharpe_eligible": result.best_sharpe_eligible,
        "best_dna_eligible_hash": (
            result.best_dna_eligible.dna_hash if result.best_dna_eligible else None
        ),
        "vs_random": None,
        "vs_random_eligible": None,
        "exit_verdict": result.exit_verdict(),
        "n_trials_search": s.n_trials,
        "n_validation_runs": s.validation_runs,
        "n_unknown_rows": s.unknown_rows,
        "n_crashes": len(result.crashes),
        "crash_summary": [
            {"type": c.exception_type, "message": c.message[:120]}
            for c in result.crashes[:5]
        ],
        "elapsed_seconds": elapsed,
        # When this report was actually computed. Without it a cached
        # search_runner_real.json cannot be told apart from the current job's
        # result, and the UI has no way to say which one it is showing.
        "completed_at_utc": datetime.now(tz=UTC).isoformat(),
    }
    for key, v in (
        ("vs_random", result.vs_random),
        ("vs_random_eligible", result.vs_random_eligible),
    ):
        if v is None:
            continue
        report[key] = {
            "p_max": float(v.p_max),
            "p_single": float(v.p_single),
            "p_mean": float(v.p_mean),
            "effect_size": float(v.effect_size),
            "verdict": v.verdict,
        }
    print(f"  elapsed: {elapsed:.1f}s")
    print(f"  funnel: {result.funnel.by_category}")
    print(f"  eligible (n_trades >= {min_trades_used}): {result.n_eligible}"
          f" of {result.n_trials}")
    print(f"  dsr_within ALL     : {result.dsr_within} "
          f"({result.dsr_within_status})")
    print(f"  dsr_within ELIGIBLE: {result.dsr_within_eligible} "
          f"({result.dsr_within_eligible_status})")
    if result.dsr_winner_sr_per_bar is not None:
        print(f"    winner SR: {result.dsr_winner_sr_per_bar:.6f}/bar  "
              f"{result.dsr_winner_sr_annualised:.4f} annualised")
    if result.dsr_sr_star_per_bar is not None:
        print(f"    SR* bar  : {result.dsr_sr_star_per_bar:.6f}/bar  "
              f"{result.dsr_sr_star_annualised:.4f} annualised  "
              f"(n_trials={population})")
    if result.dsr_eligible_winner_sr_per_bar is not None:
        print(f"    eligible winner SR: "
              f"{result.dsr_eligible_winner_sr_per_bar:.6f}/bar  "
              f"{result.dsr_eligible_winner_sr_annualised:.4f} annualised")
    if result.dsr_eligible_sr_star_per_bar is not None:
        print(f"    eligible SR* bar  : "
              f"{result.dsr_eligible_sr_star_per_bar:.6f}/bar  "
              f"{result.dsr_eligible_sr_star_annualised:.4f} annualised")
    print(f"  best_sharpe ALL={result.best_sharpe} "
          f"ELIGIBLE={result.best_sharpe_eligible}")
    print(f"  pbo: {result.pbo}  splits={result.pbo_n_splits}")
    for label, v in (("ALL     ", result.vs_random),
                     ("ELIGIBLE", result.vs_random_eligible)):
        if v:
            print(f"  vs_random {label}: p_max={v.p_max:.4f}  "
                  f"p_single={v.p_single:.4f}  p_mean={v.p_mean:.4f}  "
                  f"effect={v.effect_size:.3f}  {v.verdict}")
    print(f"  crashes: {len(result.crashes)}")
    print(f"  exit: {result.exit_verdict()}")
    return report


def run_campaign_sync(
    *,
    population: int = 5000,
    selected: set[tuple[str, str]] | None = None,
    datasets: dict[tuple[str, str], str] | None = None,
    engine_mode: str = "auto",
    on_progress: Callable[[int, int], None] | None = None,
    gene_pool: GenePool | None = None,
) -> list[dict[str, Any]]:
    """Run real S5 cells in the engine process and report actual progress."""
    if population < 1:
        raise ValueError("population must be at least 1")
    artifact_path().mkdir(parents=True, exist_ok=True)
    # Load before starting any cell so a missing audit fails the job clearly.
    _audit_header, audit = load_target_audit()

    # Cells come from the audit, not a hardcoded list. When `selected` is
    # provided it filters the audit; any requested cell with no audit row is
    # rejected — never silently dropped. Each cell uses the campaign's target
    # Sharpe and gets a deterministic seed by its position in sorted order.
    if selected is not None:
        missing = sorted(selected - set(audit.keys()))
        if missing:
            raise ValueError(f"unknown cells: {sorted(missing)}")
        cell_keys = sorted(selected)
    else:
        cell_keys = sorted(audit.keys())

    cells = [
        (symbol, timeframe, population, CAMPAIGN_TARGET_SHARPE, idx)
        for idx, (symbol, timeframe) in enumerate(cell_keys)
    ]

    total = len(cells) * population
    if on_progress is not None:
        on_progress(0, total)
    reports: list[dict[str, Any]] = []
    for cell_index, (symbol, timeframe, pop, target_sharpe, seed) in enumerate(cells):
        offset = cell_index * population

        def cell_progress(done: int, cell_total: int, *, offset: int = offset) -> None:
            if on_progress is not None:
                on_progress(min(total, offset + done), total)

        try:
            reports.append(run_cell(
                symbol=symbol,
                timeframe=timeframe,
                population=pop,
                target_sharpe=target_sharpe,
                seed=seed,
                dataset_id=(datasets or {}).get((symbol, timeframe)),
                engine_mode=engine_mode,
                on_progress=cell_progress,
                gene_pool=gene_pool,
            ))
        except SpreadBlockedError as exc:
            reports.append({
                "symbol": symbol,
                "timeframe": timeframe,
                "error": f"{type(exc).__name__}: {exc}",
                "spread_check": exc.fragment,
                "blocked_before_launch": True,
                # Otherwise the merge below would leave whatever this cell
                # reported last time under a file that looks freshly written.
                "stale": True,
                "completed_at_utc": datetime.now(tz=UTC).isoformat(),
            })
        except Exception as exc:
            reports.append({
                "symbol": symbol,
                "timeframe": timeframe,
                "error": f"{type(exc).__name__}: {exc}",
                "stale": True,
                "completed_at_utc": datetime.now(tz=UTC).isoformat(),
            })

    out = runner_json_path()
    if selected is not None and out.exists():
        previous = json.loads(out.read_text(encoding="utf-8"))
        if not isinstance(previous, list):
            previous = [previous] if isinstance(previous, dict) else []
        fresh = {(r["symbol"], r["timeframe"]): r for r in reports}
        kept: list[dict[str, Any]] = []
        for r in previous:
            key = (r.get("symbol"), r.get("timeframe"))
            replacement = fresh.pop(key, None)
            if replacement is not None:
                kept.append(replacement)
                continue
            # A cell the operator did not select keeps its earlier result. It
            # is NOT this campaign's output, so it is marked stale in the file
            # itself: anything reading search_runner_real.json then shows the
            # cell as carried over instead of presenting an old number as a
            # fresh measurement.
            r["stale"] = True
            r["stale_reason"] = "cell not re-run by the current campaign"
            kept.append(r)
        reports = kept + list(fresh.values())
    out.write_text(json.dumps(reports, indent=2, default=str), encoding="utf-8")
    if on_progress is not None:
        on_progress(total, total)
    return reports


def _selected(argv: list[str]) -> set[tuple[str, str]] | None:
    """``SYMBOL:TIMEFRAME`` arguments select a subset of cells."""
    picked = set()
    for arg in argv:
        if ":" not in arg:
            raise SystemExit(f"bad cell selector {arg!r}; expected SYMBOL:TIMEFRAME")
        symbol, timeframe = arg.split(":", 1)
        picked.add((symbol, timeframe))
    return picked or None


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("slow", "fast", "auto"), default="auto")
    args, remaining = parser.parse_known_args(argv if argv is not None else sys.argv[1:])
    only = _selected(remaining)
    print(f"ARQVIO_HOME={get_arqvio_home()}")
    print(f"TARGET_AUDIT_CSV={artifact_path(TARGET_AUDIT_CSV_NAME)}")
    _audit_header, audit = load_target_audit()
    print("Loaded target audit:")
    for k, v in audit.items():
        print(f"  {k[0]:10s} {k[1]:4s} bars={v['bars_returned']:>6s}  "
              f"years={v['years_available']}")
    reports = run_campaign_sync(population=5000, selected=only, engine_mode=args.engine)
    print(f"\nWROTE {runner_json_path()}")
    print(f"CELLS: {len(reports)}")
    total_crashes = sum(r.get("n_crashes", 0) for r in reports)
    print(f"TOTAL CRASHES: {total_crashes}")
    for r in reports:
        if "error" in r:
            print(f"  ERROR: {r['symbol']} {r['timeframe']}: {r['error']}")
        else:
            cat = r['funnel'].get('by_category', {})
            print(
                f"  {r['symbol']} {r['timeframe']}: "
                f"VAULTED={cat.get('VAULTED', 0)}  "
                f"FailedDSR={cat.get('FailedDSR', 0)}  "
                f"DSR_within={r['dsr_within']}  "
                f"pbo={r['pbo']}  "
                f"crashes={r['n_crashes']}"
            )
            print(f"    exit: {r['exit_verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
