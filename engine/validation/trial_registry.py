"""Trial registry.

The single most important missing number in almost every published backtest is
the number of trials attempted. Without it, a Sharpe ratio cannot be
interpreted, and the Deflated Sharpe Ratio cannot be computed at all.

This registry makes the count non-optional. EVERY candidate evaluation is
recorded here, including candidates rejected before their Sharpe was stored -
because those rejections are still trials, and omitting them inflates DSR.

Two operation types are recognised:
    SEARCH_TRIAL   a candidate evaluated to decide what to keep.
    VALIDATION_RUN an already-selected strategy re-run on a different cell
                   purely as a robustness check.

Only SEARCH_TRIAL rows contribute to DSR / MinBTL statistics. The legacy
``UNKNOWN`` value is preserved for rows written before op_type existed; those
rows are visible in queries but excluded from stats.

Provenance (dataset_hash, source, as_of) is mandatory on every record. It
propagates into reports so staleness is never silent.
"""
from __future__ import annotations

import json
import sqlite3
import typing
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from engine.validation.op_types import (
    OpType,
    counts_toward_dsr,
    is_valid_op_type,
)

_VALID_OP_TYPES = {OpType.SEARCH_TRIAL.value, OpType.VALIDATION_RUN.value}
_VALID_SOURCES = {"BUNDLED", "CACHED", "MT5_LIVE"}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS campaigns (
    campaign_id   TEXT PRIMARY KEY,
    symbol        TEXT NOT NULL,
    timeframe     TEXT NOT NULL,
    dataset_hash  TEXT NOT NULL,
    git_sha       TEXT NOT NULL,
    config_hash   TEXT NOT NULL,
    seed          INTEGER NOT NULL,
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    notes         TEXT
);

CREATE TABLE IF NOT EXISTS trials (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    campaign_id   TEXT NOT NULL REFERENCES campaigns(campaign_id),
    dna_hash      TEXT NOT NULL,
    symbol        TEXT,
    timeframe     TEXT,
    op_type       TEXT NOT NULL DEFAULT 'UNKNOWN',
    sharpe        REAL,           -- NULL when rejected before evaluation
    n_trades      INTEGER,
    total_return  REAL,
    max_dd        REAL,
    outcome       TEXT NOT NULL,  -- EVALUATED | LowTrades | HighDD | NegReturn | Crash | ...
    dataset_hash  TEXT,
    source        TEXT,           -- BUNDLED | CACHED | MT5_LIVE
    as_of         TEXT,           -- ISO-8601 UTC; data was current as of this moment
    -- DSR provenance, attached after the search loop by _populate_post_dsr.
    -- These four make a finished campaign re-analysable without re-running
    -- it: which candidates were eligible, how DSR was distributed, and what
    -- bar each value was measured against.
    dsr           REAL,           -- the candidate's own DSR (post_dsr)
    dsr_status    TEXT,           -- READY | UNRELIABLE | NO_TRIALS
    sr_star       REAL,           -- the luck bar in force for that cell
    n_obs         INTEGER,        -- length of that candidate's return series
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_trials_campaign ON trials(campaign_id);
CREATE INDEX IF NOT EXISTS idx_trials_outcome  ON trials(campaign_id, outcome);
CREATE INDEX IF NOT EXISTS idx_trials_scope ON trials(campaign_id, symbol, timeframe, op_type);
"""


@dataclass(frozen=True)
class CampaignStats:
    campaign_id: str
    n_trials: int                 # SEARCH_TRIAL rows only
    n_scored: int                 # scored SEARCH_TRIAL rows only
    var_sr_trials: float
    max_sharpe: float
    mean_sharpe: float
    outcomes: dict[str, int]
    validation_runs: int = 0
    unknown_rows: int = 0


@dataclass(frozen=True)
class TrialRecord:
    """A single trial row, hydrated from SQLite."""

    id: int
    campaign_id: str
    dna_hash: str
    symbol: str | None
    timeframe: str | None
    op_type: str
    sharpe: float | None
    n_trades: int | None
    total_return: float | None
    max_dd: float | None
    outcome: str
    dataset_hash: str | None
    source: str | None
    as_of: str | None
    created_at: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _normalise_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


class TrialRegistry:
    def __init__(self, path: str | Path = "data/trials.sqlite") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.executescript(_SCHEMA)
            self._ensure_columns(c)

    @contextmanager
    def _conn(self) -> typing.Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30.0)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()
        return None  # satisfy mypy "missing return type" without changing behaviour

    @staticmethod
    def _ensure_columns(c: sqlite3.Connection) -> None:
        columns = {row[1] for row in c.execute("PRAGMA table_info(trials)")}
        for name, definition in (
            ("symbol", "TEXT"),
            ("timeframe", "TEXT"),
            ("op_type", "TEXT NOT NULL DEFAULT 'UNKNOWN'"),
            ("dataset_hash", "TEXT"),
            ("source", "TEXT"),
            ("as_of", "TEXT"),
            # Added 2026-09-18. NULL on every pre-existing row, which reads
            # as "not recorded" - no rewrite of old registries.
            ("dsr", "REAL"),
            ("dsr_status", "TEXT"),
            ("sr_star", "REAL"),
            ("n_obs", "INTEGER"),
        ):
            if name not in columns:
                c.execute(f"ALTER TABLE trials ADD COLUMN {name} {definition}")
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_trials_scope "
            "ON trials(campaign_id, symbol, timeframe, op_type)"
        )

    # --- campaigns -----------------------------------------------------------

    def open_campaign(
        self,
        campaign_id: str,
        symbol: str,
        timeframe: str,
        dataset_hash: str,
        git_sha: str,
        config_hash: str,
        seed: int,
        notes: str | None = None,
    ) -> None:
        """Register a campaign. All provenance fields are mandatory by design."""
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO campaigns "
                "(campaign_id, symbol, timeframe, dataset_hash, git_sha, "
                " config_hash, seed, notes) VALUES (?,?,?,?,?,?,?,?)",
                (
                    campaign_id, symbol, timeframe, dataset_hash, git_sha,
                    config_hash, int(seed), notes,
                ),
            )

    # --- recording -----------------------------------------------------------

    def record(
        self,
        campaign_id: str,
        dna_hash: str,
        outcome: str,
        sharpe: float | None = None,
        n_trades: int | None = None,
        total_return: float | None = None,
        max_dd: float | None = None,
        *,
        symbol: str | None = None,
        timeframe: str | None = None,
        op_type: str = OpType.SEARCH_TRIAL.value,
        dataset_hash: str | None = None,
        source: str | None = None,
        as_of: str | None = None,
    ) -> None:
        """Record one operation. Only SEARCH_TRIAL rows contribute to DSR stats."""
        if not is_valid_op_type(op_type):
            raise ValueError(f"invalid op_type: {op_type!r}")
        if source is not None and source not in _VALID_SOURCES:
            raise ValueError(f"invalid source: {source!r}")
        # Every new row must carry the campaign's immutable provenance. Legacy rows
        # with absent provenance remain readable but cannot silently enter DSR.
        with self._conn() as c:
            campaign = c.execute(
                "SELECT dataset_hash, symbol, timeframe FROM campaigns WHERE campaign_id=?",
                (campaign_id,),
            ).fetchone()
        if campaign is None:
            raise KeyError(f"unknown campaign_id: {campaign_id}")
        if dataset_hash is None or source is None or as_of is None:
            raise ValueError("dataset_hash, source and as_of are required for every trial record")
        if op_type == OpType.SEARCH_TRIAL.value and str(dataset_hash) != str(campaign[0]):
            raise ValueError("search trial dataset_hash must match the campaign provenance")
        symbol, timeframe = self._resolve_cell(campaign_id, symbol, timeframe)
        with self._conn() as c:
            c.execute(
                "INSERT INTO trials (campaign_id, dna_hash, symbol, timeframe, op_type, "
                "sharpe, n_trades, total_return, max_dd, outcome, "
                "dataset_hash, source, as_of) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    campaign_id, dna_hash, symbol, timeframe, op_type,
                    None if sharpe is None or not np.isfinite(sharpe) else float(sharpe),
                    n_trades, total_return, max_dd, outcome,
                    _normalise_optional_text(dataset_hash),
                    _normalise_optional_text(source),
                    _normalise_optional_text(as_of),
                ),
            )

    def record_many(
        self,
        campaign_id: str,
        rows: Iterable[tuple[str, str, float | None, int | None, float | None, float | None]],
        *,
        symbol: str | None = None,
        timeframe: str | None = None,
        op_type: str = OpType.SEARCH_TRIAL.value,
        dataset_hash: str | None = None,
        source: str | None = None,
        as_of: str | None = None,
    ) -> None:
        """Bulk record operations; provenance fields are stamped on every row."""
        if not is_valid_op_type(op_type):
            raise ValueError(f"invalid op_type: {op_type!r}")
        if source is not None and source not in _VALID_SOURCES:
            raise ValueError(f"invalid source: {source!r}")
        symbol, timeframe = self._resolve_cell(campaign_id, symbol, timeframe)
        ds_hash = _normalise_optional_text(dataset_hash)
        src = _normalise_optional_text(source)
        asof = _normalise_optional_text(as_of)
        payload = [
            (
                campaign_id, r[0], symbol, timeframe, op_type,
                None if r[2] is None or not np.isfinite(r[2]) else float(r[2]),
                r[3], r[4], r[5], r[1], ds_hash, src, asof,
            )
            for r in rows
        ]
        with self._conn() as c:
            c.executemany(
                "INSERT INTO trials (campaign_id, dna_hash, symbol, timeframe, op_type, "
                "sharpe, n_trades, total_return, max_dd, outcome, "
                "dataset_hash, source, as_of) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                payload,
            )

    def update_dsr_many(
        self,
        campaign_id: str,
        rows: Iterable[tuple[str, float | None, str | None, float | None, int | None]],
        *,
        op_type: str = OpType.SEARCH_TRIAL.value,
    ) -> int:
        """Attach per-candidate DSR provenance to rows already recorded.

        ``rows`` is ``(dna_hash, dsr, dsr_status, sr_star, n_obs)``. One
        batched ``executemany``, not a statement per candidate.

        Why these four and nothing more. ``dsr`` and ``dsr_status`` answer
        eligible-subset and distribution questions after the fact. ``sr_star``
        is what makes a stored DSR readable later at all: a DSR is the
        probability of clearing one specific bar, and without that bar the
        number means nothing. ``n_obs`` is what separates UNRELIABLE (too few
        observations for the Gaussian approximation) from NO_TRIALS.

        ``per_period_returns`` is deliberately NOT stored: 3,915 float64 per
        candidate is ~156 MB for one cell, and it would turn a registry into
        a data store.

        ``dsr_status`` carries ``CellDSRStatus`` verbatim: READY,
        UNRELIABLE, or NO_TRIALS when the cell had no search trials to
        correct for. REJECTED belongs to the cross-cell result and never
        appears on a trial row.

        Returns the number of rows updated.
        """
        if not is_valid_op_type(op_type):
            raise ValueError(f"invalid op_type: {op_type!r}")
        payload = [
            (
                None if r[1] is None or not np.isfinite(r[1]) else float(r[1]),
                _normalise_optional_text(r[2]),
                None if r[3] is None or not np.isfinite(r[3]) else float(r[3]),
                None if r[4] is None else int(r[4]),
                campaign_id,
                r[0],
                op_type,
            )
            for r in rows
        ]
        if not payload:
            return 0
        with self._conn() as c:
            cur = c.executemany(
                "UPDATE trials SET dsr=?, dsr_status=?, sr_star=?, n_obs=? "
                "WHERE campaign_id=? AND dna_hash=? AND op_type=?",
                payload,
            )
            return int(cur.rowcount)

    # --- queries -------------------------------------------------------------

    def _resolve_cell(
        self, campaign_id: str, symbol: str | None, timeframe: str | None
    ) -> tuple[str, str]:
        if symbol is not None and timeframe is not None:
            return str(symbol), str(timeframe)
        with self._conn() as c:
            row = c.execute(
                "SELECT symbol, timeframe FROM campaigns WHERE campaign_id=?",
                (campaign_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown campaign_id: {campaign_id}")
        return symbol or str(row[0]), timeframe or str(row[1])

    def stats(
        self,
        campaign_id: str,
        *,
        symbol: str | None = None,
        timeframe: str | None = None,
    ) -> CampaignStats:
        """DSR statistics from SEARCH_TRIAL rows, optionally for one cell."""
        filters = ["campaign_id=?", f"op_type='{OpType.SEARCH_TRIAL.value}'"]
        params: list[object] = [campaign_id]
        if symbol is not None:
            filters.append("symbol=?")
            params.append(symbol)
        if timeframe is not None:
            filters.append("timeframe=?")
            params.append(timeframe)
        where = " AND ".join(filters)
        with self._conn() as c:
            n_trials_row = c.execute(
                f"SELECT COUNT(*) FROM trials WHERE {where}", params
            ).fetchone()
            rows = c.execute(
                f"SELECT sharpe FROM trials WHERE {where} AND sharpe IS NOT NULL", params
            ).fetchall()
            outcome_rows = c.execute(
                f"SELECT outcome, COUNT(*) FROM trials WHERE {where} GROUP BY outcome", params
            ).fetchall()
            validation_row = c.execute(
                "SELECT COUNT(*) FROM trials WHERE campaign_id=? AND op_type=?",
                (campaign_id, OpType.VALIDATION_RUN.value),
            ).fetchone()
            unknown_row = c.execute(
                "SELECT COUNT(*) FROM trials WHERE campaign_id=? AND op_type='UNKNOWN'",
                (campaign_id,),
            ).fetchone()

        n_trials = int(n_trials_row[0]) if n_trials_row is not None else 0
        sr = np.array([r[0] for r in rows], dtype=np.float64)
        return CampaignStats(
            campaign_id=campaign_id,
            n_trials=n_trials,
            n_scored=int(sr.size),
            var_sr_trials=float(np.var(sr, ddof=1)) if sr.size >= 2 else float("nan"),
            max_sharpe=float(sr.max()) if sr.size else float("nan"),
            mean_sharpe=float(sr.mean()) if sr.size else float("nan"),
            outcomes={k: int(v) for k, v in outcome_rows},
            validation_runs=int(validation_row[0]) if validation_row else 0,
            unknown_rows=int(unknown_row[0]) if unknown_row else 0,
        )

    def search_sharpes(
        self, campaign_id: str, symbol: str, timeframe: str
    ) -> list[float]:
        """Return finite SEARCH_TRIAL Sharpes for exactly one cell."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT sharpe FROM trials WHERE campaign_id=? AND symbol=? "
                "AND timeframe=? AND op_type=? AND sharpe IS NOT NULL",
                (campaign_id, symbol, timeframe, OpType.SEARCH_TRIAL.value),
            ).fetchall()
        return [float(row[0]) for row in rows]

    def list_trials(
        self,
        campaign_id: str,
        *,
        op_type: str | None = None,
        symbol: str | None = None,
        timeframe: str | None = None,
        outcome: str | None = None,
        limit: int | None = None,
    ) -> list[TrialRecord]:
        """Return hydrated trial rows for inspection / reports."""
        filters = ["campaign_id=?"]
        params: list[object] = [campaign_id]
        if op_type is not None:
            filters.append("op_type=?")
            params.append(op_type)
        if symbol is not None:
            filters.append("symbol=?")
            params.append(symbol)
        if timeframe is not None:
            filters.append("timeframe=?")
            params.append(timeframe)
        if outcome is not None:
            filters.append("outcome=?")
            params.append(outcome)
        where = " AND ".join(filters)
        sql = (
            "SELECT id, campaign_id, dna_hash, symbol, timeframe, op_type, "
            "sharpe, n_trades, total_return, max_dd, outcome, dataset_hash, "
            "source, as_of, created_at FROM trials WHERE {where} "
            "ORDER BY id ASC"
        )
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        with self._conn() as c:
            rows = c.execute(sql.format(where=where), params).fetchall()
        return [
            TrialRecord(
                id=int(r[0]), campaign_id=str(r[1]), dna_hash=str(r[2]),
                symbol=r[3], timeframe=r[4], op_type=str(r[5]),
                sharpe=r[6], n_trades=r[7], total_return=r[8], max_dd=r[9],
                outcome=str(r[10]),
                dataset_hash=r[11], source=r[12], as_of=r[13],
                created_at=str(r[14]),
            )
            for r in rows
        ]

    def manifest(self, campaign_id: str) -> str:
        """Reproducibility manifest as JSON. Attach to every result."""
        with self._conn() as c:
            row = c.execute(
                "SELECT symbol, timeframe, dataset_hash, git_sha, config_hash, "
                "seed, created_at FROM campaigns WHERE campaign_id=?",
                (campaign_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown campaign_id: {campaign_id}")
        s = self.stats(campaign_id)
        return json.dumps(
            {
                "campaign_id": campaign_id,
                "symbol": row[0],
                "timeframe": row[1],
                "dataset_hash": row[2],
                "git_sha": row[3],
                "config_hash": row[4],
                "seed": row[5],
                "created_at": row[6],
                "n_trials_search": s.n_trials,
                "n_scored_search": s.n_scored,
                "var_sr_trials": s.var_sr_trials,
                "validation_runs": s.validation_runs,
                "unknown_rows": s.unknown_rows,
                "outcomes": s.outcomes,
            },
            indent=2,
        )


__all__ = [
    "CampaignStats",
    "OpType",
    "TrialRecord",
    "TrialRegistry",
    "counts_toward_dsr",
    "is_valid_op_type",
]
