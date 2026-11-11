"""S5 invariant I6 - provenance propagates, staleness is never silent.

Every dataset row carries source ∈ {BUNDLED, CACHED, MT5_LIVE}, as_of,
dataset_hash, mixed_source, staleness_seconds. These propagate into every
backtest result and every report. A result computed on three-day-old
cached data prints "DATA: CACHED, 3 days old".
"""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from engine.research.provenance import (
    DataSource,
    Provenance,
    parse_source,
    report_staleness_line,
)


def test_stale_data_flagged_in_report() -> None:
    """A backtest on cached data must carry the staleness string in its output."""
    p = Provenance(
        source=DataSource.CACHED,
        dataset_hash="abc",
        as_of=datetime(2026, 9, 13, 0, 0, 0, tzinfo=UTC),
        mixed_source=False,
    )
    now = datetime(2026, 9, 16, 12, 0, 0, tzinfo=UTC)
    line = report_staleness_line(p, now=now)
    assert "CACHED" in line
    assert "3" in line
    assert "day" in line
    # The literal banner that the spec requires.
    assert line.startswith("DATA: ")


def test_bundled_data_is_fresh() -> None:
    p = Provenance(
        source=DataSource.BUNDLED,
        dataset_hash="abc",
        as_of=datetime(2026, 9, 16, 0, 0, 0, tzinfo=UTC),
        mixed_source=False,
    )
    now = datetime(2026, 9, 16, 12, 0, 0, tzinfo=UTC)
    line = report_staleness_line(p, now=now)
    assert "BUNDLED" in line


def test_mt5_live_format() -> None:
    p = Provenance(
        source=DataSource.MT5_LIVE,
        dataset_hash="abc",
        as_of=datetime(2026, 9, 16, 11, 59, 48, tzinfo=UTC),
        mixed_source=False,
    )
    now = datetime(2026, 9, 16, 12, 0, 0, tzinfo=UTC)
    line = report_staleness_line(p, now=now)
    assert line.startswith("DATA: MT5_LIVE · synced 12s ago")


def test_mixed_source_blocks_launch() -> None:
    from engine.research.campaign.preflight import PreflightError, refuse_on_provenance
    p = Provenance(
        source=DataSource.CACHED,
        dataset_hash="abc",
        as_of=datetime(2026, 9, 16, 0, 0, 0, tzinfo=UTC),
        mixed_source=True,  # !! mixing sources silently is the failure mode
    )
    with pytest.raises(PreflightError):
        refuse_on_provenance(p)


def test_parse_source_round_trip() -> None:
    for src in DataSource:
        assert parse_source(src.value) is src


def test_invalid_source_rejected() -> None:
    with pytest.raises(ValueError, match="source"):
        parse_source("MADE_UP")
