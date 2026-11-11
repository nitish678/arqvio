"""Audit header values end where their line does: no trailing terminator.

The writer (scripts/history_depth_audit_target.py) emits "# KEY value\\n"; the
readers kept the "\\n" as part of every value (AS_OF, SOURCE, TIME_BASIS), so
reports carried "utc\\n". Both readers now drop the terminator. No existing
audit file is touched; the GA's own .strip() stays as defence in depth.
"""
from __future__ import annotations

import io
from pathlib import Path

import pytest

from engine.api.routes_s5 import _parse_audit_header
from engine.research.search import run_campaign_cli as cli

HEADER = (
    "# AUDIT_CELL_SET symbol=USDCADm timeframe=H1{nl}"
    "# CELLS USDCADm:H1{nl}"
    "# AS_OF 2026-09-24T18:39:39.338452+00:00{nl}"
    "# SOURCE MT5_LIVE{nl}"
    "# TIME_BASIS utc{nl}"
    "symbol,timeframe,bar_cap,years_available,bars_returned,first_bar_utc,last_bar_utc,source,time_basis{nl}"
    "USDCADm,H1,100000,12.67,60383,2014-01-14,2026-09-24,MT5_LIVE,utc{nl}"
)
EXPECTED = {
    "AUDIT_CELL_SET": "symbol=USDCADm timeframe=H1", "CELLS": "USDCADm:H1",
    "AS_OF": "2026-09-24T18:39:39.338452+00:00", "SOURCE": "MT5_LIVE", "TIME_BASIS": "utc",
}


@pytest.mark.parametrize("nl", ["\n", "\r\n"])
def test_load_target_audit_values_have_no_terminator(isolated_arqvio_home: Path, nl: str) -> None:
    path = cli.artifact_path(cli.TARGET_AUDIT_CSV_NAME)
    path.parent.mkdir(parents=True)
    path.write_bytes(HEADER.format(nl=nl).encode("utf-8"))
    header, rows = cli.load_target_audit()
    assert header == EXPECTED
    assert rows[("USDCADm", "H1")]["time_basis"] == "utc"


@pytest.mark.parametrize("nl", ["\n", "\r\n"])
def test_the_api_audit_header_values_have_no_terminator(nl: str) -> None:
    handle = io.StringIO(HEADER.format(nl=nl), newline="")
    assert _parse_audit_header(handle) == EXPECTED
