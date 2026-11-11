"""Law 7, reader side: the API resolves every artifact path at CALL time.

routes_s5 and server read the runner report, the audits, the campaign
registries and the spread matrices that run_campaign_cli writes. Under an
ARQVIO_HOME set after import -- isolated_arqvio_home -- reader and writer name
the same files, and with a fixed ARQVIO_HOME the paths are the ones the API
has always read.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from engine.api import routes_s5
from engine.api.server import create_app
from engine.data import spread_calibration
from engine.research.search import run_campaign_cli as cli


def test_readers_resolve_what_the_writer_writes(isolated_arqvio_home: Path) -> None:
    artifacts = isolated_arqvio_home / "artifacts"
    assert routes_s5.runner_json() == cli.runner_json_path() == artifacts / "search_runner_real.json"
    assert routes_s5.target_audit_csv() == cli.artifact_path(cli.TARGET_AUDIT_CSV_NAME)
    assert routes_s5.audit_csv() == artifacts / "history_depth_audit_live.csv"
    assert routes_s5.artifact("c.sqlite") == cli.artifact_path("c.sqlite")
    assert spread_calibration.spread_dir() == cli.artifact_path("spread") == artifacts / "spread"


def test_production_paths_are_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """No override: USERPROFILE (or HOME)/ARQVIO/artifacts/<the same file names>."""
    monkeypatch.delenv("ARQVIO_HOME", raising=False)
    root = Path(os.environ.get("USERPROFILE") or os.environ["HOME"]).expanduser().resolve()
    artifacts = root / "ARQVIO" / "artifacts"
    assert routes_s5.runner_json() == artifacts / "search_runner_real.json"
    assert routes_s5.target_audit_csv() == artifacts / "history_depth_audit_target.csv"
    assert routes_s5.audit_csv() == artifacts / "history_depth_audit_live.csv"
    assert spread_calibration.spread_dir() == artifacts / "spread"


def test_the_runner_route_reads_the_isolated_home(isolated_arqvio_home: Path) -> None:
    written = cli.runner_json_path()
    written.parent.mkdir(parents=True)
    written.write_text(json.dumps([{"symbol": "USDCADm", "timeframe": "H1"}]), encoding="utf-8")
    body = TestClient(create_app()).get("/api/v1/s5/runner").json()
    assert body["exists"] is True
    assert Path(body["path"]) == written
    assert body["data"] == [{"symbol": "USDCADm", "timeframe": "H1"}]


def test_the_spread_gate_reads_the_isolated_home(isolated_arqvio_home: Path) -> None:
    assert "no artifact at READERm.json" in (routes_s5.spread_blocked_reason("READERm") or "")
    spread = cli.artifact_path("spread")
    spread.mkdir(parents=True)
    (spread / "READERm.json").write_text("{not json", encoding="utf-8")
    reason = routes_s5.spread_blocked_reason("READERm")
    assert reason is not None and "no artifact at" not in reason  # it read the isolated file
