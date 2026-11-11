"""The S7 read-only API: call-time paths, fail-closed on missing or broken artifacts, GET only, the spool's chain
verified on read, the 2-account health snapshot served as written."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from engine.api.server import create_app
from engine.research.alerts.dispatch import emit
from engine.research.alerts.schema import make_alert

S7 = "/api/v1/s7"
ROUTES = ("/portfolio", "/diversity", "/heat", "/drift", "/alerts", "/health")


@pytest.fixture
def client(isolated_arqvio_home: Path) -> TestClient:
    return TestClient(create_app())


def _write(home: Path, rel: str, obj) -> Path:
    p = home / "artifacts" / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj), encoding="utf-8")
    return p


def test_every_route_answers_exists_false_on_an_empty_home(client: TestClient, isolated_arqvio_home: Path) -> None:
    for r in ROUTES:
        body = client.get(S7 + r).json()
        assert body["exists"] is False and str(isolated_arqvio_home) in body["path"], r
    assert not (isolated_arqvio_home / "artifacts").exists()  # reading created nothing


def test_the_portfolio_route_summarises_the_replay(client: TestClient, isolated_arqvio_home: Path) -> None:
    _write(isolated_arqvio_home, "portfolio/replay.json", {
        "run_hash": "ab" * 32, "inputs": {"e0": 100000.0}, "members": [{"mid": "RSI-USDCADm"}],
        "verification": {"conservation_exact": True, "recon_ok": True, "recon_max_abs": 1.4e-11},
        "refused": {}, "events": [{"kind": "CONFLICT"}, {"kind": "CONFLICT"}, {"kind": "HEAT_HIGH"}],
        "bars": [{"equity": 100000.0}, {"equity": 99950.5}]})
    d = client.get(S7 + "/portfolio").json()["data"]
    assert d["run_hash"] == "ab" * 32 and d["verification"]["conservation_exact"] is True
    assert d["event_counts"] == {"CONFLICT": 2, "HEAT_HIGH": 1} and d["bars"] == 2 and d["final_equity"] == 99950.5
    assert "events" not in d


def test_the_analysis_artifacts_are_served_as_written(client: TestClient, isolated_arqvio_home: Path) -> None:
    for name in ("diversity", "heat", "drift"):
        _write(isolated_arqvio_home, f"portfolio/{name}.json", {"schema": f"arqvio.portfolio.{name}", "run_hash": "x"})
        assert client.get(f"{S7}/{name}").json()["data"]["schema"] == f"arqvio.portfolio.{name}"


def test_a_broken_artifact_is_reported_not_filled_in(client: TestClient, isolated_arqvio_home: Path) -> None:
    p = isolated_arqvio_home / "artifacts" / "health" / "status.json"
    p.parent.mkdir(parents=True)
    p.write_text("{not json", encoding="utf-8")
    body = client.get(S7 + "/health").json()
    assert body["exists"] is True and body["data"] is None and body["error"].startswith("unreadable: JSONDecodeError")


def test_the_alerts_route_tails_the_spool_and_verifies_its_chain(client: TestClient, isolated_arqvio_home: Path) -> None:
    for k in range(3):
        emit(make_alert("HEAT_HIGH", "warn", {"k": k}, "replay.json", clock=lambda: "c"))
    body = client.get(S7 + "/alerts", params={"n": 2}).json()
    assert body["exists"] and body["chain_ok"] is True and body["chain_lines"] == 3
    assert [r["payload"]["k"] for r in body["lines"]] == [1, 2]
    spool = isolated_arqvio_home / "artifacts" / "alerts" / "spool.jsonl"
    spool.write_text(spool.read_text(encoding="ascii").replace('"k":1', '"k":9'), encoding="ascii")
    body = client.get(S7 + "/alerts").json()
    assert body["chain_ok"] is False and "line 2" in body["chain_reason"]
    assert client.get(S7 + "/alerts", params={"n": 0}).status_code == 422


def test_the_health_snapshot_is_served_with_both_accounts(client: TestClient, isolated_arqvio_home: Path) -> None:
    snap = {"schema": "arqvio.health.status", "version": 1, "accounts": [
        {"id": "acct1", "login": 1, "demo": True, "basis": {"basis": "server_equals_utc"}, "status": "OK"},
        {"id": "acct2", "login": 2, "demo": True, "basis": {"basis": "broker_offset", "measured_offset_s": 7200},
         "status": "OK"}]}
    _write(isolated_arqvio_home, "health/status.json", snap)
    d = client.get(S7 + "/health").json()["data"]
    assert [a["id"] for a in d["accounts"]] == ["acct1", "acct2"] and d["accounts"][1]["basis"]["measured_offset_s"] == 7200


def test_the_s7_surface_is_get_only(client: TestClient) -> None:
    for r in ROUTES:
        for method in ("post", "put", "delete", "patch"):
            assert getattr(client, method)(S7 + r).status_code == 405, (method, r)
