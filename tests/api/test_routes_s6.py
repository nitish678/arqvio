"""The S6 read-only API: call-time paths, GET only, fail-closed on missing or broken artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from engine.api.server import create_app
from engine.research.mql5 import records


@pytest.fixture
def client(isolated_arqvio_home: Path) -> TestClient:
    return TestClient(create_app())


def _ea(magic: int, cell: str = "usdcadm_h1_aed3bf314539ec5a_full") -> SimpleNamespace:
    return SimpleNamespace(magic=magic, dna_hash="4fb42548f31d2741", family="MA_CROSS", strategy_id="ma",
                           cell_name=cell, template_version="astra-mql5-v1", template_digest="ab" * 32,
                           source_sha256="cd" * 32, name="arqvio_ma")


def _verdict(ok: bool) -> SimpleNamespace:
    lv = SimpleNamespace(name="D", ok=ok, summary="s", lines=[] if ok else ["row 1"], compared=3, exact=3, inexact=[])
    return SimpleNamespace(ok=ok, levels=[lv])


def test_paths_resolve_under_the_home_set_after_import(isolated_arqvio_home: Path) -> None:
    a = isolated_arqvio_home / "artifacts"
    assert records.ea_registry_path() == a / "mql5" / "ea_registry.json"
    assert records.parity_dir() == a / "mql5" / "parity"
    assert records.worker_status_path() == a / "live" / "worker_status.json"


def test_missing_artifacts_answer_exists_false(client: TestClient, isolated_arqvio_home: Path) -> None:
    for route in ("/api/v1/s6/ea-registry", "/api/v1/s6/worker", "/api/v1/s6/parity"):
        body = client.get(route).json()
        assert body["exists"] is False and body["path"].startswith(str(isolated_arqvio_home))
    assert client.get("/api/v1/s6/parity/S-GA-usdcadm-00").status_code == 404


def test_the_registry_maps_magic_to_dna_and_template(client: TestClient) -> None:
    records.record_ea(_ea(610_100_000))
    records.record_ea(_ea(610_100_000, cell="xauusdm_h1_447b843f87438a9a_full"))
    body = client.get("/api/v1/s6/ea-registry").json()
    assert body["exists"] and len(body["data"]["eas"]) == 2
    row = body["data"]["eas"]["610100000:usdcadm_h1_aed3bf314539ec5a_full"]
    assert (row["range"], row["slot"], row["dna_hash"], row["template_version"]) == (
        "ga-usdcadm_h1-s7-cbcb2ae061e70bef", 0, "4fb42548f31d2741", "astra-mql5-v1")


def test_parity_summaries_list_every_subject_and_level(client: TestClient) -> None:
    records.record_parity("S-GA-usdcadm-00", "GA finalist 0", "usdcadm_h1_aed3bf314539ec5a_full", "r0", _verdict(True))
    records.record_parity("S-MA-0", "MA draw 0", "usdcadm_h1_aed3bf314539ec5a_last800", "r1", _verdict(False))
    body = client.get("/api/v1/s6/parity").json()
    assert (body["subjects"], body["passed"]) == (2, 1)
    assert {r["subject"]: r["levels"] for r in body["rows"]} == {"S-GA-usdcadm-00": {"D": True}, "S-MA-0": {"D": False}}
    one = client.get("/api/v1/s6/parity/S-MA-0").json()
    assert one["data"]["levels"][0]["mismatches"] == 1
    assert client.get("/api/v1/s6/parity/..%2Fsecret").status_code in (400, 404)


def test_a_broken_artifact_is_reported_not_filled_in(client: TestClient) -> None:
    p = records.worker_status_path()
    p.parent.mkdir(parents=True)
    p.write_text("{not json", encoding="utf-8")
    body = client.get("/api/v1/s6/worker").json()
    assert body["exists"] is True and body["data"] is None and body["error"].startswith("unreadable: JSONDecodeError")


def test_the_worker_status_is_served_as_written(client: TestClient) -> None:
    records.write_json_atomic(records.worker_status_path(), {"state": "HALTED", "halt_reason": "Completed",
                                                             "last_reconciliation": {"ok": True, "foreign": 1}})
    body = client.get("/api/v1/s6/worker").json()
    assert body["data"]["state"] == "HALTED" and body["data"]["last_reconciliation"]["ok"] is True


def test_the_s6_surface_is_get_only(client: TestClient) -> None:
    app = client.app
    s6 = [r for r in app.routes if getattr(r, "path", "").startswith("/api/v1/s6")]
    assert len(s6) == 4 and all(r.methods == {"GET"} for r in s6)
    assert client.post("/api/v1/s6/worker").status_code == 405
