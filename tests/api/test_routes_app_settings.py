"""SETTINGS: only settings the engine applies -- validation, the config echo, the OOS-trades gate, saved purge
criteria, AUTOMATIC PURGE, the local bug report and the queue's CLEAR ALL."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from engine.api.server import create_app
from engine.core import app_settings
from engine.research.search import sampler
from tests.research.vault.test_library import CAMPAIGN, candidate, write_campaign


@pytest.fixture
def c(isolated_arqvio_home: Path) -> TestClient:
    return TestClient(create_app())


def test_settings_validate_and_save(c: TestClient) -> None:
    s = c.get("/api/v2/app/settings").json()
    assert s["settings"]["campaign"] == {"train_split": 0.7, "held_out_min_trades": 15} and s["splits"] == [0.6, 0.7, 0.8]
    bad = c.patch("/api/v2/app/settings", json={"campaign": {"train_split": 0.5}, "v1": {"generations": 0}, "x": {}})
    why = lambda r: str(r.json().get("detail") or r.json().get("error"))  # noqa: E731 -- the app's error shape
    assert bad.status_code == 422 and "unknown settings section" in why(bad)
    bad = c.patch("/api/v2/app/settings", json={"campaign": {"train_split": 0.5}, "v1": {"generations": 0}})
    assert "train_split" in why(bad) and "v1.generations" in why(bad)
    ok = c.patch("/api/v2/app/settings", json={"campaign": {"train_split": 0.8, "held_out_min_trades": 25},
                                               "v1": {"generations": 6, "risk_pct": 0.02},
                                               "purge": {"criteria": {"min_score": 40, "max_dd": None}}}).json()
    assert ok["settings"]["campaign"]["train_split"] == 0.8 and ok["settings"]["purge"]["criteria"] == {"min_score": 40}
    assert app_settings.v1_overrides() == {"generations": 6, "risk_pct": 0.02}


def test_the_echo_shows_what_the_next_launch_uses(c: TestClient) -> None:
    c.patch("/api/v2/app/settings", json={"campaign": {"train_split": 0.6}, "v1": {"generations": 7}})
    v1 = c.get("/api/v2/app/settings/echo?depth=STANDARD&engine=v1").json()
    assert v1["held_out"] == 0.4 and v1["v1"]["generations"] == 7 and v1["v1"]["preset"] == "research"
    assert v1["line"].startswith("[CONFIG] split=0.60/0.40")
    v2 = c.get("/api/v2/app/settings/echo?depth=QUICK&engine=standard").json()
    assert set(v2["v2"]) == {"pop", "gen", "budget", "final_n"}


def test_the_oos_trades_gate_follows_the_setting(c: TestClient) -> None:
    from engine.research.vault import library

    assert library.min_held_out_trades() == 15
    c.patch("/api/v2/app/settings", json={"campaign": {"held_out_min_trades": 40}})
    assert library.min_held_out_trades() == 40
    assert c.get("/api/v2/app/engine-settings").json()["certified"]["min_held_out_trades"] == 40


def test_blank_purge_uses_the_saved_criteria(c: TestClient, isolated_arqvio_home: Path) -> None:
    write_campaign(isolated_arqvio_home, [candidate(sampler.random_dna(np.random.default_rng(s))) for s in (1, 2)])
    ids = c.post("/api/v2/app/library/import", json={"campaign_id": CAMPAIGN}).json()["added"]
    c.patch("/api/v2/app/settings", json={"purge": {"criteria": {"min_score": 1000}}})
    assert sorted(c.post("/api/v2/app/library/purge-criteria", json={"criteria": {}}).json()["ids"]) == sorted(ids)
    assert c.post("/api/v2/app/library/purge-criteria", json={"criteria": {"min_score": 0}}).json()["count"] == 0


def test_automatic_purge_waits_one_interval_then_sweeps_and_caps(c: TestClient, isolated_arqvio_home: Path) -> None:
    from engine.research.automation import runner
    from engine.research.vault import library

    write_campaign(isolated_arqvio_home, [candidate(sampler.random_dna(np.random.default_rng(s))) for s in (1, 2, 3)])
    c.post("/api/v2/app/library/import", json={"campaign_id": CAMPAIGN})
    t0 = datetime.now(tz=UTC) + timedelta(days=60)  # every strategy is 60 days old by then
    assert runner.auto_purge(t0) is None  # off
    c.patch("/api/v2/app/settings", json={"auto_purge": {"enabled": True, "every_hours": 24, "older_than_days": 45,
                                                         "cap": 100}, "purge": {"criteria": {"min_score": 1000}}})
    assert runner.auto_purge(t0) is None  # switched on: the first sweep waits one interval
    assert runner.auto_purge(t0 + timedelta(hours=1)) is None
    r = runner.auto_purge(t0 + timedelta(hours=25))
    assert r is not None and len(r["purged"]) == 3 and library.load() == []
    assert any(x["name"] == "AUTOMATIC PURGE" for x in runner.activity_log())


def test_bug_report_is_saved_locally_and_clear_all_empties_the_queue(c: TestClient) -> None:
    r = c.post("/api/v2/app/settings/bug-report", json={"what": "LAUNCH did nothing", "contact": "me@x"}).json()
    text = Path(r["path"]).read_text(encoding="utf-8")
    assert "LAUNCH did nothing" in text and "me@x" in text and "ARQVIO" in text
    assert c.delete("/api/v2/app/launch/queue").json() == {"removed": 0, "items": []}
