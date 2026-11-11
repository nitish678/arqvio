"""Research VAULT actions: N SELECTED delete / parity, criteria purge (portfolio members spared), DEDUPE, RESTORE ALL,
DELETE CAMPAIGN, GET JSON passport, the extra columns."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from engine.api.server import create_app
from engine.research.search import sampler
from tests.research.vault.test_library import CAMPAIGN, candidate, write_campaign


@pytest.fixture
def c(isolated_arqvio_home: Path) -> TestClient:
    return TestClient(create_app())


def _vault(home: Path, c: TestClient, n: int = 3) -> list[str]:
    dnas = [sampler.random_dna(np.random.default_rng(s)) for s in range(1, n + 1)]
    write_campaign(home, [candidate(d) for d in dnas])
    r = c.post("/api/v2/app/library/import", json={"campaign_id": CAMPAIGN})
    assert r.status_code == 200, r.json()
    return r.json()["added"]


def test_rows_carry_the_research_columns(c: TestClient, isolated_arqvio_home: Path) -> None:
    ids = _vault(isolated_arqvio_home, c, 1)
    p = c.post("/api/v2/app/portfolios", json={"name": "Team A"}).json()["portfolio"]
    c.post(f"/api/v2/app/portfolios/{p['id']}/add", json={"ids": ids})
    row = c.get("/api/v2/app/library").json()["strategies"][0]
    for key in ("broker", "window_years", "first_bar", "cagr", "parity_grade", "win_rate", "robust", "portfolios"):
        assert key in row
    assert row["portfolios"] == ["Team A"]


def test_n_selected_delete_and_restore_all(c: TestClient, isolated_arqvio_home: Path) -> None:
    ids = _vault(isolated_arqvio_home, c, 3)
    r = c.post("/api/v2/app/library/bulk", json={"action": "delete", "ids": ids[:2] + ["nope@x"]}).json()
    assert sorted(r["done"]) == sorted(ids[:2]) and r["skipped"] == ["nope@x"]
    assert [s["id"] for s in c.get("/api/v2/app/library").json()["strategies"]] == [ids[2]]
    assert sorted(c.post("/api/v2/app/library/recycle/restore-all").json()["restored"]) == sorted(ids[:2])
    assert len(c.get("/api/v2/app/library").json()["strategies"]) == 3
    assert c.post("/api/v2/app/library/bulk", json={"action": "fly", "ids": ids}).status_code == 422


def test_parity_for_n_selected_says_why_each_cannot_run(c: TestClient, isolated_arqvio_home: Path, monkeypatch) -> None:
    monkeypatch.delenv("ARQVIO_MT5_TERMINAL", raising=False)
    ids = _vault(isolated_arqvio_home, c, 2)
    r = c.post("/api/v2/app/library/bulk", json={"action": "parity", "ids": ids}).json()
    assert r["queued"] == [] and set(r["refused"]) == set(ids)


def test_criteria_purge_previews_then_applies_and_spares_portfolio_members(c: TestClient, isolated_arqvio_home: Path) -> None:
    ids = _vault(isolated_arqvio_home, c, 3)
    p = c.post("/api/v2/app/portfolios", json={"name": "Keep"}).json()["portfolio"]
    c.post(f"/api/v2/app/portfolios/{p['id']}/add", json={"ids": [ids[0]]})
    pv = c.post("/api/v2/app/library/purge-criteria", json={"criteria": {"min_score": 1000}}).json()
    assert pv["applied"] is False and sorted(pv["ids"]) == sorted(ids[1:])
    ap = c.post("/api/v2/app/library/purge-criteria", json={"criteria": {"min_score": 1000}, "apply": True}).json()
    assert ap["applied"] is True
    assert [s["id"] for s in c.get("/api/v2/app/library").json()["strategies"]] == [ids[0]]


def test_delete_campaign_and_passport(c: TestClient, isolated_arqvio_home: Path) -> None:
    ids = _vault(isolated_arqvio_home, c, 2)
    pp = c.get(f"/api/v2/app/library/strategies/{ids[0]}/passport").json()
    assert pp["schema"] == "arqvio.passport" and pp["strategy"]["id"] == ids[0] and "address" in pp
    camp = c.get("/api/v2/app/library").json()["strategies"][0]["campaign_id"]
    assert sorted(c.post(f"/api/v2/app/library/campaign/{camp}/delete").json()["removed"]) == sorted(ids)
    assert c.post(f"/api/v2/app/library/campaign/{camp}/delete").status_code == 404
    assert c.post("/api/v2/app/library/dedupe", json={}).json()["applied"] is False


def test_a_named_portfolio_is_a_portfolio_source_and_has_a_card(c: TestClient, isolated_arqvio_home: Path) -> None:
    ids = _vault(isolated_arqvio_home, c, 2)
    p = c.post("/api/v2/app/portfolios", json={"name": "Alpha"}).json()["portfolio"]
    c.post(f"/api/v2/app/portfolios/{p['id']}/add", json={"ids": ids[:1]})
    one = c.get(f"/api/v2/app/library/portfolio?source=pf:{p['id']}").json()
    assert one["overall"] is None and [m["id"] for m in one["members"]] == ids[:1]
    card = c.get("/api/v2/app/portfolios/cards").json()["cards"][0]
    assert card["name"] == "Alpha" and card["members"] == 1 and card["in_vault"] == 1 and card["grade"] is None
    assert card["avg_return"] is not None
    assert c.get("/api/v2/app/library/portfolio?source=pf:nope").status_code == 404
