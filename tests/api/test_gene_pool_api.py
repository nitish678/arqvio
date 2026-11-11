"""W2 gene pool API: schema, default, PUT/PATCH persistence and 422s."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from engine.api import routes_s5
from engine.api.server import create_app
from engine.research.search import gene_pool as gp
from engine.research.search import run_campaign_cli, sampler
from engine.research.search.dna import SIGNAL_VALUES

URL = "/api/v1/gene-pool"


@pytest.fixture
def client(isolated_arqvio_home: Path) -> TestClient:
    return TestClient(create_app())


def _full(**overrides: dict) -> dict:
    fams = {f: {"enabled": True, "weight": 1.0} for f in SIGNAL_VALUES}
    fams.update(overrides)
    return {"families": fams}


def _problems(response) -> list[str]:
    assert response.status_code == 422, response.text
    body = response.json()
    assert body["status_code"] == 422
    assert body["error"]["error"] == "invalid gene pool"
    return body["error"]["problems"]


def test_schema_lists_families_and_sampler_ranges(client: TestClient) -> None:
    r = client.get(f"{URL}/schema")
    assert r.status_code == 200
    s = r.json()
    assert s["families"] == list(SIGNAL_VALUES)
    genes = {g["name"]: g for fam in s["genes"].values() for g in fam}
    assert (genes["max_feature_lookback"]["min"], genes["max_feature_lookback"]["max"]) \
        == sampler.MAX_FEATURE_LOOKBACK_RANGE
    assert (genes["rsi_period"]["min"], genes["rsi_period"]["max"]) \
        == (sampler.RSI_PERIOD_VALUES[0], sampler.RSI_PERIOD_VALUES[-1])
    assert (genes["macd_slow"]["min"], genes["macd_slow"]["max"]) \
        == (sampler.MACD_SLOW_VALUES[0], sampler.MACD_SLOW_VALUES[-1])
    assert (genes["donchian_period"]["min"], genes["donchian_period"]["max"]) \
        == (sampler.DONCHIAN_PERIOD_VALUES[0], sampler.DONCHIAN_PERIOD_VALUES[-1])
    assert [genes["stoch_oversold"]["min"], genes["stoch_oversold"]["max"]] \
        == list(sampler.STOCH_OVERSOLD_RANGE)
    assert genes["rsi_exit_midline"] == {"name": "rsi_exit_midline", "kind": "bool"}
    assert s == gp.schema()
    assert s["constraints"] == ["MACD_CROSS: macd_slow - macd_fast >= 5"]


def test_get_default_pool_when_nothing_stored(client: TestClient) -> None:
    r = client.get(URL)
    assert r.status_code == 200
    assert r.json() == gp.default_gene_pool().to_dict()


def test_put_valid_pool_is_persisted(client: TestClient, isolated_arqvio_home: Path) -> None:
    body = _full(STOCH_CROSS={"enabled": False, "weight": 1.0}, MA_CROSS={"enabled": True, "weight": 2.0})
    body["genes"] = {"MACD_CROSS": {"macd_fast": {"fixed": 21}}}
    r = client.put(URL, json=body)
    assert r.status_code == 200, r.text
    put = r.json()
    assert put["families"]["STOCH_CROSS"] == {"enabled": False, "weight": 0.0}
    assert put["families"]["MA_CROSS"]["weight"] == pytest.approx(0.4)
    assert put["genes"]["MACD_CROSS"]["macd_fast"] == {"fixed": 21}
    assert (isolated_arqvio_home / "configs" / "gene_pool.json").is_file()
    assert client.get(URL).json() == put


def test_put_all_off_rejected_with_problems(client: TestClient) -> None:
    body = {"families": {f: {"enabled": False, "weight": 1.0} for f in SIGNAL_VALUES}}
    problems = _problems(client.put(URL, json=body))
    assert any("all-off" in p for p in problems)
    assert client.get(URL).json() == gp.default_gene_pool().to_dict()


def test_put_non_object_body_is_422_with_problems(client: TestClient) -> None:
    assert _problems(client.put(URL, json=[1, 2]))


def test_patch_merges_onto_stored(client: TestClient) -> None:
    client.put(URL, json=_full(MA_CROSS={"enabled": True, "weight": 3.0}))
    r = client.patch(URL, json={
        "families": {"DONCHIAN_BREAKOUT": {"enabled": False}},
        "genes": {"RSI_REVERSION": {"rsi_period": {"fixed": 9}}},
    })
    assert r.status_code == 200, r.text
    pool = r.json()
    assert pool["families"]["DONCHIAN_BREAKOUT"] == {"enabled": False, "weight": 0.0}
    # MA_CROSS kept its stored 3:1 lead over the others after renormalisation.
    assert pool["families"]["MA_CROSS"]["weight"] == pytest.approx(3 * pool["families"]["RSI_REVERSION"]["weight"])
    assert pool["genes"]["RSI_REVERSION"]["rsi_period"] == {"fixed": 9}
    assert client.get(URL).json() == pool


def test_patch_out_of_range_fixed_leaves_store_unchanged(client: TestClient) -> None:
    before = client.put(URL, json=_full(STOCH_CROSS={"enabled": False, "weight": 1.0})).json()
    problems = _problems(client.patch(URL, json={
        "genes": {"DONCHIAN_BREAKOUT": {"donchian_period": {"fixed": 56}}},
    }))
    assert any("donchian_period" in p for p in problems)
    assert client.get(URL).json() == before


@pytest.mark.parametrize(
    "weights",
    [(1e308,) * 5, (1.0, 1.0, 1.0, 1.0, 5e-324), (10**400, 1, 1, 1, 1)],
)
def test_put_weights_that_cannot_normalise_are_422(client: TestClient, weights) -> None:
    body = {"families": {
        f: {"enabled": True, "weight": w} for f, w in zip(SIGNAL_VALUES, weights, strict=True)
    }}
    assert _problems(client.put(URL, json=body))
    r = client.get(URL)
    assert r.status_code == 200
    assert r.json() == gp.default_gene_pool().to_dict()


def test_put_fixed_float_too_large_for_a_float_is_422(client: TestClient) -> None:
    body = _full()
    body["genes"] = {"STOCH_CROSS": {"stoch_oversold": {"fixed": 10**400}}}
    assert any("stoch_oversold" in p for p in _problems(client.put(URL, json=body)))


def test_patch_enabling_a_disabled_family_joins_at_an_equal_share(client: TestClient) -> None:
    client.put(URL, json=_full(DONCHIAN_BREAKOUT={"enabled": False, "weight": 1.0}))
    r = client.patch(URL, json={"families": {"DONCHIAN_BREAKOUT": {"enabled": True}}})
    assert r.status_code == 200, r.text
    assert r.json()["families"] == {f: {"enabled": True, "weight": pytest.approx(0.2)}
                                    for f in SIGNAL_VALUES}


def test_patch_ignores_hash_like_put(client: TestClient) -> None:
    r = client.patch(URL, json={"hash": "0" * 16, "families": {"MA_CROSS": {"enabled": False}}})
    assert r.status_code == 200, r.text
    assert r.json()["families"]["MA_CROSS"] == {"enabled": False, "weight": 0.0}


def test_run_job_runs_the_campaign_on_the_stored_pool(
    isolated_arqvio_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = _full(STOCH_CROSS={"enabled": False, "weight": 1.0})
    body["genes"] = {"MACD_CROSS": {"macd_fast": {"fixed": 12}}}
    stored = gp.canonicalise(body)
    gp.save(stored)
    calls: list[dict] = []

    def fake_run_campaign_sync(**kwargs):
        calls.append(kwargs)
        return []

    # _run_job imports run_campaign_sync from this module at call time.
    monkeypatch.setattr(run_campaign_cli, "run_campaign_sync", fake_run_campaign_sync)
    job = routes_s5._Job("gene-pool-job", 10, None)
    routes_s5._run_job(job)
    assert job.status == "COMPLETE", job.error
    assert len(calls) == 1
    assert calls[0]["gene_pool"].to_dict() == stored.to_dict()


def test_entry_gates_patch_on_and_a_put_without_them_keeps_them(client: TestClient) -> None:
    plain = client.get("/api/v1/gene-pool").json()
    assert "gates" not in plain  # an all-off pool carries no gates key: its hash is the one it always had
    on = client.patch("/api/v1/gene-pool", json={"gates": {"session": True}}).json()
    assert on["gates"] == {"session": True} and on["hash"] != plain["hash"]
    kept = client.put("/api/v1/gene-pool", json=_full()).json()  # the editor's PUT carries no gates
    assert kept["gates"] == {"session": True}
    off = client.patch("/api/v1/gene-pool", json={"gates": {"session": False}}).json()
    assert "gates" not in off and off["hash"] == plain["hash"]
    bad = client.patch("/api/v1/gene-pool", json={"gates": {"moon_phase": True}})
    assert bad.status_code == 422 and any("moon_phase" in p for p in _problems(bad))
