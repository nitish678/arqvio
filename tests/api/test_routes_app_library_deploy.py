"""Vault and Full Deploy over HTTP: every verb answers with the engine's own record or a REFUSED reason."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from engine.api import routes_app_deploy as rd
from engine.api.server import create_app
from engine.research.search import sampler
from tests.research.vault.test_library import CAMPAIGN, candidate, write_campaign


@pytest.fixture
def c(isolated_arqvio_home: Path) -> TestClient:
    return TestClient(create_app())


def fake_mt5(tmp: Path) -> dict:
    d = {"terminal": str(tmp / "terminal64.exe"), "data_dir": str(tmp / "data"), "mql5_dir": str(tmp / "data" / "MQL5"),
         "common_files": str(tmp / "common")}
    for k in ("data_dir", "mql5_dir", "common_files"):
        Path(d[k]).mkdir(parents=True, exist_ok=True)
    return d


def vaulted(home: Path, c: TestClient) -> str:
    dna = sampler.random_dna(np.random.default_rng(5))
    write_campaign(home, [candidate(dna)])
    r = c.post("/api/v2/app/library/import", json={"campaign_id": CAMPAIGN})
    assert r.status_code == 200, r.json()
    return r.json()["added"][0]


def test_the_vault_lists_imports_and_removes(c: TestClient, isolated_arqvio_home: Path) -> None:
    empty = c.get("/api/v2/app/library").json()
    assert empty["strategies"] == [] and empty["counts"] == {"CANDIDATE": 0, "CERTIFIED": 0, "UNCERTIFIED": 0}
    assert c.post("/api/v2/app/library/import", json={"campaign_id": "nope"}).status_code == 404
    eid = vaulted(isolated_arqvio_home, c)
    over = c.get("/api/v2/app/library").json()
    assert [s["id"] for s in over["strategies"]] == [eid] and over["campaigns"][0]["imported"] is True
    s = over["strategies"][0]
    assert (s["tier"], s["score_max"], s["robustness"]) == ("CANDIDATE", 60.0, None)
    detail = c.get(f"/api/v2/app/library/strategies/{eid}").json()
    assert detail["source"]["campaign_id"] == CAMPAIGN and detail["gates"]
    assert c.delete(f"/api/v2/app/library/strategies/{eid}").json() == {"removed": eid}
    assert c.get(f"/api/v2/app/library/strategies/{eid}").status_code == 404


def test_a_robustness_run_left_by_a_stopped_sidecar_reads_interrupted(c: TestClient, isolated_arqvio_home: Path) -> None:
    from engine.research.vault import library

    eid = vaulted(isolated_arqvio_home, c)
    library.set_robustness(eid, {"state": "RUNNING"})
    assert c.get("/api/v2/app/library").json()["strategies"][0]["robustness"] == "INTERRUPTED"


def test_robustness_re_runs_a_strategy_on_its_own_snapshot(c: TestClient, isolated_arqvio_home: Path,
                                                           monkeypatch) -> None:
    """A QUICK cell refreshed with new bars after breeding must not change what 'reproduces exactly' re-runs on."""
    import time

    from engine.research.ga import runner
    from engine.research.vault import library

    eid = vaulted(isolated_arqvio_home, c)
    asked = {}

    def fake_prepare(name, **kw):
        from engine.research.search.real_backtest import _COST_MODE

        asked.update(name=name, cost_mode=_COST_MODE.get(), **kw)
        raise RuntimeError("stop here")

    monkeypatch.setattr(runner, "prepare_cell", fake_prepare)
    assert c.post(f"/api/v2/app/library/strategies/{eid}/robustness").status_code == 200
    for _ in range(100):
        if library.get(eid)["robustness"]["state"] != "RUNNING":
            break
        time.sleep(0.05)
    assert asked == {"name": "usdcadm_h1", "dataset_id": library.get(eid)["dataset_id"], "cost_mode": "hourly"}
    assert library.get(eid)["robustness"]["state"] == "FAILED"


def test_deploy_without_mt5_says_why_and_refuses_the_steps_that_need_it(c: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(rd, "_mt5", lambda chosen=None: {"terminal": None, "data_dir": None, "mql5_dir": None,
                                             "common_files": None, "problems": ["terminal: not found"]})
    over = c.get("/api/v2/app/deploy").json()
    assert over["bots"] == [] and over["stop_all"] is None and over["mt5"]["problems"] == ["terminal: not found"]
    for verb, path, body in (("post", "/api/v2/app/deploy/stage", None), ("post", "/api/v2/app/deploy/stop",
                                                                         {"stop": True})):
        r = getattr(c, verb)(path, json=body)
        assert r.status_code == 409 and "MetaTrader 5 is not resolved" in r.json()["error"]


def test_a_vault_strategy_becomes_a_staged_bot_and_start_needs_confirm(c: TestClient, isolated_arqvio_home: Path,
                                                                       tmp_path: Path, monkeypatch) -> None:
    mt5 = fake_mt5(tmp_path)
    monkeypatch.setattr(rd, "_mt5", lambda chosen=None: mt5)
    monkeypatch.setattr(rd, "_cell_limits", lambda s, t, d: ((21,), {"max_spread_points": 0, "spread_regime_multiple": 3.0,
                                                                    "spread_regime_bars": 480}))

    def compile_ok(source: Path, log: Path) -> Path:
        source.with_suffix(".ex5").write_bytes(b"EX5")
        return source.with_suffix(".ex5")

    monkeypatch.setattr(rd, "_compiler", lambda mt5: compile_ok)
    eid = vaulted(isolated_arqvio_home, c)
    bad = c.post("/api/v2/app/deploy/bots", json={"vault_id": eid, "guardrails": {"lots": -1}})
    assert bad.status_code == 422 and "lots must be > 0" in bad.json()["error"]
    row = c.post("/api/v2/app/deploy/bots", json={"vault_id": eid, "symbol": "USDCAD",
                                                  "guardrails": {"lots": 0.05}}).json()
    assert row["symbol"] == "USDCAD" and row["guardrails"]["max_spread_points"] == 0
    assert row["guardrails"]["spread_regime_multiple"] == 3.0 and row["guardrails"]["spread_regime_bars"] == 480
    assert row["no_trade_hours"] == [21] and row["source"]["tier"] == "CANDIDATE"
    assert c.post("/api/v2/app/deploy/start", json={"confirm": True}).status_code == 409  # not staged yet
    staged = c.post("/api/v2/app/deploy/stage").json()
    assert [s["bot"] for s in staged["staged"]] == [row["bot"]]
    assert c.post("/api/v2/app/deploy/start", json={"confirm": False}).status_code == 422
    assert c.post(f"/api/v2/app/deploy/bots/{row['bot']}/stage", json={"stage": 1}).json()["status"] == "PLANNED"
    assert c.post("/api/v2/app/deploy/stop", json={"stop": True}).json() == {"stop_all": True}
    over = c.get("/api/v2/app/deploy").json()
    assert over["stop_all"] is True and over["bots"][0]["lot_scale"] == 0.5 and over["bots"][0]["live"] is None
    assert c.get(f"/api/v2/app/deploy/bots/{row['bot']}/log").json()["state"] == "NO LOG YET"
    assert c.delete(f"/api/v2/app/deploy/bots/{row['bot']}").json()["status"] == "REMOVED"


def test_engine_settings_show_the_rules_the_engine_uses(c: TestClient, monkeypatch) -> None:
    from engine.config.settings import get_settings
    from engine.research.vault import library

    get_settings.cache_clear()
    monkeypatch.setenv("ARQVIO_SPREAD_COST_MODE", "hourly")
    try:
        r = c.get("/api/v2/app/engine-settings").json()
    finally:
        get_settings.cache_clear()
    assert r["costs"]["spread_cost_mode"] == "hourly" and r["costs"]["spread_gate_mode"] in ("regime", "fixed")
    assert r["certified"]["beat_random_min"] == library.BEAT_RANDOM_MIN and r["grades"][0] == {"min": 80.0, "grade": "A"}
