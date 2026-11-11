"""XAU-H1 deploy-check (DR-2026-10-03-xau-h1-campaign §5): the bundle validates through the S6 worker's OWN surface.

The fixture is a byte-identical copy of <ARQVIO artifacts>/campaign_xau_h1/deploy_bundle.json. Validation is exactly
the worker's own entry points -- WorkerConfig.__post_init__, dna_to_strategy, OracleSignalSource(...) and
Worker.__post_init__ -- on the in-memory FakeMT5. No worker is started: run / probe / act / flatten_own are replaced
by failures for the whole test, and the fake records no order.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from engine.research.live import worker as wk
from engine.research.live.kill_switch import KillSwitch
from engine.research.search.dna import DNA
from engine.research.search.real_backtest import dna_to_strategy
from tests.research.live.fake_mt5 import FakeMT5

FIXTURE = Path(__file__).parent / "fixtures" / "xau_h1_deploy_bundle.json"
SMOKE_LABEL = "PIPELINE SMOKE — NO EDGE CLAIM"


@pytest.fixture
def bundle() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def never_start(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("run", "probe", "act", "flatten_own"):
        monkeypatch.setattr(wk.Worker, name, lambda *a, _n=name, **k: pytest.fail(f"Worker.{_n} was called"))


def test_the_bundle_is_labelled_for_its_lane(bundle: dict) -> None:
    assert bundle["schema"] == "arqvio.deploy.bundle" and bundle["version"] == 1
    assert bundle["lane"] in ("SMOKE", "SURVIVOR") and bundle["started_by_this_stage"] is False
    assert bundle["cell"] == {"symbol": "XAUUSDm", "timeframe": "H1", "dataset_id": "447b843f87438a9a"}
    assert "power-check pending" in bundle["campaign"]["power_check"]
    if bundle["lane"] == "SMOKE":
        assert bundle["label"] == SMOKE_LABEL and bundle["edge_claim"] == "NONE"
        assert bundle["subject"]["n_trades"] >= 1 and "COUNT rule" in bundle["subject"]["rule"]


def test_the_dna_hash_recomputes_and_matches_the_worker_config(bundle: dict) -> None:
    dna = DNA(**bundle["dna"])
    assert dna.dna_hash == bundle["dna_hash"] == bundle["worker_config"]["dna_hash"]


def test_the_worker_config_validates_at_the_s6_micro_caps(bundle: dict) -> None:
    cfg = wk.WorkerConfig(**bundle["worker_config"])
    assert (cfg.volume, cfg.max_positions, cfg.daily_order_cap) == (0.01, 1, 6) and cfg.source == "oracle_signal"
    assert re.fullmatch(r"[0-9a-f]{16}", cfg.config_hash())


@pytest.mark.parametrize("field,value", [("volume", 0.0), ("max_positions", 0), ("daily_order_cap", 0),
                                         ("source", "martingale")])
def test_a_bundle_that_cannot_run_is_refused_by_the_worker(bundle: dict, field: str, value) -> None:
    with pytest.raises(wk.WorkerConfigError):
        wk.WorkerConfig(**(bundle["worker_config"] | {field: value}))


def test_the_worker_constructs_on_fakes_and_nothing_starts(bundle: dict, tmp_path: Path) -> None:
    mt5 = FakeMT5()
    cfg = wk.WorkerConfig(**bundle["worker_config"])
    _, strategy = dna_to_strategy(DNA(**bundle["dna"]))
    source = wk.OracleSignalSource(mt5, bundle["cell"]["symbol"], strategy)
    logs: list[str] = []
    w = wk.Worker(mt5, cfg, wk.Operational(), source, log=logs.append, sleep=lambda s: None,
                  kill=KillSwitch(mt5, flag_dir=tmp_path), max_spread_points=bundle["guardrails"]["max_spread_points"],
                  no_trade_hours=tuple(bundle["guardrails"]["no_trade_hours"]))
    assert w.state == wk.BOOT and w.magic == bundle["worker_magic"] and w.cfg.config_hash() == cfg.config_hash()
    assert source.anchor is None and source.done is False and mt5.sent == [] and logs == []
    assert not (tmp_path / "KILL").exists()
