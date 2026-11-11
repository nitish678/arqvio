"""Full Deploy: magics never reused, the incubation ladder, staging writes every file MT5 needs, heartbeats."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

from engine.research.deploy import full_deploy as fd
from engine.research.deploy.live_ea import INCUBATION_STAGES, Guardrails
from engine.research.search import sampler


def dnas(n: int) -> list:
    rng = np.random.default_rng(3)
    return [sampler.random_dna(rng) for _ in range(n)]


def fake_compile(source: Path, log: Path) -> Path:
    log.write_text("Result: 0 errors, 0 warnings", encoding="utf-8")
    ex5 = source.with_suffix(".ex5")
    ex5.write_bytes(b"EX5")
    return ex5


def test_magics_are_allocated_in_order_and_never_reused(isolated_arqvio_home: Path) -> None:
    a, b = dnas(2)
    r1 = fd.add_bot(a, symbol="EURUSD", timeframe="H1", guardrails=Guardrails(), source={"vault_id": "x"})
    r2 = fd.add_bot(b, symbol="EURUSD", timeframe="H1", guardrails=Guardrails(), source={"vault_id": "y"})
    assert (r1["magic"], r2["magic"]) == (610_500_000, 610_500_001)
    with pytest.raises(fd.DeployError, match="already deployed"):
        fd.add_bot(a, symbol="EURUSD", timeframe="H1", guardrails=Guardrails(), source={})
    fd.remove_bot(r1["bot"])
    again = fd.add_bot(a, symbol="EURUSD", timeframe="H1", guardrails=Guardrails(), source={})
    assert again["magic"] == 610_500_002  # the removed bot keeps its magic: no live position can be mistaken
    assert [r["status"] for r in fd.load_bots()] == ["REMOVED", "PLANNED", "PLANNED"]


def test_the_incubation_ladder_changes_the_lot_scale_and_needs_a_new_stage(isolated_arqvio_home: Path) -> None:
    (a,) = dnas(1)
    row = fd.add_bot(a, symbol="XAUUSDm", timeframe="M15", guardrails=Guardrails(lots=0.1), source={})
    assert row["stage"] == 0 and row["guardrails"]["lot_scale"] == INCUBATION_STAGES[0]
    up = fd.set_stage(row["bot"], 2)
    assert up["guardrails"]["lot_scale"] == 1.0 and up["status"] == "PLANNED"
    assert fd.render(up).inputs["InpLotScale"] == "1.00000000" and up["source_sha256"] != row["source_sha256"]
    with pytest.raises(fd.DeployError, match="stage must be"):
        fd.set_stage(row["bot"], 3)
    with pytest.raises(fd.DeployError, match="no bot"):
        fd.set_stage("nope", 1)


def test_stage_writes_every_file_and_the_plan(isolated_arqvio_home: Path, tmp_path: Path) -> None:
    a, b = dnas(2)
    fd.add_bot(a, symbol="EURUSD", timeframe="H1", guardrails=Guardrails(), source={})
    gone = fd.add_bot(b, symbol="GBPUSD", timeframe="M30", guardrails=Guardrails(), source={})
    kept = fd.add_bot(b, symbol="USDJPY", timeframe="D1", guardrails=Guardrails(), source={})
    fd.remove_bot(gone["bot"])
    mql5 = tmp_path / "MQL5"
    (mql5 / "Files" / "ARQVIO").mkdir(parents=True)
    (mql5 / "Files" / "ARQVIO" / "deploy_result.csv").write_text("old,run\n", encoding="ascii")
    report = fd.stage(mql5, compile_fn=fake_compile)
    assert [s["bot"] for s in report["staged"]] == [fd.load_bots()[0]["bot"], kept["bot"]]
    for s in report["staged"]:
        assert Path(s["mq5"]).is_file() and Path(s["ex5"]).is_file()
        tpl = Path(s["template"]).read_bytes()
        assert tpl[:2] == b"\xff\xfe" and "<expert>" in tpl[2:].decode("utf-16-le")
        assert b"\r" not in Path(s["preset"]).read_bytes()
    assert (mql5 / "Scripts" / "ARQVIO" / "arqvio_attach.ex5").is_file()
    plan = (mql5 / "Files" / "ARQVIO" / "deploy_plan.csv").read_text(encoding="ascii").splitlines()
    assert plan == [f"{fd.load_bots()[0]['bot']},EURUSD,60,{fd.load_bots()[0]['bot']}.tpl",
                    f"{kept['bot']},USDJPY,1440,{kept['bot']}.tpl"]
    assert not (mql5 / "Files" / "ARQVIO" / "deploy_result.csv").exists()
    assert [r["status"] for r in fd.load_bots()] == ["STAGED", "REMOVED", "STAGED"]


def test_stage_refuses_an_empty_plan(isolated_arqvio_home: Path, tmp_path: Path) -> None:
    with pytest.raises(fd.DeployError, match="no bots"):
        fd.stage(tmp_path, compile_fn=fake_compile)


def test_heartbeats_say_live_only_while_fresh(tmp_path: Path) -> None:
    d = tmp_path / "ARQVIO" / "bots"
    d.mkdir(parents=True)
    for magic, age in ((610_500_000, 5), (610_500_001, 600)):
        p = d / f"{magic}.json"
        p.write_text(json.dumps({"schema": "arqvio.bot_heartbeat", "version": 1, "magic": magic, "standby": ""}),
                     encoding="utf-8")
        t = time.time() - age
        os.utime(p, (t, t))
    (d / "junk.json").write_text("{not json", encoding="utf-8")
    beats = fd.heartbeats(tmp_path)
    assert beats[610_500_000]["state"] == "LIVE" and beats[610_500_001]["state"] == "SILENT"
    assert len(beats) == 2


def test_stop_file_start_config_and_attach_results(tmp_path: Path) -> None:
    assert fd.set_stop(tmp_path, True) is True and fd.stop_file(tmp_path).is_file()
    assert fd.set_stop(tmp_path, False) is False
    ini = fd.start_config(tmp_path, {"symbol": "EURUSD", "timeframe": "H1"}).read_text(encoding="ascii")
    assert "AllowLiveTrading=1" in ini and "Script=ARQVIO\\arqvio_attach" in ini and "Period=H1" in ini
    res = tmp_path / "MQL5" / "Files" / "ARQVIO"
    res.mkdir(parents=True)
    (res / "deploy_result.csv").write_bytes("﻿bot_a,EURUSD,60,ATTACHED,,123\r\n".encode("utf-16-le"))
    assert fd.attach_results(tmp_path / "MQL5")[0]["status"] == "ATTACHED"
