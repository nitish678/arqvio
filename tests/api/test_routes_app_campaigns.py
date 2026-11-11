"""S5-APP campaign verbs (DR-2026-10-10-s5-app §10; build rulings 2026-10-10, the H-START LAW): the guard's premise
recomputed from the runner's own helpers; a fresh START into an existing campaign directory refused at preflight, at
START and again in the child by the launcher; the measured preflight from an isolated probe home; the one-attempt
START with the pre-registration written before the spawn; STOP of an owned tree; RESUME only with the recorded argv
onto the history this run's last process left; live files read without blocking the runner's atomic replace; no
orphan outliving or bypassing the sidecar; every refusal verbatim in the event log. Hermetic: an isolated ARQVIO_HOME
and a stand-in runner (tests/api/fake_ga_runner.py) driven by the real launcher; the real runner is exercised only by
the ruled proofs in the fixture rig."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from engine.api import campaign_launch
from engine.api import routes_app_campaigns as rc
from engine.api.server import create_app
from engine.research.ga.checkpoint import CheckpointError
from engine.research.ga.parallel import cpu_count, workers_for
from engine.validation.minbtl import min_backtest_length

USD = "ga-usdcadm_h1-s7-cbcb2ae061e70bef"
XAU = "ga-xauusdm_h1-s7-f32e0d4defd9cf07"
STANDARD = {"pop": 200, "gen": 25, "budget": 8000, "final_n": 20}
LINE = re.compile(r"^(?P<ts>\S+) \| (?P<level>[A-Z]+) \| (?P<text>.*)$")
FAKE = "tests.api.fake_ga_runner"
REAL_CHECKPOINT_STATE = rc.checkpoint_state  # captured before any fixture replaces it
D3 = {"prior_trials": {"value": 376, "state": "OK"}, "database_sha256": "d" * 64,
      "caveat": "deflation-integration mechanics pending R-stage DR; any verdict from this campaign carries the caveat "
                "'376 prior unbanked trials not included in deflation'"}


def fake_checkpoint_state(path: Path, config_hash: str) -> dict:
    assert not rc.live(Path(path).parent.name), "a checkpoint was opened while its writer is alive"
    data = Path(path).read_bytes()
    state = json.loads(data)
    if state["config_hash"] != config_hash:
        raise CheckpointError(f"config hash {state['config_hash']} is not this run's {config_hash}")
    return {"generation": state["generation"], "registry_rows": state["registry_rows"],
            "sha256": hashlib.sha256(data).hexdigest()}


@pytest.fixture
def camp(isolated_arqvio_home: Path, monkeypatch) -> dict:
    art = isolated_arqvio_home / "artifacts"
    for d in ("spread", "specs", "datasets"):
        (art / d).mkdir(parents=True)
    (art / "history_depth_audit_target.csv").write_text(
        "# AS_OF test\nsymbol,timeframe,years_available\nUSDCADm,H1,12.0\nGBPJPYm,H1,12.0\nXAUUSDm,H1,12.0\n",
        encoding="utf-8")
    for name in ("USDCADm_H1_aed3bf314539ec5a", "GBPJPYm_H1_5979a019aa6b3e0c", "XAUUSDm_H1_447b843f87438a9a"):
        (art / "datasets" / f"{name}.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(rc, "RUNNER_MODULE", FAKE)
    monkeypatch.setattr(rc, "checkpoint_state", fake_checkpoint_state)
    monkeypatch.setattr(rc, "prior_trials", lambda cell: D3 if cell == "gbpjpym_h1" else None)
    monkeypatch.setattr(rc, "_procs", {})
    monkeypatch.setenv("FAKE_GEN_SLEEP", "0.1")
    c = TestClient(create_app())
    c.home = isolated_arqvio_home
    yield {"home": isolated_arqvio_home, "c": c}
    for e in list(rc._procs.values()):
        if e["proc"].poll() is None:
            rc.kill_tree(e["proc"])
        e["thread"].join(timeout=30)


def run_path(home: Path, run: str) -> Path:
    return home / "artifacts" / "app" / "campaigns" / run


def events(home: Path) -> list[dict]:
    return rc.jsonl(home / "artifacts" / "app" / "events.jsonl")


def refused(home: Path, r, phrase: str, status: int = 409) -> str:
    """A refusal is verbatim in the response AND the newest event-log record carries the same reason."""
    err = r.json()["error"]
    assert r.status_code == status and err.startswith("REFUSED: ") and phrase in err, (r.status_code, err)
    last = events(home)[-1]
    assert (last["outcome"], last["reason"]) == ("REFUSED", err[len("REFUSED: "):])
    return err


def status(c: TestClient, run: str) -> dict:
    return c.get(f"/api/v2/app/campaigns/{run}/status").json()


def wait(c: TestClient, run: str, states: tuple[str, ...], timeout: float = 90.0) -> dict:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        s = status(c, run)
        if s["state"] in states:
            return s
        time.sleep(0.2)
    raise AssertionError(f"{run} never reached {states}; last {s['state']}")


def wait_generation(c: TestClient, run: str, gen: int) -> None:
    t0 = time.monotonic()
    while status(c, run)["checkpoint_generation"].get("value") != gen:
        assert time.monotonic() - t0 < 90, "the held generation never checkpointed"
        time.sleep(0.1)


def ready_run(c: TestClient, cell: str = "gbpjpym_h1", seed: int = 123, preset: str = "SMOKE") -> str:
    r = c.post("/api/v2/app/campaigns/preflight", json={"cell": cell, "preset": preset, "seed": seed})
    assert r.status_code == 200, r.json()
    run = r.json()["run"]
    assert wait(c, run, ("READY", "PREFLIGHT UNAVAILABLE"))["state"] == "READY"
    return run


def stopped_run(c: TestClient, monkeypatch, seed: int = 123) -> str:
    monkeypatch.setenv("FAKE_HOLD_AT_GEN", "1")
    run = ready_run(c, seed=seed)
    assert c.post(f"/api/v2/app/campaigns/{run}/start", json={"confirm": True, "power": "medium"}).status_code == 200
    wait_generation(c, run, 1)
    pid = int((run_path(c.home, run) / "fake_pid").read_text(encoding="utf-8"))
    created = rc.process_created(pid)
    st = c.post(f"/api/v2/app/campaigns/{run}/stop")
    assert st.status_code == 200 and st.json()["checkpoint_generation"]["value"] == 1
    if os.name == "nt":  # the whole tree died, not just the venv launcher
        assert created and rc.process_created(pid) != created and rc._job is not None
    return run


def test_the_h_start_premise_the_cited_ids_recompute_from_seed_7_standard(isolated_arqvio_home) -> None:
    assert rc.campaign_target("usdcadm_h1", STANDARD, 7)["campaign_id"] == USD
    assert rc.campaign_target("xauusdm_h1", STANDARD, 7)["campaign_id"] == XAU
    assert not (isolated_arqvio_home / "artifacts").exists()  # computing the id writes nothing


def test_a_fresh_start_into_an_existing_campaign_directory_is_refused_at_preflight(camp) -> None:
    c, home = camp["c"], camp["home"]
    cited = home / "artifacts" / "ga" / USD
    cited.mkdir(parents=True)
    (cited / "manifest_v2.json").write_text('{"cited": true}', encoding="utf-8")
    r = c.post("/api/v2/app/campaigns/preflight", json={"cell": "usdcadm_h1", "preset": "STANDARD", "seed": 7})
    assert refused(home, r, "H-START") == (
        f"REFUSED: H-START: <ARQVIO_HOME>/artifacts/ga/{USD} already exists; a fresh start would erase its outputs "
        "(engine/research/ga/runner.py:735-740). Every new campaign is a new pre-registered decision: choose another "
        "seed or configuration.")
    assert (cited / "manifest_v2.json").read_text(encoding="utf-8") == '{"cited": true}' and rc._procs == {}
    assert not (home / "artifacts" / "app" / "campaigns").exists()


def test_a_directory_that_appears_after_preflight_is_refused_at_start_and_nothing_spawns(camp) -> None:
    c, home = camp["c"], camp["home"]
    run = ready_run(c)
    cid = status(c, run)["request"]["campaign_id"]
    (home / "artifacts" / "ga" / cid).mkdir(parents=True)
    refused(home, c.post(f"/api/v2/app/campaigns/{run}/start", json={"confirm": True}), "H-START")
    assert not (run_path(home, run) / "preregistration.json").exists()
    assert [s["kind"] for s in rc.jsonl(run_path(home, run) / "spawns.jsonl")] == ["probe"]


def test_the_launcher_re_checks_h_start_and_the_id_in_the_child(camp, capsys) -> None:
    home = camp["home"]
    t = rc.campaign_target("usdcadm_h1", STANDARD, 7)
    rargs = rc.runner_args("usdcadm_h1", STANDARD, 7, None)
    t["dir"].mkdir(parents=True)
    (t["dir"] / "manifest_v2.json").write_text("cited", encoding="utf-8")
    assert campaign_launch.main(["--runner", FAKE, "--expect-campaign-id", USD, "--mode", "fresh", "--", *rargs]) == 2
    assert campaign_launch.main(["--runner", FAKE, "--expect-campaign-id", "ga-other", "--mode", "resume", "--",
                                 *rargs]) == 2
    out = capsys.readouterr().out
    assert "REFUSED: H-START: <ARQVIO_HOME>/artifacts/ga/" + USD in out and "REFUSED: campaign id drift" in out
    assert [p.name for p in t["dir"].iterdir()] == ["manifest_v2.json"]  # the runner never ran
    assert (t["dir"] / "manifest_v2.json").read_text(encoding="utf-8") == "cited" and home.is_dir()


def test_a_tribal_run_launches_under_its_pre_registered_id(camp, capsys) -> None:
    """The launcher re-derives the id with the runner's --islands, as runner.main does (the tribal "id drift")."""
    t = rc.campaign_target("usdcadm_h1", STANDARD, 7, islands=2)
    assert t["campaign_id"] != USD  # the island block is part of the hashed config
    rargs = rc.runner_args("usdcadm_h1", STANDARD, 7, camp["home"] / "checkpoint.npz", islands=2)
    assert "--checkpoint" not in rargs and rargs[rargs.index("--islands") + 1] == "2"
    assert campaign_launch.main(["--runner", FAKE, "--expect-campaign-id", t["campaign_id"], "--mode", "fresh",
                                 "--", *rargs]) == 0
    assert f"LAUNCH fresh {t['campaign_id']}" in capsys.readouterr().out
    assert (t["dir"] / "manifest_v2.json").is_file()


def test_a_tribal_campaign_runs_end_to_end_through_the_routes_and_cannot_resume(camp) -> None:
    c, home = camp["c"], camp["home"]
    r = c.post("/api/v2/app/campaigns/preflight", json={"cell": "gbpjpym_h1", "preset": "SMOKE", "seed": 5,
                                                         "islands": 2})
    assert r.status_code == 200, r.json()
    run = r.json()["run"]
    assert wait(c, run, ("READY", "PREFLIGHT UNAVAILABLE"))["state"] == "READY"
    assert c.post(f"/api/v2/app/campaigns/{run}/start", json={"confirm": True}).status_code == 200
    assert wait(c, run, ("COMPLETED", "FAILED"))["state"] == "COMPLETED"
    prereg = rc.read_json(run_path(home, run) / "preregistration.json")
    assert "--islands" in prereg["runner_args"] and "--checkpoint" not in prereg["runner_args"]
    refused(home, c.post(f"/api/v2/app/campaigns/{run}/resume", json={}), "cannot resume")


def test_presets_are_sealed_custom_is_complete_and_the_picker_fails_closed(camp) -> None:
    c, home = camp["c"], camp["home"]
    post = lambda **b: c.post("/api/v2/app/campaigns/preflight", json=b)  # noqa: E731
    assert post(cell="eurusdm_h1", preset="SMOKE", seed=1).status_code == 404  # L2
    refused(home, post(cell="usdcadm_h1", preset="SMOKE", seed=1, pop=10), "is sealed", 422)
    refused(home, post(cell="usdcadm_h1", preset="CUSTOM", seed=1, pop=10), "CUSTOM needs", 422)
    refused(home, post(cell="usdcadm_h1", preset="CUSTOM", seed=1, pop=50, gen=2, budget=10, final_n=4),
            "trials_budget 10 cannot evaluate one population of 50", 422)
    p = c.get("/api/v2/app/campaigns/presets").json()
    assert {k: p["presets"]["STANDARD"][k] for k in STANDARD} == STANDARD
    assert p["sha256"] and "islands" not in json.dumps(p["presets"])


def test_the_small_window_warning_is_the_engines_minbtl(camp) -> None:
    ok, small = rc.window("usdcadm_h1", 400), rc.window("usdcadm_h1", 10**9)
    for w, n in ((ok, 400), (small, 10**9)):
        assert w["minbtl_required_years"] == min_backtest_length(n_trials=n, target_sharpe=1.5).min_years
        assert w["years_available"]["value"] == 12.0 and w["years_available"]["sha256"]
    assert ok["satisfied"] is True and "warning" not in ok
    assert small["satisfied"] is False and small["warning"].endswith("SMOKE-only, no edge claim from this run")
    (camp["home"] / "artifacts" / "history_depth_audit_target.csv").write_text("symbol,timeframe,years_available\n")
    assert rc.window("usdcadm_h1", 400)["state"] == "NO DATA"


def test_preflight_start_stop_resume_complete(camp, monkeypatch) -> None:
    c, home = camp["c"], camp["home"]
    run = stopped_run(c, monkeypatch)
    d = run_path(home, run)
    s = status(c, run)
    assert s["state"] == "STOPPED" and s["exits"][-1]["stopped"] is True and s["exits"][-1]["rows_sha256"]
    est = s["preflight"]["estimate"]  # the stand-in probe records prepare 2 s, search 8 s for 20 trials, cross 4 s / 4
    assert (est["kind"], est["projected_s"]) == ("upper bound", pytest.approx(2.0 + 0.4 * 400 + 1.0 * 20))
    assert s["preflight"]["manifest"].startswith(f"<ARQVIO_HOME>/artifacts/app/campaigns/{run}/probe/home/")
    pre = s["preregistration"]
    assert pre["d3"] == D3 and pre["window"]["satisfied"] is True and pre["argv"][3] == rc.LAUNCHER_MODULE
    assert pre["argv"][pre["argv"].index("--runner") + 1] == FAKE and pre["argv"][-len(pre["runner_args"]):] == \
        pre["runner_args"]
    # the power mode is recorded and passed to the runner; it is operational, outside the campaign id
    assert pre["power"] == {"mode": "medium", "workers": workers_for("medium"), "logical_cores": cpu_count()}
    assert pre["runner_args"][pre["runner_args"].index("--workers") + 1] == str(workers_for("medium"))
    refused(home, c.post(f"/api/v2/app/campaigns/{run}/start", json={"confirm": True}), "already started")
    refused(home, c.post(f"/api/v2/app/campaigns/{run}/stop"), "no live process owned")
    err = refused(home, c.post(f"/api/v2/app/campaigns/{run}/resume", json={"pop": 51}), "resume arguments differ")
    assert '"asked": 51' in err
    assert c.post(f"/api/v2/app/campaigns/{run}/resume", json={"pop": 50, "seed": 123}).status_code == 200
    done = wait(c, run, ("COMPLETED",))
    assert done["result"]["trials_used"]["value"] == 200 and done["result"]["trials_used"]["sha256"]
    spawns = rc.jsonl(d / "spawns.jsonl")
    assert [x["kind"] for x in spawns] == ["probe", "runner", "runner"]
    assert spawns[1]["argv"] == pre["argv"]  # the START spawned exactly the pre-registered argv
    assert spawns[2]["argv"] == rc.launcher_argv("resume", pre["campaign_id"],
                                                 [*pre["runner_args"], "--resume", str(d / "checkpoint.npz")])
    refused(home, c.post(f"/api/v2/app/campaigns/{run}/resume", json={}), "completed")
    log = (d / "runner.log").read_text(encoding="utf-8").splitlines()
    parsed = [LINE.match(x) for x in log]
    assert all(parsed) and any(m["level"] == "WARN" and m["text"] == "fake:main:1 | a loguru-shaped warning"
                               for m in parsed)  # loguru's level lifted, its source location kept verbatim
    assert sum(m["text"].startswith("process exited rc=") for m in parsed) == 2
    assert any(m["text"].startswith("RESUME from") for m in parsed)
    lg = c.get(f"/api/v2/app/logs?source=campaign:{run}").json()
    assert lg["total"] == len(log) and lg["lines"][0]["origin"].endswith("runner.log:1")
    assert c.get(f"/api/v2/app/logs?source=probe:{run}").json()["total"] > 0
    outcomes = [e["outcome"] for e in events(home) if e.get("run") == run]
    assert outcomes[:3] == ["PROBING", "PREFLIGHT", "STARTED"] and {"STOPPED", "RESUMED"} <= set(outcomes)
    cc = c.get("/api/v2/app/cells/gbpjpym_h1/campaigns").json()
    assert cc["app_runs"][0]["state"] == "COMPLETED" and cc["cited"]["value"] == "UNRUN"


def test_resume_refuses_any_history_that_is_not_what_this_runs_last_process_left(camp, monkeypatch) -> None:
    c, home = camp["c"], camp["home"]
    run = stopped_run(c, monkeypatch)
    d = run_path(home, run)
    ck, pr, ex = d / "checkpoint.npz", d / "preregistration.json", d / "exits.jsonl"
    cid = status(c, run)["request"]["campaign_id"]
    db = home / "artifacts" / "ga" / cid / "trials.sqlite"
    resume = lambda: c.post(f"/api/v2/app/campaigns/{run}/resume", json={})  # noqa: E731
    keep = {p: p.read_bytes() for p in (ck, pr, ex, db)}

    state = json.loads(keep[ck])
    ck.write_text(json.dumps({**state, "registry_rows": state["registry_rows"] - 1}), encoding="utf-8")
    refused(home, resume(), "the checkpoint changed after this run's last process exited")
    ck.write_text(json.dumps({**state, "config_hash": "0" * 16}), encoding="utf-8")
    refused(home, resume(), "checkpoint refused: config hash")
    ck.write_bytes(keep[ck])
    con = sqlite3.connect(db)
    con.execute("DELETE FROM trials WHERE rowid = (SELECT MIN(rowid) FROM trials)")
    con.commit()
    con.close()
    refused(home, resume(), "rows are not the history this run left")
    db.write_bytes(keep[db])
    ex.unlink()
    refused(home, resume(), "no exit fingerprint of this run's last process")
    ex.write_bytes(keep[ex])
    doc = json.loads(keep[pr])
    doc["runner_args"][doc["runner_args"].index("--checkpoint") + 1] = "C:/elsewhere/checkpoint.npz"
    pr.write_text(json.dumps(doc), encoding="utf-8")
    refused(home, resume(), "the recorded runner arguments are not this run's")
    pr.write_bytes(keep[pr])
    assert resume().status_code == 200  # the untouched history resumes
    wait(c, run, ("COMPLETED",))


def test_a_failed_probe_blocks_start_and_zero_finalists_give_only_a_lower_bound(camp, monkeypatch) -> None:
    c, home = camp["c"], camp["home"]
    monkeypatch.setenv("FAKE_PROBE_FAIL", "1")
    r = c.post("/api/v2/app/campaigns/preflight", json={"cell": "usdcadm_h1", "preset": "SMOKE", "seed": 31})
    run = r.json()["run"]
    s = wait(c, run, ("READY", "PREFLIGHT UNAVAILABLE"))
    assert s["state"] == "PREFLIGHT UNAVAILABLE" and s["preflight"]["reason"].startswith("probe exit 3")
    refused(home, c.post(f"/api/v2/app/campaigns/{run}/start", json={"confirm": True}), "no OK measured preflight")
    monkeypatch.delenv("FAKE_PROBE_FAIL")

    def too_deep(target: Path, cell: str) -> None:
        raise FileNotFoundError(2, "No such file or directory", str(target / ("x" * 300)))

    with monkeypatch.context() as m:
        m.setattr(rc, "probe_home", too_deep)
        r = c.post("/api/v2/app/campaigns/preflight", json={"cell": "usdcadm_h1", "preset": "SMOKE", "seed": 33})
        refused(home, r, "the probe could not be started: FileNotFoundError")
    lost = [x for x in c.get("/api/v2/app/campaigns").json()["runs"] if x["run"].endswith("-s33")]
    assert lost[0]["state"] == "PREFLIGHT UNAVAILABLE" and rc._procs.get(lost[0]["run"]) is None
    monkeypatch.setenv("FAKE_FINALISTS", "0")
    est = status(c, ready_run(c, cell="usdcadm_h1", seed=32))["preflight"]["estimate"]
    assert (est["kind"], est["per_finalist_cross_s"], est["projected_s"]) == ("lower bound", None,
                                                                              pytest.approx(2.0 + 0.4 * 400))


def test_one_process_at_a_time_no_unowned_survivor_and_a_frozen_sidecar_dispatches(camp, monkeypatch) -> None:
    c, home = camp["c"], camp["home"]
    monkeypatch.setenv("FAKE_GEN_SLEEP", "2.0")
    first = c.post("/api/v2/app/campaigns/preflight", json={"cell": "usdcadm_h1", "preset": "SMOKE", "seed": 5})
    assert first.status_code == 200
    refused(home, c.post("/api/v2/app/campaigns/preflight", json={"cell": "usdcadm_h1", "preset": "SMOKE", "seed": 6}),
            "another campaign process is running")
    rc.kill_tree(rc._procs[first.json()["run"]]["proc"])
    ghost = run_path(home, "20261010T000000Z-usdcadm_h1-s9")
    ghost.mkdir(parents=True)
    rc.append(ghost / "spawns.jsonl", {"kind": "runner", "pid": os.getpid(), "created": rc.process_created(os.getpid())})
    if os.name == "nt":
        refused(home, c.post("/api/v2/app/campaigns/preflight", json={"cell": "usdcadm_h1", "preset": "SMOKE",
                                                                    "seed": 7}), "earlier sidecar session")
    (ghost / "spawns.jsonl").unlink()
    exe = "C:/Program Files/ARQVIO/arqvio-engine.exe"
    monkeypatch.setattr(rc.sys, "executable", exe)  # the installed runtime: CPython itself, named arqvio-engine.exe
    assert rc.launcher_argv("fresh", "ga-x", ["--cell", "c"])[:4] == [exe, "-u", "-m", rc.LAUNCHER_MODULE]
    monkeypatch.setattr(rc.sys, "frozen", True, raising=False)  # a compiled binary has no "-m": it dispatches
    assert rc.launcher_argv("fresh", "ga-x", ["--cell", "c"]) == [
        exe, "--run-module", rc.LAUNCHER_MODULE, "--runner", rc.RUNNER_MODULE, "--expect-campaign-id", "ga-x",
        "--mode", "fresh", "--", "--cell", "c"]
    for bad in ("nope", "..%2F..%2Fconfig", "20261010T000000Z-usdcadm_h1-s1"):
        assert c.get(f"/api/v2/app/campaigns/{bad}/status").status_code == 404


def test_the_gene_pool_cannot_change_while_a_campaign_runs(camp, monkeypatch) -> None:
    monkeypatch.setattr(rc, "live", lambda run=None: {"20261010T000000Z-usdcadm_h1-s1": {}})
    for verb in ("put", "patch"):
        refused(camp["home"], getattr(camp["c"], verb)("/api/v1/gene-pool", json={}), "gene pool cannot change")


@pytest.mark.skipif(os.name != "nt", reason="the WinError 5 hazard is Windows-only")
def test_why_runner_files_are_never_opened_during_a_run(tmp_path) -> None:
    """Any open read handle on Windows makes the runner's os.replace (checkpoint.save, mv2.write_json) fail."""
    live_file, tmp = tmp_path / "checkpoint.npz", tmp_path / "checkpoint.npz.tmp"
    live_file.write_bytes(b"generation 1")
    tmp.write_bytes(b"generation 2")
    with live_file.open("rb"), pytest.raises(PermissionError):
        os.replace(tmp, live_file)
    os.replace(tmp, live_file)
    assert live_file.read_bytes() == b"generation 2"


def test_live_status_reads_progress_from_the_runners_own_lines_only(camp, monkeypatch) -> None:
    c, home = camp["c"], camp["home"]
    monkeypatch.setenv("FAKE_HOLD_AT_GEN", "1")
    run = ready_run(c, seed=77)
    c.post(f"/api/v2/app/campaigns/{run}/start", json={"confirm": True})
    wait_generation(c, run, 1)  # fake_checkpoint_state fails the request if status opened the checkpoint now
    for _ in range(10):
        s = status(c, run)
        assert s["state"] == "RUNNING" and s["checkpoint_generation"]["value"] == 1
        assert re.fullmatch(r"<ARQVIO_HOME>/artifacts/app/campaigns/.+/runner\.log:\d+",
                            s["checkpoint_generation"]["artifact_path"])
        assert s["progress"]["value"].startswith("gen  1:") and s["monitor"]["state"] == "NO DATA"
        assert "WinError 5" in s["monitor"]["reason"]
    line = int(s["checkpoint_generation"]["artifact_path"].rsplit(":", 1)[1])
    log = (run_path(home, run) / "runner.log").read_text(encoding="utf-8").splitlines()
    assert "CHECKPOINT after generation 1 -> " in log[line - 1]
    with pytest.raises(RuntimeError, match="not opened while its writer is alive"):
        REAL_CHECKPOINT_STATE(run_path(home, run) / "checkpoint.npz", "0" * 16)


def test_the_real_checkpoint_reader_refuses_what_the_engine_loader_refuses(tmp_path) -> None:
    bad = tmp_path / "checkpoint.npz"
    bad.write_bytes(b"not a checkpoint")
    with pytest.raises(Exception) as exc:
        REAL_CHECKPOINT_STATE(bad, "0" * 16)
    assert not isinstance(exc.value, AssertionError)
    with pytest.raises(FileNotFoundError):
        REAL_CHECKPOINT_STATE(tmp_path / "absent.npz", "0" * 16)


def side_process(body: str) -> subprocess.Popen:
    code = ("import subprocess, sys, time; from engine.api import routes_app_campaigns as rc; "
            "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)']); rc.adopt(p); "
            f"print(p.pid, rc.process_created(p.pid), flush=True); {body}")
    return subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True,
                            cwd=str(Path(__file__).resolve().parents[2]),
                            env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])})


@pytest.mark.skipif(os.name != "nt", reason="Windows job objects")
def test_adopted_children_die_with_the_sidecar_and_its_exit_code_survives() -> None:
    side = side_process("time.sleep(120)")
    pid, created = side.stdout.readline().split(" ", 1)
    created = created.strip()
    assert rc.process_created(int(pid)) == created
    subprocess.run(["taskkill", "/F", "/PID", str(side.pid)], capture_output=True, check=False)  # the sidecar dies
    side.wait(timeout=60)
    t0 = time.monotonic()
    while rc.process_created(int(pid)) == created and time.monotonic() - t0 < 30:
        time.sleep(0.2)
    assert rc.process_created(int(pid)) != created  # the job closed with its only handle: the child is gone
    exiting = side_process("sys.exit(3)")
    exiting.stdout.readline()
    assert exiting.wait(timeout=60) == 3  # the sidecar is not a job member: its own exit code is never masked
