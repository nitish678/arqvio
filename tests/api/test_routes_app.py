"""S5-APP read API contract (DR-2026-10-10-s5-app §7-§9): every value addressed (L1), the picker fail-closed (L2), the
owner rulings D-1/D-2/D-3/D-4/D-7/D-8 honoured verbatim, refusals and log lines traced to their source lines (LG1),
and the stats payload byte-identical on a fresh-process recompute (L5). Hermetic: a temp project root (the real
registry + synthetic reports) and an isolated ARQVIO_HOME with synthetic artifacts in the engine's own shapes."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from engine.api import routes_app as ra
from engine.api.server import create_app
from engine.core import paths
from engine.research.registry import cells as registry

USD_CAMPAIGN = "ga-usdcadm_h1-s7-cbcb2ae061e70bef"
XAU_CAMPAIGN = "ga-xauusdm_h1-s7-f32e0d4defd9cf07"
REPO = Path(__file__).resolve().parents[2]


def write(p: Path, doc) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(doc if isinstance(doc, str) else json.dumps(doc), encoding="utf-8")
    return p


def manifest(cell: str, n: int = 2) -> dict:
    cands = [{"dna_hash": f"{i:016x}", "dna": {"signal": "RSI_REVERSION"}, "dsr": {"primary_final": 0.1 * i},
              "metrics": {cell: {"total_return": 0.25 + i, "max_dd": 0.05, "sharpe": 0.8, "n_trades": 100 + i,
                                 "costs": {"spread": 1.0}, "sham": {"margin": 0.1}}}} for i in range(n)]
    return {"candidates": cands, "run": {
        "config": {"population": 200, "seed": 7, "ga": {"generations": 25, "trials_budget": 8000, "final_n": 20,
                                                       "gates": {"dsr_min": 0.5}, "gene_pool_hash": "c9c5"}},
        "dsr_n_final": 4760, "trials_used": 4760,
        "cells": {cell: {"canary": {"vs_random_p_max": 0.0005, "verdict": "WINNER BEATS RANDOM"}}}}}


@pytest.fixture
def app_root(isolated_arqvio_home: Path, tmp_path: Path, monkeypatch) -> dict:
    root = tmp_path / "repo"
    shutil.copy(REPO / "configs" / "cells_registry.json", write(root / "configs" / "x", "").parent)
    write(root / "artifacts" / "reports" / "u2" / "xau_power" / "xau_scope.json",
          {"scope_attached": "The XAU-H1 NONE verdict rules out only edges of annual Sharpe >= 2.0"})
    for name in ("verdict_citations.json", "statistical_context.json", "campaign_presets.json"):
        shutil.copy(REPO / "configs" / "app" / name, write(root / "configs" / "app" / name, "{}"))
    report = write(root / "artifacts" / "reports" / "s4_stage2_report.md",
                   "\n".join(f"line {n}" for n in range(1, 421)) + "\n- **Verdict**: 0 of 20 pass DSR.\n  WEAK.\n")
    write(root / "artifacts" / "reports" / "s4_stage3" / "power_out" / "power_table.json",
          {"verdict": {"controlled": True}, "table": [{"N": 20, "strength": 0.0, "detected": 0.005}],
           "design": "docs/x.md"})
    write(root / "docs" / "cells" / "gbpjpym_h1.md", "# x\n## Known Limitations\n- spread basis absent\n## Power\n")
    write(root / "artifacts" / "reports" / "u4" / "prior_trials_gbpjpym_h1.log",
          "  bytes 1  sha256 " + "d" * 64 + "\nPRIOR TRIALS on GBPJPYm H1: 376\nEXIT=0\n")
    write(root / "artifacts" / "reports" / "u4" / "wizard" / "audit.jsonl",
          json.dumps({"step": "pin", "symbol": "GBPJPYm", "tf": "H1", "outcome": "REFUSED",
                      "reason": "unexplained-gap", "utc": "2026-10-09T05:23:00+00:00"}) + "\n")
    monkeypatch.setattr(paths, "PROJECT_ROOT", root)
    home = isolated_arqvio_home
    art = home / "artifacts"
    write(home / "config" / "accounts.json", {"accounts": [
        {"id": "acct1", "login": 1, "server": "Fake-Demo", "terminal": "C:/fake/terminal64.exe",
         "password_env": "UNSET", "probe_symbol": "USDCADm", "pinned_symbol": "USDCADm"}]})
    write(art / "health" / "status.json", {"schema": "s7", "version": 1, "snapshot_utc": "2026-10-02T15:07:21+00:00",
                                           "accounts": [{"id": "acct1", "demo": True, "basis": {"basis": "server_equals_utc"},
                                                         "probed_utc": "2026-10-02T15:07:21+00:00"}]})
    for s in ("USDCADm", "XAUUSDm", "GBPJPYm", "EURUSDm"):
        write(art / "spread" / f"{s}.json", {"symbol": s})
    write(art / "datasets" / "USDCADm_H1_aed3bf314539ec5a.json",
          {"dataset_id": "aed3bf314539ec5a", "symbol": "USDCADm", "timeframe": "H1", "n_bars": 59957})
    usd = write(art / "ga" / USD_CAMPAIGN / "manifest_v2.json", manifest("usdcadm_h1"))
    write(art / "ga" / USD_CAMPAIGN / "individuals.json",
          [{"dna_hash": f"{0:016x}", "metrics": {"exit_reasons": {"SL": 60, "TP": 39, "SLTP_DEFERRED_ZERO_SPREAD": 1}}}])
    write(art / "cells" / "gbpjpym_h1" / "w2_manifest.json",
          {"n_bars": 59034, "gaps": {"owner_accepted_holes": [{"missing_from": "a"}] * 5, "owner_accepted_bars": 43,
                                     "acceptance": {"file": "configs/cells/gbpjpym_h1.accepted_holes.json"}}})
    return {"root": root, "home": home, "report": report, "usd_manifest": usd, "client": TestClient(create_app())}


def test_environment_probe_is_read_only(app_root, monkeypatch) -> None:
    """ld-yes (DR §2): the /environment endpoint now includes an mt5_live block
    from the read-only probe, but the probe itself is allowlisted and the frontend
    never imports MetaTrader5.  This test blocks MetaTrader5 at the module table so
    an accidental mt5.* write call would surface as a hard failure; the probe falls
    back to MT5_NOT_RUNNING honestly rather than inventing a result."""
    import sys as _sys
    monkeypatch.setitem(_sys.modules, "MetaTrader5", None)
    e = app_root["client"].get("/api/v2/app/environment").json()
    assert e["build"]["commit"] and e["build"]["state"] in ("OK", "SOURCE", "UNAVAILABLE")
    a = e["accounts"]["items"][0]
    assert (a["id"], a["demo"]["value"], a["demo"]["state"]) == ("acct1", True, "STALE")  # snapshot 2026-10-02
    assert a["demo"]["artifact_path"] == "<ARQVIO_HOME>/artifacts/health/status.json" and len(a["demo"]["sha256"]) == 64
    assert e["validated"]["cells"] == ["gbpjpym_h1", "usdcadm_h1", "xauusdm_h1"]
    assert e["detected_only"]["symbols"] == ["EURUSDm"]  # ENV2: never merged with the validated symbols
    assert e["machine"]["cpu_logical"]["value"] and "ctypes" in e["machine"]["source"]
    # ld-yes: the mt5_live block is present, honest (MT5_NOT_RUNNING since MetaTrader5 is blocked),
    # with source + as_of, and DETECTED != VALIDATED is preserved.
    ml = e["mt5_live"]
    assert ml["state"] == "MT5_NOT_RUNNING"
    assert ml["source"] == "mt5_live_probe"
    assert ml["as_of"] is not None and isinstance(ml["as_of"], str)
    assert ml["terminal_path"] is None
    assert ml["terminal_source"] == "NOT_FOUND"
    assert ml["detail"] is not None  # verbatim reason, not invented


def test_cells_carry_the_triple_and_the_ready_for_first_search_state(app_root) -> None:
    cs = {c["cell_id"]: c for c in app_root["client"].get("/api/v2/app/cells").json()["cells"]}
    g = cs["gbpjpym_h1"]
    assert g["render_state"] == "READY FOR FIRST SEARCH"
    assert g["power_check"]["value"]["status"] == "BLOCKED:no-campaign" and g["power_check"]["sha256"]
    assert (g["known_holes"]["holes"]["value"], g["known_holes"]["bars"]["value"]) == (5, 43)
    assert g["prior_trials"]["prior_trials"]["value"] == 376 and g["prior_trials"]["database_sha256"] == "d" * 64
    assert "376 prior unbanked trials not included in deflation" in g["prior_trials"]["caveat"]
    assert "spread basis absent" in g["known_limitations"]["text"]
    assert cs["usdcadm_h1"]["render_state"] == "VALIDATED" and cs["usdcadm_h1"]["prior_trials"] is None
    x = cs["xauusdm_h1"]["scope_note"]  # DR §6: XAU carries the U2 scope caveat, from its cited source
    assert x["value"].startswith("The XAU-H1 NONE verdict rules out") and x["sha256"] and x["field"] == "scope_attached"
    assert x["artifact_path"] == "artifacts/reports/u2/xau_power/xau_scope.json" and g["scope_note"] is None


def test_the_stats_card_renders_stored_metrics_only(app_root) -> None:
    s = app_root["client"].get("/api/v2/app/cells/usdcadm_h1/stats").json()
    b = s["best"]
    sha = hashlib.sha256(app_root["usd_manifest"].read_bytes()).hexdigest()
    assert b["net_return_pct"]["value"] == pytest.approx(25.0) and b["net_return_pct"]["unit"] == "%"
    assert b["net_return_pct"]["field"] == "candidates[0].metrics.usdcadm_h1.total_return"
    assert b["net_return_pct"]["sha256"] == sha and b["max_dd_pct"]["value"] == pytest.approx(5.0)
    for k in ("win_rate", "profit_factor", "expectancy"):  # owner D-1: NO DATA with the machine reason verbatim
        assert (b[k]["state"], b[k]["reason"]) == ("NO DATA", "not recorded by campaign machinery; derivation "
                                                              "deferred to R-stage — owner ledger")
    assert b["exit_anatomy"]["value"] == {"SL": 60, "TP": 39, "SLTP deferred (zero spread)": 1}  # D-2
    assert b["survivor"]["state"] == "NO DATA"  # no artifact records it; never re-derived from the gates (D-3)
    v = s["verdict"]  # D-3: the report-text citation, with the file's sha256 and the badge
    assert v["kind"] == "report_text" and v["badge"] == "verdict recorded as report text (pre-registry era)"
    assert "0 of 20 pass DSR" in v["text"]["value"] and v["text"]["field"] == "lines 421-422"
    assert v["text"]["sha256"] == hashlib.sha256(app_root["report"].read_bytes()).hexdigest()
    sc = s["statistical_context"]  # D-8
    assert sc["vs_random"]["p_max"]["value"] == 0.0005 and sc["power_curve"]["rows"][0]["N"] == 20
    assert sc["mc_fan"] == {"title": "Monte Carlo equity fan", "state": "NO DATA",
                            "reason": "Monte Carlo equity fan — NO DATA (engine records no MC artifact)"}
    assert s["power_status"]["value"]["status"] == "COMPUTED"  # L4 on every verdict


def test_a_recorded_survivor_flag_comes_from_the_verdict_artifact(app_root) -> None:
    art = app_root["home"] / "artifacts"
    write(art / "ga" / XAU_CAMPAIGN / "manifest_v2.json", manifest("xauusdm_h1"))
    write(art / "campaign_xau_h1" / "verdict.json", {"campaign_id": XAU_CAMPAIGN, "verdict": "NONE",
                                                     "finalists": [{"dna_hash": f"{0:016x}", "survivor": False}]})
    s = app_root["client"].get("/api/v2/app/cells/xauusdm_h1/stats").json()
    assert s["verdict"]["kind"] == "artifact" and s["verdict"]["text"]["value"] == "NONE"
    assert (s["best"]["survivor"]["value"], s["best"]["survivor"]["state"]) == (False, "OK")
    assert s["finalists"][1]["survivor"]["state"] == "NO DATA"


def test_the_picker_is_fail_closed_and_an_unrun_cell_has_no_campaign(app_root) -> None:
    c = app_root["client"]
    assert c.get("/api/v2/app/cells/eurusdm_h1/stats").status_code == 404  # L2
    s = c.get("/api/v2/app/cells/gbpjpym_h1/stats").json()
    assert s["state"] == "NO CAMPAIGN" and s["render_state"] == "READY FOR FIRST SEARCH"


def test_wizard_status_is_derived_and_add_cell_is_an_instruction(app_root) -> None:
    rows = {r["symbol"]: r for r in app_root["client"].get("/api/v2/app/wizard").json()["rows"]}
    assert rows["GBPJPYm"]["state"] == "VALIDATED" and rows["GBPJPYm"]["refusals"][0]["reason"] == "unexplained-gap"
    assert rows["GBPJPYm"]["refusals"][0]["origin"] == "artifacts/reports/u4/wizard/audit.jsonl:1"
    assert rows["EURUSDm"]["state"] == "NOT VALIDATED YET"
    assert "wizard ceremonies run from the CLI" in rows["EURUSDm"]["instruction"]
    assert "QUEUED" not in json.dumps(rows)  # D-4: nothing writes a queue


def test_refusals_and_logs_trace_to_their_source_lines(app_root) -> None:
    c = app_root["client"]
    write(ra.app_events_path(), json.dumps({"utc": "2026-10-10T08:00:00+00:00", "outcome": "REFUSED", "level": "WARN",
                                            "run": "r1", "reason": "resume-args-differ"}) + "\n")
    r = c.get("/api/v2/app/refusals").json()
    assert {x["reason"] for x in r["records"]} == {"unexplained-gap", "resume-args-differ"}
    assert all(x["raw"] and x["origin"].rsplit(":", 1)[1].isdigit() and len(x["sha256"]) == 64 for x in r["records"])
    lg = c.get("/api/v2/app/logs?source=wizard-audit").json()
    assert lg["lines"][0]["origin"] == "artifacts/reports/u4/wizard/audit.jsonl:1" and lg["lines"][0]["level"] == "WARN"
    assert json.loads(lg["lines"][0]["text"])["reason"] == "unexplained-gap"  # the raw line, verbatim
    assert c.get("/api/v2/app/logs?source=made-up").status_code == 404


def test_the_stats_payload_is_byte_identical_on_a_fresh_process_recompute(app_root) -> None:
    code = ("import json,sys; from fastapi.testclient import TestClient; from engine.core import paths; "
            "from pathlib import Path; paths.PROJECT_ROOT = Path(sys.argv[1]); "
            "from engine.api.server import create_app; "
            "print(json.dumps(TestClient(create_app()).get('/api/v2/app/cells/usdcadm_h1/stats').json(), "
            "sort_keys=True))")
    env = {**__import__("os").environ, "PYTHONPATH": str(REPO), "ARQVIO_HOME": str(app_root["home"])}
    runs = [subprocess.run([sys.executable, "-c", code, str(app_root["root"])], capture_output=True, text=True,
                           env=env, check=True).stdout.splitlines()[-1] for _ in range(2)]
    assert runs[0] == runs[1] and json.loads(runs[0])["state"] == "OK"  # L5


def test_machine_and_cell_campaigns_follow_the_dr_endpoint_list(app_root) -> None:
    c = app_root["client"]
    m = c.get("/api/v2/app/machine").json()
    assert m["cpu_logical"]["value"] and m["as_of"] and "ctypes" in m["source"]  # D-7
    cc = c.get("/api/v2/app/cells/usdcadm_h1/campaigns").json()
    assert cc["cited"]["value"] == f"RUN@{USD_CAMPAIGN}" and cc["cited"]["sha256"] and cc["app_runs"] == []
    assert c.get("/api/v2/app/cells/eurusdm_h1/campaigns").status_code == 404  # L2


def test_the_registry_entry_hashes_are_untouched(app_root) -> None:
    before = registry.registry_path().read_bytes()
    app_root["client"].get("/api/v2/app/cells")
    assert registry.registry_path().read_bytes() == before  # the read API never writes


def test_first_run_status_gates_on_accounts_json(isolated_arqvio_home, monkeypatch) -> None:
    """DR §3: the shell uses GET /first-run to decide whether to show the wizard.
    With no accounts.json the endpoint returns first_run=True (and the honest path);
    after scaffold it returns first_run=False."""
    from engine.api.server import create_app
    c = TestClient(create_app())
    # No accounts.json in the isolated home -> first run.
    r = c.get("/api/v2/app/first-run").json()
    assert r["first_run"] is True
    assert r["accounts_path"].endswith("accounts.json")
    # Scaffold creates accounts.json -> not first run.
    s = c.post("/api/v2/app/first-run/scaffold").json()
    assert s["scaffolded"] is True and s["existed"] is False
    r2 = c.get("/api/v2/app/first-run").json()
    assert r2["first_run"] is False
