"""The strategy vault: imports trace to the manifest, tiers come from the gates, robustness completes the score."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from engine.research.search import sampler
from engine.research.vault import library, robustness

CAMPAIGN = "ga-usdcadm_h1-s1-test"


def candidate(dna, *, sharpe=1.2, killed=None, dsr=0.9, sham=0.8, cross=0.6, trades=320, dd=0.12) -> dict:
    def cell(s, k):
        return {"n_trades": trades, "max_dd": dd, "total_return": 0.2, "engine": "kernel", "costs": {},
                "fitness": {"sharpe": s, "killed_by": k, "fitness": s, "sham_passed": sham >= 0}}
    return {"dna": dna.to_dict(), "dna_hash": dna.dna_hash, "run_hash": "r" * 64, "config_hash": "c" * 64,
            "metrics": {"usdcadm_h1": cell(sharpe, killed), "xauusdm_h1": cell(cross, None)},
            "sham_margin": {"usdcadm_h1": sham, "xauusdm_h1": 0.1},
            "dsr": {"primary_final": {"dsr": dsr, "status": "READY", "n_trials": 400}},
            "cross_cell_delta": {"alive_on_both": killed is None}}


def write_campaign(home: Path, cands: list[dict], name: str = CAMPAIGN) -> Path:
    d = home / "artifacts" / "ga" / name
    d.mkdir(parents=True)
    p = d / "manifest_v2.json"
    p.write_text(json.dumps({"schema": "arqvio.ga.manifest", "version": 2, "candidates": cands,
                             "cells": {"primary": "usdcadm_h1", "cross": "xauusdm_h1"},
                             "run": {"completed_at_utc": "2026-10-22T00:00:00+00:00"}}), encoding="utf-8")
    return p


def dnas(n: int) -> list:
    rng = np.random.default_rng(11)
    return [sampler.random_dna(rng) for _ in range(n)]


def done(share=0.9, segs=(0.5, 0.7, 0.4), p=0.02, identical=True, month_share=0.2, in_market=0.9, plateau=0.8,
         mc_dd=0.2, beaten=0.99) -> dict:
    return {"state": "DONE", "reproduction": {"identical": identical},
            "neighbourhood": {"positive_share": share, "plateau_share": plateau},
            "segments": {"sharpe": list(segs)}, "bootstrap": {"p_sharpe_le_0": p, "max_dd_p95": mc_dd},
            "random": {"n": 100, "beaten_share": beaten},
            "concentration": {"months": 12, "largest_month_share": month_share, "months_in_market_share": in_market}}


def test_import_copies_alive_finalists_with_their_trace_and_never_twice(isolated_arqvio_home: Path) -> None:
    a, b, c = dnas(3)
    write_campaign(isolated_arqvio_home, [candidate(a), candidate(b, killed="min_trades"), candidate(c, sharpe=0.3)])
    out = library.import_campaign(CAMPAIGN)
    assert out["added"] == [f"{a.dna_hash}@usdcadm_h1", f"{c.dna_hash}@usdcadm_h1"]
    assert out["skipped"] == [{"index": 1, "reason": "killed by min_trades"}]
    e = library.get(out["added"][0])
    assert e["source"]["campaign_id"] == CAMPAIGN and len(e["source"]["manifest_sha256"]) == 64
    assert (e["symbol"], e["timeframe"]) == ("USDCADm", "H1")  # from the validated cell registry
    again = library.import_campaign(CAMPAIGN)
    assert again["added"] == [] and {s["reason"] for s in again["skipped"]} >= {"already in the vault"}
    assert [x["campaign_id"] for x in library.completed_campaigns()] == [CAMPAIGN]


def test_a_strategy_remembers_the_cost_model_it_was_bred_with(isolated_arqvio_home: Path) -> None:
    a, b = dnas(2)
    p = write_campaign(isolated_arqvio_home, [candidate(a)])
    eid = library.import_campaign(CAMPAIGN)["added"][0]
    assert library.get(eid)["cost_model"] == "hourly"  # no bar_spread source in the manifest
    doc = json.loads(p.read_text(encoding="utf-8"))
    doc["candidates"] = [candidate(b)]
    doc["run"]["cells"] = {"usdcadm_h1": {"spread_cost_source": "bar_spread+mean_buckets"}}
    write_campaign(isolated_arqvio_home, doc["candidates"], name="ga-usdcadm_h1-s2-test").write_text(
        json.dumps(doc), encoding="utf-8")
    bid = library.import_campaign("ga-usdcadm_h1-s2-test")["added"][0]
    assert library.get(bid)["cost_model"] == "bar"
    assert library.recorded_cost_model({"id": "pre-1.2"}) == "hourly"


def test_an_unknown_or_unsafe_campaign_is_refused(isolated_arqvio_home: Path) -> None:
    for bad in ("nope", "../x", "a\\b"):
        with pytest.raises(library.VaultError):
            library.import_campaign(bad)


def test_the_tier_is_certified_only_when_every_strict_gate_passes(isolated_arqvio_home: Path) -> None:
    (a,) = dnas(1)
    write_campaign(isolated_arqvio_home, [candidate(a)])
    eid = library.import_campaign(CAMPAIGN)["added"][0]
    e = library.get(eid)
    assert e["tier"] == "CANDIDATE" and e["score_max"] == 60.0  # robustness and held-out not run: unjudged
    assert [g["gate"] for g in e["gates"] if g["passed"] is None] == [
        "reproduces exactly", "neighbours that make money", "every time segment positive", "bootstrap P(Sharpe <= 0)",
        "plateau: neighbours keeping half the Sharpe", "Monte Carlo p95 drawdown", "beats random-entry twins",
        "best month's share of the profit", "held-out trades (bars breeding never saw)", "held-out return",
        "months with a position"]
    r = library.set_robustness(eid, done())
    assert r["tier"] == "CANDIDATE"  # the held-out test has not run: unjudged, never assumed
    r = library.set_held_out(eid, {"state": "DONE", "trades": 40, "return": 0.08, "sharpe": 0.9})
    assert r["tier"] == "CERTIFIED" and r["score_max"] == 100.0 and r["grade"] in "AB"
    for bad_ho in ({"state": "DONE", "trades": 9, "return": 0.08}, {"state": "DONE", "trades": 40, "return": -0.01},
                   {"state": "NONE", "reason": "bred before v4"}):
        assert library.set_held_out(eid, bad_ho)["tier"] == "CANDIDATE"
    library.set_held_out(eid, {"state": "DONE", "trades": 40, "return": 0.08, "sharpe": 0.9})
    for bad in (done(share=0.5), done(segs=(0.5, -0.1, 0.4)), done(p=0.3), done(identical=False),
                done(month_share=0.89), done(month_share=None), done(in_market=0.25), done(plateau=0.4),
                done(mc_dd=0.45), done(beaten=0.9)):
        assert library.set_robustness(eid, bad)["tier"] == "CANDIDATE"


@pytest.mark.parametrize("change", [{"dsr": 0.2}, {"sham": -0.1}, {"cross": -0.2}, {"trades": 60}, {"dd": 0.45}])
def test_one_failed_record_gate_keeps_a_strategy_a_candidate(isolated_arqvio_home: Path, change: dict) -> None:
    (a,) = dnas(1)
    write_campaign(isolated_arqvio_home, [candidate(a, **change)])
    eid = library.import_campaign(CAMPAIGN)["added"][0]
    assert library.set_robustness(eid, done())["tier"] == "CANDIDATE"


def test_a_robustness_run_from_before_v4_leaves_the_new_gates_unjudged(isolated_arqvio_home: Path) -> None:
    """No plateau, Monte Carlo drawdown or random twins in the record: the gates say "re-run", never a pass, and the
    score is out of what was measured."""
    (a,) = dnas(1)
    write_campaign(isolated_arqvio_home, [candidate(a)])
    eid = library.import_campaign(CAMPAIGN)["added"][0]
    old = done()
    old["neighbourhood"].pop("plateau_share")
    old["bootstrap"].pop("max_dd_p95")
    old.pop("random")
    r = library.set_robustness(eid, old)
    judged = {g["gate"]: g["passed"] for g in r["gates"]}
    assert [judged[k] for k in ("plateau: neighbours keeping half the Sharpe", "Monte Carlo p95 drawdown",
                                "beats random-entry twins")] == [None, None, None]
    assert r["score_max"] == 60.0 + 8.0 + 5.0 and r["tier"] == "CANDIDATE"
    assert library.set_robustness(eid, done())["score_max"] == 60.0 + 15.0 + 5.0


def test_beat_random_counts_twins_beaten_strictly() -> None:
    out = robustness.beat_random(1.0, [0.5, 1.0, 2.0, float("nan"), -0.3])
    assert out["n"] == 4 and out["beaten_share"] == 0.5 and robustness.beat_random(1.0, [])["beaten_share"] is None


def test_score_and_grade_bands() -> None:
    assert [library.grade(x) for x in (95, 80, 79.9, 65, 50, 35, 34.9)] == ["A", "A", "B", "B", "C", "D", "F"]


def test_neighbours_are_valid_distinct_one_step_nudges() -> None:
    for dna in dnas(40):
        ns = robustness.neighbours(dna)
        hashes = [d.dna_hash for _, d in ns]
        assert ns and len(set(hashes)) == len(hashes) and dna.dna_hash not in hashes
        assert all(label for label, _ in ns)


def test_robustness_is_seeded_and_reports_every_check() -> None:
    (dna,) = dnas(1)
    rng = np.random.default_rng(1)
    base_returns = rng.normal(0.0004, 0.01, 3000)

    def evaluate(d):
        shift = 0.0 if d.dna_hash == dna.dna_hash else 0.0001 * (int(d.dna_hash[:2], 16) % 3 - 1)
        r = base_returns + shift
        s = robustness.annual_sharpe(r, "H1")
        return SimpleNamespace(crashed=False, engine="kernel",
                               result=SimpleNamespace(sharpe=s, n_trades=100, per_period_returns=r))

    recorded = robustness.annual_sharpe(base_returns, "H1")
    asked = []

    def twin(d, entries, k):
        asked.append((d.dna_hash, entries, k))
        return -0.5 + 0.01 * k  # twins 0..99: Sharpe -0.5 .. 0.49, all below the strategy's

    a = robustness.run(dna, timeframe="H1", recorded_sharpe=recorded, evaluate=evaluate, random_sharpe=twin)
    b = robustness.run(dna, timeframe="H1", recorded_sharpe=recorded, evaluate=evaluate, random_sharpe=twin)
    assert a == b and a["state"] == "DONE" and a["reproduction"]["identical"] is True
    assert a["neighbourhood"]["n"] == len(robustness.neighbours(dna)) and len(a["segments"]["sharpe"]) == 3
    assert 0.0 <= a["bootstrap"]["p_sharpe_le_0"] <= 1.0 and a["bootstrap"]["block_bars"] == 24
    assert "in-sample" in a["label"] and a["random"]["n"] == robustness.N_RANDOM
    assert a["random"]["beaten_share"] == (1.0 if recorded > 0.49 else a["random"]["beaten_share"])
    assert {k for _, _, k in asked} == set(range(robustness.N_RANDOM)) and {e for _, e, _ in asked} == {100}
    assert 0.0 <= a["neighbourhood"]["plateau_share"] <= 1.0
    assert robustness.run(dna, timeframe="H1", recorded_sharpe=recorded, evaluate=evaluate)["random"] is None


def test_concentration_names_the_month_that_carries_the_profit() -> None:
    """The XAGUSD M5 shape: one big month, then months with no position at all."""
    day = 86400
    times = np.array([day * (31 * m + d) for m in range(6) for d in range(20)], dtype=np.int64)
    r = np.zeros(times.size)
    r[0:20] = 0.01          # month 1: the profit
    r[20:25] = -0.001       # month 2: a little loss; months 3-6: flat, no position
    c = robustness.concentration(r, times)
    assert c["months"] == 6 and c["largest_month"] == "1970-01"
    assert c["largest_month_share"] > 1.0 - 1e-9 or c["largest_month_share"] > 0.99
    assert c["months_in_market_share"] == 2 / 6
    lost = robustness.concentration(-np.abs(r), times)
    assert lost["largest_month_share"] is None  # no profit: no share to judge
    # plain Python values: a numpy float here made the coverage gate a numpy bool, and the vault save failed
    assert type(c["months_in_market_share"]) is float and type(c["largest_month"]) is str


def test_a_real_shaped_robustness_result_saves_with_its_gates(isolated_arqvio_home: Path) -> None:
    (a,) = dnas(1)
    write_campaign(isolated_arqvio_home, [candidate(a)])
    eid = library.import_campaign(CAMPAIGN)["added"][0]
    rb = done()
    rb["concentration"] = robustness.concentration(np.r_[np.full(20, 0.01), np.full(20, -0.001)],
                                                   np.arange(40, dtype=np.int64) * 86400 * 20)
    r = library.set_robustness(eid, rb)
    assert all(type(g["passed"]) in (bool, type(None)) for g in r["gates"])
    json.dumps(library.get(eid))


def test_copies_of_one_behaviour_are_one_cluster_led_by_the_best() -> None:
    def e(i, trades=100, ret=0.25, sharpe=1.5, cell="xagusd_m5"):
        return {"id": i, "cell": cell, "metrics": {"n_trades": trades, "total_return": ret, "sharpe": sharpe}}
    rows = [e("a"), e("b"), e("c", ret=0.25 + 1e-15), e("d", trades=101), e("x", cell="eurusd_m5")]
    c = library.copies(rows)
    assert c["a"] == {"copies": 3, "copy_of": None}
    assert c["b"]["copy_of"] == "a" and c["c"]["copy_of"] == "a"
    assert c["d"] == {"copies": 1, "copy_of": None} and c["x"] == {"copies": 1, "copy_of": None}


def test_held_out_metrics_count_only_the_bars_after_the_split() -> None:
    from engine.research.vault import held_out

    eq = np.array([100.0, 110.0, 121.0, 121.0, 108.9, 119.79])  # +10%, +10%, 0, -10%, +10%
    m = held_out.metrics(eq, np.array([0, 1, 3, 4]), split=3, timeframe="H1")
    assert m["trades"] == 2 and m["bars"] == 2
    assert abs(m["return"] - (119.79 / 121.0 - 1.0)) < 1e-12 and abs(m["max_dd"] - 0.1) < 1e-12


def test_deleted_strategies_go_to_the_recycle_bin_and_come_back(isolated_arqvio_home: Path) -> None:
    a, b = dnas(2)
    write_campaign(isolated_arqvio_home, [candidate(a), candidate(b, sharpe=0.1, dd=0.29, trades=20)])
    ids = library.import_campaign(CAMPAIGN)["added"]
    library.remove(ids[0])
    assert [r["id"] for r in library.load()] == [ids[1]]
    assert [r["id"] for r in library.recycled()] == [ids[0]] and library.recycled()[0]["deleted_reason"] == "deleted"
    library.restore(ids[0])
    assert {r["id"] for r in library.load()} == set(ids) and library.recycled() == []
    with pytest.raises(library.VaultError):
        library.restore(ids[0])


def test_purge_previews_then_moves_only_what_its_rule_names(isolated_arqvio_home: Path) -> None:
    a, b = dnas(2)
    write_campaign(isolated_arqvio_home, [candidate(a), candidate(b, sharpe=0.05, dd=0.29, trades=20, sham=0.0, cross=0.0)])
    ids = library.import_campaign(CAMPAIGN)["added"]
    weak = [r["id"] for r in library.load() if r["grade"] == "F"]
    assert weak, "fixture: one strategy must grade F"
    pre = library.purge("grade_f")
    assert pre["ids"] == weak and not pre["applied"] and len(library.load()) == 2
    done = library.purge("grade_f", apply=True)
    assert done["applied"] and {r["id"] for r in library.load()} == set(ids) - set(weak)
    assert [r["id"] for r in library.recycled()] == weak


def test_audit_log_export_and_address(tmp_path, monkeypatch):
    """W4-5 5.3 / 5.7: removals and restores are logged; export carries every entry with its address."""
    from engine.research.vault import library as lib

    monkeypatch.setattr(lib, "vault_dir", lambda: tmp_path)
    monkeypatch.setattr(lib, "ga_dir", lambda: tmp_path / "ga")
    rows = [{"id": f"s{i}@c", "source": {"campaign_id": "camp1", "manifest_sha256": "ab"}, "dataset_id": "d",
             "cell": "c", "added_utc": "2026-11-01T00:00:00+00:00", "tier": "CANDIDATE", "score": 1.0} for i in range(3)]
    vault = tmp_path / "vault.json"
    lib.save(rows, vault)
    lib.remove("s1@c", vault, reason="deleted")
    lib.restore("s1@c", vault)
    log = lib.audit_log()
    assert [(x["action"], x["ids"]) for x in log] == [("recycle", ["s1@c"]), ("restore", ["s1@c"])]
    (tmp_path / "audit.jsonl").open("a", encoding="utf-8").write('{"torn')
    assert len(lib.audit_log()) == 2
    doc = lib.export(None, vault)
    assert doc["count"] == 3 and doc["schema"] == "arqvio.vault_export"
    a = doc["strategies"][0]["address"]
    assert a["campaign_id"] == "camp1" and a["file"] == "manifest_v2.json" and a["as_of"].startswith("2026-11-01")
    assert lib.export(["s2@c"], vault)["count"] == 1
    with pytest.raises(lib.VaultError):
        lib.export(["nope"], vault)
