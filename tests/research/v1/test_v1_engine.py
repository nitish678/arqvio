"""Engine V1 (the research engine): config, seeds, score, the breeding loop, checkpoint/resume, skip, gates, vault.

The loop runs on a synthetic evaluator (returns drawn from each DNA's hash, with a drift that rewards one region of
the gene space), so these tests pin the mechanism, not market results.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from collections import Counter

import numpy as np
import pytest

from engine.research.ga.runner import Evaluation
from engine.research.genes import registry
from engine.research.search.runner import TrialResult
from engine.research.v1 import config as v1c
from engine.research.v1 import score as v1s
from engine.research.v1.engine import V1Run
from engine.research.v1.gates import judge, shuffled_dd_p95
from engine.research.v1.seeds import ORIGINS, seeds


class FakeEvaluator:
    """Deterministic per DNA: noise from the dna_hash, drift highest at sl_atr 2.0 and tp_atr 3.0."""

    def __init__(self, bars: int = 600) -> None:
        self.bars = bars
        self.calls = 0

    def evaluate(self, dna):
        self.calls += 1
        rng = np.random.default_rng(int(hashlib.sha256(dna.dna_hash.encode()).hexdigest()[:12], 16))
        drift = 0.0006 - 0.0002 * abs(dna.sl_atr - 2.0) - 0.0001 * abs(dna.tp_atr - 3.0)
        r = rng.normal(drift, 0.004, self.bars)
        n = int(30 + rng.integers(0, 120))
        res = TrialResult(dna=dna, sharpe=float(np.mean(r) / np.std(r) * np.sqrt(6000)), n_trades=n,
                          per_period_returns=r)
        return Evaluation(res, False, "kernel")


def quick(**kw):
    return v1c.v1_config("quick", **kw)


def test_v1_pool_is_research_only():
    pool = v1c.v1_gene_pool()
    enabled = {f for f, s in pool.families.items() if s["enabled"]}
    assert enabled == set(registry.breedable_families("v1"))
    assert "MA_CROSS" not in enabled                       # ARQVIO's family is V2's
    assert "session" not in pool.gates                     # ARQVIO's gate is V2's
    for g in registry.genes(engine="v1"):
        assert g.origin == registry.RESEARCH


def test_config_numbers_and_refusals():
    cfg = v1c.v1_config()
    assert (cfg.proving_grounds, cfg.generations, cfg.random_n, cfg.shuffle_n) == (3000, 20, 100, 1000)
    assert cfg.held_out_min_trades == 15 and cfg.shuffle_dd_max == 0.40
    with pytest.raises(ValueError):
        v1c.v1_config("nope")
    with pytest.raises(ValueError):
        v1c.v1_config("quick", tribe_size=150)
    with pytest.raises(ValueError):
        v1c.v1_config("quick", risk_pct=0.2)


def test_sixteen_seeds_research_origins():
    pool = v1c.v1_gene_pool()
    s = seeds(pool, 0.01)
    assert len(s) == 16 and Counter(o for o, _ in s) == Counter(ORIGINS)
    assert Counter(ORIGINS) == {"Wild": 10, "TRIBE A": 3, "TRIBE B": 1, "FINAL WAR": 2}
    assert {d.signal for _, d in s} == {f for f, x in pool.families.items() if x["enabled"]}
    assert all(d.risk_pct == 0.01 for _, d in s)
    assert [d.dna_hash for _, d in s] == [d.dna_hash for _, d in seeds(pool, 0.01)]


def test_score_weights_and_verdict():
    assert sum(v1s.WEIGHTS.values()) == 100.0
    good = v1s.Metrics(sharpe=2.5, total_return=0.8, max_dd=0.05, n_trades=150, profit_factor=2.5, linearity=0.95)
    weak = v1s.Metrics(sharpe=0.2, total_return=0.01, max_dd=0.28, n_trades=20, profit_factor=1.02, linearity=0.1)
    hi, lo = v1s.score(good, good, good), v1s.score(weak, weak, weak)
    assert hi["verdict"] == "GOOD" and hi["score"] >= 90
    assert lo["verdict"] == "WEAK" and lo["score"] < 55
    assert set(hi["parts"]) == set(v1s.WEIGHTS)


def test_fitness_is_log_scaled_and_tribes_differ():
    steady = v1s.Metrics(1.5, 0.10, 0.03, 120, 1.3, 0.97)
    racy = v1s.Metrics(1.5, 0.90, 0.25, 120, 2.0, 0.40)
    for m in (steady, racy):
        assert 0.0 <= v1s.fitness(m, v1s.TRIBE_A) <= np.log1p(100.0)
    assert v1s.fitness(steady, v1s.TRIBE_A) > v1s.fitness(racy, v1s.TRIBE_A)
    assert v1s.fitness(racy, v1s.TRIBE_B) > v1s.fitness(steady, v1s.TRIBE_B)


def test_run_is_deterministic_and_tribal(tmp_path):
    a = V1Run(quick(seed=7), FakeEvaluator(), "H1", out_dir=tmp_path / "a", log=lambda m: None).run()
    b = V1Run(quick(seed=7), FakeEvaluator(), "H1", out_dir=tmp_path / "b", log=lambda m: None).run()
    assert [f["effective_hash"] for f in a["finalists"]] == [f["effective_hash"] for f in b["finalists"]]
    assert a["generations"][0]["phase"] == "PROVING GROUNDS" and a["generations"][0]["tested"] == 200
    assert [g["phase"] for g in a["generations"][1:-1]] == ["TRIBES"] * 4
    assert a["generations"][-1]["phase"] == "FINAL WAR"
    assert 0 < len(a["finalists"]) <= 3
    assert len({f["effective_hash"] for f in a["finalists"]}) == len(a["finalists"])
    assert all(f["alive"] and f["dna"]["risk_pct"] == 0.01 for f in a["finalists"])


def test_resume_ends_where_the_uninterrupted_run_ends(tmp_path):
    full = V1Run(quick(seed=3), FakeEvaluator(), "H1", out_dir=tmp_path / "full", log=lambda m: None).run()

    class Stop(Exception):
        pass

    def log(msg):
        if msg.startswith("GEN 2/"):
            raise Stop

    out = tmp_path / "cut"
    with pytest.raises(Stop):
        V1Run(quick(seed=3), FakeEvaluator(), "H1", out_dir=out, log=log).run()
    # the GEN 2 line is logged before its checkpoint: the resumed run redoes generation 2 from generation 1's state
    resumed = V1Run(quick(seed=3), FakeEvaluator(), "H1", out_dir=out, log=lambda m: None).run(resume=True)
    assert [f["effective_hash"] for f in resumed["finalists"]] == [f["effective_hash"] for f in full["finalists"]]


def test_resume_refuses_another_config(tmp_path):
    V1Run(quick(seed=1), FakeEvaluator(), "H1", out_dir=tmp_path, log=lambda m: None).run()
    with pytest.raises(RuntimeError, match="different V1 configuration"):
        V1Run(quick(seed=2), FakeEvaluator(), "H1", out_dir=tmp_path, log=lambda m: None).run(resume=True)


def test_skip_finishes_the_generation_then_final_war(tmp_path):
    skip = tmp_path / "skip"
    skip.write_text("")
    r = V1Run(quick(seed=5), FakeEvaluator(), "H1", out_dir=tmp_path, log=lambda m: None, skip_path=skip).run()
    assert r["skipped"] is True
    assert [g["phase"] for g in r["generations"]] == ["PROVING GROUNDS", "TRIBES", "FINAL WAR"]


def test_shuffle_gate_is_seeded():
    r = np.random.default_rng(0).normal(0.001, 0.01, 500)
    assert shuffled_dd_p95(r, n=200, seed=9) == shuffled_dd_p95(r, n=200, seed=9)
    assert np.isnan(shuffled_dd_p95(r[:5], n=10, seed=1))


def test_judge_gates(tmp_path):
    from engine.research.backtest.kernel import TR_EXIT_BAR, TR_FILL_BAR, TR_NET, TR_SIZE

    cfg = quick(random_n=20, shuffle_n=100)
    bars, split = 400, 280
    eq = np.cumprod(np.concatenate(([10_000.0], 1.0 + np.full(bars, 0.0008))))
    rows = np.zeros((40, TR_SIZE))
    rows[:, TR_FILL_BAR] = np.linspace(5, 395, 40)
    rows[:, TR_EXIT_BAR] = rows[:, TR_FILL_BAR] + 2
    rows[:, TR_NET] = np.where(np.arange(40) % 4 == 0, -10.0, 25.0)
    dna = seeds(v1c.v1_gene_pool(), 0.01)[0][1]
    ok = Evaluation(TrialResult(dna=dna, sharpe=1.0, n_trades=40), False, "kernel")
    noise = np.random.default_rng(1)
    g = judge(dna, cfg=cfg, timeframe="H1", split=split, full_run=lambda d: (rows, eq),
              twin_returns=lambda d, n, s: noise.normal(0.0, 0.004, bars), evaluate_bred=lambda d: ok,
              neighbours=lambda d: [("x", d), ("y", d)])
    assert g["checks"]["held_out_trades"]["value"] == int(np.sum(rows[:, TR_FILL_BAR] >= split))
    assert g["passed"] is (not g["failed"])
    assert g["verdict"] in ("GOOD", "WEAK") and 0 <= g["score"] <= 100
    few = rows[rows[:, TR_FILL_BAR] < split]
    g2 = judge(dna, cfg=cfg, timeframe="H1", split=split, full_run=lambda d: (few, eq),
               twin_returns=lambda d, n, s: noise.normal(0.0, 0.004, bars), evaluate_bred=lambda d: ok,
               neighbours=lambda d: [("x", d)])
    assert "held_out_trades" in g2["failed"]


def test_import_v1_marks_uncertified(tmp_path, monkeypatch):
    from engine.research.vault import library

    monkeypatch.setattr(library, "ga_dir", lambda: tmp_path / "ga")
    dna = seeds(v1c.v1_gene_pool(), 0.01)[0][1]
    m = {"sharpe": 1.4, "total_return": 0.3, "max_dd": 0.08, "n_trades": 90, "profit_factor": 1.6, "linearity": 0.9}
    passed = {"dna_hash": dna.dna_hash, "passed": True, "failed": [], "split_time": "2026-01-01T00:00:00+00:00",
              "checks": {"held_out_trades": {"value": 30, "passed": True}}, "score": 71.2, "verdict": "GOOD",
              "parts": {k: 1.0 for k in v1s.WEIGHTS}, "in_sample": m, "held_out": {**m, "n_trades": 30}, "full": m,
              "random": {"n": 100, "beaten_share": 0.97}, "reshuffled": {"dd_p95": 0.2}, "nudged": {"positive_share": 0.8}}
    failed = {"dna_hash": "f" * 16, "passed": False, "failed": ["beats_random"]}
    fin = {"dna": json.loads(json.dumps(dataclasses.asdict(dna), default=list)), "dna_hash": dna.dna_hash,
           "effective_hash": "e", "generation": 3, "parents": [], "operators": ["seed"], "origin": "Wild",
           "source": "seed", "engine": "kernel", "fit_a": 3.0, "fit_b": 2.0}
    doc = {"campaign": {"cell": "c1", "symbol": "XAUUSDm", "timeframe": "H1", "dataset_id": "d1",
                        "spread_cost_source": "bar_spread_v1"},
           "run": {"finalists": [fin]}, "gates": [passed, failed], "vaulted": [dna.dna_hash]}
    (tmp_path / "ga" / "v1-x").mkdir(parents=True)
    (tmp_path / "ga" / "v1-x" / "v1_result.json").write_text(json.dumps(doc), encoding="utf-8")
    vault = tmp_path / "vault.json"
    res = library.import_v1("v1-x", vault)
    assert res["added"] == [f"{dna.dna_hash}@c1"] and len(res["skipped"]) == 1
    e = library.load(vault)[0]
    assert (e["engine"], e["tier"], e["grade"], e["score"]) == ("v1", "UNCERTIFIED", "GOOD", 71.2)
    assert e["held_out"]["trades"] == 30 and e["cost_model"] == "bar"
    assert library.import_v1("v1-x", vault)["added"] == []


def test_certify_turns_a_v1_strategy_into_a_v2_judged_one(tmp_path, monkeypatch):
    from engine.research.ga.fitness import DSRReading
    from engine.research.vault import library

    test_import_v1_marks_uncertified(tmp_path, monkeypatch)
    vault = tmp_path / "vault.json"
    e = library.load(vault)[0]
    dna = seeds(v1c.v1_gene_pool(), 0.01)[0][1]
    r = np.random.default_rng(2).normal(0.0004, 0.003, 800)
    good = Evaluation(TrialResult(dna=dna, sharpe=1.6, n_trades=140, max_dd=0.07, total_return=0.4,
                                  per_period_returns=r), False, "kernel")
    cross = Evaluation(TrialResult(dna=dna, sharpe=0.7, n_trades=120, max_dd=0.09, total_return=0.1), False, "kernel")
    up = library.certify_metrics(e, primary=good, cross=cross, dsr=DSRReading(0.97, "READY", None, None, 350),
                                 sham_sharpe=0.2, n_trials=350, cross_cell="c2")
    assert up["certify"]["kind"] == "VALIDATION_RUN" and up["certify"]["dsr_n_trials"] == 350
    assert up["tier"] in ("CANDIDATE", "CERTIFIED") and up["engine"] == "v1"
    assert up["metrics"]["alive_on_both"] is True and up["metrics"]["sham_margin"] == pytest.approx(1.4)
    # without a robustness run ARQVIO's gates cannot all pass yet: CANDIDATE, never CERTIFIED by CERTIFY alone
    assert up["tier"] == "CANDIDATE"
    library.set_certify(up["id"], up, path=vault)
    assert library.load(vault)[0]["certify"]["state"] == "DONE"


def test_campaign_tab_sidecar_has_curves_and_never_changes_the_result(tmp_path: Path) -> None:
    """live.json (the CAMPAIGN tab's fan) is written after every generation and is display only."""
    from engine.research.search import live_view

    a = V1Run(quick(seed=7), FakeEvaluator(), "H1", out_dir=tmp_path / "a", log=lambda m: None).run()
    doc = live_view.read(tmp_path / "a")
    assert doc is not None and doc["phase"] == "FINAL WAR"
    assert doc["bots"] and all(len(b["curve"]) >= 2 for b in doc["bots"])
    assert doc["tape"] == []  # the fake evaluator has no kernel trade log: no invented trades
    (tmp_path / "a" / live_view.FILE).unlink()
    b = V1Run(quick(seed=7), FakeEvaluator(), "H1", out_dir=tmp_path / "b", log=lambda m: None).run()
    a.pop("elapsed_s"), b.pop("elapsed_s")  # wall-clock time, the only part allowed to differ
    assert a == b
