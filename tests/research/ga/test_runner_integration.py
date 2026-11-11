"""The GA generation loop end to end: elites, budgets, aborts, lineage, finalists,
the cross-cell pass, and one real run on both pinned cells."""
from __future__ import annotations

import ast
import inspect
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from engine.research.ga import runner as ga
from engine.research.ga.config import CAMPAIGN_GATES, Selection, ga_config
from engine.research.search import gene_pool as gp
from engine.research.search.dna import DNA
from engine.research.search.runner import TrialResult


class SyntheticCell:
    """A deterministic stand-in for a pinned cell: results are a pure function
    of the DNA, with a real selection gradient (longer stops trade better)."""

    def __init__(self, *, name="usdcadm_h1", crash_signal=None, engine=ga.ENGINE_KERNEL,
                 fallback_after=None, crash_after=None, sham_raises_for=None):
        self.name, self.symbol, self.timeframe = name, "USDCADm", "H1"
        self.engine_mode_actual = "fast"
        self.crash_signal, self.engine = crash_signal, engine
        # From the (n+1)-th evaluation on: every run falls back / crashes.
        self.fallback_after, self.crash_after = fallback_after, crash_after
        self.sham_raises_for = sham_raises_for
        self.calls = 0
        self.sham_calls = 0

    def evaluate(self, dna: DNA) -> ga.Evaluation:
        self.calls += 1
        engine = self.engine
        if self.fallback_after is not None and self.calls > self.fallback_after:
            engine = ga.ENGINE_SLOW_FALLBACK
        if dna.signal == self.crash_signal or (self.crash_after is not None and self.calls > self.crash_after):
            return ga.Evaluation(TrialResult(dna=dna, outcome="Crash", sharpe=None, n_trades=0), True, engine)
        rng = np.random.default_rng(int(dna.dna_hash, 16) % 2**32)
        edge = 0.0004 * (dna.sl_atr - 2.0) + 0.0002 * (dna.tp_atr - 3.0)
        returns = rng.normal(edge, 0.01, size=600)
        sharpe = float(np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(6048))
        n_trades = 10 + int(dna.max_feature_lookback) % 80
        return ga.Evaluation(TrialResult(dna=dna, sharpe=sharpe, n_trades=n_trades, total_return=0.0,
                                         max_dd=0.1, per_period_returns=returns), False, engine)

    def sham_sharpe(self, result: TrialResult) -> float:
        self.sham_calls += 1
        if result.dna.signal == self.sham_raises_for:
            raise RuntimeError("sham engine edge case")
        return float(np.random.default_rng(int(result.dna.dna_hash[:8], 16)).normal(0.0, 0.5))


def _cfg(**kw):
    base = dict(seed=7, gates=CAMPAIGN_GATES, gene_pool=gp.default_gene_pool(), population=24,
                generations=5, trials_budget=500, final_n=6, selection=Selection(elitism=4))
    base.update(kw)
    return ga_config(**base)


def test_generations_elites_and_budget_accounting(isolated_arqvio_home: Path) -> None:
    cfg, cell = _cfg(), SyntheticCell()
    res = ga.search(cfg, cell)
    assert res.stop_reason == ga.STOP_COMPLETED and res.aborted is None
    assert len(res.generations) == 5
    assert res.trials_used == 24 + 4 * 20 == cell.calls == len(res.archive)  # elites never re-evaluated
    assert [g["n_new"] for g in res.generations] == [24, 20, 20, 20, 20]
    assert [g["n_elites"] for g in res.generations] == [0, 4, 4, 4, 4]
    assert len({i.effective_hash for i in res.archive}) == len(res.archive)  # no intra-gen or global dup


def test_elitism_makes_the_best_fitness_non_decreasing(isolated_arqvio_home: Path) -> None:
    res = ga.search(_cfg(generations=6), SyntheticCell())
    best = [g["population_fitness"]["best"] for g in res.generations]
    assert best == sorted(best)


def test_a_generation_the_budget_cannot_pay_for_is_not_started(isolated_arqvio_home: Path) -> None:
    res = ga.search(_cfg(trials_budget=24 + 20 + 19), SyntheticCell())
    assert res.stop_reason == ga.STOP_BUDGET and len(res.generations) == 2 and res.trials_used == 44


def test_any_crash_aborts_the_run(isolated_arqvio_home: Path) -> None:
    res = ga.search(_cfg(), SyntheticCell(crash_signal="MACD_CROSS"))
    assert res.stop_reason == ga.STOP_ABORTED and res.aborted["generation"] == 0
    assert "crash rate must be 0" in res.aborted["reason"] and len(res.generations) == 1
    assert res.generations[0]["crashes"] > 0


def test_a_generation_below_the_kernel_share_is_invalid(isolated_arqvio_home: Path) -> None:
    res = ga.search(_cfg(), SyntheticCell(engine=ga.ENGINE_SLOW_FALLBACK))
    assert res.stop_reason == ga.STOP_ABORTED and "generation INVALID" in res.aborted["reason"]
    assert res.generations[0]["kernel_share"] == 0.0 and res.generations[0]["kernel_share_ok"] is False


@pytest.mark.parametrize("breaker", [{"fallback_after": 44}, {"crash_after": 44}])
def test_a_generation_that_aborts_the_run_supplies_no_finalist(isolated_arqvio_home: Path, breaker: dict) -> None:
    """Gen 0 (24) and gen 1 (20) are sound; gen 2 goes INVALID or crashes."""
    res = ga.search(_cfg(), SyntheticCell(**breaker))
    assert res.stop_reason == ga.STOP_ABORTED and res.aborted["generation"] == 2
    assert res.excluded_generation == 2 and len(res.generations) == 3
    assert any(i.generation == 2 for i in res.archive)  # its trials were run and count
    assert res.finalists and all(i.generation < 2 for i in res.finalists)
    expected = sorted((i for i in res.archive if i.fitness.alive and i.generation < 2), key=ga.rank_key)[:6]
    assert res.finalists == expected
    # their returns survived the trim, so the final DSR is computed, not UNRELIABLE for lack of data
    assert all(i.result.per_period_returns is not None for i in res.finalists)


def test_a_sham_that_raises_is_a_recorded_crash_not_a_lost_run(isolated_arqvio_home: Path) -> None:
    res = ga.search(_cfg(), SyntheticCell(sham_raises_for="RSI_REVERSION"))
    assert res.stop_reason == ga.STOP_ABORTED and "crash rate must be 0" in res.aborted["reason"]
    errors = res.generations[-1]["sham_errors"]
    assert errors and all("sham engine edge case" in e for e in errors.values())
    crashed = [i for i in res.archive if i.dna_hash in errors]
    assert crashed and all(i.fitness.killed_by == "crash" and i.fitness.fitness == 0.0 for i in crashed)


def test_a_parent_crossed_with_itself_is_recorded_as_a_clone(isolated_arqvio_home: Path) -> None:
    res = ga.search(_cfg(selection=Selection(elitism=4, tournament_k=24, crossover_p=1.0)), SyntheticCell())
    crossed = [i for i in res.archive if i.operators and i.operators[0] == "crossover"]
    assert crossed and all(len(set(i.parents)) == 2 for i in crossed)
    assert any(i.operators[0] == "clone" for i in res.archive)  # k = population: the same winner twice


def test_slow_mode_is_refused_before_any_work(isolated_arqvio_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ga, "prepare_cell", lambda *a, **k: pytest.fail("prepared a cell"))
    with pytest.raises(ga.GARunError, match="kernel-first"):
        ga.run_pinned(_cfg(), "usdcadm_h1", engine_mode="slow", echo=lambda s: None)


def test_cross_re_runs_are_validation_rows_outside_the_search_budget(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from engine.validation.trial_registry import TrialRegistry

    cfg = _cfg()
    res = ga.search(cfg, SyntheticCell())
    crossed, _ = ga.cross_check(res.finalists, cfg, SyntheticCell(name="xauusdm_h1"))
    reg = TrialRegistry(tmp_path / "t.sqlite")
    reg.open_campaign("c-cross", "XAUUSDm", "H1", "d", "g", "h", 7)
    ev = SimpleNamespace(symbol="XAUUSDm", timeframe="H1", context={"dataset_hash": "447b843f87438a9a"})
    ga._record_cross(reg, "c-cross", ev, crossed, "2026-09-30T00:00:00+00:00")
    import sqlite3
    con = sqlite3.connect(tmp_path / "t.sqlite")
    ops = [r[0] for r in con.execute("SELECT op_type FROM trials WHERE campaign_id = 'c-cross'")]
    con.close()
    assert ops == ["VALIDATION_RUN"] * len(crossed) and crossed
    assert reg.search_sharpes("c-cross", "XAUUSDm", "H1") == []  # not a search trial


def test_the_campaign_id_carries_the_config_hash() -> None:
    assert ga.campaign_id_for("usdcadm_h1", _cfg(), "0123456789abcdef") == "ga-usdcadm_h1-s7-0123456789abcdef"


def test_stale_outputs_are_gone_even_if_the_run_fails_early(
    isolated_arqvio_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = isolated_arqvio_home / "artifacts" / "ga" / "ga-stale"
    out.mkdir(parents=True)
    for name in ga.OUTPUT_FILES:
        (out / name).write_text('{"from": "an earlier run"}', encoding="utf-8")
    monkeypatch.setattr(ga, "campaign_id_for", lambda *a: "ga-stale")

    def boom(*a, **k):
        raise RuntimeError("preflight failed")

    monkeypatch.setattr(ga, "prepare_cell", boom)
    with pytest.raises(RuntimeError, match="preflight failed"):
        ga.run_pinned(_cfg(), "usdcadm_h1", echo=lambda s: None)
    assert not any((out / name).exists() for name in ga.OUTPUT_FILES)


def test_kernel_share_counts_only_backtests_that_ran() -> None:
    from collections import Counter
    assert ga.kernel_share(Counter({"kernel": 19, "slow_fallback": 1, "refused": 5})) == 0.95
    assert ga.kernel_share(Counter({"refused": 3})) is None


def test_lineage_is_recorded_for_every_individual(isolated_arqvio_home: Path) -> None:
    res = ga.search(_cfg(), SyntheticCell())
    known: set[str] = set()
    for ind in res.archive:
        assert ind.operators, ind
        if ind.generation == 0:
            assert ind.operators == ("init",) and ind.parents == ()
        elif ind.operators[0] == "resample":
            assert ind.parents == ()
        else:
            assert ind.operators[0] in {"crossover", "clone"}
            assert set(ind.parents) <= known  # parents were evaluated earlier
        known.add(ind.effective_hash)


def test_the_census_is_logged_every_generation(isolated_arqvio_home: Path) -> None:
    res = ga.search(_cfg(), SyntheticCell())
    for g in res.generations:
        census = g["census"]
        assert sum(census["trials_per_family"].values()) == g["n_new"]
        assert census["dedupe_hits"] == g["collisions"]
        assert {"zero_trade_rate_per_family", "min_trades_failures_per_family", "boundary_saturation",
                "effective_over_real"} <= set(census)
        assert sum(g["funnel"][k] for k in ("Crash", "LowTrades", "LowSharpe", "ALIVE")) == g["n_new"]


def test_shams_run_only_for_survivors(isolated_arqvio_home: Path) -> None:
    cell = SyntheticCell()
    res = ga.search(_cfg(), cell)
    assert cell.sham_calls == sum(1 for i in res.archive if i.fitness.alive)


def test_finalists_are_the_fittest_survivors_with_their_returns_kept(isolated_arqvio_home: Path) -> None:
    res = ga.search(_cfg(), SyntheticCell())
    alive = sorted((i for i in res.archive if i.fitness.alive), key=ga.rank_key)
    assert res.finalists == alive[:6]
    assert all(i.result.per_period_returns is not None for i in res.finalists)
    others = [i for i in res.archive if i not in res.finalists]
    assert all(i.result.per_period_returns is None for i in others)
    assert set(res.final_dsr) == {i.effective_hash for i in res.finalists}
    assert all(d.n_trials == res.dsr_n_final for d in res.final_dsr.values())


def test_a_killed_individual_is_never_a_finalist(isolated_arqvio_home: Path) -> None:
    """final_n above the survivor count: the finalists are the survivors, no more."""
    res = ga.search(_cfg(final_n=500), SyntheticCell())
    alive = [i for i in res.archive if i.fitness.alive]
    assert 0 < len(alive) < len(res.archive)  # the fixture kills some (LowTrades)
    assert sorted(res.finalists, key=ga.rank_key) == sorted(alive, key=ga.rank_key)


def test_the_cross_check_scores_every_finalist_on_the_other_cell(isolated_arqvio_home: Path) -> None:
    cfg = _cfg()
    res = ga.search(cfg, SyntheticCell())
    other = SyntheticCell(name="xauusdm_h1")
    crossed, n_trials = ga.cross_check(res.finalists, cfg, other)
    assert len(crossed) == len(res.finalists) == other.calls and n_trials == len(res.finalists)


def test_two_runs_are_byte_identical_and_a_seed_changes_them(isolated_arqvio_home: Path) -> None:
    from engine.research.ga.determinism import canonical, individuals_table, run_hash

    a = individuals_table(ga.search(_cfg(), SyntheticCell()).archive)
    b = individuals_table(ga.search(_cfg(), SyntheticCell()).archive)
    assert canonical(a) == canonical(b) and run_hash(a) == run_hash(b)
    c = individuals_table(ga.search(_cfg(seed=8), SyntheticCell()).archive)
    assert run_hash(c) != run_hash(a)


def test_the_cell_manifest_derives_provenance_from_the_dataset_id() -> None:
    """One Manifest(...) in the GA runner, dataset_provenance=
    dataset_provenance_for(dataset_id), dataset_id bound once from the cell."""
    tree = ast.parse(inspect.getsource(ga))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "Manifest"]
    assert len(calls) == 1
    value = {k.arg: k.value for k in calls[0].keywords}["dataset_provenance"]
    assert isinstance(value, ast.Call) and value.func.id == "dataset_provenance_for"
    assert [getattr(a, "id", None) for a in value.args] == ["dataset_id"]
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "cell_manifest")
    binds = [n for n in ast.walk(fn) if isinstance(n, ast.Name) and n.id == "dataset_id"
             and not isinstance(n.ctx, ast.Load)]
    assert len(binds) == 1


def test_the_ga_package_holds_no_import_time_path() -> None:
    """Law 7: no module-level get_artifacts_dir / get_arqvio_home / Path(...) in the GA."""
    pkg = Path(ga.__file__).parent
    for src in sorted(pkg.glob("*.py")):
        tree = ast.parse(src.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Assign | ast.AnnAssign):
                text = ast.unparse(node)
                for needle in ("get_artifacts_dir", "get_arqvio_home", "artifact_path", "Path("):
                    assert needle not in text, f"{src.name}: {text}"


# --- one real run on both pinned cells ---------------------------------------

@pytest.fixture(scope="session")
def real_artifacts() -> Path:
    """The artifacts dir of the ARQVIO_HOME this test process started with,
    resolved through engine/core/paths (Law 7). Session-scoped, so pytest
    builds it before any function-scoped isolated_arqvio_home takes effect."""
    from engine.core.paths import get_artifacts_dir

    return get_artifacts_dir()


def _seed_isolated_home(src: Path, home: Path) -> None:
    dst = home / "artifacts"
    need = [src / "datasets" / f"{s}_H1_{d}.{ext}" for s, d in (("USDCADm", "aed3bf314539ec5a"),
                                                              ("XAUUSDm", "447b843f87438a9a"))
            for ext in ("json", "npz")]
    need += [src / sub / f"{s}.json" for sub in ("spread", "specs") for s in ("USDCADm", "XAUUSDm")]
    need += [src / "history_depth_audit_target.csv"]
    missing = [p for p in need if not p.is_file()]
    if missing:
        pytest.skip(f"pinned cells not on this machine: {missing[0]}")
    for p in need:
        target = dst / p.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)


def test_a_real_run_on_both_pinned_cells(real_artifacts: Path, isolated_arqvio_home: Path) -> None:
    assert not real_artifacts.is_relative_to(isolated_arqvio_home)  # resolved before isolation
    _seed_isolated_home(real_artifacts, isolated_arqvio_home)
    cfg = _cfg(population=8, generations=2, trials_budget=40, final_n=3, selection=Selection(elitism=2))
    lines: list[str] = []
    summary = ga.run_pinned(cfg, "usdcadm_h1", echo=lines.append)
    out = isolated_arqvio_home / "artifacts" / "ga" / summary["campaign_id"]
    assert all(Path(p).is_relative_to(out) for p in summary["paths"].values())
    doc = json.loads((out / "manifest_v2.json").read_text(encoding="utf-8"))
    run = doc["run"]
    assert run["stop_reason"] in {ga.STOP_COMPLETED, ga.STOP_BUDGET, ga.STOP_ABORTED}
    assert run["engines"].get("slow_fallback", 0) == 0 and run["kernel_share"] == 1.0
    for cell, dataset_id in (("usdcadm_h1", "aed3bf314539ec5a"), ("xauusdm_h1", "447b843f87438a9a")):
        c = run["cells"][cell]
        assert c["dataset_id"] == dataset_id and c["dataset_provenance"] == "pinned_cell"
        assert c["manifest"]["dataset_provenance"] == "pinned_cell" and c["canary"]["passed"] is True
        assert len(c["spread_artifact_sha256"]) == 64 and c["audit"]["time_basis"] == "utc"
    for cand in doc["candidates"]:
        assert cand["dataset_provenance"] == {"usdcadm_h1": "pinned_cell", "xauusdm_h1": "pinned_cell"}
        assert cand["run_hash"] == run["run_hash"] and cand["config_hash"] == run["config_hash"]
    assert run["config"]["ga"] == cfg.block()
    assert run["config"]["min_trades"] == cfg.gates.min_trades  # the config cannot contradict its gate
    assert any(line.lstrip().startswith("gen  0:") for line in lines)
    table = json.loads((out / "individuals.json").read_text(encoding="utf-8"))
    assert len(table) == run["trials_used"]
    # The cross cell is its own campaign: its id, and a config naming its own data.
    cross_manifest = run["cells"]["xauusdm_h1"]["manifest"]
    assert cross_manifest["campaign_id"] == run["cross_campaign_id"] == f"{summary['campaign_id']}-cross"
    assert cross_manifest["config_hash"] == run["cross_config_hash"] != run["config_hash"]
    assert (run["cross_config"]["symbol"], run["cross_config"]["dataset"]) == ("XAUUSDm", "447b843f87438a9a")
    assert run["cells"]["usdcadm_h1"]["manifest"]["campaign_id"] == summary["campaign_id"]
    # Search rows are SEARCH_TRIAL; the cross re-runs of selected finalists are VALIDATION_RUN.
    import sqlite3
    con = sqlite3.connect(out / "trials.sqlite")
    rows = dict(((c, o), n) for c, o, n in con.execute(
        "SELECT campaign_id, op_type, COUNT(*) FROM trials GROUP BY campaign_id, op_type"))
    con.close()
    assert rows == {(summary["campaign_id"], "SEARCH_TRIAL"): run["trials_used"],
                    **({(run["cross_campaign_id"], "VALIDATION_RUN"): run["finalists"]} if run["finalists"] else {})}
    for cell in ("usdcadm_h1", "xauusdm_h1"):
        c = run["cells"][cell]
        assert c["fast_fallback_count"] == 0 and c["fast_fallback_reasons"] == []
        fills = run["sham_fills"][cell]
        assert set(fills) == {"shams", "entries_requested", "entries_drawn", "entries_filled", "truncated"}
    assert run["sham_errors"] == {"search": {}, "cross": {}}
