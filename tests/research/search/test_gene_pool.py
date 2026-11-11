"""W2 gene pool: validation, canonical form, the sampler under a pool, and
the pool reaching the config hash and the manifest."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from engine.research.campaign.budget import CampaignScope
from engine.research.campaign.manifest import Manifest, config_hash
from engine.research.provenance import DataSource, Provenance
from engine.research.search import gene_pool as gp
from engine.research.search import sampler
from engine.research.search.dna import SIGNAL_VALUES, FailureTaxonomy, validate_gene_space
from engine.research.search.run_campaign_cli import campaign_config_for
from engine.research.search.runner import TrialResult, run_campaign
from engine.research.search.sampler import random_dna
from engine.validation.trial_registry import TrialRegistry

# sha256 over the dna_hash of seeds 0..4 x 100 draws, each seed followed by
# one rng.random() so rng consumption is pinned too. Recorded from the
# sampler BEFORE the gene-pool change; gene_pool=None must reproduce it.
#
# W4-2 note: this digest changed (41cd756c... -> e49c09c9...) solely because
# DNA_SCHEMA_VERSION bumped 3 -> 4 (dna.py). Empty switch fields are omitted
# from to_dict(), so the body is byte-identical to the v3 sampler; only the
# version field differs and the SHA shifts. Verified: pre-W4 body (version=3)
# reproduces 41cd756c. gene_pool=None still reproduces this (now v4) stream.
PRE_W2_SAMPLER_DIGEST = "e49c09c9a5fd02f79b409baa97b594d81c72f2bbd42dd3f601c32e3331e601e8"


def _pool(families: dict[str, tuple[bool, float]] | None = None, genes=None) -> gp.GenePool:
    fams = {f: {"enabled": True, "weight": 1.0} for f in SIGNAL_VALUES}
    for name, (on, w) in (families or {}).items():
        fams[name] = {"enabled": on, "weight": w}
    return gp.canonicalise({"families": fams, "genes": genes or {}})


def _draw(n: int, pool: gp.GenePool | None, seed: int = 0) -> list:
    rng = np.random.default_rng(seed)
    return [random_dna(rng, gene_pool=pool) for _ in range(n)]


def _problems(data) -> list[str]:
    with pytest.raises(gp.GenePoolError) as info:
        gp.canonicalise(data)
    return info.value.problems


# ---------------------------------------------------------------------------
# pool=None is the pre-W2 sampler, draw for draw
# ---------------------------------------------------------------------------

def test_no_pool_reproduces_pre_w2_stream() -> None:
    parts: list[str] = []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        parts.extend(random_dna(rng).dna_hash for _ in range(100))
        parts.append("|" + repr(rng.random()))
    digest = hashlib.sha256(",".join(parts).encode()).hexdigest()
    assert digest == PRE_W2_SAMPLER_DIGEST


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 2026])
def test_default_pool_draws_are_the_no_pool_draws(seed: int) -> None:
    free = np.random.default_rng(seed)
    pooled = np.random.default_rng(seed)
    pool = gp.default_gene_pool()
    for _ in range(200):
        assert random_dna(pooled, gene_pool=pool).to_dict() == random_dna(free).to_dict()
    assert pooled.bit_generator.state == free.bit_generator.state


def test_default_pool_with_a_fixed_gene_overrides_only_that_gene() -> None:
    pool = _pool(genes={"RSI_REVERSION": {"rsi_period": {"fixed": 9}}})
    for a, b in zip(_draw(200, None, seed=3), _draw(200, pool, seed=3), strict=True):
        assert b.to_dict() == {**a.to_dict(), "rsi_period": 9}


# ---------------------------------------------------------------------------
# schema is the sampler's own ranges
# ---------------------------------------------------------------------------

def test_schema_covers_every_family_with_sampler_ranges() -> None:
    s = gp.schema()
    assert s["families"] == list(SIGNAL_VALUES)
    assert set(gp.FAMILY_GENES) == set(SIGNAL_VALUES)
    ranges = {g["name"]: g for fam in s["genes"].values() for g in fam}
    expected_int = {
        "max_feature_lookback": tuple(range(sampler.MAX_FEATURE_LOOKBACK_RANGE[0],
                                            sampler.MAX_FEATURE_LOOKBACK_RANGE[1] + 1)),
        "rsi_period": sampler.RSI_PERIOD_VALUES,
        "stoch_k_period": sampler.STOCH_K_PERIOD_VALUES,
        "stoch_d_period": sampler.STOCH_D_PERIOD_VALUES,
        "stoch_smoothing": sampler.STOCH_SMOOTHING_VALUES,
        "macd_fast": sampler.MACD_FAST_VALUES,
        "macd_slow": sampler.MACD_SLOW_VALUES,
        "macd_signal_period": sampler.MACD_SIGNAL_PERIOD_VALUES,
        "donchian_period": sampler.DONCHIAN_PERIOD_VALUES,
    }
    for name, values in expected_int.items():
        g = ranges[name]
        assert g["kind"] == "int"
        # The schema's [min, max] is only exact if the sampler's set is contiguous.
        assert tuple(range(g["min"], g["max"] + 1)) == values, name
    expected_float = {
        "rsi_oversold": sampler.RSI_OVERSOLD_RANGE,
        "rsi_overbought": sampler.RSI_OVERBOUGHT_RANGE,
        "stoch_oversold": sampler.STOCH_OVERSOLD_RANGE,
        "stoch_overbought": sampler.STOCH_OVERBOUGHT_RANGE,
    }
    for name, (lo, hi) in expected_float.items():
        assert (ranges[name]["kind"], ranges[name]["min"], ranges[name]["max"]) == ("float", lo, hi)
    assert ranges["rsi_exit_midline"] == {"name": "rsi_exit_midline", "kind": "bool"}
    assert set(ranges) == set(expected_int) | set(expected_float) | {"rsi_exit_midline"}
    assert s["constraints"] == ["MACD_CROSS: macd_slow - macd_fast >= 5"]


# ---------------------------------------------------------------------------
# canonical form
# ---------------------------------------------------------------------------

def test_default_pool_is_all_enabled_equal_weights() -> None:
    d = gp.default_gene_pool().to_dict()
    assert set(d) == {"families", "genes", "hash"}
    assert d["families"] == {f: {"enabled": True, "weight": 0.2} for f in SIGNAL_VALUES}
    assert all(v == "enabled" for fam in d["genes"].values() for v in fam.values())
    assert {f: set(g) for f, g in d["genes"].items()} == {
        f: {s.name for s in gp.FAMILY_GENES[f]} for f in SIGNAL_VALUES
    }
    expected = hashlib.sha256(json.dumps(
        {"families": d["families"], "genes": d["genes"]},
        sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()[:16]
    assert d["hash"] == expected == gp.default_gene_pool().hash


def test_weights_renormalised_and_disabled_zeroed() -> None:
    pool = _pool({
        "MA_CROSS": (True, 2.0), "RSI_REVERSION": (True, 1.0), "STOCH_CROSS": (True, 1.0),
        "MACD_CROSS": (False, 0.0), "DONCHIAN_BREAKOUT": (False, 7.0),
    })
    assert pool.signal_weights() == (0.5, 0.25, 0.25, 0.0, 0.0)


def test_round_trip_keeps_hash() -> None:
    pool = _pool({"MA_CROSS": (True, 3.0), "STOCH_CROSS": (False, 1.0)},
                 {"MACD_CROSS": {"macd_fast": {"fixed": 8}}})
    again = gp.canonicalise(json.loads(json.dumps(pool.to_dict())))
    assert again.to_dict() == pool.to_dict()


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def test_all_off_rejected() -> None:
    fams = {f: {"enabled": False, "weight": 1.0} for f in SIGNAL_VALUES}
    assert any("all-off" in p for p in _problems({"families": fams}))


def test_unknown_family_and_gene_rejected() -> None:
    fams = {f: {"enabled": True, "weight": 1.0} for f in SIGNAL_VALUES}
    probs = _problems({"families": {**fams, "BB_REV": {"enabled": True, "weight": 1.0}}})
    assert any("BB_REV" in p for p in probs)
    probs = _problems({"families": fams, "genes": {"RSI_REVERSION": {"macd_fast": "enabled"}}})
    assert any("unknown gene 'macd_fast'" in p for p in probs)
    probs = _problems({"families": fams, "genes": {"STOCH": {}}})
    assert any("STOCH" in p for p in probs)


def test_missing_family_rejected() -> None:
    fams = {f: {"enabled": True, "weight": 1.0} for f in SIGNAL_VALUES[:-1]}
    assert any(SIGNAL_VALUES[-1] in p for p in _problems({"families": fams}))


@pytest.mark.parametrize("weight", [0, -1.0, float("nan"), float("inf"), "1", True])
def test_bad_weight_for_enabled_family_rejected(weight) -> None:
    fams = {f: {"enabled": True, "weight": 1.0} for f in SIGNAL_VALUES}
    fams["MA_CROSS"]["weight"] = weight
    assert any(p.startswith("MA_CROSS:") for p in _problems({"families": fams}))


def _enabled_weights_are_drawable(pool: gp.GenePool) -> bool:
    ws = [e["weight"] for e in pool.families.values() if e["enabled"]]
    return all(np.isfinite(w) and w > 0 for w in ws) and abs(sum(ws) - 1.0) < 1e-9


@pytest.mark.parametrize(
    "weights",
    [
        (1e308,) * 5,                      # total overflows to inf
        (1.0, 1.0, 1.0, 1.0, 5e-324),      # 5e-324 / 4 underflows to 0
    ],
)
def test_weights_that_cannot_normalise_are_rejected(weights) -> None:
    fams = {f: {"enabled": True, "weight": w} for f, w in zip(SIGNAL_VALUES, weights, strict=True)}
    assert _problems({"families": fams})


@pytest.mark.parametrize(
    "weights",
    [(1e307,) * 5, (1e-300,) * 5, (5e-324,) * 5, (1.0, 1.0, 1.0, 1.0, 1e-300), (0.2,) * 5],
)
def test_extreme_but_normalisable_weights_are_drawable(weights) -> None:
    fams = {f: {"enabled": True, "weight": w} for f, w in zip(SIGNAL_VALUES, weights, strict=True)}
    pool = gp.canonicalise({"families": fams})
    assert _enabled_weights_are_drawable(pool)
    random_dna(np.random.default_rng(0), gene_pool=pool)


def test_integer_weight_too_large_for_a_float_is_a_problem_not_a_crash() -> None:
    fams = {f: {"enabled": True, "weight": 1.0} for f in SIGNAL_VALUES}
    fams["MA_CROSS"]["weight"] = 10**400
    assert any(p.startswith("MA_CROSS: weight") for p in _problems({"families": fams}))
    fams["MA_CROSS"] = {"enabled": False, "weight": 10**400}
    assert any(p.startswith("MA_CROSS: weight") for p in _problems({"families": fams}))


def test_integer_fixed_float_too_large_for_a_float_is_a_problem_not_a_crash() -> None:
    fams = {f: {"enabled": True, "weight": 1.0} for f in SIGNAL_VALUES}
    genes = {"RSI_REVERSION": {"rsi_oversold": {"fixed": 10**400}}}
    probs = _problems({"families": fams, "genes": genes})
    assert any("RSI_REVERSION.rsi_oversold" in p for p in probs)


@pytest.mark.parametrize(
    ("family", "gene", "value"),
    [
        ("RSI_REVERSION", "rsi_period", 14.0),       # float for int
        ("RSI_REVERSION", "rsi_period", True),       # bool for int
        ("RSI_REVERSION", "rsi_period", 22),         # out of range
        ("RSI_REVERSION", "rsi_oversold", "20"),     # str for float
        ("RSI_REVERSION", "rsi_oversold", 9.99),     # out of range
        ("RSI_REVERSION", "rsi_oversold", float("nan")),
        ("RSI_REVERSION", "rsi_exit_midline", 1),    # int for bool
        ("STOCH_CROSS", "stoch_d_period", 6),
        ("MA_CROSS", "max_feature_lookback", 200),
        ("DONCHIAN_BREAKOUT", "donchian_period", 9),
    ],
)
def test_fixed_value_kind_and_range_checked(family, gene, value) -> None:
    fams = {f: {"enabled": True, "weight": 1.0} for f in SIGNAL_VALUES}
    probs = _problems({"families": fams, "genes": {family: {gene: {"fixed": value}}}})
    assert any(f"{family}.{gene}" in p for p in probs)


def test_gene_setting_shape_checked() -> None:
    fams = {f: {"enabled": True, "weight": 1.0} for f in SIGNAL_VALUES}
    for bad in ("disabled", {"fixed": 10, "extra": 1}, {}, 10):
        probs = _problems({"families": fams, "genes": {"DONCHIAN_BREAKOUT": {"donchian_period": bad}}})
        assert any("donchian_period" in p for p in probs)


def test_macd_fixed_pair_below_gap_rejected() -> None:
    fams = {f: {"enabled": True, "weight": 1.0} for f in SIGNAL_VALUES}
    genes = {"MACD_CROSS": {"macd_fast": {"fixed": 21}, "macd_slow": {"fixed": 22}}}
    assert any("macd_slow - macd_fast" in p for p in _problems({"families": fams, "genes": genes}))


def test_every_problem_reported_not_just_the_first() -> None:
    fams = {f: {"enabled": True, "weight": 1.0} for f in SIGNAL_VALUES}
    fams["RSI_REVERSION"]["weight"] = -1
    genes = {"DONCHIAN_BREAKOUT": {"donchian_period": {"fixed": 99}}}
    assert len(_problems({"families": fams, "genes": genes, "extra": 1})) == 3


# ---------------------------------------------------------------------------
# the sampler under a pool
# ---------------------------------------------------------------------------

def test_disabled_family_is_never_drawn() -> None:
    pool = _pool({"STOCH_CROSS": (False, 1.0)})
    dnas = _draw(5000, pool, seed=11)
    signal = Counter(d.signal for d in dnas)
    signal2 = Counter(d.signal2 for d in dnas)
    print(f"\nSTOCH_CROSS disabled, 5000 draws: signal={dict(signal)} signal2={dict(signal2)}")
    assert signal["STOCH_CROSS"] == 0
    assert signal2["STOCH_CROSS"] == 0
    for fam in SIGNAL_VALUES:
        if fam != "STOCH_CROSS":
            assert signal[fam] > 0, fam
            assert signal2[fam] > 0, fam
    assert signal2[""] > 0
    for d in dnas:
        validate_gene_space(d.to_dict())


def test_family_frequencies_follow_weights() -> None:
    pool = _pool({
        "MA_CROSS": (True, 2.0), "RSI_REVERSION": (True, 1.0), "STOCH_CROSS": (True, 1.0),
        "MACD_CROSS": (False, 1.0), "DONCHIAN_BREAKOUT": (False, 1.0),
    })
    n = 20000
    counts = Counter(d.signal for d in _draw(n, pool, seed=3))
    print(f"\nweights {pool.signal_weights()}, {n} draws: {dict(counts)}")
    for fam, w in zip(SIGNAL_VALUES, pool.signal_weights(), strict=True):
        assert abs(counts[fam] / n - w) < 0.03, (fam, counts[fam], w)


def test_fixed_genes_carried_on_every_draw() -> None:
    fixed = {
        "MA_CROSS": {"max_feature_lookback": {"fixed": 50}},
        "RSI_REVERSION": {"rsi_period": {"fixed": 9}, "rsi_exit_midline": {"fixed": True},
                          "rsi_oversold": {"fixed": 20}},
        "STOCH_CROSS": {"stoch_overbought": {"fixed": 80.5}},
        "DONCHIAN_BREAKOUT": {"donchian_period": {"fixed": 33}},
    }
    pool = _pool(genes=fixed)
    for d in _draw(2000, pool, seed=5):
        assert (d.max_feature_lookback, d.rsi_period, d.rsi_exit_midline) == (50, 9, True)
        assert d.rsi_oversold == 20.0
        assert d.stoch_overbought == 80.5
        assert d.donchian_period == 33
        validate_gene_space(d.to_dict())


def test_fixing_a_gene_does_not_shift_the_other_draws() -> None:
    free = _draw(300, _pool(), seed=9)
    fixed = _draw(300, _pool(genes={"DONCHIAN_BREAKOUT": {"donchian_period": {"fixed": 33}}}), seed=9)
    for a, b in zip(free, fixed, strict=True):
        da, db = a.to_dict(), b.to_dict()
        assert db.pop("donchian_period") == 33
        da.pop("donchian_period")
        assert da == db


def test_macd_fixed_fast_constrains_slow() -> None:
    pool = _pool(genes={"MACD_CROSS": {"macd_fast": {"fixed": 21}}})
    dnas = _draw(2000, pool, seed=2)
    assert all(d.macd_fast == 21 for d in dnas)
    slows = {d.macd_slow for d in dnas}
    assert min(slows) >= 26
    assert len(slows) > 1
    for d in dnas:
        validate_gene_space(d.to_dict())


def test_macd_fixed_slow_constrains_fast() -> None:
    pool = _pool(genes={"MACD_CROSS": {"macd_slow": {"fixed": 22}}})
    dnas = _draw(2000, pool, seed=4)
    assert all(d.macd_slow == 22 for d in dnas)
    fasts = {d.macd_fast for d in dnas}
    assert max(fasts) <= 17
    assert len(fasts) > 1
    for d in dnas:
        validate_gene_space(d.to_dict())


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------

def test_merge_applies_partial_onto_stored() -> None:
    base = gp.default_gene_pool()
    merged = gp.merge(base, {
        "families": {"STOCH_CROSS": {"enabled": False}},
        "genes": {"MACD_CROSS": {"macd_fast": {"fixed": 10}}},
    })
    assert merged.families["STOCH_CROSS"] == {"enabled": False, "weight": 0.0}
    assert merged.signal_weights() == (0.25, 0.25, 0.0, 0.25, 0.25)
    assert merged.genes["MACD_CROSS"]["macd_fast"] == {"fixed": 10}
    assert merged.genes["MACD_CROSS"]["macd_slow"] == "enabled"
    assert base.to_dict() == gp.default_gene_pool().to_dict(), "merge mutated its input"


def test_merge_validates_the_whole_result() -> None:
    base = gp.merge(gp.default_gene_pool(), {"genes": {"MACD_CROSS": {"macd_fast": {"fixed": 21}}}})
    with pytest.raises(gp.GenePoolError) as info:
        gp.merge(base, {"genes": {"MACD_CROSS": {"macd_slow": {"fixed": 22}}}})
    assert any("macd_slow - macd_fast" in p for p in info.value.problems)


def test_merge_weight_is_relative_to_the_stored_normalised_weights() -> None:
    base = _pool({"STOCH_CROSS": (False, 1.0)})
    assert base.signal_weights() == (0.25, 0.25, 0.0, 0.25, 0.25)
    merged = gp.merge(base, {"families": {"MA_CROSS": {"weight": 0.5}}})
    ma, rsi, _, macd, don = merged.signal_weights()
    assert ma == pytest.approx(2 * rsi)
    assert rsi == macd == don


def test_merge_enabling_a_family_without_weight_joins_at_an_equal_share() -> None:
    base = _pool({"MA_CROSS": (True, 3.0), "STOCH_CROSS": (False, 1.0)})
    stored = [base.families[f]["weight"] for f in SIGNAL_VALUES if base.families[f]["enabled"]]
    merged = gp.merge(base, {"families": {"STOCH_CROSS": {"enabled": True}}})
    assert merged.families["STOCH_CROSS"]["enabled"] is True
    mean = sum(stored) / len(stored)
    total = sum(stored) + mean
    assert merged.families["STOCH_CROSS"]["weight"] == pytest.approx(mean / total)
    assert merged.families["MA_CROSS"]["weight"] == pytest.approx(
        base.families["MA_CROSS"]["weight"] / total
    )
    assert _enabled_weights_are_drawable(merged)


def test_merge_enabling_with_a_weight_uses_that_weight() -> None:
    base = _pool({"STOCH_CROSS": (False, 1.0)})
    merged = gp.merge(base, {"families": {"STOCH_CROSS": {"enabled": True, "weight": 0.5}}})
    assert merged.families["STOCH_CROSS"]["weight"] == pytest.approx(0.5 / 1.5)


def test_merge_ignores_hash_like_canonicalise() -> None:
    base = gp.default_gene_pool()
    merged = gp.merge(base, {"hash": "0" * 16, "families": {"MA_CROSS": {"enabled": False}}})
    assert merged.families["MA_CROSS"] == {"enabled": False, "weight": 0.0}
    assert gp.merge(base, {"hash": "stale"}).to_dict() == base.to_dict()


# ---------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------

def test_store_round_trip_and_default(tmp_path: Path) -> None:
    path = tmp_path / "gene_pool.json"
    assert gp.load(path).to_dict() == gp.default_gene_pool().to_dict()
    pool = _pool({"MA_CROSS": (False, 0.0)})
    gp.save(pool, path)
    assert gp.load(path).to_dict() == pool.to_dict()


def test_store_refuses_hash_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "gene_pool.json"
    gp.save(gp.default_gene_pool(), path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["families"]["MA_CROSS"]["enabled"] = False
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(gp.GenePoolError, match="does not match"):
        gp.load(path)


def test_store_path_resolved_at_call_time(isolated_arqvio_home: Path) -> None:
    written = gp.save(_pool({"MA_CROSS": (False, 0.0)}))
    assert written == isolated_arqvio_home / "configs" / "gene_pool.json"
    assert gp.load().families["MA_CROSS"]["enabled"] is False


# ---------------------------------------------------------------------------
# provenance: config hash and manifest
# ---------------------------------------------------------------------------

def _cfg_hash(pool: gp.GenePool | None) -> str:
    return config_hash(campaign_config_for(
        symbol="USDCADm", timeframe="H1", population=500, target_sharpe=1.5,
        seed=0, dataset_id="pinned", gene_pool=pool,
    ))


def test_gene_pool_is_in_the_config_hash(isolated_arqvio_home: Path) -> None:
    default = _cfg_hash(gp.default_gene_pool())
    assert default == _cfg_hash(gp.default_gene_pool())
    assert _cfg_hash(_pool({"STOCH_CROSS": (False, 1.0)})) != default
    assert _cfg_hash(_pool(genes={"MACD_CROSS": {"macd_fast": {"fixed": 9}}})) != default
    # The default pool draws exactly what no pool draws, but a run on it
    # records the pool it ran under; no pool records none.
    assert _cfg_hash(None) != default


def test_no_pool_config_has_no_gene_pool_key(isolated_arqvio_home: Path) -> None:
    cfg = campaign_config_for(
        symbol="USDCADm", timeframe="H1", population=500, target_sharpe=1.5,
        seed=0, dataset_id=None, gene_pool=None,
    )
    assert "gene_pool" not in cfg


def test_manifest_renders_gene_pool_summary() -> None:
    pool = _pool({"STOCH_CROSS": (False, 1.0)}, {"MACD_CROSS": {"macd_fast": {"fixed": 21}}})
    line = gp.summary(pool)
    assert line.startswith(pool.hash)
    assert "off: STOCH_CROSS" in line
    assert "fixed: macd_fast=21" in line
    m = Manifest(
        campaign_id="c", symbol="USDCADm", timeframe="H1", dataset_hash="d",
        dataset_source="MT5_LIVE", dataset_provenance="pinned_cell",
        bars_loaded=10, date_range_utc=None,
        git_sha="abc", config_hash="0123456789abcdef", gene_pool=line, seed=0,
        completed_at_utc="2026-09-29T00:00:00+00:00",
    )
    assert f"  {'gene_pool':18s} {line}" in m.render().splitlines()
    assert m.to_dict()["gene_pool"] == line
    assert m.missing_fields() == ()
    assert "gene_pool" in Manifest(**{**m.to_dict(), "gene_pool": ""}).missing_fields()


def test_run_campaign_draws_from_the_pool_it_is_given(tmp_path: Path) -> None:
    pool = _pool({"MA_CROSS": (False, 1.0), "MACD_CROSS": (True, 3.0)},
                 {"DONCHIAN_BREAKOUT": {"donchian_period": {"fixed": 33}}})
    seen: list = []

    def backtest(dna) -> TrialResult:
        seen.append(dna)
        return TrialResult(dna=dna, outcome=FailureTaxonomy.EVALUATED.value, sharpe=0.0,
                           n_trades=0, total_return=0.0, max_dd=0.0)

    run_campaign(
        campaign_id="pool-forwarding",
        scope=CampaignScope(symbols=("USDCADm",), timeframes=("H1",), population_per_cell=50),
        registry=TrialRegistry(tmp_path / "trials.sqlite"),
        backtest_fn=backtest,
        control_fn=None,
        real_scores_factory=lambda trials: [],
        provenance=Provenance(source=DataSource.MT5_LIVE, dataset_hash="d",
                              as_of=datetime(2026, 9, 29, tzinfo=UTC), mixed_source=False),
        seed=13,
        n_obs=0,
        gene_pool=pool,
    )
    assert [d.to_dict() for d in seen] == [d.to_dict() for d in _draw(50, pool, seed=13)]
