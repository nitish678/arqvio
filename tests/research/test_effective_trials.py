"""W2/G3: the luck bar scales with distinct backtests, not genome count.

A Deflated Sharpe's luck bar rises with the number of INDEPENDENT trials.
Feeding it the raw genome count defends against a search that never
happened: DNAs differing only in genes the engine ignores are one
backtest with several names, and each extra name raises the bar a real
edge must clear without a second real test behind it.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from engine.research.search.effective_hash import (
    count_effective_trials,
    effective_gene_hash,
    effective_genes,
)
from engine.research.search.real_backtest import dna_to_ma_config
from engine.research.search.sampler import random_dna


def _dna(**overrides):
    base = random_dna(np.random.default_rng(4))
    return dataclasses.replace(base, **overrides) if overrides else base


def test_effective_hash_tracks_the_projection() -> None:
    """The hash and dna_to_ma_config must not drift apart.

    effective_genes duplicates the projection rather than importing it
    (circular import). This is what keeps the duplicate honest: if the
    projection grows a field, this fails until the hash grows it too.
    Only MA_CROSS is checked here (it is the only family with a
    dna_to_*_config counterpart).
    """
    rng = np.random.default_rng(9)
    for _ in range(200):
        dna = dataclasses.replace(random_dna(rng), signal="MA_CROSS")
        cfg = dna_to_ma_config(dna)
        _, _sig, fast, slow, volume, sl_atr, tp_atr = effective_genes(dna)
        assert fast == cfg.fast, f"fast drift for {dna.dna_hash}"
        assert slow == cfg.slow, f"slow drift for {dna.dna_hash}"
        assert volume == cfg.volume, f"volume drift for {dna.dna_hash}"
        assert sl_atr == cfg.sl_atr, f"sl_atr drift for {dna.dna_hash}"
        assert tp_atr == cfg.tp_atr, f"tp_atr drift for {dna.dna_hash}"


def test_genes_the_engine_ignores_do_not_split_the_hash() -> None:
    """Two DNAs the engine cannot tell apart are ONE effective trial.

    These are exactly the genes the C5 audit found: sampled, hashed,
    stored, and discarded before the backtest.
    """
    base = _dna()
    ignored = [
        {"bias": "NONE"}, {"filter": "NONE"}, {"signal2": "NONE"},
        {"friday_close": not base.friday_close},
        {"news_filter": not base.news_filter},
        {"daily_dd_limit": base.daily_dd_limit / 2 + 1.0},
        {"max_agg_risk": base.max_agg_risk / 2 + 1.0},
        {"max_hold_time": base.max_hold_time + 17},
        {"window_start_hour_utc": (base.window_start_hour_utc + 3) % 24},
    ]
    for override in ignored:
        variant = dataclasses.replace(base, **override)
        if variant.to_dict() == base.to_dict():
            continue
        assert effective_gene_hash(variant) == effective_gene_hash(base), (
            f"{next(iter(override))} split the effective hash but never "
            "reaches the engine"
        )


def test_genes_the_engine_uses_do_split_the_hash() -> None:
    """The converse: a gene that changes the run must change the hash.

    Pinned to MA_CROSS so every listed gene actually reaches the engine:
    max_feature_lookback drives fast+slow, and the rest are common genes.
    """
    base = _dna(
        signal="MA_CROSS", sl_atr=1.0, tp_atr=1.0,
        max_feature_lookback=40, risk_pct=0.02,
    )
    h = effective_gene_hash(base)
    for override in (
        {"sl_atr": 2.0},
        {"tp_atr": 2.0},
        {"max_feature_lookback": 80},
        {"risk_pct": 0.05},
    ):
        assert effective_gene_hash(dataclasses.replace(base, **override)) != h, (
            f"{next(iter(override))} reaches the engine but not the hash"
        )


def test_the_c5_fixture_collapses_to_fewer_effective_trials() -> None:
    """The audit's five DNAs: 5 raw, fewer effective.

    mgmt is pinned and never reached the engine, so the three mgmt
    variants are one backtest. sl_atr and tp_atr now DO reach it, so
    those two stay distinct.
    """
    base = _dna(sl_atr=1.0, tp_atr=1.0)
    five = [
        base,
        dataclasses.replace(base, sl_atr=9.0),
        dataclasses.replace(base, tp_atr=9.0),
        dataclasses.replace(base, mgmt="TRAIL_ATR"),
        dataclasses.replace(base, mgmt="PARTIAL"),
    ]
    assert len({d.dna_hash for d in five}) == 5, "fixture must be 5 raw trials"
    assert count_effective_trials(five) == 3, (
        f"expected 3 distinct backtests, got {count_effective_trials(five)}"
    )


def test_effective_never_exceeds_raw() -> None:
    """Deduplication cannot invent trials."""
    rng = np.random.default_rng(12)
    pop = [random_dna(rng) for _ in range(400)]
    assert 1 <= count_effective_trials(pop) <= len(pop)


def test_hash_is_stable_across_calls() -> None:
    dna = _dna()
    assert effective_gene_hash(dna) == effective_gene_hash(dna)
    assert len(effective_gene_hash(dna)) == 16
