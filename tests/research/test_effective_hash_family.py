"""B3 effective_hash family-awareness tests.

effective_genes must be family-aware: the projection includes the family
string as its first component so that two DNAs from different families
with otherwise identical projections never collide in the dedupe cache.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from engine.research.search.dna import DNA
from engine.research.search.effective_hash import (
    EFFECTIVE_HASH_VERSION,
    effective_gene_hash,
    effective_genes,
)
from engine.research.search.sampler import random_dna


def _base_rsi_dna(**kw) -> DNA:
    rng = np.random.default_rng(0)
    d = random_dna(rng)
    return dataclasses.replace(d, **kw)


# ---------------------------------------------------------------------------
# 1.A(i): family-in-hash
# ---------------------------------------------------------------------------
def test_family_string_is_first_gene() -> None:
    """effective_genes[1] must be the DNA signal (family string)."""
    rng = np.random.default_rng(20260930)
    for _ in range(10):
        dna = random_dna(rng)
        genes = effective_genes(dna)
        assert genes[0] == EFFECTIVE_HASH_VERSION
        assert genes[1] == dna.signal


def test_different_families_same_projection_have_different_hash() -> None:
    """Same projected genes, different family -> different effective hash."""
    rng = np.random.default_rng(20261001)
    dna1 = dataclasses.replace(random_dna(rng), signal="MA_CROSS")
    dna2 = dataclasses.replace(random_dna(rng), signal="RSI_REVERSION")
    # Force identical common genes.
    dna2 = dataclasses.replace(
        dna2,
        risk_pct=dna1.risk_pct,
        sl_atr=dna1.sl_atr,
        tp_atr=dna1.tp_atr,
    )
    assert effective_gene_hash(dna1) != effective_gene_hash(dna2)


def test_colliding_dnas_now_distinct_with_family_projection() -> None:
    """Two DNAs that collided under the old (MA-only) projection now differ.

    OLD effective_genes ignored family and only hashed (fast, slow, volume,
    sl_atr, tp_atr) -- an MA_CROSS and an RSI_REVERSION sharing those values
    got the SAME hash. That is a dedupe collision: the GA would treat them
    as one trial. With family-aware projection they hash apart.
    """
    rng = np.random.default_rng(20261002)
    d_ma = dataclasses.replace(random_dna(rng), signal="MA_CROSS",
                               max_feature_lookback=40, risk_pct=0.05,
                               sl_atr=2.0, tp_atr=3.0)
    # rsi_period=10 (within [5,21]) -- the only family-varying gene.
    d_rsi = dataclasses.replace(random_dna(rng), signal="RSI_REVERSION",
                                max_feature_lookback=40, risk_pct=0.05,
                                sl_atr=2.0, tp_atr=3.0, rsi_period=10)
    assert effective_gene_hash(d_ma) != effective_gene_hash(d_rsi)


# ---------------------------------------------------------------------------
# 1.A(ii): per-family, diffing one listed gene changes the hash
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "signal,field,value",
    [
        ("MA_CROSS", "max_feature_lookback", 60),     # fast+slow change
        ("RSI_REVERSION", "rsi_period", 9),
        ("RSI_REVERSION", "rsi_oversold", 20.0),
        ("RSI_REVERSION", "rsi_overbought", 80.0),
        ("RSI_REVERSION", "rsi_exit_midline", True),
        ("STOCH_CROSS", "stoch_k_period", 18),
        ("STOCH_CROSS", "stoch_oversold", 15.0),
        ("STOCH_CROSS", "stoch_overbought", 85.0),
        ("MACD_CROSS", "macd_fast", 10),
        ("DONCHIAN_BREAKOUT", "donchian_period", 15),
    ],
)
def test_single_family_gene_change_changes_hash(signal, field, value) -> None:
    """Changing one engine-reaching gene for its family must change the hash."""
    rng = np.random.default_rng(20261003)
    dna_a = dataclasses.replace(random_dna(rng), signal=signal)
    if isinstance(value, bool) and getattr(dna_a, field) == value:
        # The draw already holds the target boolean; start from its complement.
        dna_a = dataclasses.replace(dna_a, **{field: not value})
    dna_b = dataclasses.replace(dna_a, **{field: value})
    assert dna_a.dna_hash != dna_b.dna_hash, "test seed did not differ"
    assert effective_gene_hash(dna_a) != effective_gene_hash(dna_b), (
        f"effective hash did not change when {field}={value} "
        f"was varied for family {signal}"
    )


# ---------------------------------------------------------------------------
# 1.A(iii): dedupe-level -- two DNAs colliding under old projection now distinct
# ---------------------------------------------------------------------------
def test_old_ma_only_collision_now_uses_family() -> None:
    """Under the old MA-only projection, an MA_CROSS and an STOCH_CROSS with
    the same (fast, slow, volume, sl_atr, tp_atr) collided. With family-aware
    projection they are distinct groups.
    """
    rng = np.random.default_rng(20261004)
    d_ma = dataclasses.replace(random_dna(rng), signal="MA_CROSS",
                               max_feature_lookback=40, risk_pct=0.05,
                               sl_atr=2.0, tp_atr=3.0)
    d_stoch = dataclasses.replace(random_dna(rng), signal="STOCH_CROSS",
                                  max_feature_lookback=40, risk_pct=0.05,
                                  sl_atr=2.0, tp_atr=3.0)
    # Different families => different effective groups.
    assert effective_gene_hash(d_ma) != effective_gene_hash(d_stoch)


def test_unbacked_signal_raises() -> None:
    """A signal not in _FAMILY_PROJECTIONS must not get an effective hash."""
    rng = np.random.default_rng(20261005)
    dna = dataclasses.replace(random_dna(rng), signal="MA_CROSS")
    # Manually craft a DNA with a phantom signal (bypass __post_init__ since
    # it validates against registered features).
    bad = dataclasses.replace(dna, signal="PHANTOM_FAMILY")
    with pytest.raises(Exception, match="no family projection"):
        effective_genes(bad)
