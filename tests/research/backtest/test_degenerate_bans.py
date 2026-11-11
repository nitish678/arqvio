"""C3-1.C: Degenerate bans completion + raise-tests for all families.

Every family's config or DNA validator must reject degenerate gene
combinations that produce non-actionable or near-zero indicator output:

- RSI: oversold >= overbought; overbought - oversold < 5 (RSIConfig.__post_init__)
- STOCH: k_oversold >= k_overbought; k_period < 2 (StochCrossConfig.__post_init__)
- MACD: macd_slow - macd_fast < 5 (MACDCrossConfig.__post_init__ + DNA validation);
  subsumes the mission's slow <= fast + 3 ban
- DONCHIAN: period < 2 (DonchianBreakoutConfig.__post_init__); the mission's
  exit_ch >= entry_ch ban is N/A (single donchian_period gene)

All bans are enforced at the config constructor and asserted through the
DNA projection (dna_to_*_config) as well, so neither the fast-path fallback nor the slow engine
can ever construct a degenerate config.
"""
from __future__ import annotations

import pytest

from engine.core.errors import DataError
from engine.research.backtest.strategies.rsi import RSIConfig
from engine.research.backtest.strategies.stoch_cross import StochCrossConfig
from engine.research.backtest.strategies.macd_cross import MACDCrossConfig


# --- RSI: oversold >= overbought must reject ---

def test_rsi_rejects_oversold_equal_overbought() -> None:
    """RSIConfig with oversold == overbought is degenerate (no signal zone)."""
    with pytest.raises(DataError, match="oversold"):
        RSIConfig(oversold=50.0, overbought=50.0)


def test_rsi_rejects_oversold_greater_overbought() -> None:
    """RSIConfig with oversold > overbought is logically inverted."""
    with pytest.raises(DataError, match="oversold"):
        RSIConfig(oversold=60.0, overbought=40.0)


def test_rsi_allows_strictly_oversold_below_overbought() -> None:
    """RSIConfig with oversold < overbought is valid."""
    cfg = RSIConfig(oversold=30.0, overbought=70.0)
    assert cfg.oversold == 30.0
    assert cfg.overbought == 70.0


# --- STOCH: k_oversold >= k_overbought must reject ---

def test_stoch_rejects_k_oversold_equal_k_overbought() -> None:
    """StochCrossConfig with k_oversold == k_overbought (both in active zone) is degenerate."""
    with pytest.raises(DataError, match="k_oversold"):
        StochCrossConfig(k_oversold=50.0, k_overbought=50.0)


def test_stoch_rejects_k_oversold_greater_k_overbought() -> None:
    """StochCrossConfig with inverted gates is logically wrong."""
    with pytest.raises(DataError, match="k_oversold"):
        StochCrossConfig(k_oversold=80.0, k_overbought=20.0)


def test_stoch_allows_k_oversold_below_k_overbought() -> None:
    """StochCrossConfig with valid gate ordering is accepted."""
    cfg = StochCrossConfig(k_oversold=20.0, k_overbought=80.0)
    assert cfg.k_oversold == 20.0
    assert cfg.k_overbought == 80.0


# --- MACD: slow - fast < 5 must reject ---

def test_macd_rejects_slow_equal_fast() -> None:
    """MACD with slow == fast produces zero MACD line — degenerate."""
    with pytest.raises(DataError, match="fast_period"):
        MACDCrossConfig(fast_period=12, slow_period=12)


def test_macd_rejects_slow_less_than_fast() -> None:
    """MACD with slow < fast inverts the MACD line sign — degenerate."""
    with pytest.raises(DataError, match="fast_period"):
        MACDCrossConfig(fast_period=26, slow_period=12)


@pytest.mark.parametrize("gap", [1, 2, 3, 4])
def test_macd_rejects_slow_fast_gap_below_5(gap: int) -> None:
    """MACD slow-fast gap < 5 produces near-zero MACD line — degenerate."""
    with pytest.raises(DataError, match="gap"):
        MACDCrossConfig(fast_period=12, slow_period=12 + gap)


def test_macd_allows_gap_5_or_more() -> None:
    """MACD with slow - fast >= 5 is accepted."""
    cfg = MACDCrossConfig(fast_period=12, slow_period=26)
    assert cfg.slow_period - cfg.fast_period == 14


@pytest.mark.parametrize("fast,slow", [(12, 13), (12, 15), (12, 16)])
def test_macd_rejects_gap_below_5(fast: int, slow: int) -> None:
    """Direct gap validation: slow - fast must be >= 5."""
    if slow - fast >= 5:
        pytest.skip(f"gap {slow - fast} >= 5, should pass")
    with pytest.raises(DataError, match="gap"):
        MACDCrossConfig(fast_period=fast, slow_period=slow)


# --- DNA-level validation ---

def test_dna_rejects_macd_slow_fast_gap_below_5() -> None:
    """DNA validation layer must reject macd_slow - macd_fast < 5."""
    from engine.research.search.dna import DNA, validate_gene_space

    dna = DNA(
        bias="DAILY_TREND", filter="NONE", signal="MACD_CROSS",
        signal2="", exec_mode="MARKET2", risk="FIXED_ATR",
        mgmt="NONE",
        sl_atr=2.0, tp_atr=3.0, risk_pct=0.01,
        friday_close=False, news_filter=False,
        daily_dd_limit=0.05, max_agg_risk=0.03,
        max_feature_lookback=50, max_hold_time=200,
        window_start_hour_utc=8, window_end_hour_utc=20,
        rsi_period=14, rsi_oversold=30.0, rsi_overbought=70.0,
        rsi_exit_midline=False,
        stoch_k_period=14, stoch_d_period=3, stoch_smoothing=3,
        stoch_oversold=20.0, stoch_overbought=80.0,
        macd_fast=20, macd_slow=22,  # gap=2 < 5
        macd_signal_period=9,
        donchian_period=20,
        uses_future_bar=False, schema_version=1,
    )
    with pytest.raises(ValueError, match="macd gap"):
        validate_gene_space(dna.to_dict())


def test_dna_allows_macd_slow_fast_gap_5() -> None:
    """DNA validation layer must accept macd_slow - macd_fast >= 5."""
    from engine.research.search.dna import DNA, validate_gene_space

    dna = DNA(
        bias="DAILY_TREND", filter="NONE", signal="MACD_CROSS",
        signal2="", exec_mode="MARKET2", risk="FIXED_ATR",
        mgmt="NONE",
        sl_atr=2.0, tp_atr=3.0, risk_pct=0.01,
        friday_close=False, news_filter=False,
        daily_dd_limit=0.05, max_agg_risk=0.03,
        max_feature_lookback=50, max_hold_time=200,
        window_start_hour_utc=8, window_end_hour_utc=20,
        rsi_period=14, rsi_oversold=30.0, rsi_overbought=70.0,
        rsi_exit_midline=False,
        stoch_k_period=14, stoch_d_period=3, stoch_smoothing=3,
        stoch_oversold=20.0, stoch_overbought=80.0,
        macd_fast=12, macd_slow=22,  # gap=10 >= 5
        macd_signal_period=9,
        donchian_period=20,
        uses_future_bar=False, schema_version=1,
    )
    validate_gene_space(dna.to_dict())  # should not raise
    assert dna.macd_slow - dna.macd_fast >= 5


def test_dna_rejects_stoch_k_oversold_geq_k_overbought() -> None:
    """DNA range validation prevents stoch_oversold >= stoch_overbought:
    oversold is in [5,25], overbought in [75,95]. Verify the range guard
    rejects an inverted assignment."""
    from engine.research.search.dna import DNA, validate_gene_space

    dna = DNA(
        bias="DAILY_TREND", filter="NONE", signal="STOCH_CROSS",
        signal2="", exec_mode="MARKET2", risk="FIXED_ATR",
        mgmt="NONE",
        sl_atr=2.0, tp_atr=3.0, risk_pct=0.01,
        friday_close=False, news_filter=False,
        daily_dd_limit=0.05, max_agg_risk=0.03,
        max_feature_lookback=50, max_hold_time=200,
        window_start_hour_utc=8, window_end_hour_utc=20,
        rsi_period=14, rsi_oversold=30.0, rsi_overbought=70.0,
        rsi_exit_midline=False,
        stoch_k_period=14, stoch_d_period=3, stoch_smoothing=3,
        stoch_oversold=80.0, stoch_overbought=20.0,  # out of range
        macd_fast=12, macd_slow=26,
        macd_signal_period=9,
        donchian_period=20,
        uses_future_bar=False, schema_version=1,
    )
    with pytest.raises(ValueError, match="stoch_oversold"):
        validate_gene_space(dna.to_dict())


# --- S3 PART 2.2 completion: remaining bans at BOTH layers ---
#
# Every ban below is asserted twice: directly on the *_Config constructor and
# through the DNA projection (dna_to_*_config), which is the path every trial
# takes. DONCHIAN's exit_ch>=entry_ch ban is N/A: the committed gene space has
# a single donchian_period (spec-owner ruling), so only period<2 applies.

from engine.research.backtest.strategies.donchian_breakout import (  # noqa: E402
    DonchianBreakoutConfig,
)
from engine.research.search.dna import DNA  # noqa: E402
from engine.research.search.real_backtest import (  # noqa: E402
    dna_to_donchian_config,
    dna_to_macd_config,
    dna_to_rsi_config,
    dna_to_stoch_config,
)


def _dna(signal: str, **genes) -> DNA:
    base = dict(
        bias="NONE", filter="NONE", signal=signal, signal2="",
        exec_mode="MARKET2", risk="FIXED_ATR", mgmt="NONE",
        sl_atr=2.0, tp_atr=3.0, risk_pct=0.01,
        friday_close=False, news_filter=False,
        daily_dd_limit=0.05, max_agg_risk=0.03,
        max_feature_lookback=50, max_hold_time=200,
        window_start_hour_utc=0, window_end_hour_utc=23,
    )
    base.update(genes)
    return DNA(**base)


@pytest.mark.parametrize("oversold,overbought", [(30.0, 34.0), (30.0, 34.999)])
def test_rsi_rejects_zone_gap_below_5(oversold: float, overbought: float) -> None:
    """overbought - oversold < 5 leaves no room between the entry zones."""
    with pytest.raises(DataError, match="gap"):
        RSIConfig(oversold=oversold, overbought=overbought)


def test_rsi_allows_zone_gap_exactly_5() -> None:
    cfg = RSIConfig(oversold=30.0, overbought=35.0)
    assert cfg.overbought - cfg.oversold == 5.0


def test_rsi_zone_gap_ban_via_dna_projection() -> None:
    with pytest.raises(DataError, match="gap"):
        dna_to_rsi_config(_dna("RSI_REVERSION", rsi_oversold=40.0, rsi_overbought=42.0))


@pytest.mark.parametrize("k_period", [0, 1])
def test_stoch_rejects_k_period_below_2(k_period: int) -> None:
    """k_period < 2 makes HH(k)==LL(k) on every bar: raw %K is undefined."""
    with pytest.raises(DataError, match="k_period"):
        StochCrossConfig(k_period=k_period)


def test_stoch_allows_k_period_2() -> None:
    assert StochCrossConfig(k_period=2).k_period == 2


def test_stoch_k_period_ban_via_dna_projection() -> None:
    with pytest.raises(DataError, match="k_period"):
        dna_to_stoch_config(_dna("STOCH_CROSS", stoch_k_period=1))


def test_stoch_inverted_zones_ban_via_dna_projection() -> None:
    with pytest.raises(DataError, match="k_oversold"):
        dna_to_stoch_config(
            _dna("STOCH_CROSS", stoch_oversold=80.0, stoch_overbought=20.0)
        )


@pytest.mark.parametrize("fast,slow", [(12, 12), (12, 15), (20, 23)])
def test_macd_slow_le_fast_plus_3_ban_via_dna_projection(fast: int, slow: int) -> None:
    """Mission ban slow<=fast+3 is subsumed by the committed gap>=5 ban."""
    with pytest.raises(DataError):
        dna_to_macd_config(_dna("MACD_CROSS", macd_fast=fast, macd_slow=slow))


@pytest.mark.parametrize("period", [0, 1])
def test_donchian_rejects_period_below_2(period: int) -> None:
    """A 1-bar prior channel is the previous bar itself: degenerate."""
    with pytest.raises(DataError, match="period"):
        DonchianBreakoutConfig(period=period)


def test_donchian_allows_period_2() -> None:
    assert DonchianBreakoutConfig(period=2).period == 2


def test_donchian_period_ban_via_dna_projection() -> None:
    with pytest.raises(DataError, match="period"):
        dna_to_donchian_config(_dna("DONCHIAN_BREAKOUT", donchian_period=1))
