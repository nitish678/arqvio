"""W4 degeneracy fields and campaign cost totals."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from engine.research.search.degeneracy import (
    FLOAT_EDGE_FRACTION,
    boundary_saturation,
    cost_totals,
    degeneracy_report,
)
from engine.research.search.dna import FailureTaxonomy
from engine.research.search.runner import TrialResult
from engine.research.search.sampler import random_dna

EVAL = FailureTaxonomy.EVALUATED.value


def _trial(signal: str, *, n_trades: int = 50, sharpe: float = 0.0, costs=None, **genes) -> TrialResult:
    dna = dataclasses.replace(random_dna(np.random.default_rng(7)), signal=signal, **genes)
    return TrialResult(dna=dna, sharpe=sharpe, n_trades=n_trades, costs=costs or {})


def test_family_counts_zero_trade_rates_and_min_trade_failures() -> None:
    trials = [
        _trial("MA_CROSS", n_trades=0),
        _trial("MA_CROSS", n_trades=10),
        _trial("MA_CROSS", n_trades=40),
        _trial("RSI_REVERSION", n_trades=0),
    ]
    rep = degeneracy_report(trials, [EVAL] * 4, min_trades=30, trials_raw=4, trials_effective=3)
    assert rep["trials_per_family"] == {
        "MA_CROSS": 3, "RSI_REVERSION": 1, "STOCH_CROSS": 0,
        "MACD_CROSS": 0, "DONCHIAN_BREAKOUT": 0,
    }
    assert rep["zero_trade_rate_per_family"]["MA_CROSS"] == pytest.approx(1 / 3)
    assert rep["zero_trade_rate_per_family"]["RSI_REVERSION"] == 1.0
    assert rep["zero_trade_rate_per_family"]["STOCH_CROSS"] is None
    assert rep["min_trades_failures_per_family"]["MA_CROSS"] == 2
    assert rep["effective_over_real"] == 0.75
    assert rep["dedupe_hits"] == 1


def test_boundary_saturation_flags_a_gene_pinned_to_its_range_end() -> None:
    trials = [
        _trial("DONCHIAN_BREAKOUT", sharpe=3.0 - i * 0.1, donchian_period=55 if i < 3 else 30)
        for i in range(4)
    ]
    sat = boundary_saturation(trials, [EVAL] * 4, min_trades=30)
    assert "top quartile" in sat["basis"]
    # Top quartile of 4 eligible = the best one, which sits at 55 (the max).
    # The whole entry, exactly: the three original keys and the uniform-baseline
    # keys beside them (donchian_period ranges over 10..55: k = 46).
    assert sat["per_gene"]["DONCHIAN_BREAKOUT.donchian_period"] == {
        "n": 1, "at_boundary": 1, "share": 1.0,
        "raw_share": 1.0, "distinct_values": 1, "value_set_size": 46,
        "uniform_baseline": 2 / 46, "adjusted_margin": 1.0 - 2 / 46,
        "small_value_set_caveat": False,
    }
    assert sat["flagged"] == ["DONCHIAN_BREAKOUT.donchian_period"]


def test_boundary_saturation_uses_survivors_when_there_are_enough() -> None:
    vaulted = FailureTaxonomy.VAULTED.value
    trials = [
        _trial("RSI_REVERSION", rsi_period=5 if i % 2 else 13, rsi_oversold=10.0 + 25.0 * FLOAT_EDGE_FRACTION / 2)
        for i in range(4)
    ]
    sat = boundary_saturation(trials, [vaulted] * 4, min_trades=30)
    assert sat["basis"] == "survivors (n=4)"
    assert sat["per_gene"]["RSI_REVERSION.rsi_period"]["share"] == 0.5
    assert sat["per_gene"]["RSI_REVERSION.rsi_oversold"]["share"] == 1.0
    assert sat["flagged"] == ["RSI_REVERSION.rsi_oversold"]
    assert "RSI_REVERSION.rsi_exit_midline" not in sat["per_gene"]  # bool: no boundary


def test_cost_totals_sum_every_line() -> None:
    trials = [
        _trial("MA_CROSS", costs={"spread": 1.5, "commission": 0.5, "slippage": 0.25, "swap": -2.0}),
        _trial("MA_CROSS", costs={"spread": 2.5, "commission": 0.5, "slippage": 0.75, "swap": 1.0}),
        _trial("MA_CROSS"),
    ]
    assert cost_totals(trials) == {"spread": 4.0, "commission": 1.0, "slippage": 1.0, "swap": -1.0}


def test_run_campaign_reports_degeneracy_and_cost_totals(tmp_path) -> None:
    from datetime import UTC, datetime

    from engine.research.campaign.budget import CampaignScope
    from engine.research.provenance import DataSource, Provenance
    from engine.research.search.real_backtest import real_scores
    from engine.research.search.runner import run_campaign
    from engine.validation.trial_registry import TrialRegistry

    def bt(dna):
        return TrialResult(
            dna=dna, sharpe=0.1, n_trades=40, total_return=0.01, max_dd=0.05,
            per_period_returns=np.full(200, 0.0001),
            costs={"spread": 1.0, "commission": 0.0, "slippage": 0.5, "swap": 0.0},
        )

    result = run_campaign(
        campaign_id="degeneracy", scope=CampaignScope(("USDCADm",), ("H1",), 12),
        registry=TrialRegistry(tmp_path / "r.sqlite"), backtest_fn=bt, control_fn=None,
        real_scores_factory=real_scores,
        provenance=Provenance(
            source=DataSource.MT5_LIVE, dataset_hash="t",
            as_of=datetime.now(tz=UTC), mixed_source=False,
        ),
        n_obs=200,
    )
    assert sum(result.degeneracy["trials_per_family"].values()) == 12
    assert result.degeneracy["min_trades"] == 30
    assert result.cost_totals == {"spread": 12.0, "commission": 0.0, "slippage": 6.0, "swap": 0.0}
