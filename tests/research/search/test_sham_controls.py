"""W3: the sham random-entry control.

Pinned to the USDCADm H1 snapshot (first 3,000 bars) so every number the
shams produce is reproducible; skipped where the snapshot was never taken.
The interface and strategy tests need no data and always run.
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from engine.research.backtest.strategies.sham import ShamConfig, ShamStrategy
from engine.research.campaign.budget import CampaignScope
from engine.research.data.dataset_snapshot import find_snapshot, load_snapshot
from engine.research.provenance import DataSource, Provenance
from engine.research.search.canary import CanaryResult
from engine.research.search.dna import FailureTaxonomy
from engine.research.search.real_backtest import (
    GAUSSIAN_CONTROL_KIND,
    CellBars,
    dna_to_strategy,
    fast_backtest_factory,
    real_scores,
    slow_cell_setup,
)
from engine.research.search.runner import TrialResult, run_campaign
from engine.research.search.sampler import random_dna
from engine.research.search.sham_control import (
    CONTROLS_REGIME,
    SHAM_CONTROL_KIND,
    SHAM_SALT,
    attempted_entries,
    controls_block,
    draw_entries,
    eligible_entry_range,
    sham_backtest_factory,
    sham_control_factory,
)
from engine.validation.trial_registry import TrialRegistry
from engine.validation.vs_random import vs_random

DATASET_ID = "aed3bf314539ec5a"
SYMBOL, TIMEFRAME = "USDCADm", "H1"
SLICE_BARS = 3_000
N_DNAS = 8


def _provenance(dataset_hash: str) -> Provenance:
    return Provenance(
        source=DataSource.MT5_LIVE, dataset_hash=dataset_hash,
        as_of=datetime(2026, 9, 29, tzinfo=UTC), mixed_source=False,
    )


@pytest.fixture(scope="module")
def cell() -> CellBars:
    if find_snapshot(DATASET_ID) is None:
        pytest.skip(f"snapshot {DATASET_ID} not taken on this machine")
    from engine.research.data.mt5_source import BarData

    bars = load_snapshot(DATASET_ID)
    n = SLICE_BARS
    sliced = BarData(
        time=np.asarray(bars.time)[:n],
        open=np.asarray(bars.open)[:n],
        high=np.asarray(bars.high)[:n],
        low=np.asarray(bars.low)[:n],
        close=np.asarray(bars.close)[:n],
        tick_volume=np.asarray(bars.tick_volume)[:n],
        spread=np.asarray(bars.spread)[:n],
        real_volume=np.asarray(bars.real_volume)[:n],
        symbol=bars.symbol,
        timeframe=bars.timeframe,
    )
    log_ret = np.diff(np.log(np.asarray(sliced.close, dtype=np.float64)))
    return CellBars(
        symbol=SYMBOL, timeframe=TIMEFRAME, bars=sliced, n_bars=n,
        n_returns=int(log_ret.size), returns=log_ret,
        source="MT5_SNAPSHOT", dataset_hash=f"{DATASET_ID}-first{n}",
    )


@pytest.fixture(scope="module")
def real_trials(cell: CellBars) -> list[TrialResult]:
    """Real DNAs from a fixed seed, on the fast kernel (which reproduces the
    oracle; the parity fixtures in tests/research/backtest pin that). Only
    the shams need the slow engine, and they get it."""
    rng = np.random.default_rng(11)
    bt = fast_backtest_factory(cell)
    return [bt(random_dna(rng)) for _ in range(N_DNAS)]


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------
def test_same_dna_and_salt_give_an_identical_sham(cell, real_trials) -> None:
    t = real_trials[0]
    n = attempted_entries(t)
    a = sham_backtest_factory(cell)(t.dna, n)
    b = sham_backtest_factory(cell)(t.dna, n)
    assert a.sharpe == b.sharpe
    assert a.n_trades == b.n_trades
    assert a.exit_reasons == b.exit_reasons
    assert np.array_equal(a.per_period_returns, b.per_period_returns)


def test_a_different_salt_draws_a_different_entry_set(cell, real_trials) -> None:
    t = real_trials[0]
    cfg, _ = dna_to_strategy(t.dna)
    lo, hi = eligible_entry_range(cfg, cell.n_bars)
    n = attempted_entries(t)
    assert n > 0, "fixture DNA never traded; the comparison is vacuous"
    bars_a, _ = draw_entries(t.dna.dna_hash, SHAM_SALT, lo, hi, n)
    bars_b, _ = draw_entries(t.dna.dna_hash, "another-salt", lo, hi, n)
    assert bars_a != bars_b
    assert draw_entries(t.dna.dna_hash, SHAM_SALT, lo, hi, n)[0] == bars_a


# ---------------------------------------------------------------------------
# isolation
# ---------------------------------------------------------------------------
def test_a_sham_does_not_depend_on_what_was_evaluated_before_it(
    cell, real_trials,
) -> None:
    target = real_trials[0]
    first = sham_backtest_factory(cell)(target.dna, attempted_entries(target))
    later_fn = sham_backtest_factory(cell)
    for t in real_trials[1:4]:
        later_fn(t.dna, attempted_entries(t))
    later = later_fn(target.dna, attempted_entries(target))
    assert first.sharpe == later.sharpe
    assert np.array_equal(first.per_period_returns, later.per_period_returns)


def test_the_sham_control_does_not_change_the_campaigns_dna_sequence(
    cell, tmp_path: Path,
) -> None:
    def _campaign(control_fn, tag: str) -> tuple[list[str], object]:
        seen: list[str] = []
        base = fast_backtest_factory(cell)

        def _bt(dna) -> TrialResult:
            seen.append(dna.dna_hash)
            return base(dna)

        result = run_campaign(
            campaign_id=f"sham-isolation-{tag}",
            scope=CampaignScope(
                symbols=(SYMBOL,), timeframes=(TIMEFRAME,), population_per_cell=12,
            ),
            registry=TrialRegistry(tmp_path / f"{tag}.sqlite"),
            backtest_fn=_bt,
            control_fn=control_fn,
            real_scores_factory=real_scores,
            provenance=_provenance(cell.dataset_hash),
            dataset_hash=cell.dataset_hash,
            seed=3,
            n_obs=cell.n_returns,
        )
        return seen, result

    without, r0 = _campaign(None, "none")
    with_sham, r1 = _campaign(sham_control_factory(cell), "sham")
    assert without == with_sham
    assert r0.vs_random is None
    assert r1.vs_random is not None


# ---------------------------------------------------------------------------
# fired and gated
# ---------------------------------------------------------------------------
def test_shams_fire_their_entries_on_both_sides(cell, real_trials) -> None:
    sham = sham_backtest_factory(cell)
    active = [t for t in real_trials if int(t.n_trades or 0) >= 10]
    assert len(active) >= 3, "too few trading DNAs in the fixture"
    real_fills = sham_fills = 0
    for t in active:
        requested = attempted_entries(t)
        s = sham(t.dna, requested)
        # Every drawn entry is emitted: it either filled or was refused by
        # the gate. Only a reversal on the final bar can lose one.
        emitted = int(s.n_trades) + int(s.blocked_max_spread)
        assert emitted >= 0.9 * requested, (t.dna.signal, emitted, requested)
        assert s.parity_class == t.parity_class
        real_fills += int(t.n_trades)
        sham_fills += int(s.n_trades)
        cfg, _ = dna_to_strategy(t.dna)
        lo, hi = eligible_entry_range(cfg, cell.n_bars)
        _, sides = draw_entries(t.dna.dna_hash, SHAM_SALT, lo, hi, requested)
        assert {"long", "short"} <= set(sides)
    assert sham_fills >= 0.9 * real_fills, (sham_fills, real_fills)


def test_shams_meet_the_production_gates(cell, real_trials) -> None:
    setup = slow_cell_setup(cell)
    if setup.max_spread_points is None and not setup.no_trade_hours:
        pytest.skip(f"{SYMBOL} has no calibrated gate on this machine")
    sham = sham_backtest_factory(cell)
    gated = 0
    for t in real_trials[:4]:
        s = sham(t.dna, attempted_entries(t))
        gated += int(s.blocked_max_spread) + int(s.blocked_zero_spread_events)
    assert gated > 0


# ---------------------------------------------------------------------------
# interface and report
# ---------------------------------------------------------------------------
def test_run_campaign_hands_control_fn_the_scored_trials(tmp_path: Path) -> None:
    calls: list[list[TrialResult]] = []
    state = {"i": 0}

    def _bt(dna) -> TrialResult:
        i = state["i"]
        state["i"] += 1
        return TrialResult(
            dna=dna, outcome=FailureTaxonomy.EVALUATED.value,
            sharpe=None if i % 4 == 0 else 0.1 * i,
            n_trades=40 if i % 2 else 5, total_return=0.0, max_dd=0.0,
        )

    def _ctrl(trials: list[TrialResult]) -> list[float]:
        calls.append(list(trials))
        return [0.0] * len(trials)

    run_campaign(
        campaign_id="control-interface",
        scope=CampaignScope(
            symbols=(SYMBOL,), timeframes=(TIMEFRAME,), population_per_cell=20,
        ),
        registry=TrialRegistry(tmp_path / "trials.sqlite"),
        backtest_fn=_bt,
        control_fn=_ctrl,
        real_scores_factory=real_scores,
        provenance=_provenance("d"),
        seed=5,
        n_obs=0,
    )
    assert len(calls) == 2
    everything, eligible = calls
    assert all(isinstance(t, TrialResult) for t in everything + eligible)
    assert all(t.sharpe is not None and np.isfinite(t.sharpe) for t in everything)
    assert len(everything) == 15  # 20 trials, every fourth unscored
    assert eligible == [t for t in everything if t.n_trades >= 30]


def test_controls_block_describes_the_sham_regime(cell, real_trials) -> None:
    ctrl = sham_control_factory(cell)
    scored = [t for t in real_trials if t.sharpe is not None]
    control_scores = ctrl(scored)
    again = ctrl(scored[:3])
    assert again == control_scores[:3]
    assert len(ctrl.shams) == len({(t.dna.dna_hash, attempted_entries(t)) for t in scored})
    v = vs_random(real_scores(scored), control_scores,
                  n_bootstrap=200, n_permutations=100, seed=0)
    block = controls_block(ctrl, vs_all=v, vs_eligible=None)
    assert block["kind"] == SHAM_CONTROL_KIND
    assert block["salt"] == SHAM_SALT
    assert block["engine"] == "slow"
    assert block["side"] == "50/50"
    assert block["seed"] == "sha256(dna_hash:salt)"
    assert block["n_all"] == len(scored)
    assert block["n_eligible"] == 0
    assert block["random_max_q95"] == {"all": v.random_max_q95, "eligible": None}
    assert block["entries_requested"] == sum(attempted_entries(t) for t in scored)
    assert all(w.startswith(("all: ", "sham ")) for w in block["warnings"])
    assert "v1 iid N(0,0.5)" in CONTROLS_REGIME
    assert "not comparable" in CONTROLS_REGIME


def test_the_canary_states_its_control_is_gaussian() -> None:
    result = CanaryResult(
        passed=True, survivors=1, dsr_within=1.0, vs_random_p_max=0.0,
        exit_verdict="", crashes=0, git_sha="x",
    )
    assert result.control == GAUSSIAN_CONTROL_KIND


def test_sham_strategy_reverses_through_pending_and_loses_no_draw() -> None:
    """Draws at 5 (long), 8 (short), 9 (long). 8 finds the long open: close,
    short on 9. 9's own draw collides with that pending entry, so it is taken
    on 10: close the short, long on 11."""
    strategy = ShamStrategy(ShamConfig(
        entry_bars=(5, 8, 9), sides=("long", "short", "long"),
        volume=0.1, strategy_id="sham_unit",
    ))
    state: dict[str, object] = {}
    emitted = {}
    for i in range(14):
        intent = strategy.on_bar(i, None, state)  # no stops: bars are unread
        if intent is not None:
            emitted[i] = (intent.direction, intent.volume)
    assert emitted == {
        5: ("long", 0.1),
        8: ("short", 0.0),
        9: ("short", 0.1),
        10: ("long", 0.0),
        11: ("long", 0.1),
    }
