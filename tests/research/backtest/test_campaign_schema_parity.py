"""S2-D: a fast campaign must be indistinguishable from a slow one.

Not merely "the numbers agree" -- the REPORT must carry the same
fields. A fast path that quietly dropped trials_effective, or
parity_class, or the grade, would produce a report that looks fine and
answers fewer questions than the slow one. Any field present in slow
mode and missing in fast mode is a failure here.

The 50-trial tripwire runs both modes over the same pinned dataset and
compares the campaign result field by field.
"""
from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from engine.research.campaign.budget import CampaignScope
from engine.research.data.dataset_snapshot import find_snapshot
from engine.research.provenance import DataSource, Provenance
from engine.research.search.real_backtest import (
    fast_backtest_factory,
    gaussian_control_factory,
    load_cell_bars_from_snapshot,
    real_backtest_factory,
    real_scores,
)
from engine.research.search.runner import run_campaign
from engine.validation.trial_registry import TrialRegistry

DATASET_ID = "aed3bf314539ec5a"
SYMBOL, TIMEFRAME = "USDCADm", "H1"
N_TRIALS = 50
SLICE_BARS = 4_000

# Every field the campaign result carries that a reader would act on.
# Missing in fast mode == FAIL.
REQUIRED_FIELDS = (
    "trials_raw",
    "trials_effective",
    "dsr_N_used",
    "exit_reasons",
    "parity_classes",
    "blocked_zero_spread_events",
    "deferred_zero_spread_fills",
    "sltp_deferred_zero_spread",
    "distinct_dsr_count",
    "trades_by_bucket",
)


def _cell():
    from engine.research.data.mt5_source import BarData
    from engine.research.search.real_backtest import CellBars

    full = load_cell_bars_from_snapshot(SYMBOL, TIMEFRAME, DATASET_ID)
    n = min(SLICE_BARS, full.n_bars)
    bars = full.bars
    sliced = BarData(
        time=np.asarray(bars.time)[-n:],
        open=np.asarray(bars.open)[-n:],
        high=np.asarray(bars.high)[-n:],
        low=np.asarray(bars.low)[-n:],
        close=np.asarray(bars.close)[-n:],
        tick_volume=np.asarray(bars.tick_volume)[-n:],
        spread=np.asarray(bars.spread)[-n:],
        real_volume=np.asarray(bars.real_volume)[-n:],
        symbol=bars.symbol,
        timeframe=bars.timeframe,
    )
    closes = np.asarray(sliced.close, dtype=np.float64)
    log_ret = np.diff(np.log(closes))
    return CellBars(
        symbol=SYMBOL, timeframe=TIMEFRAME, bars=sliced, n_bars=n,
        n_returns=int(log_ret.size), returns=log_ret,
        source="MT5_SNAPSHOT", dataset_hash=f"{DATASET_ID}-sliced{n}",
    )


def _run(cell, tmp_path, mode: str):
    """One campaign in ``mode``. The control is the gaussian one
    (gaussian_control_factory): this file
    compares fast against slow, and a sham control would add slow-engine
    runs to both sides without changing what is compared."""
    factory = (
        fast_backtest_factory if mode == "fast" else real_backtest_factory
    )
    registry = TrialRegistry(tmp_path / f"{mode}.sqlite")
    return run_campaign(
        campaign_id=f"schema-{mode}",
        scope=CampaignScope(
            symbols=(SYMBOL,), timeframes=(TIMEFRAME,),
            population_per_cell=N_TRIALS,
        ),
        registry=registry,
        backtest_fn=factory(cell),
        control_fn=gaussian_control_factory(np.random.default_rng(1)),
        real_scores_factory=real_scores,
        provenance=Provenance(
            source=DataSource.MT5_LIVE, dataset_hash=cell.dataset_hash,
            as_of=datetime.now(tz=UTC), mixed_source=False,
        ),
        target_is_annual_sharpe=1.5,
        dataset_hash=cell.dataset_hash,
        git_sha="testsha",
        config_hash="testcfg",
        seed=4,
        n_obs=cell.n_returns,
        bars_loaded=cell.n_bars,
        audit_bars_expected=cell.n_bars,
    )


@pytest.fixture(scope="module")
def both(tmp_path_factory):
    if find_snapshot(DATASET_ID) is None:
        pytest.skip(f"snapshot {DATASET_ID} not taken on this machine")
    cell = _cell()
    base = tmp_path_factory.mktemp("schema")
    slow = _run(cell, base, "slow")
    fast = _run(cell, base, "fast")
    return slow, fast


def test_every_required_field_is_present_in_fast_mode(both) -> None:
    slow, fast = both
    missing = [
        name for name in REQUIRED_FIELDS
        if getattr(fast, name, None) is None and getattr(slow, name, None) is not None
    ]
    assert not missing, f"fast mode dropped: {missing}"


def test_field_by_field_comparison(both) -> None:
    """The report table. Printed so the comparison is in the log."""
    slow, fast = both
    lines = []
    mismatches = []
    for name in REQUIRED_FIELDS:
        s_val = getattr(slow, name, None)
        f_val = getattr(fast, name, None)
        same = s_val == f_val
        lines.append(f"  {name:32s} slow={s_val!s:40.40s} fast={f_val!s:40.40s} {'OK' if same else 'DIFFER'}")
        if not same:
            mismatches.append(f"{name}: slow={s_val!r} fast={f_val!r}")
    print("\ncampaign schema parity, 50 trials, pinned dataset:")
    print("\n".join(lines))
    assert not mismatches, "fields differ:\n  " + "\n  ".join(mismatches)


def test_funnel_and_verdict_agree(both) -> None:
    slow, fast = both
    assert dict(slow.funnel.by_category) == dict(fast.funnel.by_category), (
        f"funnel slow={dict(slow.funnel.by_category)} "
        f"fast={dict(fast.funnel.by_category)}"
    )
    assert slow.funnel.survivors == fast.funnel.survivors
    assert slow.exit_verdict() == fast.exit_verdict()


def test_the_tripwire_actually_exercised_the_engine(both) -> None:
    """A campaign where nothing traded would compare two empty reports."""
    slow, _ = both
    traded = sum(slow.trades_by_bucket.values())
    assert traded > 0, "no trades in either mode; the comparison is vacuous"
    assert slow.trials_raw == N_TRIALS
