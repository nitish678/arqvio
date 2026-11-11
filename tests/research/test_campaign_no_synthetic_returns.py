"""The campaign path must never fabricate a return series.

Defect (2026-09-17): ``run_campaign`` built its cell return series with
``np.random.default_rng(seed + 1).normal(0.0, 0.01, size=n_obs)`` and
``n_obs`` carried a silent default of 1500. Every DSR / PBO / vs-random
input must instead come from ``TrialResult.per_period_returns``, the real
bar-level series the backtester produces.

Two random draws are legitimate in a campaign and are NOT what these tests
forbid:

  - ``rng = np.random.default_rng(seed)`` -> ``random_dna(rng)``: the search
    sampler. A random search is random by definition.
  - the caller's ``control_fn`` and ``vs_random``'s own bootstrap /
    permutation resampling: the "unskilled" benchmark. Those live in the
    caller and in ``engine.validation.vs_random``, not in the runner.

Anything else drawing from ``np.random`` inside
``engine.research.search.runner`` is data fabrication, and that is what
fails here.
"""
from __future__ import annotations

import inspect
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from engine.research.campaign.budget import CampaignScope
from engine.research.provenance import DataSource, Provenance
from engine.research.search import runner as runner_module
from engine.research.search.dna import DNA, FailureTaxonomy
from engine.research.search.runner import TrialResult, run_campaign
from engine.validation.trial_registry import TrialRegistry

N_OBS = 300
POPULATION = 40  # >= 2 * PBO n_splits(16) so the PBO path actually runs


# ---------------------------------------------------------------------------
# Stand-in for loaded bars. Built ONCE, before the campaign starts, so that
# nothing inside the campaign path needs an rng.
# ---------------------------------------------------------------------------
def _real_bar_series(population: int = POPULATION) -> list[np.ndarray]:
    """Distinct per-candidate bar-level return series, deterministic."""
    seed_rng = np.random.default_rng(20260917)
    base = seed_rng.normal(0.0, 0.01, size=N_OBS)
    out: list[np.ndarray] = []
    for k in range(population):
        # Each candidate gets its own drift and its own shuffle of the same
        # bars, so Sharpes differ candidate by candidate.
        idx = seed_rng.permutation(N_OBS)
        out.append(base[idx] + (k - population / 2.0) * 1e-4)
    return out


def _series_sharpe(series: np.ndarray) -> float:
    sd = float(np.std(series, ddof=1))
    if sd == 0.0:
        return 0.0
    return float(np.mean(series) / sd * np.sqrt(252.0))


def _backtest_from_series(
    series: list[np.ndarray],
) -> Callable[[DNA], TrialResult]:
    """A backtester that hands back real bar-level returns, like the MT5 one."""
    state = {"i": 0}

    def _bt(dna: DNA) -> TrialResult:
        arr = series[state["i"] % len(series)]
        state["i"] += 1
        return TrialResult(
            dna=dna,
            outcome=FailureTaxonomy.EVALUATED.value,
            sharpe=_series_sharpe(arr),
            n_trades=60,
            total_return=float(np.sum(arr)),
            max_dd=0.05,
            spread_uncalibrated=False,
            per_period_returns=arr,
        )

    return _bt


def _provenance() -> Provenance:
    return Provenance(
        source=DataSource.MT5_LIVE,
        dataset_hash="real-bars",
        as_of=datetime(2026, 9, 17, tzinfo=UTC),
        mixed_source=False,
    )


def _scope() -> CampaignScope:
    return CampaignScope(
        symbols=("USDCADm",),
        timeframes=("D1",),
        population_per_cell=POPULATION,
    )


# ---------------------------------------------------------------------------
# The guard: numpy as the runner module sees it.
# ---------------------------------------------------------------------------
class _RandomModuleGuard:
    """Stands in for ``np.random`` inside the runner module.

    ``default_rng`` is allowed (the DNA sampler) and counted. Every other
    attribute raises on access, which catches ``np.random.normal``,
    ``np.random.standard_normal``, ``np.random.RandomState`` and friends.
    """

    def __init__(self, real_random) -> None:
        self._real = real_random
        self.attr_uses: list[str] = []
        self.default_rng_calls = 0

    def __getattr__(self, name: str):
        self.attr_uses.append(name)
        if name != "default_rng":
            raise AssertionError(
                f"np.random.{name} was called inside the campaign path. "
                "Return series must come from TrialResult.per_period_returns, "
                "never from an rng."
            )

        def _default_rng(*args, **kwargs):
            self.default_rng_calls += 1
            return self._real.default_rng(*args, **kwargs)

        return _default_rng


class _NumpyProxy:
    """numpy with a guarded ``.random``; everything else forwards."""

    def __init__(self, real_np, random_guard: _RandomModuleGuard) -> None:
        self._np = real_np
        self.random = random_guard

    def __getattr__(self, name: str):
        return getattr(self._np, name)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_campaign_makes_no_random_draws_when_real_bars_loaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With real bars loaded, the runner's only rng use is the DNA sampler.

    control_fn receives the scored TrialResults, so the fixed control is
    sliced to their count.
    """
    series = _real_bar_series()
    control = [0.1 * i for i in range(POPULATION)]  # deterministic control

    guard = _RandomModuleGuard(np.random)
    monkeypatch.setattr(runner_module, "np", _NumpyProxy(np, guard))

    result = run_campaign(
        campaign_id="no-synth",
        scope=_scope(),
        registry=TrialRegistry(tmp_path / "trials.sqlite"),
        backtest_fn=_backtest_from_series(series),
        control_fn=lambda trials: control[:len(trials)],
        real_scores_factory=lambda trials: [
            float(t.sharpe)
            for t in trials
            if t.sharpe is not None and np.isfinite(t.sharpe)
        ],
        provenance=_provenance(),
        seed=7,
        n_obs=N_OBS,
        bars_loaded=N_OBS + 1,
        audit_bars_expected=N_OBS + 1,
    )

    # default_rng is constructed exactly once: the DNA sampler. A second
    # construction is the fabricated-return-series defect coming back.
    assert guard.default_rng_calls == 1, (
        f"expected 1 default_rng call (the DNA sampler), got "
        f"{guard.default_rng_calls}; uses={guard.attr_uses}"
    )
    assert guard.attr_uses == ["default_rng"], guard.attr_uses
    assert result.n_trials == POPULATION
    assert result.n_obs_real == N_OBS


def test_runner_source_has_no_fabricated_return_series() -> None:
    """Static guard: the runner module holds exactly one rng construction."""
    src = inspect.getsource(runner_module)
    hits = [
        line.strip()
        for line in src.splitlines()
        if "np.random" in line or "default_rng" in line
    ]
    assert hits == ["rng = np.random.default_rng(seed)"], hits
    # The specific fabrication that was here.
    assert ".normal(" not in src, "runner draws from a normal distribution"


def test_run_campaign_requires_n_obs(tmp_path: Path) -> None:
    """A missing n_obs must raise, not silently become 1500."""
    with pytest.raises(TypeError, match="n_obs"):
        run_campaign(
            campaign_id="missing-n-obs",
            scope=_scope(),
            registry=TrialRegistry(tmp_path / "trials.sqlite"),
            backtest_fn=_backtest_from_series(_real_bar_series()),
            control_fn=None,
            real_scores_factory=lambda trials: [],
            provenance=_provenance(),
            seed=7,
        )
    assert "n_obs: int = 1500" not in inspect.getsource(run_campaign)


def test_dsr_and_pbo_come_from_per_candidate_returns(tmp_path: Path) -> None:
    """DSR must differ candidate by candidate; PBO must be computable."""
    series = _real_bar_series()
    result = run_campaign(
        campaign_id="per-candidate",
        scope=_scope(),
        registry=TrialRegistry(tmp_path / "trials.sqlite"),
        backtest_fn=_backtest_from_series(series),
        control_fn=None,
        real_scores_factory=lambda trials: [
            float(t.sharpe)
            for t in trials
            if t.sharpe is not None and np.isfinite(t.sharpe)
        ],
        provenance=_provenance(),
        seed=7,
        n_obs=N_OBS,
        bars_loaded=N_OBS + 1,
        audit_bars_expected=N_OBS + 1,
    )
    # Defect #6 sentinel: one distinct DSR across the whole cell means DSR
    # is not candidate-specific.
    assert result.distinct_dsr_count > 1, (
        f"distinct_dsr_count={result.distinct_dsr_count}; DSR is not "
        "responding to per-candidate returns"
    )
    # PBO is built from the (S, T) matrix of per_period_returns.
    assert result.pbo is not None
    assert result.pbo_n_splits > 0


def test_pbo_matrix_is_observations_by_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """pbo_cscv must receive (n_obs, n_candidates), not its transpose.

    Both orientations pass pbo_cscv's own shape validation, so a swap is
    silent: it blocks over candidates and ranks over bars, and still
    returns a plausible-looking number.
    """
    captured: dict[str, object] = {}

    class _FakeResult:
        pbo = 0.25
        n_splits = 12870

    def _spy(perf_matrix, n_blocks: int = 16, higher_is_better: bool = True):
        captured["shape"] = np.asarray(perf_matrix).shape
        captured["n_blocks"] = n_blocks
        return _FakeResult()

    monkeypatch.setattr(runner_module, "pbo_cscv", _spy)

    series = _real_bar_series()
    result = run_campaign(
        campaign_id="pbo-orientation",
        scope=_scope(),
        registry=TrialRegistry(tmp_path / "trials.sqlite"),
        backtest_fn=_backtest_from_series(series),
        control_fn=None,
        real_scores_factory=lambda trials: [],
        provenance=_provenance(),
        seed=7,
        n_obs=N_OBS,
        bars_loaded=N_OBS + 1,
        audit_bars_expected=N_OBS + 1,
    )
    assert captured["shape"] == (N_OBS, POPULATION), (
        f"pbo_cscv got {captured['shape']}, expected "
        f"(n_obs, n_candidates) = {(N_OBS, POPULATION)}"
    )
    assert captured["n_blocks"] == 16
    assert result.pbo == 0.25
    assert result.pbo_n_splits == 12870


def test_pbo_needs_two_candidates_not_thirty_two(tmp_path: Path) -> None:
    """The candidate floor is 2 (CSCV ranks them); the n_blocks * 2 bound
    belongs to the TIME axis."""
    from engine.research.search.runner import _compute_pbo

    reg = TrialRegistry(tmp_path / "trials.sqlite")
    series = _real_bar_series(population=2)
    two = [
        TrialResult(
            dna=None,  # type: ignore[arg-type]
            sharpe=0.1,
            per_period_returns=s,
        )
        for s in series
    ]
    pbo, n_splits = _compute_pbo(
        two, symbol="USDCADm", timeframe="D1",
        registry=reg, campaign_id="two",
    )
    assert pbo is not None, "2 candidates x 300 obs must produce a PBO"
    assert n_splits > 0

    # One candidate cannot be ranked against anything.
    one, _ = _compute_pbo(
        two[:1], symbol="USDCADm", timeframe="D1",
        registry=reg, campaign_id="one",
    )
    assert one is None

    # Long enough candidate list, series too short for 16 blocks.
    short = [
        TrialResult(
            dna=None,  # type: ignore[arg-type]
            sharpe=0.1,
            per_period_returns=np.zeros(31),
        )
        for _ in range(40)
    ]
    short_pbo, _ = _compute_pbo(
        short, symbol="USDCADm", timeframe="D1",
        registry=reg, campaign_id="short",
    )
    assert short_pbo is None

# ---------------------------------------------------------------------------
# The DSR-provenance guard (replaced 2026-09-18: pairwise -> population)
# ---------------------------------------------------------------------------
def test_dsr_guard_fires_when_every_candidate_shares_one_series(
    tmp_path: Path,
) -> None:
    """The defect the guard exists for: one series, one DSR, whole population.

    Every candidate reports its own Sharpe but is handed the SAME return
    series, so compute_cell_dsr returns one identical value for all of them.
    """
    shared = _real_bar_series(population=1)[0]
    population = 200  # above MIN_DSR_POPULATION
    state = {"i": 0}

    def _bt(dna: DNA) -> TrialResult:
        state["i"] += 1
        return TrialResult(
            dna=dna,
            outcome=FailureTaxonomy.EVALUATED.value,
            # Distinct Sharpe per candidate, but one shared series: exactly
            # the shape of the original defect.
            sharpe=0.5 + state["i"] * 1e-3,
            n_trades=60,
            total_return=0.02,
            max_dd=0.05,
            per_period_returns=shared,
        )

    with pytest.raises(AssertionError, match="not candidate-specific"):
        run_campaign(
            campaign_id="shared-series",
            scope=CampaignScope(
                symbols=("USDCADm",),
                timeframes=("D1",),
                population_per_cell=population,
            ),
            registry=TrialRegistry(tmp_path / "trials.sqlite"),
            backtest_fn=_bt,
            control_fn=None,
            real_scores_factory=lambda trials: [],
            provenance=_provenance(),
            seed=7,
            n_obs=N_OBS,
            bars_loaded=N_OBS + 1,
            audit_bars_expected=N_OBS + 1,
        )


def _trial(sharpe: float, post_dsr: float) -> TrialResult:
    return TrialResult(
        dna=None,  # type: ignore[arg-type]
        sharpe=sharpe,
        n_trades=146,
        post_dsr=post_dsr,
    )


def test_dsr_guard_ignores_the_gbpjpy_tail_collision() -> None:
    """The 2026-09-18 GBPJPYm D1 abort must not happen again.

    Two candidates with Sharpes 0.576525 and 0.576179 shared DSR=1.677e-09
    under the old round(dsr, 12) key. That key is an absolute 1e-12 grid;
    genuine candidate spacing at DSR ~1e-9 is ~1e-11 or finer, so the check
    was firing below its own resolution. A healthy population containing
    that exact pair must pass.
    """
    results = [_trial(1.0 + i * 1e-3, 0.4 + i * 1e-4) for i in range(198)]
    results += [
        _trial(0.576525, 1.677e-09),
        _trial(0.576179, 1.677e-09),
    ]
    runner_module._assert_dsr_is_candidate_specific(results)  # must not raise


def test_dsr_guard_skips_populations_too_small_to_judge() -> None:
    """A ratio over a handful of candidates carries no signal.

    50 candidates all sharing one DSR is 0.02 distinct - above the 1% floor
    purely because the population is small. The guard declines to judge;
    distinct_dsr_count carries the raw number instead.
    """
    tiny = [_trial(1.0 + i * 1e-3, 0.25) for i in range(50)]
    runner_module._assert_dsr_is_candidate_specific(tiny)  # must not raise


def test_dsr_guard_ratio_boundary() -> None:
    """Healthy cells sit two orders of magnitude above the 1% floor."""
    # 1 distinct value in 1000 -> 0.001, fires.
    with pytest.raises(AssertionError, match="not candidate-specific"):
        runner_module._assert_dsr_is_candidate_specific(
            [_trial(1.0 + i * 1e-4, 0.25) for i in range(1000)]
        )
    # The measured healthy ratio (0.327 on USDCADm H4) is nowhere near it.
    healthy = [_trial(1.0 + i * 1e-4, 0.2 + (i % 327) * 1e-5) for i in range(1000)]
    runner_module._assert_dsr_is_candidate_specific(healthy)


# ---------------------------------------------------------------------------
# all vs eligible
# ---------------------------------------------------------------------------
def test_headline_statistics_are_reported_for_both_populations(
    tmp_path: Path,
) -> None:
    """dsr_within / best_sharpe / vs_random, over all AND over eligible.

    Cell 1 of the 2026-09-18 campaign had all three belonging to a 23-trade
    candidate the funnel bucketed LowTrades. The fixture reproduces that: the
    highest-Sharpe candidate trades too little to be eligible.

    control_fn receives the scored TrialResults, so the ramp control is
    sized from their count.
    """
    series = _real_bar_series()
    min_trades = 30
    state = {"i": 0}

    def _bt(dna: DNA) -> TrialResult:
        i = state["i"]
        state["i"] += 1
        arr = series[i % len(series)]
        # Candidate 0 is the best on Sharpe and the worst on trade count.
        n_trades = 23 if i == 0 else 60
        sharpe = 9.0 if i == 0 else _series_sharpe(arr)
        return TrialResult(
            dna=dna,
            outcome=FailureTaxonomy.EVALUATED.value,
            sharpe=sharpe,
            n_trades=n_trades,
            total_return=0.02,
            max_dd=0.05,
            per_period_returns=arr,
        )

    result = run_campaign(
        campaign_id="all-vs-eligible",
        scope=_scope(),
        registry=TrialRegistry(tmp_path / "trials.sqlite"),
        backtest_fn=_bt,
        control_fn=lambda trials: [0.01 * k for k in range(len(trials))],
        real_scores_factory=lambda trials: [
            float(t.sharpe)
            for t in trials
            if t.sharpe is not None and np.isfinite(t.sharpe)
        ],
        provenance=_provenance(),
        seed=7,
        min_trades=min_trades,
        n_obs=N_OBS,
        bars_loaded=N_OBS + 1,
        audit_bars_expected=N_OBS + 1,
    )

    assert result.n_eligible == POPULATION - 1
    # The 23-trade candidate owns the all-population winner and is absent
    # from the eligible one.
    assert result.best_sharpe == 9.0
    assert result.best_sharpe_eligible is not None
    assert result.best_sharpe_eligible < 9.0
    # Both DSR views are present and independently computed.
    assert result.dsr_within is not None
    assert result.dsr_within_eligible is not None
    assert result.dsr_within_eligible_status is not None
    # Both vs_random views are present.
    assert result.vs_random is not None
    assert result.vs_random_eligible is not None
    # The luck bar is reported on both, so neither DSR is uninterpretable.
    assert result.dsr_sr_star_per_bar is not None
    assert result.dsr_eligible_sr_star_per_bar is not None


def test_exit_verdict_reads_the_eligible_population(tmp_path: Path) -> None:
    """exit_verdict() must not be decided by candidates the funnel rejected."""
    from engine.research.search.runner import CampaignResult
    from engine.research.search.funnel import summarise

    funnel = summarise([FailureTaxonomy.VAULTED.value])

    class _VS:
        p_max = 0.001
        p_mean = 0.001
        p_single = 0.001
        effect_size = 1.0
        verdict = "STRUCTURE FOUND - winner and population both beat random"

    # All-population stats scream STRUCTURE FOUND; eligible ones do not.
    result = CampaignResult(
        campaign_id="c",
        n_trials=1,
        funnel=funnel,
        best_dna=None,
        best_sharpe=9.0,
        vs_random=_VS(),  # type: ignore[arg-type]
        provenance=_provenance(),
        as_of="now",
        dsr_within=0.99,
        vs_random_eligible=None,
        dsr_within_eligible=None,
    )
    assert result.exit_verdict().startswith("INCONCLUSIVE")


def test_dsr_within_eligible_is_reconstructable_from_the_registry(
    tmp_path: Path,
) -> None:
    """The registry must answer the eligible question without a re-run.

    This is the whole point of storing dsr / dsr_status / sr_star / n_obs.
    Before them, dsr_within_eligible for a finished campaign could only be
    recovered by re-running every backtest, because post_dsr lived nowhere
    but in memory. If this reconstruction does not match what run_campaign
    reported, the columns are not doing their job.
    """
    import sqlite3

    from engine.validation.op_types import OpType

    series = _real_bar_series()
    min_trades = 30
    db = tmp_path / "trials.sqlite"
    state = {"i": 0}

    def _bt(dna: DNA) -> TrialResult:
        i = state["i"]
        state["i"] += 1
        # Candidate 0 gets the strongest series AND too few trades, so the
        # all-population and eligible answers are genuinely different.
        arr = series[-1] if i == 0 else series[i % (len(series) - 1)]
        return TrialResult(
            dna=dna,
            outcome=FailureTaxonomy.EVALUATED.value,
            sharpe=_series_sharpe(arr),
            n_trades=23 if i == 0 else 60,
            total_return=0.02,
            max_dd=0.05,
            per_period_returns=arr,
        )

    result = run_campaign(
        campaign_id="reconstruct",
        scope=_scope(),
        registry=TrialRegistry(db),
        backtest_fn=_bt,
        control_fn=None,
        real_scores_factory=lambda trials: [],
        provenance=_provenance(),
        seed=7,
        min_trades=min_trades,
        n_obs=N_OBS,
        bars_loaded=N_OBS + 1,
        audit_bars_expected=N_OBS + 1,
    )
    assert result.dsr_within is not None
    assert result.dsr_within_eligible is not None
    # The fixture must actually separate the two, or this proves nothing.
    assert result.dsr_within != result.dsr_within_eligible

    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rebuilt_eligible = conn.execute(
            "SELECT MAX(dsr) FROM trials WHERE campaign_id=? AND op_type=? "
            "AND n_trades >= ? AND dsr IS NOT NULL",
            ("reconstruct", OpType.SEARCH_TRIAL.value, min_trades),
        ).fetchone()[0]
        rebuilt_all = conn.execute(
            "SELECT MAX(dsr) FROM trials WHERE campaign_id=? AND op_type=? "
            "AND dsr IS NOT NULL",
            ("reconstruct", OpType.SEARCH_TRIAL.value),
        ).fetchone()[0]
        n_scored, n_status, n_sr_star, n_obs_rows = conn.execute(
            "SELECT COUNT(dsr), COUNT(dsr_status), COUNT(sr_star), COUNT(n_obs) "
            "FROM trials WHERE campaign_id=? AND op_type=?",
            ("reconstruct", OpType.SEARCH_TRIAL.value),
        ).fetchone()
        star, obs, status = conn.execute(
            "SELECT sr_star, n_obs, dsr_status FROM trials "
            "WHERE campaign_id=? AND op_type=? AND dsr IS NOT NULL "
            "ORDER BY dsr DESC LIMIT 1",
            ("reconstruct", OpType.SEARCH_TRIAL.value),
        ).fetchone()
    finally:
        conn.close()

    assert rebuilt_eligible == result.dsr_within_eligible
    assert rebuilt_all == result.dsr_within
    # Every candidate carries all four columns, not just the winner.
    assert n_scored == POPULATION
    assert n_status == POPULATION
    assert n_sr_star == POPULATION
    assert n_obs_rows == POPULATION
    # The luck bar is stored, which is what makes a stored DSR readable.
    assert star == result.dsr_sr_star_per_bar
    assert obs == N_OBS
    assert status == "READY"


def test_registry_dsr_columns_migrate_onto_an_existing_database(
    tmp_path: Path,
) -> None:
    """ALTER TABLE ADD COLUMN; pre-existing rows read as 'not recorded'."""
    import sqlite3

    db = tmp_path / "old.sqlite"
    # A registry written before the columns existed.
    conn = sqlite3.connect(db)
    conn.executescript(
        "CREATE TABLE campaigns (campaign_id TEXT PRIMARY KEY, symbol TEXT, "
        "timeframe TEXT, dataset_hash TEXT, git_sha TEXT, config_hash TEXT, "
        "seed INTEGER, created_at TEXT, notes TEXT);"
        "CREATE TABLE trials (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "campaign_id TEXT, dna_hash TEXT, symbol TEXT, timeframe TEXT, "
        "op_type TEXT NOT NULL DEFAULT 'UNKNOWN', sharpe REAL, n_trades INTEGER, "
        "total_return REAL, max_dd REAL, outcome TEXT NOT NULL, "
        "dataset_hash TEXT, source TEXT, as_of TEXT, created_at TEXT);"
        "INSERT INTO trials (campaign_id, dna_hash, outcome) "
        "VALUES ('old', 'abc', 'EVALUATED');"
    )
    conn.commit()
    conn.close()

    TrialRegistry(db)  # opening it runs the migration

    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(trials)")}
        row = conn.execute(
            "SELECT dsr, dsr_status, sr_star, n_obs FROM trials WHERE dna_hash='abc'"
        ).fetchone()
    finally:
        conn.close()

    assert {"dsr", "dsr_status", "sr_star", "n_obs"} <= cols
    assert row == (None, None, None, None)  # not recorded, still valid


def test_campaign_computes_dsr_across_instead_of_copying(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """dsr_across must come from compute_cross_cell_dsr, not an assignment.

    run_campaign used to do `dsr_across = dsr_all.dsr_within` with the real
    call sitting dead behind `if False`. For one cell the two agree, so no
    value check can tell them apart - only watching whether the function is
    called can.
    """
    from engine.validation.cell_dsr import CellDSRResult

    real = runner_module.compute_cross_cell_dsr
    calls: list[tuple] = []

    def _spy(cells):
        materialised = tuple(cells)
        calls.append(materialised)
        return real(materialised)

    monkeypatch.setattr(runner_module, "compute_cross_cell_dsr", _spy)

    result = run_campaign(
        campaign_id="across",
        scope=_scope(),
        registry=TrialRegistry(tmp_path / "trials.sqlite"),
        backtest_fn=_backtest_from_series(_real_bar_series()),
        control_fn=None,
        real_scores_factory=lambda trials: [],
        provenance=_provenance(),
        seed=7,
        n_obs=N_OBS,
        bars_loaded=N_OBS + 1,
        audit_bars_expected=N_OBS + 1,
    )

    assert len(calls) == 1, "compute_cross_cell_dsr was never called"
    (cells,) = calls
    assert len(cells) == 1
    assert isinstance(cells[0], CellDSRResult)
    # The cell handed over must describe THIS cell, not a placeholder.
    assert cells[0].symbol == "USDCADm"
    assert cells[0].dsr == result.dsr_within
    assert cells[0].winner_sharpe == result.dsr_winner_sr_annualised

    # One cell means there is no second selection to apply. It must be
    # LABELLED degenerate, never reported as a number equal to dsr_within --
    # a copy and a correction are indistinguishable at n_cells = 1, so the
    # label is the only thing that tells them apart.
    assert result.dsr_across_n_cells == 1
    assert result.dsr_across is None
    assert result.dsr_across_status == "DEGENERATE"
    assert result.dsr_across != result.dsr_within
