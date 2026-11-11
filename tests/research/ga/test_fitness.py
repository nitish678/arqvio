"""GA fitness: gate order, dead-anywhere, every component recorded, DSR = the runner's."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from engine.research.ga import fitness as fit
from engine.research.ga.config import CAMPAIGN_GATES, Gates
from engine.research.search.runner import TrialResult, _populate_post_dsr
from engine.research.search.sampler import random_dna
from engine.validation.cell_dsr import CellDSRStatus
from engine.validation.trial_registry import TrialRegistry

GATES = Gates(min_trades=30, sharpe_min=0.0, dsr_min=0.5, sham_margin_min=0.0)


def _result(seed: int = 0, *, sharpe: float | None = 2.0, n_trades: int = 50, returns=None) -> TrialResult:
    return TrialResult(dna=random_dna(np.random.default_rng(seed)), sharpe=sharpe, n_trades=n_trades,
                       per_period_returns=returns)


def _never(_: TrialResult):
    raise AssertionError("a killed individual must not be scored further")


def _dsr(value: float | None) -> fit.DSRReading:
    return fit.DSRReading(value, "READY" if value is not None else "UNRELIABLE", 0.01, 0.005, 100)


# --- gate order: dead anywhere -> 0, later gates never evaluated -------------

def test_a_crash_is_killed_first_whatever_else_it_carries() -> None:
    rec = fit.score(_result(sharpe=5.0, n_trades=500), crashed=True, gates=GATES, dsr_of=_never, sham_of=_never)
    assert (rec.fitness, rec.killed_by) == (0.0, fit.KILLED_CRASH)


def test_too_few_trades_is_killed_before_the_sharpe_gate() -> None:
    rec = fit.score(_result(sharpe=5.0, n_trades=29), crashed=False, gates=GATES, dsr_of=_never, sham_of=_never)
    assert (rec.fitness, rec.killed_by, rec.n_trades, rec.sharpe) == (0.0, fit.KILLED_MIN_TRADES, 29, 5.0)


def test_failing_several_gates_names_the_earliest() -> None:
    rec = fit.score(_result(sharpe=-2.0, n_trades=3), crashed=False, gates=GATES, dsr_of=_never, sham_of=_never)
    assert rec.killed_by == fit.KILLED_MIN_TRADES
    rec = fit.score(_result(sharpe=-2.0, n_trades=3), crashed=True, gates=GATES, dsr_of=_never, sham_of=_never)
    assert rec.killed_by == fit.KILLED_CRASH


@pytest.mark.parametrize("sharpe", [0.0, -1.0, None, float("nan")])
def test_sharpe_at_or_below_the_floor_is_killed(sharpe) -> None:
    rec = fit.score(_result(sharpe=sharpe), crashed=False, gates=GATES, dsr_of=_never, sham_of=_never)
    assert (rec.fitness, rec.killed_by) == (0.0, fit.KILLED_SHARPE_MIN)
    assert rec.dsr is None and rec.sham_margin is None  # never computed


def test_exactly_min_trades_passes_that_gate() -> None:
    rec = fit.score(_result(n_trades=30), crashed=False, gates=GATES, dsr_of=lambda r: _dsr(None), sham_of=lambda r: 9.0)
    assert rec.alive


# --- the score itself ---------------------------------------------------------

def test_both_bonuses_add_exactly_as_specified() -> None:
    rec = fit.score(_result(sharpe=2.0), crashed=False, gates=GATES,
                    dsr_of=lambda r: _dsr(0.6), sham_of=lambda r: 1.2)
    assert rec.base == 2.0 and rec.dsr_passed and rec.sham_passed
    assert rec.dsr_bonus == 0.25 * 2.0
    assert rec.sham_margin == pytest.approx(0.8) and rec.sham_bonus == pytest.approx(0.25 * 0.8)
    assert rec.fitness == pytest.approx(2.0 + 0.5 + 0.2)


def test_a_failed_dsr_or_sham_gate_drops_only_its_bonus_and_is_recorded() -> None:
    rec = fit.score(_result(sharpe=2.0), crashed=False, gates=GATES,
                    dsr_of=lambda r: _dsr(0.4), sham_of=lambda r: 2.5)
    assert rec.alive and rec.fitness == 2.0
    assert rec.dsr_passed is False and rec.dsr.dsr == 0.4 and rec.dsr_bonus == 0.0
    assert rec.sham_passed is False and rec.sham_margin == pytest.approx(-0.5) and rec.sham_bonus == 0.0


def test_an_unreliable_dsr_earns_no_bonus() -> None:
    rec = fit.score(_result(sharpe=1.0), crashed=False, gates=GATES,
                    dsr_of=lambda r: _dsr(None), sham_of=lambda r: 0.0)
    assert rec.dsr_passed is False and rec.dsr.status == "UNRELIABLE"
    assert rec.fitness == pytest.approx(1.0 + 0.25 * 1.0)


def test_the_margin_gate_is_inclusive() -> None:
    gates = Gates(min_trades=30, sharpe_min=0.0, dsr_min=0.5, sham_margin_min=0.5)
    rec = fit.score(_result(sharpe=2.0), crashed=False, gates=gates,
                    dsr_of=lambda r: _dsr(None), sham_of=lambda r: 1.5)
    assert rec.sham_passed and rec.sham_bonus == pytest.approx(0.125)


def test_every_component_is_recorded_as_plain_json() -> None:
    rec = fit.score(_result(), crashed=False, gates=CAMPAIGN_GATES,
                    dsr_of=lambda r: _dsr(0.9), sham_of=lambda r: 0.5)
    d = rec.to_dict()
    assert json.loads(json.dumps(d)) == d
    assert set(d) == {"fitness", "killed_by", "n_trades", "sharpe", "base", "dsr", "dsr_passed",
                      "dsr_bonus", "sham_sharpe", "sham_margin", "sham_passed", "sham_bonus"}
    assert set(d["dsr"]) == {"dsr", "status", "sr_hat_per_bar", "sr_star_per_bar", "n_trials"}


# --- DSR is the runner's, candidate by candidate -----------------------------

def test_dsr_trials_is_run_campaigns_n_and_is_order_free() -> None:
    assert fit.dsr_trials([1.0, None, 2.0, float("nan")], 10) == (2, [1.0, 2.0])
    assert fit.dsr_trials([3.0, 1.0, 2.0], 2) == (2, [1.0, 2.0, 3.0])
    assert fit.dsr_trials([], 0) == (1, [])


def test_candidate_dsr_reproduces_populate_post_dsr(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    results = []
    for i in range(12):
        r = rng.normal(0.0002 * (i - 4), 0.01, size=400)
        sharpe = float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(6048))
        results.append(_result(i, sharpe=sharpe, returns=r))
    reg = TrialRegistry(tmp_path / "t.sqlite")
    reg.open_campaign("c", "USDCADm", "H1", "d", "g", "h", 0)
    for r in results:
        reg.record("c", dna_hash=r.dna.dna_hash, outcome="EVALUATED", sharpe=r.sharpe, n_trades=50,
                   symbol="USDCADm", timeframe="H1", source="MT5_LIVE")
    n_used = _populate_post_dsr(results, symbol="USDCADm", timeframe="H1", registry=reg,
                                campaign_id="c", n_effective_trials=12)
    n, sharpes = fit.dsr_trials([r.sharpe for r in results], 12)
    assert n == n_used
    for r in results:
        got = fit.candidate_dsr(r, n_trials=n, trial_sharpes=sharpes, symbol="USDCADm", timeframe="H1")
        assert got.dsr == r.post_dsr  # bit-for-bit: same call, sorted sharpes, same variance
        assert (got.sr_hat_per_bar, got.sr_star_per_bar) == (r.post_dsr_sr_per_bar, r.post_dsr_sr_star)


def test_a_dsr_that_is_not_ready_is_no_dsr(monkeypatch: pytest.MonkeyPatch) -> None:
    """As in the runner: only READY carries a usable number, whatever else is set."""
    from engine.validation.cell_dsr import CellDSRResult

    def rejected(*a, **k):
        return CellDSRResult(symbol="USDCADm", timeframe="H1", status=CellDSRStatus.REJECTED, dsr=0.9,
                             n_obs=400, n_trials=5, var_sr_trials=0.1, winner_sharpe=1.0, min_obs_threshold=100)

    monkeypatch.setattr(fit, "compute_cell_dsr", rejected)
    got = fit.candidate_dsr(_result(sharpe=1.0, returns=np.full(400, 0.001)), n_trials=5,
                            trial_sharpes=[1.0], symbol="USDCADm", timeframe="H1")
    assert got.dsr is None and got.status == CellDSRStatus.REJECTED.value


def test_too_short_a_series_is_unreliable_not_scored() -> None:
    r = _result(sharpe=1.0, returns=np.full(10, 0.001))
    got = fit.candidate_dsr(r, n_trials=5, trial_sharpes=[0.5, 1.0], symbol="USDCADm", timeframe="H1")
    assert got.dsr is None and got.status == CellDSRStatus.UNRELIABLE.value
