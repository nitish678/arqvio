"""The shared-account replay: caps refuse whole trades, conflicts are recorded, and on the REAL pinned cells the
replay is verified (DR §2): the tap is transparent, every member reconstructs, conservation is exact every bar, the
member-sum reconciles < 1e-8, and two builds are byte-identical."""
from __future__ import annotations

from pathlib import Path

import pytest

from engine.research.portfolio.heat import HEAT_CAP
from engine.research.portfolio.sim import RECON_TOL, replay
from tests.research.portfolio.fakes import T, fake_run

TIMES = [3600 * k for k in range(1, 5)]


def test_an_entry_that_would_breach_the_heat_cap_is_refused_and_itemized() -> None:
    # A risks 0.05 of 1000 (heat 0.05); B's entry would add 0.06 -> 0.11 > HEAT_CAP.
    a = fake_run("A", "AAA", TIMES, [T("a1", 0, 1.0, sl=0.95, volume=1000.0)], [2.0, 3.0, 4.0, 5.0])
    b = fake_run("B", "BBB", TIMES, [T("b1", 1, 0.5, exit_bar=3, gross=7.0, sl=0.94, volume=1000.0)],
                 [0.0, 1.0, 2.0, 0.0], [(2, "b1", 0.25)])
    rep = replay([a, b], e0=1000.0)
    assert list(rep.refused) == ["b1"] and rep.refused["b1"]["cap"] == "HEAT"
    assert rep.refused["b1"]["values"]["heat"] > HEAT_CAP
    assert rep.refused["b1"]["contribution_total"] == -0.5 - 0.25 + 7.0
    assert [b["equity"] for b in rep.bars] == [1000.0 - 1.0 + 2.0, 1002.0, 1003.0, 1004.0]
    assert rep.bars[0]["heat_per_symbol"] == {"AAA": pytest.approx(50.0 / 1001.0, rel=1e-12)}
    assert rep.bars[0]["heat"] == rep.bars[0]["heat_per_symbol"]["AAA"]  # one symbol: the rollup is the total
    assert rep.recon_max_abs < RECON_TOL and rep.conservation_bars == 4
    assert [e["kind"] for e in rep.events if e["kind"] == "REFUSED"] == ["REFUSED"]


def test_margin_and_floor_caps_refuse() -> None:
    big = fake_run("M", "XAU", TIMES, [T("m1", 0, 0.0, entry=2000.0, sl=1999.99, volume=1.0)], [0.0] * 4,
                   contract_size=100.0, usd_base=False)  # margin 1 * 100 * 2000 / 100 = 2000 > 0.5 * 1000
    rep = replay([big], e0=1000.0)
    assert rep.refused["m1"]["cap"] == "MARGIN"
    loser = fake_run("L", "AAA", TIMES, [T("l1", 0, 0.0, exit_bar=1, gross=-850.0, sl=0.999),
                                         T("l2", 2, 0.0, sl=0.999)], [0.0] * 4)
    rep = replay([loser], e0=1000.0)
    assert list(rep.refused) == ["l2"] and rep.refused["l2"]["cap"] == "FLOOR"


def test_a_position_without_a_stop_is_unbounded_heat_and_refused() -> None:
    rep = replay([fake_run("U", "AAA", TIMES, [T("u1", 0, 0.0, sl=None)], [0.0] * 4)], e0=1000.0)
    assert rep.refused["u1"]["cap"] == "HEAT" and rep.refused["u1"]["values"]["unbounded"] is True


def test_opposite_positions_on_one_symbol_coexist_and_are_recorded() -> None:
    a = fake_run("A", "AAA", TIMES, [T("a1", 0, 0.0, direction="long")], [0.0] * 4)
    b = fake_run("B", "AAA", TIMES, [T("b1", 1, 0.0, direction="short")], [0.0] * 4)
    rep = replay([a, b], e0=1000.0)
    conflicts = [e for e in rep.events if e["kind"] == "CONFLICT"]
    assert [e["t"] for e in conflicts] == TIMES[1:] and conflicts[0]["members"] == ["A", "B"] and not rep.refused


@pytest.fixture(scope="module")
def real_set(tmp_path_factory: pytest.TempPathFactory, real_artifacts_dir: Path):
    import shutil

    from engine.config import settings as settings_module
    from tests.research.portfolio.conftest import PINNED

    need = [real_artifacts_dir / "datasets" / f"{s}_H1_{d}.{ext}" for s, d in PINNED for ext in ("json", "npz")]
    need += [real_artifacts_dir / sub / f"{s}.json" for sub in ("spread", "specs") for s, _ in PINNED]
    if any(not p.is_file() for p in need):
        pytest.skip("pinned cells not on this machine")
    home = tmp_path_factory.mktemp("s7_home") / "ARQVIO"
    for p in need:
        target = home / "artifacts" / p.relative_to(real_artifacts_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("ARQVIO_HOME", str(home))
        settings_module.get_settings.cache_clear()
        from engine.research.portfolio.members import member_set

        yield member_set()
    settings_module.get_settings.cache_clear()


def test_the_tap_does_not_change_the_engine(real_set) -> None:
    import numpy as np

    from engine.research.portfolio.members import member_share, sim_cells
    from engine.research.portfolio.sim import run_member

    cells = sim_cells()
    for m, r in zip(real_set.members, real_set.runs):
        plain = run_member(m.mid, m.dna, cells[m.symbol], initial_equity=member_share(), tap=False)
        assert np.array_equal(plain.result.equity_curve, r.result.equity_curve), m.mid
        assert len(r.marks) == len(r.times)


def test_every_real_member_reconstructs_from_its_events(real_set) -> None:
    from engine.research.portfolio.sim import member_reconstruction_max_abs

    assert len(real_set.members) == 8 and {m.family for m in real_set.members} == {
        "RSI_REVERSION", "STOCH_CROSS", "MACD_CROSS", "DONCHIAN_BREAKOUT"}
    for r in real_set.runs:
        assert member_reconstruction_max_abs(r) < RECON_TOL, r.mid


def test_the_real_replay_is_verified_and_deterministic(real_set) -> None:
    from engine.research.portfolio.members import E0
    from engine.research.portfolio.run import build, canonical, run_hash

    rep = replay(list(real_set.runs), e0=E0)
    art = build(real_set, rep)
    v = art["verification"]
    print(f"\nreal replay: conservation {v['conservation_bars']}/{v['axis_bars']} exact={v['conservation_exact']}; "
          f"recon max|d| {v['recon_max_abs']:.3e}; member recon max {v['member_reconstruction_max_abs']:.3e}; "
          f"refused {len(art['refused'])}; run_hash {art['run_hash']}")
    assert v["conservation_exact"] and v["conservation_bars"] == len(rep.axis) and v["recon_ok"]
    again = build(real_set, replay(list(real_set.runs), e0=E0))
    assert canonical(art) == canonical(again) and run_hash(real_set) == art["run_hash"]
