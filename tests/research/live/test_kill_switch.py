"""The kill switch flattens every own-magic position and order and never touches another magic."""
from __future__ import annotations

import copy
from pathlib import Path

import pytest

from engine.research.live import guard
from engine.research.live.kill_switch import KillSwitch, KillSwitchIncomplete
from engine.research.mql5.magic import live_magic
from tests.research.live.fake_mt5 import FakeMT5


def _planted() -> tuple[FakeMT5, dict]:
    mt5 = FakeMT5()
    t = {
        "own_a": mt5.plant_position(symbol="USDCADm", side="long", volume=0.01, magic=live_magic("plant_own_a")),
        "own_b": mt5.plant_position(symbol="USDCADm", side="short", volume=0.01, magic=live_magic("plant_own_b")),
        "own_pending": mt5.plant_order(symbol="USDCADm", volume=0.01, price=1.36500,
                                       magic=live_magic("plant_own_pending")),
        "foreign": mt5.plant_position(symbol="USDCADm", side="long", volume=0.01, magic=424242),
        "foreign_arqvio_other_range": mt5.plant_position(symbol="USDCADm", side="long", volume=0.01,
                                                        magic=610_100_000),
        "foreign_pending": mt5.plant_order(symbol="USDCADm", volume=0.01, price=1.36000, magic=0),
    }
    return mt5, t


def test_flatten_closes_own_and_leaves_foreign_exactly_as_found(tmp_path: Path) -> None:
    mt5, t = _planted()
    foreign_before = {k: copy.deepcopy(vars(mt5.positions[t[k]])) for k in ("foreign", "foreign_arqvio_other_range")}
    pending_before = copy.deepcopy(vars(mt5.orders[t["foreign_pending"]]))
    lines: list[str] = []
    rep = KillSwitch(mt5, flag_dir=tmp_path, log=lines.append).flatten()
    assert sorted(r["ticket"] for r in rep.closed) == sorted([t["own_a"], t["own_b"]])
    assert [r["ticket"] for r in rep.removed] == [t["own_pending"]]
    assert all(r["retcode"] == mt5.TRADE_RETCODE_DONE for r in rep.closed + rep.removed)
    assert sorted(rep.foreign_positions) == sorted([t["foreign"], t["foreign_arqvio_other_range"]])
    assert rep.foreign_orders == [t["foreign_pending"]] and rep.own_left == []
    for k, before in foreign_before.items():
        assert vars(mt5.positions[t[k]]) == before
    assert vars(mt5.orders[t["foreign_pending"]]) == pending_before
    assert all(int(r["magic"]) in {live_magic("plant_own_a"), live_magic("plant_own_b"),
                                   live_magic("plant_own_pending")} for r in mt5.sent)
    assert any("skip FOREIGN position" in s for s in lines) and lines[-1].startswith("KILL FLATTEN end own_left=[]")


def test_an_own_position_that_will_not_close_is_named(tmp_path: Path) -> None:
    mt5, t = _planted()
    mt5.refuse_close.add(t["own_b"])
    with pytest.raises(KillSwitchIncomplete, match=str(t["own_b"])):
        KillSwitch(mt5, flag_dir=tmp_path, log=lambda s: None).flatten()
    assert t["own_a"] not in mt5.positions


def test_a_real_account_is_flattened_even_unarmed(tmp_path: Path, isolated_arqvio_home: Path) -> None:
    """Taking own exposure off never needs arming: a disarm or a kill must always work."""
    mt5, t = _planted()
    mt5.trade_mode = mt5.ACCOUNT_TRADE_MODE_REAL
    KillSwitch(mt5, flag_dir=tmp_path, log=lambda s: None).flatten()
    left = {p.ticket for p in mt5.positions_get()}
    assert t["own_a"] not in left and t["own_b"] not in left
    assert t["foreign"] in left and t["foreign_arqvio_other_range"] in left


def test_a_contest_account_is_refused_before_any_send(tmp_path: Path) -> None:
    mt5, _ = _planted()
    mt5.trade_mode = mt5.ACCOUNT_TRADE_MODE_CONTEST
    with pytest.raises(guard.UnknownAccountMode):
        KillSwitch(mt5, flag_dir=tmp_path, log=lambda s: None).flatten()
    assert mt5.sent == []


def test_the_flag_file_triggers_and_clears(tmp_path: Path) -> None:
    ks = KillSwitch(FakeMT5(), flag_dir=tmp_path, log=lambda s: None)
    assert not ks.triggered()
    p = ks.trigger("test")
    assert p == tmp_path / "KILL" and ks.triggered() and "test" in p.read_text(encoding="ascii")
    ks.clear()
    assert not ks.triggered()


def test_an_empty_account_flattens_to_nothing(tmp_path: Path) -> None:
    mt5 = FakeMT5()
    rep = KillSwitch(mt5, flag_dir=tmp_path, log=lambda s: None).flatten()
    assert (rep.closed, rep.removed, rep.own_left, mt5.sent) == ([], [], [], [])
