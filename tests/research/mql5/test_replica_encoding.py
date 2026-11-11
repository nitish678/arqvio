"""The replica's order-preserving price encoding (S6 parity run 2: one-ulp broker noise vs the tester's grid)."""
from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from engine.research.mql5 import replica
from engine.research.mql5.replica import ENC_OFFSET_RP, encode

RP = 1e-8  # one replica point at 5 + 3 digits


def _q(p: float) -> float:
    return float(encode(p, 5))


def test_the_parity_run_2_price_keeps_its_side_of_the_level() -> None:
    level = round(1.31892, 5)
    raw_high = 1.3189199999999999          # pinned USDCADm bar 832 high, one ulp under the grid
    assert raw_high < level and not raw_high >= level  # the oracle: TP NOT touched
    assert encode(raw_high, 5) == "1.31891998" and _q(raw_high) < level   # the replica keeps it under
    assert encode(level, 5) == "1.31892000"


def test_every_stop_type_sees_the_oracle_relation_through_the_tester_ask() -> None:
    """long SL bid<=L, long TP bid>=L, short SL ask>=L, short TP (at L+1rp) ask<=L+1rp; ask = bid + 1rp."""
    rng = np.random.default_rng(7)
    for _ in range(4000):
        level = round(float(rng.uniform(1.2, 1.5)), 5)
        for raw in (np.nextafter(level, -np.inf), level, np.nextafter(level, np.inf),
                    level - 1e-5, level + 1e-5):
            bid = _q(float(raw))
            ask = float(Decimal(encode(float(raw), 5)) + Decimal("0.00000001"))
            assert (bid <= level) == (raw <= level)                       # long SL
            assert (bid >= level) == (raw >= level)                       # long TP
            assert (ask >= level) == (raw >= level)                       # short SL
            assert (ask <= float(Decimal(repr(level)) + Decimal("0.00000001"))) == (raw <= level)  # short TP


def test_a_fill_still_rounds_to_the_oracle_quote() -> None:
    for raw in (1.3189199999999999, 1.31892, 1.3189200000000001, 1.40001):
        assert round(_q(raw), 5) == round(raw, 5) and round(_q(raw) + RP, 5) == round(raw, 5)


def test_on_grid_prices_and_metals_are_unchanged() -> None:
    assert encode(1277.989, 3) == "1277.989000" and float(encode(1277.989, 3)) == 1277.989
    assert ENC_OFFSET_RP == 2


class _Cell:
    def __init__(self, names: tuple[str, str]) -> None:
        self.replica_names = names


def test_the_purge_removes_only_this_cells_tester_copies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """DEV-12: the tester served the previous build's history; the purge drops exactly our two symbols."""
    tester, data = tmp_path / "tester", tmp_path / "data"
    monkeypatch.setattr(replica, "tester_data_dir", lambda: tester)
    monkeypatch.setattr(replica, "data_dir", lambda: data)
    keep = []
    for sym in ("AS6_X_abc123_full_L", "AS6_X_abc123_full_H", "AS6_X_abc123_last800_L", "USDCADm"):
        d = tester / "bases" / "Srv" / "history" / sym
        d.mkdir(parents=True)
        (d / "2026.hcs").write_bytes(b"x")
        f = data / "Tester" / "cache" / f"ea.{sym}.H1.20140114_20260902.1.ABC.tst"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"x")
        if "full" not in sym:
            keep += [d, f]
    removed = replica.purge_tester_copies(_Cell(("AS6_X_abc123_full_L", "AS6_X_abc123_full_H")))
    assert len(removed) == 4 and all(not p.exists() for p in removed) and all(p.exists() for p in keep)
    with pytest.raises(replica.ReplicaError, match="non-replica"):
        replica.purge_tester_copies(_Cell(("USDCADm", "AS6_X_abc123_full_H")))
    assert (tester / "bases" / "Srv" / "history" / "USDCADm").is_dir()
