"""The magic registry: disjoint ranges, deterministic and injective allocation, foreign stays foreign."""
from __future__ import annotations

import hashlib
import itertools

import pytest

from engine.research.mql5 import magic as mg


def _hashes(n: int) -> list[str]:
    return [hashlib.sha256(f"dna-{i}".encode()).hexdigest()[:16] for i in range(n)]


def test_the_committed_table_loads_and_its_ranges_are_disjoint() -> None:
    t = mg.load_table()
    assert (t.space_min, t.space_max) == (610_000_000, 610_999_999)
    for a, b in itertools.combinations(t.ranges, 2):
        assert a.last < b.base or b.last < a.base, (a, b)
    assert {r.name for r in t.ranges} == {
        "s6-parity-fixtures", "ga-usdcadm_h1-s7-cbcb2ae061e70bef", "arqvio-deploy", "s6-live-demo"}


def test_every_known_non_arqvio_magic_lies_outside_the_space() -> None:
    t = mg.load_table()
    assert set(t.outside) == {0, 77789, 999000001, 424242}
    for m in t.outside:
        assert mg.owner_of(m, table=t) is None


def test_allocation_is_deterministic_and_injective_over_a_full_range() -> None:
    source = _hashes(2000)
    first = [mg.allocate("s6-parity-fixtures", h, source) for h in source]
    again = [mg.allocate("s6-parity-fixtures", h, source) for h in source]
    assert first == again
    assert len(set(first)) == len(first)
    assert first[0] == 610_000_000 and first[-1] == 610_001_999


def test_two_ranges_never_hand_out_the_same_magic() -> None:
    source = _hashes(500)
    a = {mg.allocate("s6-parity-fixtures", h, source) for h in source}
    b = {mg.allocate("ga-usdcadm_h1-s7-cbcb2ae061e70bef", h, source) for h in source}
    assert not a & b


def test_owner_of_round_trips_every_allocation() -> None:
    source = _hashes(50)
    for slot, h in enumerate(source):
        m = mg.allocate("ga-usdcadm_h1-s7-cbcb2ae061e70bef", h, source)
        assert mg.owner_of(m) == ("ga-usdcadm_h1-s7-cbcb2ae061e70bef", slot)


def test_a_repeated_hash_in_the_source_is_refused() -> None:
    source = _hashes(3) + _hashes(1)
    with pytest.raises(mg.MagicError, match="not be injective"):
        mg.allocate("s6-parity-fixtures", source[0], source)


def test_an_unknown_hash_or_range_or_slot_is_refused() -> None:
    with pytest.raises(mg.MagicError, match="not in the range's source list"):
        mg.allocate("s6-parity-fixtures", "ffffffffffffffff", _hashes(3))
    with pytest.raises(mg.MagicError, match="no magic range named"):
        mg.magic_for("no-such-campaign", 0)
    with pytest.raises(mg.MagicError, match="outside range"):
        mg.magic_for("s6-live-demo", 100_000)


def test_live_roles_are_fixed_distinct_and_own_only_inside_the_live_range() -> None:
    magics = {role: mg.live_magic(role) for role in mg.LIVE_ROLES}
    assert magics == {"worker": 610_900_000, "plant_own_a": 610_900_001,
                      "plant_own_b": 610_900_002, "plant_own_pending": 610_900_003}
    for m in magics.values():
        assert mg.is_own(m, mg.LIVE_RANGE)
    assert not mg.is_own(424242, mg.LIVE_RANGE)
    assert not mg.is_own(610_100_000, mg.LIVE_RANGE)  # an ARQVIO magic of another range is foreign here
    with pytest.raises(mg.MagicError, match="unknown live role"):
        mg.live_magic("plant_foreign")
