"""The ARQVIO magic-number registry: which MT5 magic belongs to which strategy.

The committed range table (``magic_ranges.json``, read at call time) assigns
each campaign a disjoint range inside the ARQVIO space 610,000,000-610,999,999.
Within a range, a strategy's slot is the index of its dna_hash in the range's
ordered source list (a GA campaign's manifest order, the pre-registered S6
subject order), so

    magic(range, dna_hash) = base + index of dna_hash in the source list

is deterministic, and injective by construction: the source list holds unique
hashes (checked) and the ranges are disjoint (checked on every load). The
live-demo range has fixed roles instead of hashes.

Every magic the account history shows, the F2 probe magic and the planted
foreign magic of the kill-switch proof lie OUTSIDE the space (checked), so an
ARQVIO allocation can never collide with them. ``owner_of`` answers the only
question order-capable code may ask: is this magic one of ours?
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources

from engine.core.schema import same_schema

TABLE_SCHEMA = "arqvio.mql5.magic_ranges"
TABLE_VERSION = 1

LIVE_RANGE = "s6-live-demo"
LIVE_ROLES = {"worker": 0, "plant_own_a": 1, "plant_own_b": 2, "plant_own_pending": 3}


class MagicError(ValueError):
    """A magic that cannot be allocated, or a range table that cannot be trusted."""


@dataclass(frozen=True)
class MagicRange:
    name: str
    base: int
    span: int

    @property
    def last(self) -> int:
        return self.base + self.span - 1

    def contains(self, magic: int) -> bool:
        return self.base <= magic <= self.last


@dataclass(frozen=True)
class MagicTable:
    space_min: int
    space_max: int
    ranges: tuple[MagicRange, ...]
    outside: tuple[int, ...]

    def range(self, name: str) -> MagicRange:
        for r in self.ranges:
            if r.name == name:
                return r
        raise MagicError(f"no magic range named {name!r}; ranges are {[r.name for r in self.ranges]}")


def load_table() -> MagicTable:
    """Read and validate the committed range table."""
    raw = json.loads(resources.files("engine.research.mql5").joinpath("magic_ranges.json").read_text("utf-8"))
    if not same_schema(raw.get("schema"), TABLE_SCHEMA) or raw.get("version") != TABLE_VERSION:
        raise MagicError(f"magic table schema {raw.get('schema')!r} v{raw.get('version')!r} is not {TABLE_SCHEMA} v{TABLE_VERSION}")
    lo, hi = int(raw["space"]["min"]), int(raw["space"]["max"])
    ranges = tuple(MagicRange(str(r["name"]), int(r["base"]), int(r["span"])) for r in raw["ranges"])
    outside = tuple(sorted({int(m) for group in raw["outside"].values() for m in group}))
    names = [r.name for r in ranges]
    if len(set(names)) != len(names):
        raise MagicError(f"duplicate range names: {names}")
    for r in ranges:
        if r.span < 1 or r.base < lo or r.last > hi:
            raise MagicError(f"range {r.name} [{r.base}, {r.last}] is outside the space [{lo}, {hi}]")
    ordered = sorted(ranges, key=lambda r: r.base)
    for a, b in zip(ordered, ordered[1:]):
        if b.base <= a.last:
            raise MagicError(f"ranges {a.name} and {b.name} overlap")
    for m in outside:
        if lo <= m <= hi:
            raise MagicError(f"magic {m} is listed outside the ARQVIO space but lies inside it")
    return MagicTable(lo, hi, ranges, outside)


def magic_for(range_name: str, slot: int, *, table: MagicTable | None = None) -> int:
    """The magic of ``slot`` in ``range_name``."""
    r = (table or load_table()).range(range_name)
    if not 0 <= slot < r.span:
        raise MagicError(f"slot {slot} is outside range {r.name} (span {r.span})")
    return r.base + slot


def slot_of(dna_hash: str, source: list[str] | tuple[str, ...]) -> int:
    """The index of ``dna_hash`` in an ordered source list of unique hashes."""
    if len(set(source)) != len(source):
        raise MagicError("the source list repeats a dna_hash; slots would not be injective")
    try:
        return list(source).index(dna_hash)
    except ValueError:
        raise MagicError(f"dna_hash {dna_hash!r} is not in the range's source list") from None


def allocate(range_name: str, dna_hash: str, source: list[str] | tuple[str, ...]) -> int:
    """magic(range, dna_hash) = base + the hash's index in the ordered source list."""
    return magic_for(range_name, slot_of(dna_hash, source))


def live_magic(role: str) -> int:
    """The fixed-role magic of the live-demo range."""
    if role not in LIVE_ROLES:
        raise MagicError(f"unknown live role {role!r}; roles are {sorted(LIVE_ROLES)}")
    return magic_for(LIVE_RANGE, LIVE_ROLES[role])


def owner_of(magic: int, *, table: MagicTable | None = None) -> tuple[str, int] | None:
    """(range name, slot) when ``magic`` is an ARQVIO allocation, else None."""
    t = table or load_table()
    for r in t.ranges:
        if r.contains(int(magic)):
            return r.name, int(magic) - r.base
    return None


def is_own(magic: int, range_name: str, *, table: MagicTable | None = None) -> bool:
    """True only when ``magic`` lies in ``range_name``: anything else is foreign."""
    return (table or load_table()).range(range_name).contains(int(magic))


__all__ = [
    "LIVE_RANGE",
    "LIVE_ROLES",
    "MagicError",
    "MagicRange",
    "MagicTable",
    "allocate",
    "is_own",
    "live_magic",
    "load_table",
    "magic_for",
    "owner_of",
    "slot_of",
]
