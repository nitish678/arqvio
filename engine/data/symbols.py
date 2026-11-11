"""Symbol discovery + classification.

The registry is dynamic: it is rebuilt from ``mt5.symbols_get()`` on
startup. New symbols default to disabled. The category mapping follows
the spec.

Broker suffix auto-detection: Exness uses ``m`` (e.g. ``XAUUSDm``). The
suffix is whatever follows the base 6/3-character pair. The detection
function returns the suffix and the base name; both are recorded so
campaigns can be scoped by base name (i.e. "all XAUUSD variants").
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

# Exness uses "m" for micro accounts and similar; the pattern is
# "<BASE><suffix>". The base can be 6 chars (FX) or 3 chars (metals/indices).
_BROKER_SUFFIX_PATTERN = re.compile(r"^(?P<base>[A-Z]{6}|[A-Z]{3})(?P<suffix>[A-Za-z]+)?$")


@dataclass(frozen=True)
class SymbolRecord:
    name: str
    base: str
    suffix: str
    category: str
    visible: bool
    digits: int
    point: float
    spread_points: int

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "base": self.base,
            "suffix": self.suffix,
            "category": self.category,
            "visible": self.visible,
            "digits": self.digits,
            "point": self.point,
            "spread_points": self.spread_points,
        }


# Categorisation rules. The MT5 ``path`` field is "broker\\section\\name";
# section is the category for most brokers. Where the section is "Forex" we
# further split into MAJOR / MINOR / EXOTIC by base pair membership.
FOREX_MAJOR_BASES: frozenset[str] = frozenset({
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD",
})
FOREX_MINOR_BASES: frozenset[str] = frozenset({
    "EURGBP", "EURJPY", "EURCHF", "EURAUD", "EURCAD", "EURNZD",
    "GBPJPY", "GBPCHF", "GBPAUD", "GBPCAD", "GBPNZD",
    "AUDJPY", "AUDCHF", "AUDCAD", "AUDNZD",
    "CADJPY", "CADCHF", "CHFJPY", "NZDJPY", "NZDCHF", "NZDCAD",
})
METALS_BASES: frozenset[str] = frozenset({"XAUUSD", "XAGUSD", "XPTUSD", "XPDUSD"})
ENERGY_BASES: frozenset[str] = frozenset({"XTIUSD", "XBRUSD", "XNGUSD"})
CRYPTO_BASES: frozenset[str] = frozenset({
    "BTCUSD", "ETHUSD", "LTCUSD", "XRPUSD", "BCHUSD", "ADAUSD",
    "DOGUSD", "SOLUSD", "DOTUSD", "AVXUSD", "LINKUSD",
})


def detect_suffix(symbol_name: str) -> tuple[str, str]:
    """Return (base, suffix). The suffix may be empty.

    >>> detect_suffix("XAUUSDm")
    ('XAUUSD', 'm')
    >>> detect_suffix("EURUSD")
    ('EURUSD', '')
    """
    match = _BROKER_SUFFIX_PATTERN.match(symbol_name)
    if not match:
        return symbol_name, ""
    return match.group("base"), match.group("suffix") or ""


def categorise(base: str, section: str) -> str:
    """Return one of: FOREX_MAJOR, FOREX_MINOR, FOREX_EXOTIC, METALS,
    INDICES, ENERGY, CRYPTO, OTHER."""
    if section.lower() in {"forex", "fx"}:
        if base in FOREX_MAJOR_BASES:
            return "FOREX_MAJOR"
        if base in FOREX_MINOR_BASES:
            return "FOREX_MINOR"
        if base in METALS_BASES:
            return "METALS"  # exotic-broker sometimes puts metals under forex
        return "FOREX_EXOTIC"
    if base in METALS_BASES or section.lower() in {"metals", "bullion"}:
        return "METALS"
    if section.lower() in {"indices", "index", "cfd indices"}:
        return "INDICES"
    if base in ENERGY_BASES or section.lower() in {"energy", "commodities"}:
        return "ENERGY"
    if base in CRYPTO_BASES or section.lower() in {"crypto", "cryptocurrency"}:
        return "CRYPTO"
    return "OTHER"


def build_record(name: str, path: str, visible: bool, digits: int, point: float, spread_points: int) -> SymbolRecord:
    base, suffix = detect_suffix(name)
    parts = (path or "").split("\\")
    section = parts[1] if len(parts) >= 2 else "Other"
    return SymbolRecord(
        name=name, base=base, suffix=suffix,
        category=categorise(base, section),
        visible=bool(visible),
        digits=int(digits),
        point=float(point),
        spread_points=int(spread_points),
    )


def by_category(records: Iterable[SymbolRecord]) -> dict[str, list[SymbolRecord]]:
    out: dict[str, list[SymbolRecord]] = {}
    for record in records:
        out.setdefault(record.category, []).append(record)
    return out
