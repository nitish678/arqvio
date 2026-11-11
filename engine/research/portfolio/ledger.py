"""The shared cash ledger: one account, many members, conservation checked per bar, in exact rationals.

Every amount enters as ``Fraction(float)`` -- the exact value of the IEEE double, no rounding. ``cash`` is E0 plus
every applied FEE / SWAP / REALIZE; ``equity`` is updated INCREMENTALLY by each applied event and by each member's
mark change. ``close_bar`` asserts ``equity == cash + sum(marks)`` exactly (DR §2 check (a)) and records the bar's
doubles. A broken invariant raises ``ConservationError`` naming the bar and both sides.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

CASH_KINDS = ("FEE", "SWAP", "REALIZE")


class ConservationError(RuntimeError):
    """equity != cash + marks at a bar close."""


@dataclass(frozen=True)
class CashEvent:
    t: int
    member: str
    kind: str                 # FEE | SWAP | REALIZE
    trade_id: str
    amount: float | Fraction  # signed, account currency; a SWAP is the exact difference of two engine doubles


@dataclass(frozen=True)
class BarState:
    t: int
    cash: float
    marks: float
    equity: float


@dataclass
class Ledger:
    e0: float
    cash: Fraction = field(init=False)
    equity: Fraction = field(init=False)
    marks: dict[str, Fraction] = field(default_factory=dict, init=False)
    applied: list[CashEvent] = field(default_factory=list, init=False)
    bars: list[BarState] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.cash = Fraction(self.e0)
        self.equity = Fraction(self.e0)

    def apply(self, ev: CashEvent) -> None:
        if ev.kind not in CASH_KINDS:
            raise ValueError(f"not a cash event kind: {ev.kind!r}")
        x = ev.amount if isinstance(ev.amount, Fraction) else Fraction(ev.amount)
        self.cash += x
        self.equity += x
        self.applied.append(ev)

    def set_mark(self, member: str, value: float) -> None:
        new = Fraction(value)
        self.equity += new - self.marks.get(member, Fraction(0))
        self.marks[member] = new

    def close_bar(self, t: int) -> BarState:
        total = sum(self.marks.values(), Fraction(0))
        if self.equity != self.cash + total:
            raise ConservationError(f"bar {t}: equity {self.equity} != cash {self.cash} + marks {total}")
        state = BarState(t, float(self.cash), float(total), float(self.equity))
        self.bars.append(state)
        return state

    def equity_float(self) -> float:
        return float(self.equity)


__all__ = ["BarState", "CASH_KINDS", "CashEvent", "ConservationError", "Ledger"]
