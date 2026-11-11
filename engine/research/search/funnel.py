"""Failure taxonomy + survivor pipeline.

The funnel applies, in order:
    1. Lookahead guard
    2. Spread calibration check
    3. Trade-count filter (LowTrades)
    4. Drawdown filter (HighDD)
    5. Return sign (NegReturn)
    6. DSR gate (FailedDSR) - per-cell, with min_obs gate (DsrUnreliable)
    7. PBO gate (HighPBO)
    8. Crash recovery (Crash)
    9. VAULTED

``summarise`` reduces the funnel to the failure-taxonomy counts the
final report must include.
"""
from __future__ import annotations

import collections
from collections.abc import Iterable
from dataclasses import dataclass

from engine.research.search.dna import FailureTaxonomy


@dataclass(frozen=True)
class FunnelCounts:
    total: int
    by_category: dict[str, int]
    survivors: int
    n_vaulted: int
    n_unreliable: int
    n_failed_dsr: int
    n_high_pbo: int
    n_low_trades: int
    n_high_dd: int
    n_neg_return: int
    n_spread_uncalibrated: int
    n_crashed: int

    def to_dict(self) -> dict[str, object]:
        return {
            "total": self.total,
            "by_category": dict(self.by_category),
            "survivors": self.survivors,
            "VAULTED": self.n_vaulted,
            "DsrUnreliable": self.n_unreliable,
            "FailedDSR": self.n_failed_dsr,
            "HighPBO": self.n_high_pbo,
            "LowTrades": self.n_low_trades,
            "HighDD": self.n_high_dd,
            "NegReturn": self.n_neg_return,
            "SpreadUncalibrated": self.n_spread_uncalibrated,
            "Crash": self.n_crashed,
        }


def summarise(outcomes: Iterable[str]) -> FunnelCounts:
    counter: collections.Counter[str] = collections.Counter()
    for outcome in outcomes:
        counter[outcome] += 1
    n_vaulted = counter.get(FailureTaxonomy.VAULTED.value, 0)
    return FunnelCounts(
        total=int(sum(counter.values())),
        by_category=dict(counter),
        survivors=n_vaulted,
        n_vaulted=n_vaulted,
        n_unreliable=counter.get(FailureTaxonomy.DsrUnreliable.value, 0),
        n_failed_dsr=counter.get(FailureTaxonomy.FailedDSR.value, 0),
        n_high_pbo=counter.get(FailureTaxonomy.HighPBO.value, 0),
        n_low_trades=counter.get(FailureTaxonomy.LowTrades.value, 0),
        n_high_dd=counter.get(FailureTaxonomy.HighDD.value, 0),
        n_neg_return=counter.get(FailureTaxonomy.NegReturn.value, 0),
        n_spread_uncalibrated=counter.get(FailureTaxonomy.SpreadUncalibrated.value, 0),
        n_crashed=counter.get(FailureTaxonomy.Crash.value, 0),
    )


__all__ = ["FunnelCounts", "summarise"]
