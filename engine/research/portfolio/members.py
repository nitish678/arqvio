"""The declared S7 portfolio member set and its sim cells (DR §1). The rule, not a result, chooses the members.

Rule: for each family in FAMILIES order, the FIRST DNA of its seeded parity-battery list
(``engine.research.mql5.subjects.battery_dnas``, the battery's FAMILY_SEEDS verbatim) whose run has >= 1 entry fill
on BOTH sim cells -- a trade COUNT, nothing else -- is that family's DNA, run on both cells: 4 families x 2 symbols =
8 members, in (family, cell) order. Candidates are indices 0..MAX_INDEX; none passing refuses (MemberSetError).

Sim cells: the LAST ``SIM_BARS`` bars of each pinned cell under the campaign's own slow setup
(``campaign_cell``: calibrated hourly spreads, matrix no-trade hours, the percentile gate, risk_pct sizing, seed 0).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from engine.research.data.mt5_source import BarData
from engine.research.mql5.cell_config import CellConfig, campaign_cell
from engine.research.mql5.subjects import FAMILY_SHORT, battery_dnas
from engine.research.portfolio.sim import MemberRun, run_member
from engine.research.search.dna import DNA

CELLS = (("USDCADm", "aed3bf314539ec5a"), ("XAUUSDm", "447b843f87438a9a"))
FAMILIES = ("RSI_REVERSION", "STOCH_CROSS", "MACD_CROSS", "DONCHIAN_BREAKOUT")
SIM_BARS = 800
E0 = 100_000.0
MAX_INDEX = 5


class MemberSetError(RuntimeError):
    """No battery DNA of a family passes the sanity rule on both cells."""


@dataclass(frozen=True)
class Member:
    mid: str
    family: str
    symbol: str
    dataset_id: str
    battery_index: int
    dna: DNA


def sim_cell(symbol: str, dataset_id: str) -> CellConfig:
    full = campaign_cell(symbol, dataset_id)
    s = slice(-SIM_BARS, None)
    b = full.bars
    bars = BarData(time=np.asarray(b.time)[s], open=np.asarray(b.open)[s], high=np.asarray(b.high)[s],
                   low=np.asarray(b.low)[s], close=np.asarray(b.close)[s], tick_volume=np.asarray(b.tick_volume)[s],
                   spread=np.asarray(b.spread)[s], real_volume=np.asarray(b.real_volume)[s],
                   symbol=b.symbol, timeframe=b.timeframe)
    return dataclasses.replace(full, name=f"{symbol.lower()}_h1_{dataset_id}_last{SIM_BARS}c", bars=bars)


def sim_cells() -> dict[str, CellConfig]:
    return {sym: sim_cell(sym, did) for sym, did in CELLS}


def member_share() -> float:
    return E0 / (len(FAMILIES) * len(CELLS))


@dataclass(frozen=True)
class MemberSet:
    members: tuple[Member, ...]
    runs: tuple[MemberRun, ...]
    skipped: tuple[dict, ...]


def member_set(cells: dict[str, CellConfig] | None = None) -> MemberSet:
    cells = cells or sim_cells()
    members, runs, skipped = [], [], []
    for family in FAMILIES:
        dnas = battery_dnas(family)
        for idx in range(MAX_INDEX + 1):
            dna = dnas[idx]
            trial = {sym: run_member(f"{FAMILY_SHORT[family]}-{sym}", dna, cells[sym], initial_equity=member_share())
                     for sym, _ in CELLS}
            fills = {sym: sum(1 for t in r.result.trades if t.entry_fill is not None) for sym, r in trial.items()}
            if all(n >= 1 for n in fills.values()):
                for sym, did in CELLS:
                    members.append(Member(f"{FAMILY_SHORT[family]}-{sym}", family, sym, did, idx, dna))
                    runs.append(trial[sym])
                break
            skipped.append({"family": family, "battery_index": idx, "dna_hash": dna.dna_hash, "entry_fills": fills})
        else:
            raise MemberSetError(f"{family}: no battery DNA in 0..{MAX_INDEX} has >= 1 entry fill on both cells")
    return MemberSet(tuple(members), tuple(runs), tuple(skipped))


__all__ = ["CELLS", "E0", "FAMILIES", "MAX_INDEX", "Member", "MemberSet", "MemberSetError", "SIM_BARS",
           "member_set", "member_share", "sim_cell", "sim_cells"]
