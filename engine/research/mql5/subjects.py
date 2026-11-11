"""The pre-registered S6 parity subject set (DR-2026-10-01-s6-dual-mode-parity-design, "Parity subject set").

76 subjects, in this order:

* S-GA (40): the 20 GA finalists (read-only, ``ga_subjects.load_finalists``) on
  the full USDCADm window, then the same 20 on the full XAUUSDm window -- the
  campaign's own cell setup;
* S-BAT (24): 6 battery indices per family, drawn with ONE
  ``default_rng(20261001)`` by ``choice(60, 6, replace=False)`` family by family
  (RSI, STOCH, MACD, DONCHIAN), each the battery's own DNA at that index (its
  ``FAMILY_SEEDS`` stream, ``random_dna`` with the signal replaced); battery cell;
* S-MA (6): the first 6 ``random_dna`` draws of ``default_rng(20261005)`` as
  MA_CROSS; battery cell;
* S-EDGE (6): rules e1-e6 (DR), each over a FRESH ``default_rng(20261006)``
  stream; battery cell.

``SUBJECT_IDS`` is a pure function of the pre-registration (no data needed);
``subject_set`` resolves the DNAs. Magic: S-GA through the GA campaign's range
(manifest order); the other 36 through ``s6-parity-fixtures``, slot = their index in this order.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from engine.research.mql5.cell_config import CellConfig
from engine.research.mql5.ga_subjects import GA_CAMPAIGN, load_finalists
from engine.research.mql5.magic import allocate, magic_for
from engine.research.mql5.oracle_trace import run_oracle
from engine.research.search.dna import DNA
from engine.research.search.sampler import random_dna

BATTERY_SEEDS = {  # tests/research/backtest/test_family_kernel_parity.py FAMILY_SEEDS, verbatim
    "RSI_REVERSION": 20261001, "STOCH_CROSS": 20261202, "MACD_CROSS": 20261003, "DONCHIAN_BREAKOUT": 20261004,
}
BAT_SELECT_SEED, MA_SEED, EDGE_SEED = 20261001, 20261005, 20261006
FAMILY_SHORT = {"RSI_REVERSION": "RSI", "STOCH_CROSS": "STOCH", "MACD_CROSS": "MACD", "DONCHIAN_BREAKOUT": "DONCH"}
FIXTURES_RANGE = "s6-parity-fixtures"
EDGE_SCAN_CAP = 200

SUBJECT_IDS: tuple[str, ...] = (
    tuple(f"S-GA-usdcadm-{k:02d}" for k in range(20)) + tuple(f"S-GA-xauusdm-{k:02d}" for k in range(20))
    + tuple(f"S-BAT-{FAMILY_SHORT[f]}-{j}" for f in BATTERY_SEEDS for j in range(6))
    + tuple(f"S-MA-{j}" for j in range(6)) + tuple(f"S-EDGE-e{j}" for j in range(1, 7))
)


class SubjectSetError(RuntimeError):
    """A pre-registered selection rule that cannot be resolved."""


@dataclass(frozen=True)
class Subject:
    sid: str
    group: str
    dna: DNA
    cell_key: str  # "usdcadm_full" | "xauusdm_full" | "battery"
    magic: int
    note: str


def battery_dnas(family: str, n: int = 60) -> list[DNA]:
    rng = np.random.default_rng(BATTERY_SEEDS[family])
    return [dataclasses.replace(random_dna(rng), signal=family) for _ in range(n)]


def _stream(seed: int) -> Iterator[DNA]:
    rng = np.random.default_rng(seed)
    while True:
        yield random_dna(rng)


def battery_indices() -> dict[str, list[int]]:
    rng = np.random.default_rng(BAT_SELECT_SEED)
    return {f: [int(i) for i in rng.choice(60, 6, replace=False)] for f in BATTERY_SEEDS}


def edge_dnas(battery: CellConfig) -> list[tuple[DNA, str]]:
    out: list[tuple[DNA, str]] = []
    for k, d in enumerate(_stream(EDGE_SEED)):
        if k >= EDGE_SCAN_CAP:
            raise SubjectSetError(f"e1: no qualifying draw in the first {EDGE_SCAN_CAP}")
        dna = dataclasses.replace(d, signal="RSI_REVERSION", rsi_period=5)
        twin = dataclasses.replace(dna, sl_atr=0.0, tp_atr=0.0)
        early = [x.bar_index for x in run_oracle(twin, battery).intents if x.bar_index < 13]
        if early:
            out.append((dna, f"e1 ATR suppression: draw {k}; the stop-free twin emits at bars {early}"))
            break
    for idx, dna in enumerate(battery_dnas("RSI_REVERSION")):
        if run_oracle(dna, battery).result.n_trades == 0:
            out.append((dna, f"e2 zero-trade: battery RSI index {idx}"))
            break
    else:
        raise SubjectSetError("e2: no zero-trade battery RSI DNA")
    first = next(_stream(EDGE_SEED))
    out.append((dataclasses.replace(first, signal="RSI_REVERSION", rsi_exit_midline=True), "e3 RSI midline exit"))
    out.append((dataclasses.replace(first, signal="MA_CROSS", sl_atr=0.0, tp_atr=0.0), "e4 MA_CROSS without stops"))
    out.append((dataclasses.replace(first, signal="MA_CROSS", max_feature_lookback=20), "e5 reversal-heavy MA_CROSS"))
    out.append((dataclasses.replace(first, signal="RSI_REVERSION"), "e6 RSI with sl_atr > 0 and tp_atr > 0"))
    return out


def subject_set(battery: CellConfig) -> tuple[Subject, ...]:
    fins = load_finalists()
    fin_hashes = [f.dna_hash for f in fins]
    out: list[Subject] = []
    for key, short in (("usdcadm_full", "usdcadm"), ("xauusdm_full", "xauusdm")):
        for f in fins:
            out.append(Subject(f"S-GA-{short}-{f.index:02d}", "S-GA", f.dna, key,
                               allocate(GA_CAMPAIGN, f.dna_hash, fin_hashes), f"GA finalist {f.index}"))
    rest: list[tuple[str, str, DNA, str]] = []
    for fam, idxs in battery_indices().items():
        dnas = battery_dnas(fam)
        for j, i in enumerate(idxs):
            rest.append((f"S-BAT-{FAMILY_SHORT[fam]}-{j}", "S-BAT", dnas[i], f"battery {fam} index {i}"))
    rng = np.random.default_rng(MA_SEED)
    for j in range(6):
        rest.append((f"S-MA-{j}", "S-MA", dataclasses.replace(random_dna(rng), signal="MA_CROSS"), f"MA_CROSS draw {j}"))
    for j, (dna, note) in enumerate(edge_dnas(battery), 1):
        rest.append((f"S-EDGE-e{j}", "S-EDGE", dna, note))
    # The DR's fixture slot is the SUBJECT INDEX (not the hash): two rules may resolve to the same DNA;
    # both stay as registered, each duplicate named in its note (never silently re-drawn).
    first_sid: dict[str, str] = {}
    for slot, (sid, group, dna, note) in enumerate(rest):
        if dna.dna_hash in first_sid:
            note = f"{note}; same DNA as {first_sid[dna.dna_hash]} (the pre-registered rules coincide)"
        else:
            first_sid[dna.dna_hash] = sid
        out.append(Subject(sid, group, dna, "battery", magic_for(FIXTURES_RANGE, slot), note))
    if tuple(s.sid for s in out) != SUBJECT_IDS:
        raise SubjectSetError("the resolved subject set does not follow the pre-registered order")
    return tuple(out)


__all__ = ["SUBJECT_IDS", "Subject", "SubjectSetError", "battery_dnas", "battery_indices", "edge_dnas", "subject_set"]
