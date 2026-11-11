"""The sham control: one random-entry twin per real candidate.

vs_random asks whether the real campaign's scores beat a control campaign's.
The control has to differ from the real one in exactly one thing -- whether
entries carry a signal -- or a difference in the scores measures something
else. So each real DNA gets a sham that:

  * attempts as many entries as the DNA did: its n_trades plus the entries
    the max-spread gate refused it (blocked_max_spread). The sham faces the
    same gate, so matching attempts is what keeps its FILLED count near the
    DNA's n_trades. Requesting n_trades alone leaves the sham short by
    whatever share of its entries the gate refuses;
  * enters on bars drawn uniformly, without replacement, from the bars the
    DNA itself could have signalled on (after its own warmup, and after the
    ATR warmup when it wants stops), up to the last bar that can still fill;
  * takes each side with probability 1/2;
  * keeps every risk and exit gene: lot size, risk_pct, sl_atr, tp_atr;
  * runs on the slow engine (the oracle) with the identical BacktestConfig
    the real candidate gets -- costs, spread model, no-trade hours and the
    max-spread gate -- through the same scoring code (run_slow_trial).

The draw is seeded by sha256(f"{dna_hash}:{salt}"), never Python's hash()
(salted per process), and never by the campaign rng: computing the shams
cannot change which DNAs the campaign samples.

vs_random itself takes plain score arrays (engine/validation/vs_random.py:80),
so the sham Sharpes pass straight through run_campaign's control_fn; nothing
in the frozen validation layer changes.
"""
from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any

import numpy as np

from engine.research.backtest.strategies.sham import ShamConfig, ShamStrategy
from engine.research.search.dna import DNA
from engine.research.search.real_backtest import (
    CellBars,
    dna_to_strategy,
    run_slow_trial,
    slow_cell_setup,
)
from engine.research.search.runner import TrialResult
from engine.validation.vs_random import VsRandomResult

SHAM_SALT = "astra-s3-sham-v1"  # frozen: seeds every sham (the pre-ARQVIO name)
SHAM_CONTROL_KIND = "sham_random_entry"
CONTROLS_REGIME = (
    "v2 sham_random_entry (was v1 iid N(0,0.5); p-values are not comparable "
    "across the change)"
)


def sham_seed_hex(dna_hash: str, salt: str) -> str:
    """The first 16 hex digits of sha256(f"{dna_hash}:{salt}")."""
    return hashlib.sha256(f"{dna_hash}:{salt}".encode()).hexdigest()[:16]


def eligible_entry_range(cfg: Any, n_bars: int) -> tuple[int, int]:
    """[lo, hi] bar indices a sham may enter on, inclusive.

    lo is the DNA's own strategy warmup, raised to the first bar with a
    defined ATR when the DNA wants a stop or target. hi is n_bars - 2: the
    strategy is last consulted there and its intent fills on the final bar.
    """
    lo = int(cfg.warmup)
    if cfg.sl_atr > 0.0 or cfg.tp_atr > 0.0:
        lo = max(lo, int(cfg.atr_period) - 1)
    return lo, int(n_bars) - 2


def draw_entries(
    dna_hash: str, salt: str, lo: int, hi: int, n_entries: int,
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """min(n_entries, eligible) distinct bars in [lo, hi], sorted, each with
    a fair-coin side."""
    rng = np.random.default_rng(int(sham_seed_hex(dna_hash, salt), 16))
    eligible = np.arange(lo, hi + 1, dtype=np.int64)
    k = min(max(0, int(n_entries)), int(eligible.size))
    if k == 0:
        return (), ()
    bars = np.sort(rng.choice(eligible, size=k, replace=False))
    longs = rng.random(k) < 0.5
    return (
        tuple(int(b) for b in bars),
        tuple("long" if is_long else "short" for is_long in longs),
    )


def attempted_entries(trial: TrialResult) -> int:
    """Entries the real candidate tried to open: the ones that filled plus
    the ones the max-spread gate refused."""
    return int(trial.n_trades or 0) + int(trial.blocked_max_spread or 0)


def sham_backtest_factory(
    cell: CellBars,
    *,
    flat_cost: bool = False,
    salt: str = SHAM_SALT,
) -> Callable[[DNA, int], TrialResult]:
    """fn(dna, n_entries) -> the sham's TrialResult, on the slow engine.

    ``truncated`` on the returned function maps dna_hash to
    (requested, drawn) for every sham whose eligible range held fewer bars
    than the entries requested; all eligible bars are used then.
    """
    setup = slow_cell_setup(cell, flat_cost=flat_cost, caller="sham_backtest_factory")
    truncated: dict[str, tuple[int, int]] = {}

    def _sham(dna: DNA, n_entries: int, *, salt: str = salt) -> TrialResult:  # another salt: another draw
        cfg, _ = dna_to_strategy(dna)
        lo, hi = eligible_entry_range(cfg, cell.n_bars)
        bars, sides = draw_entries(dna.dna_hash, salt, lo, hi, n_entries)
        if len(bars) < n_entries:
            truncated[dna.dna_hash] = (int(n_entries), len(bars))
        sham_cfg = ShamConfig(
            entry_bars=bars,
            sides=sides,
            volume=cfg.volume,
            strategy_id=f"sham_{sham_seed_hex(dna.dna_hash, salt)}",
            sl_atr=cfg.sl_atr,
            tp_atr=cfg.tp_atr,
            atr_period=cfg.atr_period,
        )
        return run_slow_trial(
            cell, setup, dna, sham_cfg, ShamStrategy(sham_cfg),
            notes=f"sham of dna_hash={dna.dna_hash} salt={salt}",
        )

    _sham.salt = salt
    _sham.truncated = truncated
    return _sham


def sham_control_factory(
    cell: CellBars,
    *,
    flat_cost: bool = False,
    salt: str = SHAM_SALT,
) -> Callable[[list[TrialResult]], list[float]]:
    """A run_campaign control_fn: one sham Sharpe per scored trial.

    Each sham attempts ``attempted_entries(trial)`` entries. Shams are
    memoised on (dna_hash, attempted entries), so the eligible subset's
    vs_random reuses the shams computed for the all-trials one. ``shams``
    on the returned function is that memo.
    """
    sham = sham_backtest_factory(cell, flat_cost=flat_cost, salt=salt)
    memo: dict[tuple[str, int], TrialResult] = {}

    def _ctrl(trials: list[TrialResult]) -> list[float]:
        out: list[float] = []
        for t in trials:
            key = (t.dna.dna_hash, attempted_entries(t))
            result = memo.get(key)
            if result is None:
                result = sham(t.dna, key[1])
                memo[key] = result
            out.append(float(result.sharpe))
        return out

    _ctrl.kind = SHAM_CONTROL_KIND
    _ctrl.salt = salt
    _ctrl.shams = memo
    _ctrl.truncated = sham.truncated
    return _ctrl


def controls_block(
    control_fn: Any,
    *,
    vs_all: VsRandomResult | None,
    vs_eligible: VsRandomResult | None,
) -> dict[str, Any]:
    """The report's ``controls`` block for a sham-controlled campaign."""
    warnings: list[str] = []
    for label, v in (("all", vs_all), ("eligible", vs_eligible)):
        if v is not None:
            warnings.extend(f"{label}: {w}" for w in v.warnings)
    for dna_hash, (requested, drawn) in sorted(control_fn.truncated.items()):
        warnings.append(
            f"sham {dna_hash[:12]}: {drawn} of {requested} entries drawn "
            "(eligible bars exhausted)"
        )
    shams = control_fn.shams
    return {
        "kind": control_fn.kind,
        "salt": control_fn.salt,
        "engine": "slow",
        "entry_count": (
            "matched to each real DNA's attempted entries "
            "(n_trades + blocked_max_spread)"
        ),
        "side": "50/50",
        "seed": "sha256(dna_hash:salt)",
        "n_all": vs_all.n_random if vs_all is not None else 0,
        "n_eligible": vs_eligible.n_random if vs_eligible is not None else 0,
        "entries_requested": sum(n for _, n in shams),
        "entries_filled": sum(int(r.n_trades or 0) for r in shams.values()),
        "random_max_q95": {
            "all": vs_all.random_max_q95 if vs_all is not None else None,
            "eligible": (
                vs_eligible.random_max_q95 if vs_eligible is not None else None
            ),
        },
        "warnings": warnings,
    }


__all__ = [
    "CONTROLS_REGIME",
    "SHAM_CONTROL_KIND",
    "SHAM_SALT",
    "attempted_entries",
    "controls_block",
    "draw_entries",
    "eligible_entry_range",
    "sham_backtest_factory",
    "sham_control_factory",
    "sham_seed_hex",
]
