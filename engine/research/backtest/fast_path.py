"""Fast path driver: prep -> kernel -> the same TrialResult the slow path fills.

THE FALLBACK RULE IS THE POINT. A candidate the kernel cannot express
EXACTLY does not get approximated -- it runs on the slow engine and is
counted. One silently approximated candidate in a thousand is
indistinguishable from a real edge at the population level, which is
the failure this stage exists not to cause. ``expressible()`` is cheap,
pure Python and runs before anything is compiled.

Zero fallbacks is the goal for every allowlisted family. The mechanism
exists so that the goal is checkable rather than assumed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from engine.research.backtest.kernel import (
    REC_BLOCKED_DAILY_LOSS,
    REC_BLOCKED_MAX_SPREAD,
    REC_BLOCKED_ZERO_SPREAD,
    REC_COST_COMMISSION,
    REC_COST_SLIPPAGE,
    REC_COST_SPREAD,
    REC_COST_SWAP,
    REC_DEFERRED_ZERO_SPREAD_FILLS,
    REC_EXIT_EOD,
    REC_EXIT_FRIDAY_CLOSE,
    REC_EXIT_SIGNAL,
    REC_EXIT_SL,
    REC_EXIT_SLTP_DEFERRED,
    REC_EXIT_TP,
    REC_FINAL_EQUITY,
    REC_FRIDAY_FLATTENS,
    REC_N_OPEN,
    REC_N_ROUND_TRIPS,
    REC_N_TRADES,
    REC_SIZE,
    REC_SLTP_DEFERRED_ZERO_SPREAD,
    TR_SIZE,
    run_candidate,
)
from engine.research.backtest.prep import MA_CROSS_ARRAYS, FamilyArrays, PreppedCell

# Everything the kernel implements. A gene family, exec mode or
# management type outside these lists is a fallback, not an
# approximation.
SUPPORTED_EXEC_MODES = frozenset({"MARKET2"})
SUPPORTED_MGMT = frozenset({"NONE"})
# B1: allowlist of signal families the kernel CAN express. Everything else
# falls back to the slow engine — the kernel never approximates, so the
# allowlist is the safety boundary, not a denylist.
#
#   MA_CROSS          — means computed in-kernel (_pairwise_sum).
#   RSI_REVERSION     — W1.F1: kernel family branch + 60-DNA oracle parity.
#   STOCH_CROSS       — W1.F2: kernel family branch + 60-DNA oracle parity.
#   MACD_CROSS        — W1.F3: kernel family branch + 60-DNA oracle parity.
#   DONCHIAN_BREAKOUT — W1.F4: kernel family branch + 60-DNA oracle parity.
#
# A family joins here only AFTER its test_family_kernel_parity case is green
# (trade lists exact vs the slow engine) and its fired step fixtures pass.
SUPPORTED_SIGNALS = frozenset({
    "MA_CROSS", "RSI_REVERSION", "STOCH_CROSS", "MACD_CROSS", "DONCHIAN_BREAKOUT", "SIGNALS",
})


@dataclass(frozen=True)
class Expressibility:
    ok: bool
    reason: str = ""


def expressible(dna, config) -> Expressibility:
    """Can the kernel reproduce this candidate EXACTLY?

    Conservative by construction: anything not positively known to be
    supported is refused. A wrong "yes" here is a silent approximation;
    a wrong "no" costs only speed.
    """
    exec_mode = getattr(dna, "exec_mode", None)
    if exec_mode not in SUPPORTED_EXEC_MODES:
        return Expressibility(False, f"exec_mode={exec_mode!r} not in {sorted(SUPPORTED_EXEC_MODES)}")

    mgmt = getattr(dna, "mgmt", None)
    if mgmt not in SUPPORTED_MGMT:
        return Expressibility(False, f"mgmt={mgmt!r} not implemented in the kernel")

    signal = getattr(dna, "signal", None)
    if signal not in SUPPORTED_SIGNALS:
        return Expressibility(False, f"signal={signal!r} not in kernel allowlist {sorted(SUPPORTED_SIGNALS)}")

    if bool(getattr(dna, "uses_future_bar", False)):
        return Expressibility(False, "uses_future_bar=True")
    # W4-2: any V4 switch gene that is ON means the kernel cannot reproduce the
    # bias/filter/management/risk logic — fall back to the slow oracle.
    # v4 batch B: bias/filter switches reach the kernel as gate arrays (backtest/switch_gates.py), the same gate
    # objects the oracle steps; management/risk switches still fall back to the slow oracle.
    # v4 batch A: breakeven, sl_reduce, sl_lock, atr_trail are kernel steps (4b, engine.py _breakeven /
    # _stop_rules); other management/risk switches fall back.
    from engine.research.backtest.strategies.switch import ENGINE_MGMT, KERNEL_CLOSE_MGMT

    kernel_mgmt = ENGINE_MGMT + KERNEL_CLOSE_MGMT
    if any(m not in kernel_mgmt for m in getattr(dna, "mgmt_switches", ())) or getattr(dna, "risk_switches", ()):
        return Expressibility(False, "management/risk switches are not implemented in the kernel")
    from engine.research.backtest.switch_gates import READY_BIAS, READY_FILTERS

    if any(b not in READY_BIAS for b in dna.bias_switches) or any(f not in READY_FILTERS for f in dna.filter_switches):
        return Expressibility(False, "a bias/filter switch that is not READY")
    from engine.research.backtest.signal_switches import READY_SIGNALS

    if any(s not in READY_SIGNALS for s in getattr(dna, "signal_switches", ())):
        return Expressibility(False, "a signal switch that is not READY")

    # Config knobs the kernel does not read. Present and non-default
    # means the slow engine would do something the kernel would not.
    if getattr(config, "charge_swap", False) and getattr(config, "swap_mode", 1) not in (None, 1):
        return Expressibility(False, "swap_mode other than points is not implemented")

    return Expressibility(True)


def _slippage_draw_budget(n_bars: int) -> int:
    """How many draws the run can possibly consume.

    engine.py calls _simulate_fill once per intent processed, and the
    strategy emits at most one intent per bar, so the run cannot need
    more than one per bar plus the final-bar path. Over-allocating is
    free; running out mid-run would desync every later fill, so the
    budget is generous and the kernel reports what it used.
    """
    return int(n_bars) + 16


def run_fast(
    prepped: PreppedCell,
    *,
    fast: int = 0,
    slow: int = 0,
    volume: float,
    sl_atr: float,
    tp_atr: float,
    initial_equity: float = 10_000.0,
    max_spread_points: float | None = None,
    daily_loss_limit_pct: float | None = None,
    friday_close_hour: int | None = None,
    breakeven_at_r: float | None = None,
    sl_reduce_at_r: float | None = None,
    sl_lock_at_r: float | None = None,
    atr_trail: tuple[float, float] | None = None,
    friday_close_profit_pct: float | None = None,
    eod_close_hour: int | None = None,
    charge_swap: bool = True,
    charge_commission: bool = True,
    commission_per_lot_per_side: float = 0.0,
    slippage_mean_points: float = 0.0,
    slippage_std_points: float = 0.0,
    rng_seed: int = 0,
    want_equity: bool = True,
    family: FamilyArrays = MA_CROSS_ARRAYS,
    gates: tuple[np.ndarray, np.ndarray] | None = None,
    trades_out: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Run one candidate through the kernel.

    Returns (record, equity, status). Status 0 is success; anything
    negative means the caller must fall back. ``fast``/``slow`` are the
    MA_CROSS genes; any other family passes its prep.family_arrays().
    ``gates`` is (long_ok, short_ok) from gates.gate_arrays for a gated DNA (None: every entry may open).
    ``trades_out`` (rows x TR_SIZE) receives the trade log when given.
    """
    n = prepped.n_bars
    if gates is None:
        open_gate = np.ones(n, dtype=np.bool_)
        gates = (open_gate, open_gate)
    # Drawn in numpy, from the same generator and seed the slow engine
    # uses, so the kernel consumes the identical sequence. numba's RNG
    # is a different generator and could not reproduce it.
    rng = np.random.default_rng(rng_seed)
    if slippage_std_points == 0.0:
        draws = np.zeros(1, dtype=np.float64)
    else:
        draws = rng.normal(
            loc=slippage_mean_points,
            scale=slippage_std_points,
            size=_slippage_draw_budget(n),
        )

    record = np.zeros(REC_SIZE, dtype=np.float64)
    equity = np.zeros(n + 1, dtype=np.float64) if want_equity else np.zeros(0)

    status = run_candidate(
        prepped.open, prepped.high, prepped.low, prepped.close,
        prepped.bar_hour_index, prepped.day_index, prepped.is_friday,
        prepped.no_trade_mask, prepped.bar_spread_points, prepped.atr14,
        prepped.swap_multiplier, prepped.day_index_min,
        int(fast), int(slow), float(volume), float(sl_atr), float(tp_atr),
        int(family.family), family.ind_a, family.ind_b, int(family.warmup),
        float(family.thr_lo), float(family.thr_hi), bool(family.midline),
        gates[0], gates[1],
        float(initial_equity),
        float("nan") if max_spread_points is None else float(max_spread_points),
        float("nan") if daily_loss_limit_pct is None else float(daily_loss_limit_pct),
        -1 if friday_close_hour is None else int(friday_close_hour),
        float("nan") if breakeven_at_r is None else float(breakeven_at_r),
        np.array([float("nan") if sl_reduce_at_r is None else float(sl_reduce_at_r),
                  float("nan") if sl_lock_at_r is None else float(sl_lock_at_r),
                  float("nan") if atr_trail is None else float(atr_trail[0]),
                  float("nan") if atr_trail is None else float(atr_trail[1]),
                  float("nan") if friday_close_profit_pct is None else float(friday_close_profit_pct),
                  float("nan") if eod_close_hour is None else float(eod_close_hour)], dtype=np.float64),
        bool(charge_swap), bool(charge_commission),
        float(commission_per_lot_per_side),
        float(slippage_mean_points), float(slippage_std_points),
        float(prepped.slippage_cap_points),
        float(prepped.point), int(prepped.digits),
        prepped.fill_spread_price,
        float(prepped.per_unit_profit), float(prepped.per_unit_loss),
        float(prepped.swap_long_points), float(prepped.swap_short_points),
        float(prepped.volume_min), float(prepped.volume_max),
        float(prepped.volume_step),
        draws, record, equity,
        np.zeros((0, TR_SIZE), dtype=np.float64) if trades_out is None else trades_out,
    )
    return record, equity, status


def record_to_dict(record: np.ndarray) -> dict[str, float]:
    """The aggregate, named. Counts come back as ints."""
    counts = {
        "n_trades": REC_N_TRADES,
        "n_round_trips": REC_N_ROUND_TRIPS,
        "n_open": REC_N_OPEN,
        "blocked_zero_spread_events": REC_BLOCKED_ZERO_SPREAD,
        "deferred_zero_spread_fills": REC_DEFERRED_ZERO_SPREAD_FILLS,
        "sltp_deferred_zero_spread": REC_SLTP_DEFERRED_ZERO_SPREAD,
        "blocked_max_spread": REC_BLOCKED_MAX_SPREAD,
        "blocked_daily_loss": REC_BLOCKED_DAILY_LOSS,
        "friday_flattens": REC_FRIDAY_FLATTENS,
    }
    out: dict[str, float] = {k: int(record[i]) for k, i in counts.items()}
    out["cost_commission"] = float(record[REC_COST_COMMISSION])
    out["cost_spread"] = float(record[REC_COST_SPREAD])
    out["cost_slippage"] = float(record[REC_COST_SLIPPAGE])
    out["cost_swap"] = float(record[REC_COST_SWAP])
    out["final_equity"] = float(record[REC_FINAL_EQUITY])
    return out


def record_exit_reasons(record: np.ndarray) -> dict[str, int]:
    """The exit partition, in the slow path's own vocabulary.

    Zero-count reasons are omitted, because engine.py's dict only
    contains reasons that actually happened and the two are compared
    directly.
    """
    pairs = (
        ("SIGNAL", REC_EXIT_SIGNAL),
        ("SL", REC_EXIT_SL),
        ("TP", REC_EXIT_TP),
        ("EOD", REC_EXIT_EOD),
        ("SLTP_DEFERRED_ZERO_SPREAD", REC_EXIT_SLTP_DEFERRED),
        ("FRIDAY_CLOSE", REC_EXIT_FRIDAY_CLOSE),
    )
    return {name: int(record[i]) for name, i in pairs if int(record[i]) > 0}


__all__ = [
    "SUPPORTED_EXEC_MODES",
    "SUPPORTED_MGMT",
    "SUPPORTED_SIGNALS",
    "Expressibility",
    "expressible",
    "record_exit_reasons",
    "record_to_dict",
    "run_fast",
]


@dataclass
class BatchOutcome:
    """What a batch produced, including what it skipped and why."""

    records: np.ndarray                 # (n, REC_SIZE)
    equity: np.ndarray                  # (n, n_bars+1) or (0, 0)
    status: np.ndarray                  # (n,) int64
    dedupe_hits: int = 0
    fallback_count: int = 0
    fallback_reasons: tuple[str, ...] = ()


def run_batch(
    prepped: PreppedCell,
    genes: list[tuple[int, int, float, float, float]],
    *,
    initial_equity: float = 10_000.0,
    max_spread_points: float | None = None,
    daily_loss_limit_pct: float | None = None,
    friday_close_hour: int | None = None,
    charge_swap: bool = True,
    charge_commission: bool = True,
    commission_per_lot_per_side: float = 0.0,
    slippage_mean_points: float = 0.0,
    slippage_std_points: float = 0.0,
    rng_seed: int = 0,
    want_equity: bool = True,
    n_threads: int | None = None,
) -> BatchOutcome:
    """Evaluate many candidates in one kernel call.

    ``genes`` is (fast, slow, volume, sl_atr, tp_atr) per candidate.

    Each candidate gets its OWN pre-drawn slippage row, seeded exactly
    as the slow engine seeds that candidate. Sharing one draw cursor
    across threads would give a candidate different slippage depending
    on which thread reached it first -- the results would still be
    self-consistent and would silently stop matching the oracle.
    """
    from numba import set_num_threads

    from engine.research.backtest.kernel import REC_SIZE, eval_batch

    if n_threads is not None:
        set_num_threads(int(n_threads))

    n = len(genes)
    n_bars = prepped.n_bars
    g_fast = np.array([g[0] for g in genes], dtype=np.int64)
    g_slow = np.array([g[1] for g in genes], dtype=np.int64)
    g_volume = np.array([g[2] for g in genes], dtype=np.float64)
    g_sl = np.array([g[3] for g in genes], dtype=np.float64)
    g_tp = np.array([g[4] for g in genes], dtype=np.float64)

    budget = _slippage_draw_budget(n_bars) if slippage_std_points != 0.0 else 1
    draws = np.zeros((n, budget), dtype=np.float64)
    if slippage_std_points != 0.0:
        for c in range(n):
            # Per candidate, from the seed the slow engine would use.
            draws[c] = np.random.default_rng(rng_seed).normal(
                loc=slippage_mean_points, scale=slippage_std_points, size=budget
            )

    records = np.zeros((n, REC_SIZE), dtype=np.float64)
    equity = (
        np.zeros((n, n_bars + 1), dtype=np.float64) if want_equity
        else np.zeros((0, 0), dtype=np.float64)
    )
    status = np.zeros(n, dtype=np.int64)

    eval_batch(
        prepped.open, prepped.high, prepped.low, prepped.close,
        prepped.bar_hour_index, prepped.day_index, prepped.is_friday,
        prepped.no_trade_mask, prepped.bar_spread_points, prepped.atr14,
        prepped.swap_multiplier, prepped.day_index_min,
        g_fast, g_slow, g_volume, g_sl, g_tp,
        float(initial_equity),
        float("nan") if max_spread_points is None else float(max_spread_points),
        float("nan") if daily_loss_limit_pct is None else float(daily_loss_limit_pct),
        -1 if friday_close_hour is None else int(friday_close_hour),
        bool(charge_swap), bool(charge_commission),
        float(commission_per_lot_per_side),
        float(slippage_mean_points), float(slippage_std_points),
        float(prepped.slippage_cap_points),
        float(prepped.point), int(prepped.digits),
        prepped.fill_spread_price,
        float(prepped.per_unit_profit), float(prepped.per_unit_loss),
        float(prepped.swap_long_points), float(prepped.swap_short_points),
        float(prepped.volume_min), float(prepped.volume_max),
        float(prepped.volume_step),
        draws, records, equity, status,
    )
    return BatchOutcome(records=records, equity=equity, status=status)


class DedupeCache:
    """One backtest per DISTINCT projection, one row per DNA.

    W2 established that two DNAs differing only in genes the engine
    ignores are the same backtest with two names. Running both is
    wasted work; reporting only one would lose a trial the luck bar
    counted. So the result is computed once and written twice, and the
    reuse is counted rather than inferred.

    Keyed on effective_gene_hash, which hashes exactly the projection
    dna_to_ma_config hands the engine.
    """

    def __init__(self) -> None:
        self._by_hash: dict[str, int] = {}
        self.hits = 0

    def index_for(self, effective_hash: str, next_index: int) -> tuple[int, bool]:
        """(index of the computed result, whether this was a reuse)."""
        existing = self._by_hash.get(effective_hash)
        if existing is not None:
            self.hits += 1
            return existing, True
        self._by_hash[effective_hash] = next_index
        return next_index, False

    def __len__(self) -> int:
        return len(self._by_hash)
