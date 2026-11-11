"""Random sampler over the gene space.

The sampler draws DNA tuples from a uniform distribution over each gene.
It is deliberately simple: this is the random-search step. If random
search finds no structure, a GA will overfit harder - never better.

A gene pool (engine/research/search/gene_pool.py) can narrow the draw:
families switched off or reweighted, individual genes fixed. Without one
the draw is the pre-W2 stream, value for value and draw for draw, and so
is the draw under the default pool (every family enabled at equal weight,
nothing fixed): an API campaign on the default pool reproduces a CLI run
of the same seed DNA for DNA.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from engine.research.backtest.strategies.switch import READY_MGMT
from engine.research.backtest.switch_gates import READY_BIAS, READY_FILTERS
from engine.research.search.dna import (
    SWITCH_FAMILY,
    BIAS_VALUES,
    DNA,
    DNA_SCHEMA_VERSION,
    EXEC_VALUES,
    FILTER_VALUES,
    RISK_VALUES,
    SESSION_VALUES,
    SIGNAL2_VALUES,
    SIGNAL_VALUES,
    TREND_EMA_VALUES,
)

if TYPE_CHECKING:
    from engine.research.search.gene_pool import GenePool

# MA_CROSS lookback gene, inclusive bounds.
MAX_FEATURE_LOOKBACK_RANGE: tuple[int, int] = (10, 199)

# The genes every family hands the engine (effective_hash._common_genes).
# rng.uniform draws the half-open [lo, hi).
SL_ATR_RANGE: tuple[float, float] = (1.0, 4.0)
TP_ATR_RANGE: tuple[float, float] = (1.0, 6.0)
RISK_PCT_RANGE: tuple[float, float] = (0.005, 0.02)

# B2: rsi_period gene range. Uniform over {5..21} inclusive.
RSI_PERIOD_VALUES: tuple[int, ...] = tuple(range(5, 22))
# C3-A': threshold genes. Non-overlapping halves guarantee oversold <
# overbought by construction.
RSI_OVERSOLD_RANGE: tuple[float, float] = (10.0, 35.0)
RSI_OVERBOUGHT_RANGE: tuple[float, float] = (65.0, 90.0)

# W1: STOCH_CROSS gene ranges.
STOCH_K_PERIOD_VALUES: tuple[int, ...] = tuple(range(5, 22))
STOCH_D_PERIOD_VALUES: tuple[int, ...] = tuple(range(3, 6))
STOCH_SMOOTHING_VALUES: tuple[int, ...] = tuple(range(1, 6))
STOCH_OVERSOLD_RANGE: tuple[float, float] = (5.0, 25.0)
STOCH_OVERBOUGHT_RANGE: tuple[float, float] = (75.0, 95.0)

# W1: MACD_CROSS gene ranges.
MACD_FAST_VALUES: tuple[int, ...] = tuple(range(5, 22))
MACD_SLOW_VALUES: tuple[int, ...] = tuple(range(22, 53))
MACD_SIGNAL_PERIOD_VALUES: tuple[int, ...] = tuple(range(5, 14))
# C3-1.C: min-gap-5 between macd_slow and macd_fast.
MACD_MIN_GAP = 5


def _macd_slow_choices(fast: int) -> tuple[int, ...]:
    """SLOW values that maintain >= MACD_MIN_GAP from the given FAST."""
    return tuple(ms for ms in MACD_SLOW_VALUES if ms - fast >= MACD_MIN_GAP)


def _macd_fast_choices(slow: int) -> tuple[int, ...]:
    """FAST values that maintain >= MACD_MIN_GAP below the given SLOW."""
    return tuple(mf for mf in MACD_FAST_VALUES if slow - mf >= MACD_MIN_GAP)


# W1: DONCHIAN_BREAKOUT gene range.
DONCHIAN_PERIOD_VALUES: tuple[int, ...] = tuple(range(10, 56))

# The only management type the engine implements is "no management".
# See the comment at the mgmt= line below.
PINNED_MGMT = "NONE"


def _draw_choice(
    rng: np.random.Generator, values: tuple[str, ...], p: tuple[float, ...] | None
) -> str:
    # Equal weights over the full tuple (every value enabled) are the
    # unweighted draw, and must take the unweighted call: Generator.choice
    # with p consumes one double, without p a bounded integer, so the two
    # calls leave the stream in different places and the default pool would
    # otherwise diverge from pool=None.
    if p is None or len(set(p)) == 1:
        return str(rng.choice(values))
    return str(rng.choice(values, p=p))


def random_dna(
    rng: np.random.Generator,
    *,
    allow_lookahead: bool = False,
    gene_pool: GenePool | None = None,
) -> DNA:
    """Draw one DNA tuple. ``allow_lookahead`` is False in production; the
    test for I4 uses True to exercise the detector.

    Fixed genes in ``gene_pool`` are still drawn and then overwritten, so
    fixing one leaves every later draw where it was, with one exception:
    fixing macd_fast or macd_slow draws its partner from the gap-constrained
    set for the fixed value. Generator draws a bounded integer by rejection
    sampling whose rejection threshold depends on the size of the set, so a
    differently sized set can consume a different number of random values
    and shift every later draw (rarely: a rejection has probability below
    1e-8 per draw).
    """
    fixed: dict[str, Any] = {} if gene_pool is None else gene_pool.fixed_genes()
    # C3-1.C: macd_slow must be >= macd_fast + MACD_MIN_GAP. Draw fast first,
    # then sample slow from the constrained set. A fixed slow constrains the
    # fast draw instead. A fixed fast is still drawn (the .get default is
    # evaluated) and discarded, keeping the draw count.
    if "macd_slow" in fixed and "macd_fast" not in fixed:
        macd_fast = int(rng.choice(_macd_fast_choices(fixed["macd_slow"])))
    else:
        macd_fast = fixed.get("macd_fast", int(rng.choice(MACD_FAST_VALUES)))
    macd_slow = int(rng.choice(_macd_slow_choices(macd_fast)))
    genes: dict[str, Any] = {
        "bias": str(rng.choice(BIAS_VALUES)),
        "filter": str(rng.choice(FILTER_VALUES)),
        "signal": _draw_choice(
            rng, SIGNAL_VALUES,
            None if gene_pool is None else gene_pool.signal_weights(),
        ),
        "signal2": _draw_choice(
            rng, SIGNAL2_VALUES,
            None if gene_pool is None else gene_pool.signal2_weights(),
        ),
        "exec_mode": str(rng.choice(EXEC_VALUES)),
        "risk": str(rng.choice(RISK_VALUES)),
        # PINNED to NONE (W1). BREAKEVEN / TRAIL_ATR / PARTIAL are not
        # implemented: dna_to_ma_config never passed mgmt to the engine
        # and still does not, so sampling it produced DNAs that differed
        # in their hash, were stored as distinct trials, and ran the
        # identical backtest -- the C5 finding. Every such duplicate
        # raises the trial count the luck bar is set against without
        # adding a single independent test.
        #
        # Sampling resumes when the management types actually exist.
        # test_mgmt_pinned_none_until_implemented fails the moment a
        # non-NONE mgmt reaches the engine.
        "mgmt": PINNED_MGMT,
        "sl_atr": float(rng.uniform(*SL_ATR_RANGE)),
        "tp_atr": float(rng.uniform(*TP_ATR_RANGE)),
        "risk_pct": float(rng.uniform(*RISK_PCT_RANGE)),
        "friday_close": bool(rng.random() < 0.5),
        "news_filter": bool(rng.random() < 0.3),
        "daily_dd_limit": float(rng.uniform(0.03, 0.10)),
        "max_agg_risk": float(rng.uniform(0.02, 0.10)),
        "max_feature_lookback": int(
            rng.integers(MAX_FEATURE_LOOKBACK_RANGE[0], MAX_FEATURE_LOOKBACK_RANGE[1] + 1)
        ),
        "max_hold_time": int(rng.integers(20, 500)),
        "window_start_hour_utc": int(rng.integers(0, 14)),
        "window_end_hour_utc": int(rng.integers(15, 24)),
        "rsi_period": int(rng.choice(RSI_PERIOD_VALUES)),
        "rsi_oversold": float(rng.uniform(*RSI_OVERSOLD_RANGE)),
        "rsi_overbought": float(rng.uniform(*RSI_OVERBOUGHT_RANGE)),
        "rsi_exit_midline": bool(rng.random() < 0.5),
        "stoch_k_period": int(rng.choice(STOCH_K_PERIOD_VALUES)),
        "stoch_d_period": int(rng.choice(STOCH_D_PERIOD_VALUES)),
        "stoch_smoothing": int(rng.choice(STOCH_SMOOTHING_VALUES)),
        "stoch_oversold": float(rng.uniform(*STOCH_OVERSOLD_RANGE)),
        "stoch_overbought": float(rng.uniform(*STOCH_OVERBOUGHT_RANGE)),
        "macd_fast": macd_fast,
        "macd_slow": macd_slow,
        "macd_signal_period": int(rng.choice(MACD_SIGNAL_PERIOD_VALUES)),
        "donchian_period": int(rng.choice(DONCHIAN_PERIOD_VALUES)),
        "uses_future_bar": bool(rng.random() < 0.05) if allow_lookahead else False,
        "schema_version": DNA_SCHEMA_VERSION,
    }
    genes.update(fixed)
    # The ARQVIO entry gates, drawn last and only when the pool turns them on: an ungated pool consumes exactly
    # the draws it always did, so every campaign bred before the gates samples the same genomes.
    if gene_pool is not None and "session" in gene_pool.gates:
        genes["session"] = str(rng.choice(SESSION_VALUES))
    if gene_pool is not None and "trend_ema" in gene_pool.gates:
        genes["trend_ema"] = int(rng.choice(TREND_EMA_VALUES))
    # v4 batch B: each READY bias / filter switch on with probability SWITCH_P, drawn after the older gates so a
    # pool without them consumes exactly the draws it always did.
    if gene_pool is not None and "bias_switches" in gene_pool.gates:
        genes["bias_switches"] = gene_pool.allowed("bias_switches", draw_switches(rng, READY_BIAS))
    if gene_pool is not None and "filter_switches" in gene_pool.gates:
        genes["filter_switches"] = gene_pool.allowed("filter_switches", draw_switches(rng, READY_FILTERS))
    if gene_pool is not None and "mgmt_switches" in gene_pool.gates:  # v4 batch A, last: older pools draw as before
        genes["mgmt_switches"] = gene_pool.allowed("mgmt_switches", draw_switches(rng, READY_MGMT))
    # v4 batch 2.6, last of all: with the signal_switches gate on, the SIGNALS family (research entry signals) takes
    # an equal share beside the pool's enabled families; a pool without the gate draws exactly as before.
    if gene_pool is not None and "signal_switches" in gene_pool.gates:
        n_fam = sum(1 for w in gene_pool.signal_weights() if w > 0)
        if rng.random() < 1.0 / (n_fam + 1):
            genes.update(signal=SWITCH_FAMILY, signal2="", signal_switches=draw_signal_switches(rng, gene_pool))
    return DNA(**genes)


SWITCH_P = 0.25


def draw_signal_switches(rng: np.random.Generator, pool: Any = None) -> tuple[str, ...]:
    """The SIGNALS family's switches: each READY research signal with probability SWITCH_P, at least one; signals
    the pool turns off are dropped after the draw (the random stream is unchanged)."""
    from engine.research.backtest.signal_switches import READY_SIGNALS

    on = draw_switches(rng, READY_SIGNALS)
    out = on if on else (READY_SIGNALS[int(rng.integers(0, len(READY_SIGNALS)))],)
    if pool is not None:
        out = pool.allowed("signal_switches", out)
        if not out:  # every drawn signal is off: the first one left on (the pool keeps at least one on)
            out = pool.allowed("signal_switches", READY_SIGNALS)[:1]
    return out


def draw_switches(rng: np.random.Generator, ready: tuple[str, ...]) -> tuple[str, ...]:
    on = rng.random(len(ready)) < SWITCH_P
    return tuple(name for name, x in zip(ready, on, strict=True) if x)


def sample(
    n: int, *, seed: int = 0, allow_lookahead: bool = False
) -> list[DNA]:
    rng = np.random.default_rng(seed)
    return [random_dna(rng, allow_lookahead=allow_lookahead) for _ in range(int(n))]


__all__ = ["random_dna", "sample"]
