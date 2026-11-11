"""Effective gene hash -- which trials were actually independent.

The luck bar a Deflated Sharpe sets scales with the number of INDEPENDENT
trials. Feed it the genome count and it defends against a search that
never happened: two DNAs that differ only in a gene the engine ignores
are one backtest with two names, and counting them twice raises the bar
a real edge has to clear without a second real test having been run.

The C5 audit found exactly that. Five DNAs differing only in sl_atr,
tp_atr and mgmt produced five distinct ``dna_hash`` values and
byte-identical results, because ``dna_to_ma_config`` dropped all three.
W1 connected sl_atr and tp_atr; mgmt stays pinned and unimplemented.

So there are two counts, and they answer different questions:

    trials_raw        how many genomes were evaluated
    trials_effective  how many DISTINCT backtests that amounted to

``effective_gene_hash`` hashes the PROJECTION -- exactly the values
``dna_to_ma_config`` hands the engine -- so two DNAs share a hash if and
only if they produce the same run. The projection is duplicated here
rather than imported to avoid a circular import; the pairing is pinned
by ``test_effective_hash_tracks_the_projection``, which fails if the two
drift apart.

The float fields are hashed at full precision. That can only ever
OVERCOUNT (two stops differing in the twelfth decimal are effectively
one backtest but hash apart), and overcounting keeps the luck bar
higher. Undercounting would lower it, which is the direction that
manufactures significance.

FAMILY-AWARE PROJECTION (v3 — threshold genes):

    effective_genes(dna) = (VERSION, dna.signal, *family_genes(dna.signal), *common_genes)

where family_genes() is the set of engine-reaching parameters for that
family only, and common_genes() is the tail every family consumes:
(volume, sl_atr, tp_atr), then -- only when one is on -- the ARQVIO entry
gates (session, trend_ema). Genes sampled but not reaching the engine
(bias, filter, signal2, exec_mode, risk, mgmt, friday_close,
news_filter, daily_dd_limit, max_agg_risk, max_hold_time, the
trading-window hours, rsi_period when signal != RSI_REVERSION, etc.)
are EXCLUDED. The family string is the first gene so that two DNAs from
different families with otherwise identical projections never collide:
the dedupe cache must never group an MA_CROSS candidate with an
RSI_REVERSION candidate that happens to share a period.
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterable

from engine.core.errors import DataError
from engine.research.search.dna import DNA

# Bumped when the projection changes shape, so hashes from two different
# engine versions can never be compared as if they meant the same thing.
EFFECTIVE_HASH_VERSION = 4


# ---------------------------------------------------------------------------
# Per-family gene projections.
#
# These mirror the engine-relevant fields each ``dna_to_*_config`` function
# hands to its strategy. Adding a family to this dict forces wiring the
# strategy, the config, and a parity fixture before it can contribute to
# ``effective_gene_hash`` -- the contract from PART 1.A.
# ---------------------------------------------------------------------------
def _ma_cross_genes(dna: DNA) -> tuple[object, ...]:
    fast = max(5, int(dna.max_feature_lookback / 4))
    slow = max(fast + 5, int(dna.max_feature_lookback))
    return (fast, slow)


def _rsi_genes(dna: DNA) -> tuple[object, ...]:
    return (
        int(dna.rsi_period),
        float(dna.rsi_oversold),
        float(dna.rsi_overbought),
        bool(dna.rsi_exit_midline),
    )


def _stoch_genes(dna: DNA) -> tuple[object, ...]:
    return (
        int(dna.stoch_k_period),
        int(dna.stoch_d_period),
        int(dna.stoch_smoothing),
        float(dna.stoch_oversold),
        float(dna.stoch_overbought),
    )


def _macd_genes(dna: DNA) -> tuple[object, ...]:
    return (
        int(dna.macd_fast),
        int(dna.macd_slow),
        int(dna.macd_signal_period),
    )


def _donchian_genes(dna: DNA) -> tuple[object, ...]:
    return (int(dna.donchian_period),)


def _signals_genes(dna: DNA) -> tuple[object, ...]:
    return ("SWITCHES", *dna.signal_switches)


_FAMILY_PROJECTIONS: dict[str, object] = {
    "SIGNALS": _signals_genes,
    "MA_CROSS": _ma_cross_genes,
    "RSI_REVERSION": _rsi_genes,
    "STOCH_CROSS": _stoch_genes,
    "MACD_CROSS": _macd_genes,
    "DONCHIAN_BREAKOUT": _donchian_genes,
}


def _common_genes(dna: DNA) -> tuple[object, ...]:
    """What every family hands to the engine, regardless of signal."""
    volume = max(0.01, min(1.0, float(dna.risk_pct) * 10.0))
    sl_atr = max(0.0, float(dna.sl_atr))
    tp_atr = max(0.0, float(dna.tp_atr))
    return (volume, sl_atr, tp_atr)


def effective_genes(dna: DNA) -> tuple[object, ...]:
    """The values that actually reach the engine, in a fixed order.

    Projection per DNA = (VERSION, dna.signal, *family_genes(signal), *common_genes).
    Genes absent from this tuple are genes the backtest cannot see. Only the
    genes that reach the engine are included, and the family string comes first
    so that DNAs from different families never collide in the dedupe cache.
    """
    proj = _FAMILY_PROJECTIONS.get(dna.signal)
    if proj is None:
        # A signal not in the projection table is unbacked; it must not get
        # an effective hash, or a phantom family would silently join a trial.
        raise DataError(
            f"effective_genes: signal {dna.signal!r} has no family projection; "
            f"add it to _FAMILY_PROJECTIONS in effective_hash.py"
        )
    family_genes = proj(dna)
    common = _common_genes(dna)
    # The ARQVIO entry gates reach the engine only when on; an ungated DNA keeps the projection it always had.
    gates = ("GATES", str(dna.session), int(dna.trend_ema)) if dna.gated else ()
    # v4 switches reach the engine too (gate arrays, the breakeven step): 2026-11-01 they were missing here, so two
    # DNAs differing only in their switches shared one hash -- the run registry refused the second as already tested
    # and the trial count undercounted. Added only when present: a DNA without switches keeps its hash.
    sw = tuple((f, tuple(getattr(dna, f))) for f in ("bias_switches", "filter_switches", "mgmt_switches",
                                                     "risk_switches") if getattr(dna, f))
    switches = ("SWITCHES", *sw) if sw else ()
    return (EFFECTIVE_HASH_VERSION, dna.signal, *family_genes, *common, *gates, *switches)


def effective_gene_hash(dna: DNA) -> str:
    """Stable 16-hex-char digest of the projected genes."""
    payload = "|".join(repr(part) for part in effective_genes(dna))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def count_effective_trials(dnas: Iterable[DNA]) -> int:
    """How many DISTINCT backtests this population amounts to."""
    return len({effective_gene_hash(d) for d in dnas})


__all__ = [
    "EFFECTIVE_HASH_VERSION",
    "count_effective_trials",
    "effective_gene_hash",
    "effective_genes",
]
