"""DNA: serializable, hashable, versioned strategy description.

The DNA is a typed dict. The hash is stable per version. Any change to
the gene space produces a new version and a new hash. DNA carries the
lookahead guard flag and the embargo parameters used by I4.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

DNA_SCHEMA_VERSION = 4  # W4-2: switch-genome fields (bias_switches/filter_switches/mgmt_switches/risk_switches)

# W4-2: V4 genome = V1 family choice (signal) + on/off switch lists.
# Switches are research on/off genes (registry groups bias/filter/management/risk).
# A DNA carries the ON switches; off switches are absent -> bit-stable hash:
# every DNA bred before the switches carries [] for all of them (empty list).
BIAS_SWITCHES: tuple[str, ...] = (
    "sma", "rsi", "trailing", "momentum", "chandelier", "donchian_mid", "market_struct",
    "daily_mid", "vwap", "adx", "ema", "psar", "htf",
)
FILTER_SWITCHES: tuple[str, ...] = (
    "sma", "rsi", "bb", "cci", "receding", "doji", "no_open_friday", "volatility", "consec",
    "adr_exhaust", "adx", "keltner", "regime", "event",
)
MANAGEMENT_SWITCHES: tuple[str, ...] = (
    "breakeven", "sl_reduce", "sl_lock", "atr_trail", "partial_tp", "friday_close_profit", "eod_close",
)
RISK_SWITCHES: tuple[str, ...] = (
    "daily_dd", "friday_close", "max_agg_risk", "news_filter", "no_open_news_day",
    "profit_target", "slippage", "symbol_lock",
)
# v4 batch 2.6: the research entry signals (``use_sig_*``) as switches of the SIGNALS family
# (backtest/signal_switches.py). SIGNALS is not in SIGNAL_VALUES on purpose: the sampler draws it only when a gene
# pool turns the ``signal_switches`` gate on, so every pool bred before it draws exactly as it did.
SIGNAL_SWITCHES: tuple[str, ...] = (
    "macd", "cci", "mom_break", "engulfing", "inside_break", "wick_rejection", "fib", "breakout", "liqsweep",
    "atr_run", "fade", "vwap", "orb", "squeeze", "three_soldiers", "rsi", "bb", "pin_bar", "stoch", "williams",
    "kalman",
)
SWITCH_FAMILY = "SIGNALS"

BIAS_VALUES: tuple[str, ...] = ("DAILY_TREND", "H4_TREND", "WEEKLY_TREND", "NONE")
FILTER_VALUES: tuple[str, ...] = ("CCI", "ADX", "ATR_PCT", "RANGE_COMP", "NONE")
# B4: the DNA signal gene is restricted to the enabled family set. A family
# in SIGNAL_VALUES but missing its feature backing is rejected at DNA
# construction (FAMILY_FEATURES lookup in registry.py). The allowlist here is
# the contract — adding a family forces wiring a feature, the kernel branch,
# and a parity fixture before it can reach a trial.
#
#   MA_CROSS          — gate strategy (kernel). Fast-path expressible.
#   RSI_REVERSION     — first non-MA family (W1). Slow engine only.
#   STOCH_CROSS       — W1 (pending).
#   MACD_CROSS        — W1 (pending).
#   DONCHIAN_BREAKOUT — W1 (pending).
#
# Phantom families (the old STOCH, MACD, DONCHIAN, M_BREAK, ENGULFING, BB_REV)
# are GONE: they were never wired and never produced a trial, but they
# polluted the gene space with names that read as implemented.
SIGNAL_VALUES: tuple[str, ...] = (
    "MA_CROSS", "RSI_REVERSION",
    "STOCH_CROSS", "MACD_CROSS", "DONCHIAN_BREAKOUT",
)
SIGNAL2_VALUES: tuple[str, ...] = SIGNAL_VALUES + ("",)
EXEC_VALUES: tuple[str, ...] = ("MARKET2",)
RISK_VALUES: tuple[str, ...] = ("FIXED_ATR",)
MGMT_VALUES: tuple[str, ...] = ("NONE", "BREAKEVEN", "TRAIL_ATR", "PARTIAL")
# ARQVIO entry gates (engine/research/backtest/gates.py): "" / 0 is off, the value every DNA bred before them carries.
SESSION_VALUES: tuple[str, ...] = ("", "ASIA", "LONDON", "NEWYORK", "LONDON_NY")
TREND_EMA_VALUES: tuple[int, ...] = (0, 50, 100, 200)


class FailureTaxonomy(str, Enum):
    """Failure categories the funnel emits per trial."""

    EVALUATED = "EVALUATED"
    LowTrades = "LowTrades"
    HighDD = "HighDD"
    NegReturn = "NegReturn"
    FailedDSR = "FailedDSR"
    HighPBO = "HighPBO"
    DsrUnreliable = "DsrUnreliable"
    SpreadUncalibrated = "SpreadUncalibrated"
    Crash = "Crash"
    VAULTED = "VAULTED"


@dataclass(frozen=True)
class DNA:
    """Versioned, hashable strategy description."""

    bias: str
    filter: str
    signal: str
    signal2: str
    exec_mode: str
    risk: str
    mgmt: str
    sl_atr: float
    tp_atr: float
    risk_pct: float
    friday_close: bool
    news_filter: bool
    daily_dd_limit: float
    max_agg_risk: float
    max_feature_lookback: int
    max_hold_time: int
    window_start_hour_utc: int
    window_end_hour_utc: int
    # B2: gene-varying RSI period, uniform in [5, 21]. Only consumed when
    # signal == "RSI_REVERSION"; MA_CROSS ignores it.
    rsi_period: int = 14
    # C3-A': RSI threshold genes. oversold in [10, 35], overbought in [65, 90].
    # Non-overlapping ranges guarantee oversold < overbought.
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    # C3-A': rsi_exit_midline — when 1, an open position exits when RSI crosses
    # the 50 midline (mean-reversion signal that the move has run its course).
    rsi_exit_midline: bool = False
    # W1: STOCH_CROSS gene-varying periods. k_period in [5, 21], d_period in [3, 5],
    # smoothing in [1, 5]. Consumed only when signal == "STOCH_CROSS".
    stoch_k_period: int = 14
    stoch_d_period: int = 3
    stoch_smoothing: int = 3
    # C3-A': STOCH threshold genes. oversold in [5, 25], overbought in [75, 95].
    stoch_oversold: float = 20.0
    stoch_overbought: float = 80.0
    # W1: MACD_CROSS gene-varying periods. fast in [5, 21], slow in [22, 52],
    # signal_period in [5, 13]. Consumed only when signal == "MACD_CROSS".
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal_period: int = 9
    # W1: DONCHIAN_BREAKOUT gene-varying period, uniform in [10, 55].
    # Consumed only when signal == "DONCHIAN_BREAKOUT".
    donchian_period: int = 20
    uses_future_bar: bool = False
    schema_version: int = DNA_SCHEMA_VERSION
    # ARQVIO entry gates: an entry opens only inside the session (UTC hours) and on the side of the trend EMA.
    # Off by default; an off gate is absent from to_dict, so every DNA bred before the gates keeps its hash.
    session: str = ""
    trend_ema: int = 0
    # W4-2: V4 switch genome. ON switches only; off switches are omitted (defaults to []).
    # Empty lists for all four keep V3 dna_hash bit-identical, so bred trials keep their IDs.
    bias_switches: tuple[str, ...] = ()
    filter_switches: tuple[str, ...] = ()
    mgmt_switches: tuple[str, ...] = ()
    risk_switches: tuple[str, ...] = ()
    # v4 batch 2.6: the SIGNALS family's research entry signals (non-empty exactly when signal == SWITCH_FAMILY)
    signal_switches: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # Structural leakage safety, step 3: a genome naming a family whose
        # features are not registered is refused HERE, at construction -- not
        # later in validation, by which point it would already have been
        # scored, stored and reported.
        from engine.research.features.registry import require_registered_features

        require_registered_features(
            (self.bias, self.filter, self.signal, self.signal2, self.mgmt)
        )
        # W4-2: validate switch genes against the research vocabulary.
        _validate_switches(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "bias": self.bias,
            "filter": self.filter,
            "signal": self.signal,
            "signal2": self.signal2,
            "exec_mode": self.exec_mode,
            "risk": self.risk,
            "mgmt": self.mgmt,
            "sl_atr": float(self.sl_atr),
            "tp_atr": float(self.tp_atr),
            "risk_pct": float(self.risk_pct),
            "friday_close": bool(self.friday_close),
            "news_filter": bool(self.news_filter),
            "daily_dd_limit": float(self.daily_dd_limit),
            "max_agg_risk": float(self.max_agg_risk),
            "max_feature_lookback": int(self.max_feature_lookback),
            "max_hold_time": int(self.max_hold_time),
            "window_start_hour_utc": int(self.window_start_hour_utc),
            "window_end_hour_utc": int(self.window_end_hour_utc),
            "rsi_period": int(self.rsi_period),
            "rsi_oversold": float(self.rsi_oversold),
            "rsi_overbought": float(self.rsi_overbought),
            "rsi_exit_midline": bool(self.rsi_exit_midline),
            "stoch_k_period": int(self.stoch_k_period),
            "stoch_d_period": int(self.stoch_d_period),
            "stoch_smoothing": int(self.stoch_smoothing),
            "stoch_oversold": float(self.stoch_oversold),
            "stoch_overbought": float(self.stoch_overbought),
            "macd_fast": int(self.macd_fast),
            "macd_slow": int(self.macd_slow),
            "macd_signal_period": int(self.macd_signal_period),
            "donchian_period": int(self.donchian_period),
            "uses_future_bar": bool(self.uses_future_bar),
            **({"bias_switches": list(self.bias_switches)} if self.bias_switches else {}),
            **({"filter_switches": list(self.filter_switches)} if self.filter_switches else {}),
            **({"mgmt_switches": list(self.mgmt_switches)} if self.mgmt_switches else {}),
            **({"risk_switches": list(self.risk_switches)} if self.risk_switches else {}),
            **({"signal_switches": list(self.signal_switches)} if self.signal_switches else {}),
            **({"session": str(self.session)} if self.session else {}),
            **({"trend_ema": int(self.trend_ema)} if self.trend_ema else {}),
        }

    @property
    def gated(self) -> bool:
        """True when an entry gate is on."""
        return bool(self.session) or bool(self.trend_ema) or bool(self.bias_switches) or bool(self.filter_switches)

    def ungated(self) -> DNA:
        """The same genome with its entry gates off (session, trend EMA, bias and filter switches): the family
        strategy the gates wrap."""
        return dataclasses.replace(self, session="", trend_ema=0, bias_switches=(), filter_switches=())

    @property
    def dna_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]

    @property
    def embargo_bars(self) -> int:
        """Embargo derived from the strategy's DNA - never a fixed percentage."""
        return int(self.max_feature_lookback) + int(self.max_hold_time)


def is_lookahead_dna(d: dict[str, Any] | DNA) -> bool:
    """The runner-side guard: any DNA that uses a future bar is rejected."""
    if isinstance(d, DNA):
        return bool(d.uses_future_bar)
    return bool(d.get("uses_future_bar", False))


def dna_embargo_bars(d: dict[str, Any] | DNA) -> int:
    if isinstance(d, DNA):
        return d.embargo_bars
    return int(d.get("max_feature_lookback", 0)) + int(d.get("max_hold_time", 0))


def _validate_switches(d: DNA) -> None:
    """W4-2: a V4 DNA names only valid research switch genes (no typos, no cross-group)."""
    for field, table in (
        ("bias_switches", BIAS_SWITCHES),
        ("filter_switches", FILTER_SWITCHES),
        ("mgmt_switches", MANAGEMENT_SWITCHES),
        ("risk_switches", RISK_SWITCHES),
        ("signal_switches", SIGNAL_SWITCHES),
    ):
        got = getattr(d, field)
        seen: set[str] = set()
        table_set = set(table)
        for name in got:
            if name not in table_set:
                raise ValueError(f"invalid {field}: {name!r} not in {table}")
            if name in seen:
                raise ValueError(f"{field} has a duplicate switch {name!r}")
            seen.add(name)
    if (d.signal == SWITCH_FAMILY) != bool(d.signal_switches):
        raise ValueError(f"signal_switches must be non-empty exactly for the {SWITCH_FAMILY} family "
                         f"(signal={d.signal!r}, signal_switches={list(d.signal_switches)})")


def validate_gene_space(d: dict[str, Any]) -> None:
    """Reject DNAs whose values are outside the documented gene space."""
    if d.get("bias") not in BIAS_VALUES:
        raise ValueError(f"invalid bias: {d.get('bias')!r}")
    if d.get("filter") not in FILTER_VALUES:
        raise ValueError(f"invalid filter: {d.get('filter')!r}")
    if d.get("signal") not in (*SIGNAL_VALUES, SWITCH_FAMILY):
        raise ValueError(f"invalid signal: {d.get('signal')!r}")
    if d.get("signal2", "") not in SIGNAL2_VALUES:
        raise ValueError(f"invalid signal2: {d.get('signal2')!r}")
    if d.get("exec_mode") not in EXEC_VALUES:
        raise ValueError(f"invalid exec_mode: {d.get('exec_mode')!r}")
    if d.get("risk") not in RISK_VALUES:
        raise ValueError(f"invalid risk: {d.get('risk')!r}")
    if d.get("mgmt") not in MGMT_VALUES:
        raise ValueError(f"invalid mgmt: {d.get('mgmt')!r}")
    period = d.get("rsi_period")
    if period is not None and not (5 <= int(period) <= 21):
        raise ValueError(f"invalid rsi_period: {period!r} not in [5, 21]")
    stoch_k = d.get("stoch_k_period")
    if stoch_k is not None and not (5 <= int(stoch_k) <= 21):
        raise ValueError(f"invalid stoch_k_period: {stoch_k!r} not in [5, 21]")
    stoch_d = d.get("stoch_d_period")
    if stoch_d is not None and not (3 <= int(stoch_d) <= 5):
        raise ValueError(f"invalid stoch_d_period: {stoch_d!r} not in [3, 5]")
    stoch_s = d.get("stoch_smoothing")
    if stoch_s is not None and not (1 <= int(stoch_s) <= 5):
        raise ValueError(f"invalid stoch_smoothing: {stoch_s!r} not in [1, 5]")
    mf = d.get("macd_fast")
    if mf is not None and not (5 <= int(mf) <= 21):
        raise ValueError(f"invalid macd_fast: {mf!r} not in [5, 21]")
    ms = d.get("macd_slow")
    if ms is not None and not (22 <= int(ms) <= 52):
        raise ValueError(f"invalid macd_slow: {ms!r} not in [22, 52]")
    if mf is not None and ms is not None and int(ms) - int(mf) < 5:
        raise ValueError(
            f"invalid macd gap: slow_fast_gap={int(ms) - int(mf)} < 5 "
            f"(macd_fast={mf!r}, macd_slow={ms!r})"
        )
    msp = d.get("macd_signal_period")
    if msp is not None and not (5 <= int(msp) <= 13):
        raise ValueError(f"invalid macd_signal_period: {msp!r} not in [5, 13]")
    dp = d.get("donchian_period")
    if dp is not None and not (10 <= int(dp) <= 55):
        raise ValueError(f"invalid donchian_period: {dp!r} not in [10, 55]")
    rsi_os = d.get("rsi_oversold")
    if rsi_os is not None and not (10.0 <= float(rsi_os) <= 35.0):
        raise ValueError(f"invalid rsi_oversold: {rsi_os!r} not in [10, 35]")
    rsi_ob = d.get("rsi_overbought")
    if rsi_ob is not None and not (65.0 <= float(rsi_ob) <= 90.0):
        raise ValueError(f"invalid rsi_overbought: {rsi_ob!r} not in [65, 90]")
    st_os = d.get("stoch_oversold")
    if st_os is not None and not (5.0 <= float(st_os) <= 25.0):
        raise ValueError(f"invalid stoch_oversold: {st_os!r} not in [5, 25]")
    st_ob = d.get("stoch_overbought")
    if st_ob is not None and not (75.0 <= float(st_ob) <= 95.0):
        raise ValueError(f"invalid stoch_overbought: {st_ob!r} not in [75, 95]")
    if d.get("session", "") not in SESSION_VALUES:
        raise ValueError(f"invalid session: {d.get('session')!r} not in {SESSION_VALUES}")
    if d.get("trend_ema", 0) not in TREND_EMA_VALUES:
        raise ValueError(f"invalid trend_ema: {d.get('trend_ema')!r} not in {TREND_EMA_VALUES}")
    # W4-2: validate V4 switch genes when present (schema >= 4).
    if d.get("schema_version", DNA_SCHEMA_VERSION) >= 4:
        for field, table in (
            ("bias_switches", BIAS_SWITCHES), ("filter_switches", FILTER_SWITCHES),
            ("mgmt_switches", MANAGEMENT_SWITCHES), ("risk_switches", RISK_SWITCHES),
            ("signal_switches", SIGNAL_SWITCHES),
        ):
            for name in d.get(field, ()) or ():
                if name not in table:
                    raise ValueError(f"invalid {field}: {name!r} not in {table}")


__all__ = [
    "BIAS_SWITCHES",
    "DNA",
    "DNA_SCHEMA_VERSION",
    "FILTER_SWITCHES",
    "FailureTaxonomy",
    "MANAGEMENT_SWITCHES",
    "RISK_SWITCHES",
    "SESSION_VALUES",
    "SIGNAL_SWITCHES",
    "SWITCH_FAMILY",
    "TREND_EMA_VALUES",
    "dna_embargo_bars",
    "is_lookahead_dna",
    "validate_gene_space",
]
