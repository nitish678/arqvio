"""Level B of the parity harness: the oracle's per-bar indicator values and state-free signal conditions.

``expected_rows(dna, cell)`` is the row the EA writes to ``indicators.csv`` at every bar close, computed on the
Python side with the strategy modules' OWN series functions on the cell's pinned bars:

    i, bar_time, ATR(atr_period) at i (atr_at's rule: empty when undefined or <= 0), family fields...

Family fields (doubles as 16-hex-digit IEEE bit patterns, conditions as 0/1, empty while undefined):

* MA_CROSS: fast_now, slow_now, fast_prev, slow_prev (``np.mean`` over the strategy's slices), cross_up, cross_dn
* RSI_REVERSION: rsi, rsi < oversold, rsi > overbought, midline down-cross, midline up-cross
* STOCH_CROSS: %K, %D, cross_up, k and d <= oversold, cross_dn, k and d >= overbought
* MACD_CROSS: macd line, signal line, cross_up, cross_dn
* DONCHIAN_BREAKOUT: upper, lower, close > upper[i-1], close < lower[i-1]

The conditions are the comparisons the strategies' ``on_bar`` makes, without position state (state-dependent
decisions are level C). The series are causal, so the full-window value at ``i`` is the value the strategy sees on
its visible prefix ``[0, i]``; ``prefix_mismatches`` proves that through the strategy's own accessor (its caches
included) on every prefix of windows up to ``PREFIX_ALL`` bars, and on ``PREFIX_SAMPLES`` prefixes drawn with
``default_rng(PREFIX_SEED)`` beyond (plus the first and the last).

``primitive_inputs(cell)`` is the seeded vector the EA quantizes and normalises at init (``prims.csv``);
``primitive_expected`` is the engine's ``round(price, digits)`` and ``spec.normalise_volume`` of each input.
"""
from __future__ import annotations

import struct
from decimal import Decimal
from types import SimpleNamespace

import numpy as np

from engine.research.backtest.strategies.donchian_breakout import donchian_channels
from engine.research.backtest.strategies.macd_cross import macd_series
from engine.research.backtest.strategies.stoch_cross import stoch_series
from engine.research.features.atr import atr_at, sma_atr_series, true_range_series
from engine.research.features.rsi import rsi_series
from engine.research.mql5.cell_config import CellConfig
from engine.research.search.dna import DNA
from engine.research.search.real_backtest import dna_to_strategy

PREFIX_ALL = 2000
PREFIX_SAMPLES = 60
PRIMS_SEED = 20261002
PREFIX_SEED = 20261003
FIELDS = {"MA_CROSS": 6, "RSI_REVERSION": 5, "STOCH_CROSS": 6, "MACD_CROSS": 4, "DONCHIAN_BREAKOUT": 4}


def hx(x: float) -> str:
    return f"{struct.unpack('<Q', struct.pack('<d', float(x)))[0]:016X}"


def _b(c: bool) -> str:
    return "1" if c else "0"


def _ohlc(cell: CellConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    b = cell.bars
    return tuple(np.asarray(x, dtype=np.float64) for x in (b.open, b.high, b.low, b.close))


def _atr_series(cell: CellConfig, period: int) -> np.ndarray:
    _, h, lo, c = _ohlc(cell)
    return sma_atr_series(true_range_series(h, lo, c), period=period)


def _atr_field(atr: np.ndarray, i: int, period: int) -> str:
    if i + 1 < period:
        return ""
    v = float(atr[i])
    return hx(v) if np.isfinite(v) and v > 0.0 else ""


def _family_fields(signal: str, cfg, cell: CellConfig) -> list[list[str]]:
    _, h, lo, c = _ohlc(cell)
    n = c.size
    rows: list[list[str]] = []
    if signal == "MA_CROSS":
        f, s = cfg.fast, cfg.slow
        for i in range(n):
            if i < s:
                rows.append([""] * 6)
                continue
            fn, sn = float(np.mean(c[i - f + 1: i + 1])), float(np.mean(c[i - s + 1: i + 1]))
            fp, sp = float(np.mean(c[i - f: i])), float(np.mean(c[i - s: i]))
            rows.append([hx(fn), hx(sn), hx(fp), hx(sp), _b(fp <= sp and fn > sn), _b(fp >= sp and fn < sn)])
    elif signal == "RSI_REVERSION":
        r = rsi_series(c, period=cfg.period)
        for i in range(n):
            if i < cfg.period:
                rows.append([""] * 5)
                continue
            v = float(r[i])
            row = [hx(v), _b(v < cfg.oversold), _b(v > cfg.overbought)]
            if i - 1 < cfg.period:
                row += ["", ""]
            else:
                p = float(r[i - 1])
                row += [_b(p >= 50.0 and v < 50.0), _b(p <= 50.0 and v > 50.0)]
            rows.append(row)
    elif signal == "STOCH_CROSS":
        k, d = stoch_series(h, lo, c, k_period=cfg.k_period, d_period=cfg.d_period, smoothing=cfg.smoothing)
        for i in range(n):
            row = [hx(k[i]) if np.isfinite(k[i]) else "", hx(d[i]) if np.isfinite(d[i]) else ""]
            if i < 1 or not all(np.isfinite(x) for x in (k[i], d[i], k[i - 1], d[i - 1])):
                rows.append(row + [""] * 4)
                continue
            kn, dn, kp, dp = float(k[i]), float(d[i]), float(k[i - 1]), float(d[i - 1])
            rows.append(row + [_b(kp <= dp and kn > dn), _b(kn <= cfg.k_oversold and dn <= cfg.k_oversold),
                               _b(kp >= dp and kn < dn), _b(kn >= cfg.k_overbought and dn >= cfg.k_overbought)])
    elif signal == "MACD_CROSS":
        m, sig, _ = macd_series(c, fast_period=cfg.fast_period, slow_period=cfg.slow_period,
                                signal_period=cfg.signal_period)
        warm = cfg.slow_period + cfg.signal_period - 1
        for i in range(n):
            row = [hx(m[i]) if np.isfinite(m[i]) else "", hx(sig[i]) if np.isfinite(sig[i]) else ""]
            if i < warm:
                rows.append(row + ["", ""])
                continue
            mn, sn, mp, sp = float(m[i]), float(sig[i]), float(m[i - 1]), float(sig[i - 1])
            rows.append(row + [_b(mp <= sp and mn > sn), _b(mp >= sp and mn < sn)])
    elif signal == "DONCHIAN_BREAKOUT":
        up, dn = donchian_channels(h, lo, cfg.period)
        for i in range(n):
            row = [hx(up[i]), hx(dn[i])] if i >= cfg.period - 1 else ["", ""]
            if i < cfg.period:
                rows.append(row + ["", ""])
                continue
            rows.append(row + [_b(float(c[i]) > float(up[i - 1])), _b(float(c[i]) < float(dn[i - 1]))])
    else:
        raise ValueError(f"no level-B definition for signal {signal!r}")
    return rows


def expected_rows(dna: DNA, cell: CellConfig) -> list[list[str]]:
    cfg, _ = dna_to_strategy(dna)
    atr = _atr_series(cell, cfg.atr_period)
    times = np.asarray(cell.bars.time, dtype=np.int64)
    fam = _family_fields(dna.signal, cfg, cell)
    return [[str(i), str(int(times[i])), _atr_field(atr, i, cfg.atr_period)] + fam[i] for i in range(len(fam))]


def _prefix_indices(n: int) -> list[int]:
    if n <= PREFIX_ALL:
        return list(range(n))
    picks = np.random.default_rng(PREFIX_SEED).choice(n, PREFIX_SAMPLES, replace=False)
    return sorted({int(x) for x in picks} | {0, n - 1})


def _accessor_values(signal: str, strategy, view: SimpleNamespace, i: int) -> list[float]:
    if signal == "RSI_REVERSION":
        return [float(strategy._get_rsi(view)[i])]
    if signal == "STOCH_CROSS":
        k, d = strategy._get_series(view)
        return [float(k[i]), float(d[i])]
    if signal == "MACD_CROSS":
        m, s, _ = strategy._get_series(view)
        return [float(m[i]), float(s[i])]
    if signal == "DONCHIAN_BREAKOUT":
        up, dn = strategy._get_rolling(view)
        return [float(up[i]), float(dn[i])]
    return []  # MA_CROSS: np.mean over slices that end at i -- the expected rows ARE the strategy's expressions


def prefix_mismatches(dna: DNA, cell: CellConfig) -> tuple[int, list[str]]:
    """(prefixes checked, mismatches): the strategy's own accessor on bars [0, i] vs the full-window value at i."""
    cfg, strategy = dna_to_strategy(dna)
    o, h, lo, c = _ohlc(cell)
    full_atr = _atr_series(cell, cfg.atr_period)
    full = {"RSI_REVERSION": lambda: [rsi_series(c, period=cfg.period)],
            "STOCH_CROSS": lambda: list(stoch_series(h, lo, c, k_period=cfg.k_period, d_period=cfg.d_period,
                                                      smoothing=cfg.smoothing)),
            "MACD_CROSS": lambda: list(macd_series(c, fast_period=cfg.fast_period, slow_period=cfg.slow_period,
                                                   signal_period=cfg.signal_period)[:2]),
            "DONCHIAN_BREAKOUT": lambda: list(donchian_channels(h, lo, cfg.period)),
            "MA_CROSS": lambda: []}[dna.signal]()
    bad: list[str] = []
    idx = _prefix_indices(c.size)
    for i in idx:
        view = SimpleNamespace(open=o[: i + 1], high=h[: i + 1], low=lo[: i + 1], close=c[: i + 1])
        got = _accessor_values(dna.signal, strategy, view, i)
        for name, a, b in zip(("v0", "v1"), got, (float(s[i]) for s in full)):
            if not (hx(a) == hx(b) or (np.isnan(a) and np.isnan(b))):
                bad.append(f"prefix {i}: accessor {name} {a!r} != full-window {b!r}")
        pa = atr_at(h[: i + 1], lo[: i + 1], c[: i + 1], end_index=i, period=cfg.atr_period)
        fa = _atr_field(full_atr, i, cfg.atr_period)
        if (hx(pa) if pa is not None else "") != fa:
            bad.append(f"prefix {i}: atr_at {pa!r} != full-window field {fa!r}")
    return len(idx), bad


def primitive_inputs(cell: CellConfig) -> list[tuple[str, float]]:
    """Seeded stop-price and volume inputs (prims.csv): 1,500 stop-like prices, 300 nearest-double-to-a-tie
    prices, 100 exact (dyadic) price ties, 400 volumes and 200 volume ties."""
    rng = np.random.default_rng(PRIMS_SEED)
    _, h, lo, c = _ohlc(cell)
    d = int(cell.spec.digits)
    out: list[tuple[str, float]] = []
    ks = rng.integers(0, c.size, 1500)
    for k, m, sgn in zip(ks, rng.uniform(0.5, 6.0, ks.size), rng.choice([-1.0, 1.0], ks.size)):
        rng_k = float(h[k] - lo[k]) or float(cell.spec.point)
        out.append(("p", float(float(c[k]) + float(sgn) * float(m) * rng_k)))
    half = Decimal(5).scaleb(-(d + 1))
    for k in ks[:300]:
        out.append(("p", float(Decimal(f"{float(c[k]):.{d}f}") + half)))   # the nearest double to a tie
    for k in ks[:100]:  # EXACT ties: j / 2^(d+1), j odd, is (2n + 1) / (2 * 10^d) exactly -> half to even
        out.append(("p", (int(float(c[k]) * 2 ** (d + 1)) | 1) / 2 ** (d + 1)))
    step = float(cell.spec.volume_step)
    for v in rng.uniform(0.0, 3.0, 400):
        out.append(("v", float(v)))
    for j in range(200):
        out.append(("v", (j + 0.5) * step))
    return out


def primitive_expected(cell: CellConfig, kind: str, x: float) -> float:
    x = float(x)  # the engine rounds a Python float (round(float(price), digits)), never a numpy scalar
    return round(x, int(cell.spec.digits)) if kind == "p" else cell.spec.normalise_volume(x)


__all__ = ["FIELDS", "PREFIX_ALL", "PREFIX_SAMPLES", "PREFIX_SEED", "PRIMS_SEED", "expected_rows", "hx",
           "prefix_mismatches", "primitive_expected", "primitive_inputs"]
