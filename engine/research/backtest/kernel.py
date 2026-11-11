"""Numba fast path for the kernel-expressible signal families.

THE SLOW ENGINE IS THE SPECIFICATION. Every branch here mirrors a branch
in engine.py, in the same order, and the parity tests compare the two
candidate by candidate rather than against a hand-written expectation.
Where this file and engine.py disagree, this file is wrong.

WHAT IS NOT DONE HERE
---------------------

fastmath is OFF. It is not a tuning knob that happens to be disabled --
it licenses the compiler to reassociate float addition, and the moment
it does, equity stops being reproducible against the oracle. Numba's
default is off; it is stated explicitly at each njit so nobody adds it
later thinking it is free.

No approximation, ever. A candidate whose genes or config the kernel
cannot express EXACTLY is refused by ``expressible()`` and runs on the
slow engine instead, counted. Silently approximating one candidate in a
thousand would be indistinguishable from a real edge at the population
level, which is the failure this whole stage exists not to cause.

SLIPPAGE, AND WHY IT IS AN INPUT
--------------------------------

engine.py draws slippage from ``np.random.default_rng(seed)`` once per
simulated fill (engine.py:599, costs/slippage.py). Numba's RNG is a
different generator and cannot reproduce numpy's PCG64 stream, so the
draws are made in numpy BEFORE the kernel runs and passed in as an
array the kernel consumes in order. Same generator, same seed, same
sequence, same fills.

The engine draws ONLY when ``slippage_std_points != 0`` (slippage.py:
realised_slippage_points returns the mean without touching the rng when
std is zero), so the kernel consumes a draw under exactly that
condition and no other.

EQUITY IS AN OUTPUT ARRAY, NOT AN AGGREGATE
-------------------------------------------

The brief asks for a fixed-size aggregate record and no equity curves.
The aggregate is here, but the curve cannot be dropped: PBO/CSCV
consumes ``TrialResult.per_period_returns`` (runner.py), and a funnel
without PBO is not the same funnel. The slow path already materialises
the same series, so this costs no memory the pipeline was not already
spending. Callers that do not need it pass a zero-row buffer.

FAMILIES (W1.F1-F4)
-------------------

Step 5 dispatches on ``family``. MA_CROSS computes its means in-kernel
(see _pairwise_sum). Every other family's indicator series arrives as
``ind_a`` / ``ind_b``, computed in prep by the ORACLE's own indicator
functions on the full arrays -- they are causal (value at t depends only
on bars 0..t), so the full-array value at t is bit-identical to what the
strategy computes on its visible window [0, t]. ``_family_signal`` then
replays that strategy's ``on_bar`` decision order exactly, including
where each family checks finiteness relative to the pending reversal.

TRADE LOG
---------

``trades_out`` (rows x TR_SIZE, or zero rows to skip) records every
trade the way engine.py's Trade does: entry/exit bar indices are SIGNAL
bars for strategy-driven fills and the bar itself for SL/TP, deferred
and Friday exits; an EOD trade keeps exit_bar = -1 (exit_fill None).
It exists so parity can compare trade lists, not just aggregates.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

# --- aggregate record layout -------------------------------------------
# One fixed-size float64 row per candidate. Indices are named so the
# mapping back to TrialResult cannot drift silently.
REC_N_TRADES = 0
REC_N_ROUND_TRIPS = 1
REC_N_OPEN = 2
REC_EXIT_SIGNAL = 3
REC_EXIT_SL = 4
REC_EXIT_TP = 5
REC_EXIT_EOD = 6
REC_EXIT_SLTP_DEFERRED = 7
REC_EXIT_FRIDAY_CLOSE = 8
REC_BLOCKED_ZERO_SPREAD = 9
REC_DEFERRED_ZERO_SPREAD_FILLS = 10
REC_SLTP_DEFERRED_ZERO_SPREAD = 11
REC_BLOCKED_MAX_SPREAD = 12
REC_BLOCKED_DAILY_LOSS = 13
REC_FRIDAY_FLATTENS = 14
REC_COST_COMMISSION = 15
REC_COST_SPREAD = 16
REC_COST_SLIPPAGE = 17
REC_COST_SWAP = 18
REC_FINAL_EQUITY = 19
REC_SLIPPAGE_DRAWS_USED = 20
REC_SIZE = 21

RECORD_FIELDS = (
    "n_trades", "n_round_trips", "n_open",
    "exit_SIGNAL", "exit_SL", "exit_TP", "exit_EOD",
    "exit_SLTP_DEFERRED_ZERO_SPREAD", "exit_FRIDAY_CLOSE",
    "blocked_zero_spread_events", "deferred_zero_spread_fills",
    "sltp_deferred_zero_spread", "blocked_max_spread",
    "blocked_daily_loss", "friday_flattens",
    "cost_commission", "cost_spread", "cost_slippage", "cost_swap",
    "final_equity", "slippage_draws_used",
)
assert len(RECORD_FIELDS) == REC_SIZE

# --- signal families (step 5 dispatch) ----------------------------------
FAM_MA_CROSS = 0
FAM_RSI_REVERSION = 1
FAM_STOCH_CROSS = 2
FAM_MACD_CROSS = 3
FAM_DONCHIAN_BREAKOUT = 4
FAM_SIGNALS = 5  # v4 batch 2.6: ind_a / ind_b = long / short vote (1.0 / 0.0) from backtest/signal_switches.py

# --- trade log row layout -------------------------------------------------
TR_ENTRY_BAR = 0     # signal bar of the opening intent (Trade.entry_bar_index)
TR_FILL_BAR = 1      # bar whose open filled it (entry_fill.bar_index)
TR_DIR = 2           # +1 long, -1 short
TR_ENTRY_PRICE = 3
TR_VOLUME = 4
TR_SL = 5            # NaN when no stop
TR_TP = 6
TR_EXIT_BAR = 7      # -1 while open (EOD)
TR_EXIT_PRICE = 8    # NaN while open
TR_EXIT_REASON = 9   # TR_REASON_* code
TR_NET = 10          # realized_pnl; 0.0 while open
TR_SIZE = 11

TR_REASON_OPEN = 0   # EOD: still open when the data ran out
TR_REASON_SIGNAL = 1
TR_REASON_SL = 2
TR_REASON_TP = 3
TR_REASON_SLTP_DEFERRED = 4
TR_REASON_FRIDAY_CLOSE = 5


@njit(cache=True, fastmath=False)
def _two_product(a: float, b: float) -> tuple[float, float]:
    """(p, e) with p = fl(a*b) and a*b == p + e EXACTLY (Dekker/Veltkamp)."""
    p = a * b
    c = 134217729.0 * a  # 2**27 + 1
    a_hi = c - (c - a)
    a_lo = a - a_hi
    c = 134217729.0 * b
    b_hi = c - (c - b)
    b_lo = b - b_hi
    e = ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo
    return p, e


@njit(cache=True, fastmath=False)
def _round_to_digits(price: float, digits: int) -> float:
    """engine.py _quantize -- Python's round(price, digits), exactly.

    Python rounds the EXACT binary value of ``price`` to ``digits``
    decimals (half to even) and returns the double nearest that decimal.
    np.round instead rounds the already-rounded product ``price*10**d``,
    which decides near-ties the other way often enough to move an SL/TP
    level by one point (review finding, S3 W1). Here the product is kept
    exact as p + e, the tie is decided on p + e, and k / 10**d is the
    correctly rounded quotient -- the double nearest k * 10**-d, which is
    what Python returns.
    """
    scale = 1.0
    for _ in range(digits):
        scale *= 10.0
    p, e = _two_product(price, scale)
    k = np.floor(p)
    t = (p - k) - 0.5          # exact: p - floor(p) and the 0.5 shift
    if t > -e:
        k += 1.0
    elif t == -e and (k % 2.0) != 0.0:  # exact half: to even
        k += 1.0
    return k / scale


@njit(cache=True, fastmath=False)
def _normalise_volume(volume: float, vmin: float, vmax: float, vstep: float) -> float:
    """data/symbol_spec.normalise_volume, verbatim: snap to step, then clamp.

    ``round(volume / step) * step`` -- NOT ``vmin + round((v - vmin)/step)
    * step``, which lands on the same lot count but a different last bit
    for about a third of production volumes (0.01 + k*0.01 != (k+1)*0.01
    in binary), so every money figure downstream differed by an ulp. The
    oracle raises on step <= 0; prep only builds cells from real specs.
    """
    v = np.round(volume / vstep) * vstep
    if v < vmin:
        v = vmin
    if v > vmax:
        v = vmax
    return v


@njit(cache=True, fastmath=False)
def _slippage_points(
    mean_points: float,
    std_points: float,
    draws: np.ndarray,
    cursor: int,
    cap_points: float,
) -> tuple[float, int]:
    """costs/slippage.realised_slippage_points, capped.

    Returns (points, new_cursor). A draw is consumed ONLY when std is
    non-zero, matching the oracle: with std == 0 it returns the mean
    without touching the generator, so consuming one here would desync
    every later fill.
    """
    if std_points == 0.0:
        value = mean_points
    else:
        value = abs(draws[cursor])
        cursor += 1
    if value > cap_points:
        value = cap_points
    return value, cursor


# numpy's pairwise-summation block size (numpy/_core/src/umath/loops.c.src,
# PW_BLOCKSIZE). Reproduced exactly, not approximated: see _pairwise_sum.
_PW_BLOCKSIZE = 128


@njit(cache=True, fastmath=False)
def _pairwise_sum(a: np.ndarray, start: int, n: int) -> float:
    """numpy's pairwise summation for float64, bit-for-bit.

    np.add.reduce on a contiguous float64 array does NOT sum left to
    right -- it uses eight accumulators in blocks of eight and combines
    them in a fixed tree. A sequential sum, or a prefix-sum difference,
    gives a different last bit.

    That matters here and not merely in principle: the MA-cross
    decision is ``fast_prev <= slow_prev and fast_now > slow_now``. On
    a bar where two means are equal to within one ulp, a different
    rounding flips the comparison, the candidate takes a different
    trade, and the fast path diverges from the oracle in a way that
    looks like a logic bug and is not.

    Mirrors pairwise_sum_DOUBLE in numpy/_core/src/umath/loops.c.src.
    """
    if n < 8:
        res = 0.0
        for i in range(n):
            res += a[start + i]
        return res
    if n <= _PW_BLOCKSIZE:
        r0 = a[start + 0]
        r1 = a[start + 1]
        r2 = a[start + 2]
        r3 = a[start + 3]
        r4 = a[start + 4]
        r5 = a[start + 5]
        r6 = a[start + 6]
        r7 = a[start + 7]
        i = 8
        limit = n - (n % 8)
        while i < limit:
            r0 += a[start + i + 0]
            r1 += a[start + i + 1]
            r2 += a[start + i + 2]
            r3 += a[start + i + 3]
            r4 += a[start + i + 4]
            r5 += a[start + i + 5]
            r6 += a[start + i + 6]
            r7 += a[start + i + 7]
            i += 8
        res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
        while i < n:
            res += a[start + i]
            i += 1
        return res
    n2 = n // 2
    n2 -= n2 % 8
    return _pairwise_sum(a, start, n2) + _pairwise_sum(a, start + n2, n - n2)


@njit(cache=True, fastmath=False)
def _slice_mean(a: np.ndarray, start: int, n: int) -> float:
    """np.mean(a[start:start+n]) -- same summation, same divide."""
    return _pairwise_sum(a, start, n) / n


@njit(cache=True, fastmath=False)
def _stop_distances(
    atr14: np.ndarray, i: int, sl_atr: float, tp_atr: float,
) -> tuple[bool, float, float]:
    """ma_crossover._stop_distances: (suppress, sl_dist, tp_dist).

    G9 rulings 2 and 3. A DNA that asked for ATR-derived stops and
    cannot have them -- warmup, or a degenerate flat series where ATR is
    zero -- emits NO intent at all. Trading it without the stop would
    file a different strategy's results under this one's hash.
    """
    want_sl = sl_atr > 0.0
    want_tp = tp_atr > 0.0
    if (not want_sl) and (not want_tp):
        return False, 0.0, 0.0
    a = atr14[i]
    # atr_at: `not np.isfinite(value) or value <= 0.0` -> None -> suppress.
    # +inf (a non-finite high/low inside the window) must suppress too.
    if not np.isfinite(a) or a <= 0.0:
        return True, 0.0, 0.0
    sl_d = sl_atr * a if want_sl else 0.0
    tp_d = tp_atr * a if want_tp else 0.0
    return False, sl_d, tp_d


@njit(cache=True, fastmath=False)
def _log_open(
    trades_out: np.ndarray, row: int, signal_bar: int, fill_bar: int,
    d: int, price: float, volume: float, sl: float, tp: float,
) -> None:
    trades_out[row, TR_ENTRY_BAR] = signal_bar
    trades_out[row, TR_FILL_BAR] = fill_bar
    trades_out[row, TR_DIR] = d
    trades_out[row, TR_ENTRY_PRICE] = price
    trades_out[row, TR_VOLUME] = volume
    trades_out[row, TR_SL] = sl
    trades_out[row, TR_TP] = tp
    trades_out[row, TR_EXIT_BAR] = -1
    trades_out[row, TR_EXIT_PRICE] = np.nan
    trades_out[row, TR_EXIT_REASON] = TR_REASON_OPEN
    trades_out[row, TR_NET] = 0.0


@njit(cache=True, fastmath=False)
def _log_close(
    trades_out: np.ndarray, row: int, exit_bar: int, price: float,
    reason: int, net: float,
) -> None:
    trades_out[row, TR_EXIT_BAR] = exit_bar
    trades_out[row, TR_EXIT_PRICE] = price
    trades_out[row, TR_EXIT_REASON] = reason
    trades_out[row, TR_NET] = net


@njit(cache=True, fastmath=False)
def _family_signal(
    family: int,
    i: int,
    ind_a: np.ndarray,
    ind_b: np.ndarray,
    bar_close: np.ndarray,
    warmup: int,
    thr_lo: float,
    thr_hi: float,
    midline: bool,
    st_direction: int,
    st_pending_dir: int,
    atr14: np.ndarray,
    volume_req: float,
    sl_atr: float,
    tp_atr: float,
) -> tuple[bool, int, float, float, float, int, int]:
    """One ``on_bar`` call of a non-MA family, verbatim in decision order.

    Returns (emit, dir, volume, sl_dist, tp_dist, st_direction,
    st_pending_dir). The oracles (strategies/*.py) differ in WHERE they
    bail out relative to the pending reversal, and that order is kept:

      RSI_REVERSION     rsi.py        warmup i < period; RSI[i] finite;
                                      pending; midline exit; level entry
      STOCH_CROSS       stoch_cross   warmup i < k+d+s-2; K,D at i and i-1
                                      finite; pending; gated cross
      MACD_CROSS        macd_cross    warmup i < slow+signal-1; line and
                                      signal at i and i-1 finite; pending;
                                      cross
      DONCHIAN_BREAKOUT donchian      warmup i < period; pending; breakout
                                      of the PRIOR channel [i-p, i-1]
    """
    if i < warmup:
        return False, 0, 0.0, 0.0, 0.0, st_direction, st_pending_dir

    a_now = ind_a[i]
    if family == FAM_RSI_REVERSION:
        if not np.isfinite(a_now):
            return False, 0, 0.0, 0.0, 0.0, st_direction, st_pending_dir
    elif family == FAM_STOCH_CROSS or family == FAM_MACD_CROSS:
        if not (np.isfinite(a_now) and np.isfinite(ind_b[i])):
            return False, 0, 0.0, 0.0, 0.0, st_direction, st_pending_dir
        if not (np.isfinite(ind_a[i - 1]) and np.isfinite(ind_b[i - 1])):
            return False, 0, 0.0, 0.0, 0.0, st_direction, st_pending_dir

    # A reversal pended last bar: open it now (the close already went).
    if st_pending_dir != 0:
        suppress, sl_d, tp_d = _stop_distances(atr14, i, sl_atr, tp_atr)
        if suppress:
            return False, 0, 0.0, 0.0, 0.0, st_direction, st_pending_dir
        return True, st_pending_dir, volume_req, sl_d, tp_d, st_pending_dir, 0

    cur_long = st_direction == 1
    cur_short = st_direction == -1

    go_long = False
    go_short = False
    if family == FAM_RSI_REVERSION:
        if midline and (cur_long or cur_short) and i >= 1:
            prev = ind_a[i - 1]
            if np.isfinite(prev):
                # rsi.py emits the close with the POSITION's own side.
                if cur_long and prev >= 50.0 and a_now < 50.0:
                    return True, 1, 0.0, 0.0, 0.0, 0, st_pending_dir
                if cur_short and prev <= 50.0 and a_now > 50.0:
                    return True, -1, 0.0, 0.0, 0.0, 0, st_pending_dir
        if a_now < thr_lo:
            go_long = True
        elif a_now > thr_hi:
            go_short = True
    elif family == FAM_STOCH_CROSS:
        k_now = a_now
        d_now = ind_b[i]
        k_prev = ind_a[i - 1]
        d_prev = ind_b[i - 1]
        if k_prev <= d_prev and k_now > d_now:
            go_long = k_now <= thr_lo and d_now <= thr_lo
        elif k_prev >= d_prev and k_now < d_now:
            go_short = k_now >= thr_hi and d_now >= thr_hi
    elif family == FAM_MACD_CROSS:
        m_prev = ind_a[i - 1]
        s_now = ind_b[i]
        s_prev = ind_b[i - 1]
        if m_prev <= s_prev and a_now > s_now:
            go_long = True
        elif m_prev >= s_prev and a_now < s_now:
            go_short = True
    elif family == FAM_DONCHIAN_BREAKOUT:  # ind_a / ind_b: upper / lower channels
        c_now = bar_close[i]
        if c_now > ind_a[i - 1]:
            go_long = True
        elif c_now < ind_b[i - 1]:
            go_short = True
    elif family == FAM_SIGNALS:  # strategies/signals.py: the combined vote of bar i (never both sides)
        if a_now > 0.5:
            go_long = True
        elif ind_b[i] > 0.5:
            go_short = True

    if go_long:
        if cur_short:
            return True, 1, 0.0, 0.0, 0.0, 0, 1
        if not cur_long:
            suppress, sl_d, tp_d = _stop_distances(atr14, i, sl_atr, tp_atr)
            if not suppress:
                return True, 1, volume_req, sl_d, tp_d, 1, st_pending_dir
    elif go_short:
        if cur_long:
            return True, -1, 0.0, 0.0, 0.0, 0, -1
        if not cur_short:
            suppress, sl_d, tp_d = _stop_distances(atr14, i, sl_atr, tp_atr)
            if not suppress:
                return True, -1, volume_req, sl_d, tp_d, -1, st_pending_dir
    return False, 0, 0.0, 0.0, 0.0, st_direction, st_pending_dir


@njit(cache=True, fastmath=False)
def run_candidate(
    bar_open: np.ndarray,
    bar_high: np.ndarray,
    bar_low: np.ndarray,
    bar_close: np.ndarray,
    bar_hour_index: np.ndarray,
    day_index: np.ndarray,
    is_friday: np.ndarray,
    no_trade_mask: np.ndarray,
    bar_spread_points: np.ndarray,
    atr14: np.ndarray,
    swap_multiplier: np.ndarray,
    day_index_min: int,
    fast: int,
    slow: int,
    volume_req: float,
    sl_atr: float,
    tp_atr: float,
    family: int,
    ind_a: np.ndarray,
    ind_b: np.ndarray,
    warmup: int,
    thr_lo: float,
    thr_hi: float,
    midline: bool,
    gate_long: np.ndarray,
    gate_short: np.ndarray,
    initial_equity: float,
    max_spread_points: float,
    daily_loss_limit_pct: float,
    friday_close_hour: int,
    breakeven_at_r: float,
    stop_rules: np.ndarray,
    charge_swap: bool,
    charge_commission: bool,
    commission_per_lot_per_side: float,
    slippage_mean_points: float,
    slippage_std_points: float,
    slippage_cap_points: float,
    point: float,
    digits: int,
    fill_spread_price: np.ndarray,
    per_unit_profit: float,
    per_unit_loss: float,
    swap_long_points: float,
    swap_short_points: float,
    volume_min: float,
    volume_max: float,
    volume_step: float,
    slippage_draws: np.ndarray,
    record: np.ndarray,
    equity_out: np.ndarray,
    trades_out: np.ndarray,
) -> int:
    """One candidate, mirroring engine.py run() step for step.

    The numbered comments are engine.py's own step numbers, kept so the
    two can be read side by side. Returns 0 on success or a negative
    code the caller converts into a slow-path fallback. ``fast`` /
    ``slow`` are read only by MA_CROSS; ``ind_a`` .. ``midline`` only by
    the other families (see _family_signal). ``gate_long`` / ``gate_short`` are the ARQVIO entry gates per signal
    bar (gates.gate_arrays; all True when the DNA has none).
    """
    n = bar_close.size
    if n < 2:
        return -1
    want_trades = trades_out.shape[0] > 0
    n_logged = 0
    cur_row = -1

    pos_open = False
    pos_dir = 0
    pos_entry_price = 0.0
    pos_volume = 0.0
    pos_sl = np.nan
    pos_tp = np.nan
    pos_entry_cost = 0.0
    pos_swap = 0.0
    pos_be_dist = np.nan  # engine.py _Position.initial_sl_dist
    pos_be_done = False
    pos_rd_done = False  # engine.py _Position.reduce_done / lock_done
    pos_lk_done = False
    # engine.py _stop_rules: [sl_reduce_at_r, sl_lock_at_r, trail k, trail act], NaN = off
    rd_at = stop_rules[0]
    lk_at = stop_rules[1]
    tr_k = stop_rules[2]
    tr_act = stop_rules[3]
    any_rule = not (np.isnan(rd_at) and np.isnan(lk_at) and np.isnan(tr_k))
    # strategies/switch.py SwitchStrategy close rules: [4] friday_close_profit trigger % (NaN = off),
    # [5] eod_close UTC hour (NaN = off). st_entry_ref is its state["entry_price"]: the signal bar's close of the
    # last opening the gates let through.
    fcp_pct = stop_rules[4]
    eod_hour = stop_rules[5]
    any_close = not (np.isnan(fcp_pct) and np.isnan(eod_hour))
    st_entry_ref = 0.0

    # The strategy emits at most one intent per bar; they only pile up
    # while a no-trade window is shut.
    max_pending = 64
    p_dir = np.zeros(max_pending, dtype=np.int64)
    p_vol = np.zeros(max_pending, dtype=np.float64)
    p_sl_dist = np.zeros(max_pending, dtype=np.float64)
    p_tp_dist = np.zeros(max_pending, dtype=np.float64)
    p_blocked = np.zeros(max_pending, dtype=np.bool_)
    p_bar = np.zeros(max_pending, dtype=np.int64)
    n_pending = 0

    st_direction = 0
    st_pending_dir = 0

    realised_total = 0.0
    n_trades = 0
    n_round_trips = 0
    ex_signal = 0
    ex_sl = 0
    ex_tp = 0
    ex_eod = 0
    ex_sltp_def = 0
    ex_friday = 0
    blocked_events = 0
    deferred_fills = 0
    sltp_deferred = 0
    blocked_max_spread = 0
    blocked_daily_loss = 0
    friday_flattens = 0
    cost_commission = 0.0
    cost_spread = 0.0
    cost_slippage = 0.0
    cost_swap = 0.0

    deferred_exit = False
    last_swap_day = 0
    have_last_swap_day = False
    current_day = 0
    have_current_day = False
    day_start_realised = 0.0
    slip_cursor = 0
    want_equity = equity_out.size > 0
    if want_equity:
        equity_out[0] = initial_equity
    eq_len = 1

    use_gate = not np.isnan(max_spread_points)
    use_daily = not np.isnan(daily_loss_limit_pct)
    use_friday = friday_close_hour >= 0

    for i in range(n - 1):
        o = bar_open[i]
        h = bar_high[i]
        lo = bar_low[i]
        c = bar_close[i]
        hour = bar_hour_index[i]
        blocked = no_trade_mask[i]

        # 0. An exit owed from inside the window, taken at THIS open.
        if deferred_exit and (not blocked) and pos_open:
            signed = pos_dir * (o - pos_entry_price)
            per_unit = per_unit_profit if signed >= 0.0 else per_unit_loss
            net = signed * per_unit * pos_volume - pos_entry_cost - pos_swap
            realised_total += net
            n_round_trips += 1
            ex_sltp_def += 1
            sltp_deferred += 1
            if want_trades:
                _log_close(trades_out, cur_row, i, o, TR_REASON_SLTP_DEFERRED, net)
            cost_swap += pos_swap
            pos_open = False
            pos_swap = 0.0
            deferred_exit = False
        elif deferred_exit and (not blocked):
            deferred_exit = False

        # B4 day boundary: reset the daily loss baseline.
        bar_day = day_index[i]
        if (not have_current_day) or bar_day != current_day:
            current_day = bar_day
            have_current_day = True
            day_start_realised = realised_total

        # B4 guardrail 2: Friday close, at a TRADEABLE bar only.
        if (
            use_friday and pos_open and (not blocked)
            and is_friday[i] and hour >= friday_close_hour
        ):
            signed = pos_dir * (o - pos_entry_price)
            per_unit = per_unit_profit if signed >= 0.0 else per_unit_loss
            net = signed * per_unit * pos_volume - pos_entry_cost - pos_swap
            realised_total += net
            n_round_trips += 1
            ex_friday += 1
            friday_flattens += 1
            if want_trades:
                _log_close(trades_out, cur_row, i, o, TR_REASON_FRIDAY_CLOSE, net)
            cost_swap += pos_swap
            pos_open = False
            pos_swap = 0.0
            st_direction = 0
            keep = 0
            for k in range(n_pending):
                if p_vol[k] <= 0.0:
                    p_dir[keep] = p_dir[k]
                    p_vol[keep] = p_vol[k]
                    p_sl_dist[keep] = p_sl_dist[k]
                    p_tp_dist[keep] = p_tp_dist[k]
                    p_blocked[keep] = p_blocked[k]
                    p_bar[keep] = p_bar[k]
                    keep += 1
            n_pending = keep

        # B4 guardrail 1: daily loss limit blocks NEW entries only.
        if use_daily and n_pending > 0:
            day_loss = day_start_realised - realised_total
            limit = initial_equity * daily_loss_limit_pct / 100.0
            if limit > 0.0 and day_loss >= limit:
                keep = 0
                n_entries = 0
                for k in range(n_pending):
                    if p_vol[k] > 0.0:
                        n_entries += 1
                    else:
                        p_dir[keep] = p_dir[k]
                        p_vol[keep] = p_vol[k]
                        p_sl_dist[keep] = p_sl_dist[k]
                        p_tp_dist[keep] = p_tp_dist[k]
                        p_blocked[keep] = p_blocked[k]
                        p_bar[keep] = p_bar[k]
                        keep += 1
                if n_entries > 0:
                    blocked_daily_loss += n_entries
                    n_pending = keep

        # A5 max-spread entry gate. Exits are never gated.
        if (
            use_gate and (not blocked) and n_pending > 0
            and bar_spread_points[i] > max_spread_points
        ):
            keep = 0
            n_entries = 0
            for k in range(n_pending):
                if p_vol[k] > 0.0:
                    n_entries += 1
                else:
                    p_dir[keep] = p_dir[k]
                    p_vol[keep] = p_vol[k]
                    p_sl_dist[keep] = p_sl_dist[k]
                    p_tp_dist[keep] = p_tp_dist[k]
                    p_blocked[keep] = p_blocked[k]
                    p_bar[keep] = p_bar[k]
                    keep += 1
            if n_entries > 0:
                blocked_max_spread += n_entries
                n_pending = keep

        # 1. Fill pending intents at THIS bar's open.
        if blocked:
            blocked_events += n_pending
            for k in range(n_pending):
                p_blocked[k] = True
        else:
            for k in range(n_pending):
                if p_blocked[k]:
                    deferred_fills += 1
            for k in range(n_pending):
                d = p_dir[k]
                v_req = p_vol[k]
                slip_pts, slip_cursor = _slippage_points(
                    slippage_mean_points, slippage_std_points,
                    slippage_draws, slip_cursor, slippage_cap_points,
                )
                slip_price = slip_pts * point
                fill_price = o + slip_price if d > 0 else o - slip_price

                if v_req == 0.0:
                    if not pos_open:
                        continue
                    signed = pos_dir * (fill_price - pos_entry_price)
                    per_unit = per_unit_profit if signed >= 0.0 else per_unit_loss
                    net = signed * per_unit * pos_volume - pos_entry_cost - pos_swap
                    realised_total += net
                    n_round_trips += 1
                    ex_signal += 1
                    cost_swap += pos_swap
                    pos_open = False
                    pos_swap = 0.0
                    if want_trades:
                        _log_close(trades_out, cur_row, p_bar[k], fill_price,
                                   TR_REASON_SIGNAL, net)
                else:
                    req = v_req if v_req > volume_min else volume_min
                    v = _normalise_volume(req, volume_min, volume_max, volume_step)
                    bigger = per_unit_profit
                    if per_unit_loss > bigger:
                        bigger = per_unit_loss
                    # engine.py:808 / costs/spread.py: price * volume * per-unit.
                    sp_one_way = fill_spread_price[i] * v * bigger
                    comm = 0.0
                    if charge_commission:
                        comm = commission_per_lot_per_side * v * 2.0
                    slip_ccy = slip_price * bigger * v
                    entry_cost = sp_one_way + slip_ccy + comm

                    if pos_open:
                        signed = pos_dir * (fill_price - pos_entry_price)
                        per_unit = per_unit_profit if signed >= 0.0 else per_unit_loss
                        net = signed * per_unit * pos_volume - pos_entry_cost - pos_swap
                        realised_total += net
                        n_round_trips += 1
                        ex_signal += 1
                        cost_swap += pos_swap
                        pos_swap = 0.0
                        if want_trades:
                            _log_close(trades_out, cur_row, p_bar[k], fill_price,
                                       TR_REASON_SIGNAL, net)

                    pos_open = True
                    pos_dir = d
                    pos_entry_price = fill_price
                    pos_volume = v
                    pos_entry_cost = entry_cost
                    pos_swap = 0.0
                    n_trades += 1
                    cost_commission += comm
                    cost_spread += sp_one_way
                    cost_slippage += slip_ccy
                    sl_d = p_sl_dist[k]
                    tp_d = p_tp_dist[k]
                    if sl_d > 0.0:
                        raw = fill_price - sl_d if d > 0 else fill_price + sl_d
                        pos_sl = _round_to_digits(raw, digits)
                        pos_be_dist = abs(fill_price - pos_sl)
                    else:
                        pos_sl = np.nan
                        pos_be_dist = np.nan
                    pos_be_done = False
                    pos_rd_done = False
                    pos_lk_done = False
                    if tp_d > 0.0:
                        raw = fill_price + tp_d if d > 0 else fill_price - tp_d
                        pos_tp = _round_to_digits(raw, digits)
                    else:
                        pos_tp = np.nan
                    if want_trades:
                        if n_logged >= trades_out.shape[0]:
                            return -3
                        cur_row = n_logged
                        n_logged += 1
                        _log_open(trades_out, cur_row, p_bar[k], i, d,
                                  fill_price, v, pos_sl, pos_tp)
            n_pending = 0

        # 2. Swap at day-roll, only the nights since the last roll.
        if pos_open and have_last_swap_day and bar_day != last_swap_day and charge_swap:
            base_pts = swap_long_points if pos_dir > 0 else swap_short_points
            total_pts = 0.0
            dd = last_swap_day + 1
            while dd <= bar_day:
                total_pts += base_pts * swap_multiplier[dd - day_index_min]
                dd += 1
            bigger = per_unit_profit
            if per_unit_loss > bigger:
                bigger = per_unit_loss
            pos_swap += -(total_pts * point * pos_volume * bigger)
        if pos_open:
            last_swap_day = bar_day
            have_last_swap_day = True
        else:
            have_last_swap_day = False

        # 3. SL/TP against THIS bar's range. SL first when both touched.
        if pos_open:
            touched = 0
            if not np.isnan(pos_sl):
                if pos_dir > 0:
                    if lo <= pos_sl:
                        touched = 1
                elif h >= pos_sl:
                    touched = 1
            if touched == 0 and not np.isnan(pos_tp):
                if pos_dir > 0:
                    if h >= pos_tp:
                        touched = 2
                elif lo <= pos_tp:
                    touched = 2
            if touched != 0 and blocked:
                if not deferred_exit:
                    blocked_events += 1
                deferred_exit = True
            elif touched != 0:
                level = pos_sl if touched == 1 else pos_tp
                if touched == 1:
                    gapped = o < level if pos_dir > 0 else o > level
                else:
                    gapped = o > level if pos_dir > 0 else o < level
                exit_price = o if gapped else level
                signed = pos_dir * (exit_price - pos_entry_price)
                per_unit = per_unit_profit if signed >= 0.0 else per_unit_loss
                net = signed * per_unit * pos_volume - pos_entry_cost - pos_swap
                realised_total += net
                n_round_trips += 1
                if touched == 1:
                    ex_sl += 1
                else:
                    ex_tp += 1
                if want_trades:
                    _log_close(
                        trades_out, cur_row, i, exit_price,
                        TR_REASON_SL if touched == 1 else TR_REASON_TP, net,
                    )
                cost_swap += pos_swap
                pos_open = False
                pos_swap = 0.0
                st_direction = 0

        # 4. Mark to market.
        unreal = 0.0
        if pos_open:
            signed = pos_dir * (c - pos_entry_price)
            per_unit = per_unit_profit if signed >= 0.0 else per_unit_loss
            unreal = signed * per_unit * pos_volume - pos_entry_cost - pos_swap
        if want_equity:
            equity_out[eq_len] = initial_equity + realised_total + unreal
        eq_len += 1

        # 4b. engine.py _breakeven: judged on this close, the stop acts from the next bar's step 3.
        if (pos_open and not np.isnan(breakeven_at_r) and not pos_be_done and not np.isnan(pos_sl)
                and not np.isnan(pos_be_dist) and pos_be_dist > 0.0):
            if (c - pos_entry_price) * pos_dir >= breakeven_at_r * pos_be_dist:
                level = _round_to_digits(pos_entry_price, digits)
                if (level - pos_sl) * pos_dir > 0.0:
                    pos_sl = level
                    if want_trades:  # the oracle's Trade.sl_price is the moved stop
                        trades_out[cur_row, TR_SL] = pos_sl
                pos_be_done = True

        # 4b. engine.py _stop_rules: sl_reduce, sl_lock, atr_trail, in that order, only ever towards the trade.
        if (any_rule and pos_open and not np.isnan(pos_sl) and not np.isnan(pos_be_dist)
                and pos_be_dist > 0.0):
            gain = (c - pos_entry_price) * pos_dir
            moved = False
            if not np.isnan(rd_at) and not pos_rd_done and gain >= rd_at * pos_be_dist:
                level = _round_to_digits(pos_entry_price - pos_dir * (0.5 * pos_be_dist), digits)
                if (level - pos_sl) * pos_dir > 0.0:
                    pos_sl = level
                    moved = True
                pos_rd_done = True
            if not np.isnan(lk_at) and not pos_lk_done and gain >= lk_at * pos_be_dist:
                level = _round_to_digits(pos_entry_price + pos_dir * (0.5 * lk_at * pos_be_dist), digits)
                if (level - pos_sl) * pos_dir > 0.0:
                    pos_sl = level
                    moved = True
                pos_lk_done = True
            if not np.isnan(tr_k) and not np.isnan(atr14[i]) and gain >= tr_act * pos_be_dist:
                level = _round_to_digits(c - pos_dir * (tr_k * atr14[i]), digits)
                if (level - pos_sl) * pos_dir > 0.0:
                    pos_sl = level
                    moved = True
            if moved and want_trades:  # the oracle's Trade.sl_price is the moved stop
                trades_out[cur_row, TR_SL] = pos_sl

        # 5. The strategy sees bars 0..i and emits for i+1.
        emit = False
        emit_dir = 0
        emit_vol = 0.0
        sl_d = 0.0
        tp_d = 0.0
        # SwitchStrategy.on_bar: a management close runs first while the strategy holds a side, and the family
        # does not see the bar it fires on.
        mg_exit = False
        if any_close and st_direction != 0:
            if not np.isnan(fcp_pct) and is_friday[i]:
                move = (c - st_entry_ref) * st_direction
                if move > 0.0:
                    a = atr14[i]
                    risk = max(move, 1e-9) if (not np.isfinite(a) or a <= 0.0) else sl_atr * a  # atr_at: None
                    if move >= (fcp_pct / 100.0) * risk:
                        mg_exit = True
            if not mg_exit and not np.isnan(eod_hour) and bar_hour_index[i] >= eod_hour:
                mg_exit = True
        if mg_exit:
            emit = True
            emit_dir = st_direction
            st_direction = 0
        elif family == FAM_MA_CROSS:
            if i >= slow:
                if st_pending_dir != 0:
                    suppress, sl_d, tp_d = _stop_distances(atr14, i, sl_atr, tp_atr)
                    if not suppress:
                        emit = True
                        emit_dir = st_pending_dir
                        emit_vol = volume_req
                        st_pending_dir = 0
                        st_direction = emit_dir
                else:
                    fast_now = _slice_mean(bar_close, i - fast + 1, fast)
                    slow_now = _slice_mean(bar_close, i - slow + 1, slow)
                    fast_prev = _slice_mean(bar_close, i - fast, fast)
                    slow_prev = _slice_mean(bar_close, i - slow, slow)
                    cur_long = st_direction == 1
                    cur_short = st_direction == -1
                    if fast_prev <= slow_prev and fast_now > slow_now:
                        if cur_short:
                            st_direction = 0
                            st_pending_dir = 1
                            emit = True
                            emit_dir = 1
                            emit_vol = 0.0
                        elif not cur_long:
                            suppress, sl_d, tp_d = _stop_distances(
                                atr14, i, sl_atr, tp_atr
                            )
                            if not suppress:
                                st_direction = 1
                                emit = True
                                emit_dir = 1
                                emit_vol = volume_req
                    elif fast_prev >= slow_prev and fast_now < slow_now:
                        if cur_long:
                            st_direction = 0
                            st_pending_dir = -1
                            emit = True
                            emit_dir = -1
                            emit_vol = 0.0
                        elif not cur_short:
                            suppress, sl_d, tp_d = _stop_distances(
                                atr14, i, sl_atr, tp_atr
                            )
                            if not suppress:
                                st_direction = -1
                                emit = True
                                emit_dir = -1
                                emit_vol = volume_req
        else:
            (emit, emit_dir, emit_vol, sl_d, tp_d,
             st_direction, st_pending_dir) = _family_signal(
                family, i, ind_a, ind_b, bar_close, warmup, thr_lo, thr_hi,
                midline, st_direction, st_pending_dir, atr14, volume_req,
                sl_atr, tp_atr,
            )
        # ARQVIO entry gates (gates.py, strategies/gated.py): an opening the gate refuses is dropped and the
        # strategy is flat, as if its signal had not fired. Closing intents are never gated.
        if emit and emit_vol > 0.0 and not (gate_long[i] if emit_dir > 0 else gate_short[i]):
            emit = False
            st_direction = 0
        if emit and emit_vol > 0.0:
            st_entry_ref = c  # SwitchStrategy._observe_open
        # engine.py:550 -- a close intent with no position is dropped.
        if emit and not (emit_vol == 0.0 and not pos_open):
            if n_pending >= max_pending:
                return -2
            p_dir[n_pending] = emit_dir
            p_vol[n_pending] = emit_vol
            p_sl_dist[n_pending] = sl_d
            p_tp_dist[n_pending] = tp_d
            p_blocked[n_pending] = False
            p_bar[n_pending] = i
            n_pending += 1

    # 6. Final bar. A second fill path, obeying the same window.
    last = n - 1
    last_open = bar_open[last]
    if no_trade_mask[last]:
        blocked_events += n_pending
        n_pending = 0
    for k in range(n_pending):
        if p_blocked[k]:
            deferred_fills += 1
    for k in range(n_pending):
        d = p_dir[k]
        v_req = p_vol[k]
        slip_pts, slip_cursor = _slippage_points(
            slippage_mean_points, slippage_std_points,
            slippage_draws, slip_cursor, slippage_cap_points,
        )
        slip_price = slip_pts * point
        fill_price = last_open + slip_price if d > 0 else last_open - slip_price
        if v_req == 0.0:
            if not pos_open:
                continue
            signed = pos_dir * (fill_price - pos_entry_price)
            per_unit = per_unit_profit if signed >= 0.0 else per_unit_loss
            net = signed * per_unit * pos_volume - pos_entry_cost - pos_swap
            realised_total += net
            n_round_trips += 1
            ex_signal += 1
            cost_swap += pos_swap
            pos_open = False
            pos_swap = 0.0
            if want_trades:
                _log_close(trades_out, cur_row, p_bar[k], fill_price,
                           TR_REASON_SIGNAL, net)
        else:
            req = v_req if v_req > volume_min else volume_min
            v = _normalise_volume(req, volume_min, volume_max, volume_step)
            bigger = per_unit_profit
            if per_unit_loss > bigger:
                bigger = per_unit_loss
            sp_one_way = fill_spread_price[last] * v * bigger
            comm = 0.0
            if charge_commission:
                comm = commission_per_lot_per_side * v * 2.0
            slip_ccy = slip_price * bigger * v
            if pos_open:
                signed = pos_dir * (fill_price - pos_entry_price)
                per_unit = per_unit_profit if signed >= 0.0 else per_unit_loss
                net = signed * per_unit * pos_volume - pos_entry_cost - pos_swap
                realised_total += net
                n_round_trips += 1
                ex_signal += 1
                cost_swap += pos_swap
                pos_swap = 0.0
                if want_trades:
                    _log_close(trades_out, cur_row, p_bar[k], fill_price,
                               TR_REASON_SIGNAL, net)
            pos_open = True
            pos_dir = d
            pos_entry_price = fill_price
            pos_volume = v
            pos_entry_cost = sp_one_way + slip_ccy + comm
            pos_swap = 0.0
            n_trades += 1
            cost_commission += comm
            cost_spread += sp_one_way
            cost_slippage += slip_ccy
            if want_trades:
                # engine.py _open_position anchors the levels on the Trade
                # even here, though no bar remains to trigger them.
                sl_d = p_sl_dist[k]
                tp_d = p_tp_dist[k]
                lvl_sl = np.nan
                lvl_tp = np.nan
                if sl_d > 0.0:
                    raw = fill_price - sl_d if d > 0 else fill_price + sl_d
                    lvl_sl = _round_to_digits(raw, digits)
                if tp_d > 0.0:
                    raw = fill_price + tp_d if d > 0 else fill_price - tp_d
                    lvl_tp = _round_to_digits(raw, digits)
                if n_logged >= trades_out.shape[0]:
                    return -3
                cur_row = n_logged
                n_logged += 1
                _log_open(trades_out, cur_row, p_bar[k], last, d,
                          fill_price, v, lvl_sl, lvl_tp)

    unreal = 0.0
    if pos_open:
        signed = pos_dir * (bar_close[last] - pos_entry_price)
        per_unit = per_unit_profit if signed >= 0.0 else per_unit_loss
        unreal = signed * per_unit * pos_volume - pos_entry_cost - pos_swap
        ex_eod += 1
        cost_swap += pos_swap
    final_equity = initial_equity + realised_total + unreal
    if want_equity:
        equity_out[eq_len] = final_equity

    record[REC_N_TRADES] = n_trades
    record[REC_N_ROUND_TRIPS] = n_round_trips
    record[REC_N_OPEN] = n_trades - n_round_trips
    record[REC_EXIT_SIGNAL] = ex_signal
    record[REC_EXIT_SL] = ex_sl
    record[REC_EXIT_TP] = ex_tp
    record[REC_EXIT_EOD] = ex_eod
    record[REC_EXIT_SLTP_DEFERRED] = ex_sltp_def
    record[REC_EXIT_FRIDAY_CLOSE] = ex_friday
    record[REC_BLOCKED_ZERO_SPREAD] = blocked_events
    record[REC_DEFERRED_ZERO_SPREAD_FILLS] = deferred_fills
    record[REC_SLTP_DEFERRED_ZERO_SPREAD] = sltp_deferred
    record[REC_BLOCKED_MAX_SPREAD] = blocked_max_spread
    record[REC_BLOCKED_DAILY_LOSS] = blocked_daily_loss
    record[REC_FRIDAY_FLATTENS] = friday_flattens
    record[REC_COST_COMMISSION] = cost_commission
    record[REC_COST_SPREAD] = cost_spread
    record[REC_COST_SLIPPAGE] = cost_slippage
    record[REC_COST_SWAP] = cost_swap
    record[REC_FINAL_EQUITY] = final_equity
    record[REC_SLIPPAGE_DRAWS_USED] = slip_cursor
    return 0


@njit(cache=True, fastmath=False, parallel=True)
def eval_batch(
    bar_open: np.ndarray,
    bar_high: np.ndarray,
    bar_low: np.ndarray,
    bar_close: np.ndarray,
    bar_hour_index: np.ndarray,
    day_index: np.ndarray,
    is_friday: np.ndarray,
    no_trade_mask: np.ndarray,
    bar_spread_points: np.ndarray,
    atr14: np.ndarray,
    swap_multiplier: np.ndarray,
    day_index_min: int,
    # --- per-candidate genes, one row each -------------------------------
    g_fast: np.ndarray,
    g_slow: np.ndarray,
    g_volume: np.ndarray,
    g_sl_atr: np.ndarray,
    g_tp_atr: np.ndarray,
    # --- config, shared --------------------------------------------------
    initial_equity: float,
    max_spread_points: float,
    daily_loss_limit_pct: float,
    friday_close_hour: int,
    charge_swap: bool,
    charge_commission: bool,
    commission_per_lot_per_side: float,
    slippage_mean_points: float,
    slippage_std_points: float,
    slippage_cap_points: float,
    point: float,
    digits: int,
    fill_spread_price: np.ndarray,
    per_unit_profit: float,
    per_unit_loss: float,
    swap_long_points: float,
    swap_short_points: float,
    volume_min: float,
    volume_max: float,
    volume_step: float,
    # --- rng: ONE pre-drawn row per candidate ----------------------------
    slippage_draws: np.ndarray,     # (n_candidates, budget)
    # --- outputs ----------------------------------------------------------
    records: np.ndarray,            # (n_candidates, REC_SIZE)
    equity_out: np.ndarray,         # (n_candidates, n_bars+1) or (0, 0)
    status_out: np.ndarray,         # (n_candidates,) int64
) -> None:
    """Evaluate every candidate. Parallel over candidates, not over bars.

    DETERMINISM. Each candidate is a wholly independent sequential run
    that writes only its own row, so the thread count cannot change any
    result -- there is no cross-candidate reduction whose order could
    vary. Parallelising over BARS would have been the opposite: the
    equity accumulation is inherently sequential and reassociating it
    would change the sum.

    Each candidate also gets its OWN pre-drawn slippage row, because
    sharing one cursor across threads would hand candidates different
    draws depending on who ran first.
    """
    n_candidates = g_fast.size
    want_equity = equity_out.shape[0] > 0
    no_ind = np.zeros(0, dtype=np.float64)
    no_trades = np.zeros((0, TR_SIZE), dtype=np.float64)
    open_gate = np.ones(bar_close.size, dtype=np.bool_)  # the batch path runs ungated MA_CROSS candidates
    no_rules = np.full(6, np.nan)  # nor any management rule
    for c in prange(n_candidates):
        eq = equity_out[c] if want_equity else np.zeros(0, dtype=np.float64)
        status_out[c] = run_candidate(
            bar_open, bar_high, bar_low, bar_close,
            bar_hour_index, day_index, is_friday, no_trade_mask,
            bar_spread_points, atr14, swap_multiplier, day_index_min,
            g_fast[c], g_slow[c], g_volume[c], g_sl_atr[c], g_tp_atr[c],
            FAM_MA_CROSS, no_ind, no_ind, 0, np.nan, np.nan, False, open_gate, open_gate,
            initial_equity, max_spread_points, daily_loss_limit_pct,
            friday_close_hour, np.nan, no_rules, charge_swap, charge_commission,
            commission_per_lot_per_side, slippage_mean_points,
            slippage_std_points, slippage_cap_points,
            point, digits, fill_spread_price,
            per_unit_profit, per_unit_loss,
            swap_long_points, swap_short_points,
            volume_min, volume_max, volume_step,
            slippage_draws[c], records[c], eq, no_trades,
        )
