"""W1: SL/TP genes reach the engine and actually close positions.

Before S1 the exit engine had one exit: the strategy's own signal.
``_check_sl_tp`` returned None unconditionally, ``OrderIntent.sl_price``
and ``tp_price`` were declared and never assigned, and
``dna_to_ma_config`` dropped ``sl_atr`` and ``tp_atr`` before they could
reach anything. Five DNAs differing only in those genes produced
byte-identical results (the C5 fixture).

ATR is SMA-of-TR, MT5 iATR semantics (G9). Every hand calculation below
is written out so the expected numbers can be checked without running
the code.
"""
from __future__ import annotations

import numpy as np
import pytest

from engine.research.backtest.engine import BacktestConfig, BacktestEngine
from engine.research.backtest.order import (
    EXIT_SL,
    EXIT_SLTP_DEFERRED_ZERO_SPREAD,
    EXIT_TP,
    OrderIntent,
)
from engine.research.backtest.strategies.ma_crossover import MACrossConfig
from engine.research.data.mt5_source import BarData, Timeframe
from engine.research.search.real_backtest import dna_to_ma_config, spec_for

SYMBOL = "USDCADm"
HOUR_S = 3600


def _bars(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    opens: list[float] | None = None,
    start_hour: int = 0,
) -> BarData:
    n = len(highs)
    base = 86400 + start_hour * HOUR_S
    return BarData(
        time=np.array([base + i * HOUR_S for i in range(n)], dtype=np.int64),
        open=np.array(opens if opens is not None else closes, dtype=np.float64),
        high=np.array(highs, dtype=np.float64),
        low=np.array(lows, dtype=np.float64),
        close=np.array(closes, dtype=np.float64),
        tick_volume=np.ones(n),
        spread=np.zeros(n, dtype=np.int32),
        real_volume=np.zeros(n),
        symbol=SYMBOL,
        timeframe=Timeframe.H1,
    )


class _EnterOnBar:
    """One long entry on a chosen bar, with explicit stop distances."""

    strategy_id = "enter-with-stops"

    def __init__(
        self,
        signal_bar: int,
        sl_dist: float | None,
        tp_dist: float | None,
        direction: str = "long",
        volume: float = 0.1,
    ) -> None:
        self.signal_bar = signal_bar
        self.sl_dist = sl_dist
        self.tp_dist = tp_dist
        self.direction = direction
        self.volume = volume

    def on_bar(self, bar_index, bars, state):
        if bar_index != self.signal_bar:
            return None
        return OrderIntent(
            direction=self.direction,
            volume=self.volume,
            bar_index=bar_index,
            sl_dist=self.sl_dist,
            tp_dist=self.tp_dist,
        )


def _engine(no_trade_hours: tuple[int, ...] = ()) -> BacktestEngine:
    config = BacktestConfig(
        strategy_id="w1",
        symbol=SYMBOL,
        timeframe="H1",
        initial_equity=10_000.0,
        charge_swap=False,
        charge_commission=False,
        slippage_mean_points=0.0,
        slippage_std_points=0.0,
        no_trade_hours=no_trade_hours,
    )
    return BacktestEngine(spec=spec_for(SYMBOL), config=config, rng_seed=0)


# ---------------------------------------------------------------------------
# The genes reach the engine at all
# ---------------------------------------------------------------------------

def test_sl_tp_genes_change_results() -> None:
    """The C5 fixture: DNAs differing ONLY in sl_atr/tp_atr must differ.

    Byte-identical output is the failure this test exists to catch --
    it is what the audit found, and it meant the two genes were being
    sampled, hashed and stored while changing nothing.
    """
    import dataclasses

    from engine.research.backtest.strategies.ma_crossover import MACrossStrategy
    from engine.research.search.sampler import random_dna

    # A series with enough movement for stops to bite, and enough bars
    # for ATR(14) plus the MA warmup.
    rng = np.random.default_rng(11)
    n = 400
    walk = np.cumsum(rng.normal(0.0, 0.0015, size=n)) + 1.10
    highs = (walk + 0.0010).tolist()
    lows = (walk - 0.0010).tolist()
    closes = walk.tolist()
    bars = _bars(highs, lows, closes)

    base = random_dna(np.random.default_rng(3))
    base = dataclasses.replace(base, max_feature_lookback=40, sl_atr=1.0, tp_atr=1.0)
    variants = [
        base,
        dataclasses.replace(base, sl_atr=0.5),
        dataclasses.replace(base, sl_atr=6.0),
        dataclasses.replace(base, tp_atr=0.5),
        dataclasses.replace(base, tp_atr=6.0),
    ]

    seen = set()
    for dna in variants:
        cfg = dna_to_ma_config(dna)
        result = _engine().run(bars, MACrossStrategy(cfg))
        seen.add((result.n_trades, round(result.final_equity, 9)))

    assert len(seen) >= 2, (
        f"all {len(variants)} DNAs produced the same result {seen}; "
        "sl_atr/tp_atr are not reaching the engine"
    )


def test_dna_to_ma_config_carries_the_stop_genes() -> None:
    """The projection drops neither gene, and the id distinguishes them."""
    import dataclasses

    from engine.research.search.sampler import random_dna

    dna = random_dna(np.random.default_rng(5))
    dna = dataclasses.replace(dna, sl_atr=2.5, tp_atr=4.0)
    cfg = dna_to_ma_config(dna)
    assert cfg.sl_atr == pytest.approx(2.5)
    assert cfg.tp_atr == pytest.approx(4.0)

    other = dna_to_ma_config(dataclasses.replace(dna, sl_atr=9.0))
    assert cfg.strategy_id != other.strategy_id, (
        "two different stop configurations share a strategy_id"
    )


# ---------------------------------------------------------------------------
# Hand-computed price and P&L
# ---------------------------------------------------------------------------

def test_sl_hit_at_exact_price() -> None:
    """LONG stopped out at the level, P&L to the cent.

    Construction. 20 bars. Bars 0..14 have a constant 0.00100 range so
    ATR(14) is exactly 0.00100:
        every TR = high - low = 0.00100 (flat closes, so the gap terms
        are smaller), SMA of 14 of them = 0.00100.

    Signal on bar 14, fill at bar 15's open = 1.10000 (slippage is
    configured to zero). sl_atr = 2.0:
        sl_dist = 2.0 * 0.00100 = 0.00200
        SL      = 1.10000 - 0.00200 = 1.09800

    Bar 16 dips to 1.09799, which is <= 1.09800, so the stop is touched.
    The bar's OPEN is 1.10000, which is above the level, so price passed
    THROUGH the stop rather than gapping past it: the fill is the level
    itself, 1.09800.

    P&L. USDCADm spec: trade_tick_size and tick values give
        per_unit_per_lot = max(tvp, tvl) / trade_tick_size
    and the engine charges the round trip on the entry fill. With
    commission and swap switched off and zero slippage, the only cost is
    the entry side of the spread, so:
        gross = (1.09800 - 1.10000) * per_unit_per_lot * volume
    which is asserted directly against the spec rather than hard-coded,
    because the spec is a live MT5 snapshot and pinning its numbers here
    would make this test fail on a legitimate re-ingest.
    """
    spec = spec_for(SYMBOL)
    flat = 1.10000
    n = 20
    highs = [flat + 0.00050] * n
    lows = [flat - 0.00050] * n
    closes = [flat] * n
    opens = [flat] * n
    # Bar 16 pierces the stop.
    lows[16] = 1.09799
    bars = _bars(highs, lows, closes, opens)

    atr_expected = 0.00100
    strategy = _EnterOnBar(signal_bar=14, sl_dist=2.0 * atr_expected, tp_dist=None)
    result = _engine().run(bars, strategy)

    assert result.n_trades == 1
    trade = result.trades[0]
    assert trade.entry_fill.filled_price == pytest.approx(1.10000, abs=1e-10)
    assert trade.sl_price == pytest.approx(1.09800, abs=1e-10)
    assert trade.exit_reason == EXIT_SL
    assert trade.exit_fill is not None
    assert trade.exit_fill.filled_price == pytest.approx(1.09800, abs=1e-10)

    per_unit = max(
        spec.trade_tick_value_profit, spec.trade_tick_value_loss
    ) / spec.trade_tick_size
    expected_gross = (1.09800 - 1.10000) * per_unit * trade.volume
    assert trade.gross_price_pnl == pytest.approx(expected_gross, abs=1e-6)
    # Net is gross minus the entry-side cost the engine charged.
    assert trade.realized_pnl == pytest.approx(
        expected_gross - trade.entry_fill.total_cost, abs=1e-6
    )
    assert trade.realized_pnl < 0.0, "a stop-out must be a loss here"


def test_tp_hit_at_exact_price_is_a_profit() -> None:
    """The mirror of the stop case: TP touched, P&L positive."""
    flat = 1.10000
    n = 20
    highs = [flat + 0.00050] * n
    lows = [flat - 0.00050] * n
    closes = [flat] * n
    opens = [flat] * n
    highs[16] = 1.10201          # pierces TP at 1.10200
    bars = _bars(highs, lows, closes, opens)

    strategy = _EnterOnBar(signal_bar=14, sl_dist=None, tp_dist=2.0 * 0.00100)
    result = _engine().run(bars, strategy)

    trade = result.trades[0]
    assert trade.tp_price == pytest.approx(1.10200, abs=1e-10)
    assert trade.exit_reason == EXIT_TP
    assert trade.exit_fill.filled_price == pytest.approx(1.10200, abs=1e-10)
    assert trade.gross_price_pnl > 0.0
    assert trade.realized_pnl > 0.0


# ---------------------------------------------------------------------------
# Pessimism
# ---------------------------------------------------------------------------

def test_pessimistic_both_touched_sl_first() -> None:
    """One bar spans both levels -> SL. Not negotiable.

    An OHLC bar does not record the order in which the high and the low
    were made. Resolving the ambiguity in the strategy's favour is how a
    backtest manufactures profit it never had.
    """
    flat = 1.10000
    n = 20
    highs = [flat + 0.00050] * n
    lows = [flat - 0.00050] * n
    closes = [flat] * n
    opens = [flat] * n
    # Bar 16 spans BOTH: low under the 1.09800 stop, high over the
    # 1.10200 target.
    lows[16] = 1.09700
    highs[16] = 1.10300
    bars = _bars(highs, lows, closes, opens)

    strategy = _EnterOnBar(
        signal_bar=14, sl_dist=2.0 * 0.00100, tp_dist=2.0 * 0.00100
    )
    result = _engine().run(bars, strategy)

    trade = result.trades[0]
    assert trade.exit_reason == EXIT_SL, (
        "a bar spanning both levels resolved to TP; SL must win"
    )
    assert trade.exit_fill.filled_price == pytest.approx(1.09800, abs=1e-10)
    assert result.exit_reasons.get(EXIT_TP, 0) == 0


def test_tp_same_bar_after_entry_pessimistic() -> None:
    """The ENTRY bar itself is evaluated, and it is pessimistic too.

    Evaluation runs from the entry bar onward INCLUSIVE, so a stop
    touched on the very bar the position opened still counts.
    """
    flat = 1.10000
    n = 20
    highs = [flat + 0.00050] * n
    lows = [flat - 0.00050] * n
    closes = [flat] * n
    opens = [flat] * n
    # Bar 15 is the FILL bar. Make it span both levels.
    lows[15] = 1.09700
    highs[15] = 1.10300
    bars = _bars(highs, lows, closes, opens)

    strategy = _EnterOnBar(
        signal_bar=14, sl_dist=2.0 * 0.00100, tp_dist=2.0 * 0.00100
    )
    result = _engine().run(bars, strategy)

    trade = result.trades[0]
    assert trade.exit_bar_index == 15, "the entry bar was not evaluated"
    assert trade.exit_reason == EXIT_SL


# ---------------------------------------------------------------------------
# Policy edges W1 must honour
# ---------------------------------------------------------------------------

def test_sltp_deferral_in_zero_spread_window() -> None:
    """SL due inside a no-trade hour fills at the first tradeable OPEN.

    Not at the stop price: that price was never available to trade at.
    """
    flat = 1.10000
    n = 26
    highs = [flat + 0.00050] * n
    lows = [flat - 0.00050] * n
    closes = [flat] * n
    opens = [flat] * n
    # Bars start at 00:00, so index 21 is hour 21.
    lows[21] = 1.09700
    opens[22] = 1.09950          # the first tradeable bar's open
    bars = _bars(highs, lows, closes, opens, start_hour=0)
    assert (int(bars.time[21]) // HOUR_S) % 24 == 21

    strategy = _EnterOnBar(signal_bar=14, sl_dist=2.0 * 0.00100, tp_dist=None)
    result = _engine(no_trade_hours=(21,)).run(bars, strategy)

    trade = result.trades[0]
    assert trade.exit_reason == EXIT_SLTP_DEFERRED_ZERO_SPREAD
    assert trade.exit_bar_index == 22
    assert trade.exit_fill.filled_price == pytest.approx(1.09950, abs=1e-10), (
        "the deferred exit took the stop price, not the reopening open"
    )
    assert result.sltp_deferred_zero_spread == 1
    assert result.blocked_zero_spread_events >= 1


def test_weekend_gap_fill_at_open() -> None:
    """Gapped past the stop -> fill at the open, which is worse.

    The spec's case: LONG stop at 1.09950, entry side closes at 1.10000,
    the next session OPENS at 1.09900 -- below the stop. Price was never
    available at 1.09950, so the fill is 1.09900 and the trade takes the
    extra loss. Filling at the level would hand the backtest a price
    that did not exist.

        sl_dist = 0.5 * ATR(14) = 0.5 * 0.00100 = 0.00050
        SL      = 1.10000 - 0.00050 = 1.09950
        open    = 1.09900  ->  1.09900 < 1.09950, so gapped
        fill    = 1.09900, NOT 1.09950
    """
    flat = 1.10000
    n = 20
    highs = [flat + 0.00050] * n
    lows = [flat - 0.00050] * n
    closes = [flat] * n
    opens = [flat] * n
    # Bar 16 opens BELOW the stop and never trades back up to it.
    opens[16] = 1.09900
    highs[16] = 1.09920
    lows[16] = 1.09850
    bars = _bars(highs, lows, closes, opens)

    strategy = _EnterOnBar(signal_bar=14, sl_dist=0.5 * 0.00100, tp_dist=None)
    result = _engine().run(bars, strategy)

    trade = result.trades[0]
    # The stop sits ABOVE the gapped open, so the two prices differ and
    # the assertion below is not satisfied by both branches at once.
    assert trade.sl_price == pytest.approx(1.09950, abs=1e-10)
    assert trade.exit_reason == EXIT_SL
    assert trade.exit_fill.filled_price == pytest.approx(
        1.09900, abs=1e-10
    ), "a gapped stop filled at its level instead of the available price"
    assert trade.exit_fill.filled_price < trade.sl_price, (
        "the gap must make the stop WORSE, not better"
    )


# ---------------------------------------------------------------------------
# Tripwires
# ---------------------------------------------------------------------------

def test_mgmt_pinned_none_until_implemented() -> None:
    """mgmt is pinned to NONE because nothing implements the others.

    BREAKEVEN / TRAIL_ATR / PARTIAL never reached the engine:
    dna_to_ma_config does not pass mgmt at all. Sampling them produced
    DNAs with distinct hashes and identical backtests, inflating the
    trial count the luck bar is set against.

    This fails the moment a non-NONE mgmt is sampled OR the projection
    starts carrying it, so the management types get implemented rather
    than silently re-enabled.
    """
    import dataclasses

    from engine.research.search.sampler import random_dna

    rng = np.random.default_rng(0)
    sampled = {random_dna(rng).mgmt for _ in range(500)}
    assert sampled == {"NONE"}, (
        f"sampler produced mgmt values {sampled}; only NONE is implemented"
    )

    # And the projection still ignores it, so nothing downstream can be
    # fooled into thinking management is active.
    dna = random_dna(np.random.default_rng(1))
    a = dna_to_ma_config(dna)
    b = dna_to_ma_config(dataclasses.replace(dna, mgmt="TRAIL_ATR"))
    assert a == b, (
        "dna_to_ma_config now varies with mgmt -- implement the management "
        "types, give them tests, and update this tripwire"
    )


def test_stops_are_quantized_to_symbol_digits() -> None:
    """A stop the broker could not quote is not a stop that could exist."""
    spec = spec_for(SYMBOL)
    flat = 1.10000
    n = 20
    bars = _bars(
        [flat + 0.00050] * n, [flat - 0.00050] * n, [flat] * n, [flat] * n
    )
    # A distance with far more precision than the symbol quotes.
    strategy = _EnterOnBar(signal_bar=14, sl_dist=0.0012345678901, tp_dist=None)
    result = _engine().run(bars, strategy)
    sl = result.trades[0].sl_price
    assert sl == pytest.approx(round(sl, spec.digits), abs=1e-12)


def test_no_stops_means_the_old_signal_only_behaviour() -> None:
    """A DNA asking for no stops is unaffected by any of this."""
    flat = 1.10000
    n = 20
    highs = [flat + 0.00050] * n
    lows = [flat - 0.00050] * n
    lows[16] = 1.09000           # would blow through any stop
    bars = _bars(highs, lows, [flat] * n, [flat] * n)

    result = _engine().run(
        bars, _EnterOnBar(signal_bar=14, sl_dist=None, tp_dist=None)
    )
    trade = result.trades[0]
    assert trade.sl_price is None
    assert trade.tp_price is None
    assert trade.exit_reason != EXIT_SL


def test_exit_reason_distribution_covers_every_trade() -> None:
    """The distribution is a partition, not a sample.

    It silently was not: _charge_swap used to rebuild the Trade object,
    orphaning the instance already in the trades list, so a position
    held across a day roll was recorded as never closed. On USDCADm D1
    that hid 121 of 125 trades.
    """
    from engine.research.backtest.strategies.ma_crossover import MACrossStrategy

    rng = np.random.default_rng(2)
    n = 600
    walk = np.cumsum(rng.normal(0.0, 0.0012, size=n)) + 1.10
    bars = _bars(
        (walk + 0.0008).tolist(), (walk - 0.0008).tolist(), walk.tolist()
    )
    cfg = MACrossConfig(fast=5, slow=20, volume=0.1, sl_atr=1.5, tp_atr=2.0)
    config = BacktestConfig(
        strategy_id="partition",
        symbol=SYMBOL,
        timeframe="H1",
        initial_equity=10_000.0,
        charge_swap=True,           # the condition that used to orphan trades
        charge_commission=True,
        slippage_mean_points=0.0,
        slippage_std_points=0.0,
    )
    result = BacktestEngine(
        spec=spec_for(SYMBOL), config=config, rng_seed=0
    ).run(bars, MACrossStrategy(cfg))

    assert result.n_trades > 0
    assert sum(result.exit_reasons.values()) == result.n_trades, (
        f"{sum(result.exit_reasons.values())} reasons for "
        f"{result.n_trades} trades: {result.exit_reasons}"
    )
    assert all(t.exit_reason is not None for t in result.trades)
