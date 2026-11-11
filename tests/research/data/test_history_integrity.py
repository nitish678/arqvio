"""Real-money data safety: window, chronology and duplicate bars are never guessed."""
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from engine.research.cells import add_market

RATES = np.dtype([(name, dtype) for name, dtype in (
    ('time', '<i8'), ('open', '<f8'), ('high', '<f8'), ('low', '<f8'),
    ('close', '<f8'), ('tick_volume', '<u8'), ('spread', '<i4'), ('real_volume', '<u8'))])
START = datetime(2026, 1, 1, tzinfo=UTC)
END = START + timedelta(minutes=30_000)


def bars(start, count):
    out = np.zeros(count, dtype=RATES)
    out['time'] = int(START.timestamp()) + np.arange(start, start + count) * 60
    out['open'] = out['close'] = 1.2
    out['low'], out['high'], out['spread'] = 1.1, 1.3, 10
    return out


@pytest.fixture(autouse=True)
def no_wait(monkeypatch):
    monkeypatch.setattr(add_market, 'HISTORY_PAUSE_S', 0)


class Paged:
    TIMEFRAME_M1 = 1

    def terminal_info(self):
        return SimpleNamespace(maxbars=20_000)

    def copy_rates_range(self, *args):
        return None

    def copy_rates_from_pos(self, symbol, tf, pos, count):
        # MT5 page 0 is newest: rows within each page ascend, pages descend.
        return bars(20_000 - pos - count, count)


def test_positional_pages_return_oldest_first_with_all_fields_aligned():
    result = add_market.pull_bars(Paged(), 'GBPUSDm', 'M1', START, END)
    assert len(result['time']) == 20_000
    assert np.all(np.diff(result['time']) > 0)
    assert int(result['time'][0]) == int(START.timestamp())
    assert int(result['time'][-1]) == int(START.timestamp()) + 19_999 * 60


def test_identical_boundary_duplicates_are_counted_once():
    class Overlap(Paged):
        def copy_rates_from_pos(self, symbol, tf, pos, count):
            return bars(0 if pos else 9_999, count)
    result = add_market.pull_bars(Overlap(), 'GBPUSDm', 'M1', START, END)
    assert len(result['time']) == 19_999
    assert np.all(np.diff(result['time']) > 0)


def test_conflicting_duplicate_is_refused_instead_of_picking_a_price():
    class Conflict(Paged):
        def copy_rates_from_pos(self, symbol, tf, pos, count):
            result = bars(0 if pos else 9_999, count)
            if pos:
                result['close'][-1] = 1.25
            return result
    with pytest.raises(add_market.AddMarketRefused, match='conflicting duplicate'):
        add_market.pull_bars(Conflict(), 'GBPUSDm', 'M1', START, END)


def test_out_of_window_legacy_adapter_cannot_inflate_valid_bar_count():
    class FutureOnly:
        TIMEFRAME_M1 = 1
        def terminal_info(self):
            return SimpleNamespace(maxbars=3000)
        def copy_rates_range(self, *args):
            return None
        def copy_rates_from(self, *args):
            return bars(40_000, 3000)
    with pytest.raises(add_market.AddMarketRefused, match='0 M1 bars'):
        add_market.pull_bars(FutureOnly(), 'GBPUSDm', 'M1', START, END)


def test_direct_range_is_filtered_before_minimum_count_gate():
    class Outside(Paged):
        def copy_rates_range(self, *args):
            return bars(40_000, 3000)
        def copy_rates_from_pos(self, *args):
            return None
    with pytest.raises(add_market.AddMarketRefused, match='0 M1 bars'):
        add_market.pull_bars(Outside(), 'GBPUSDm', 'M1', START, END)


def test_range_calls_keep_utc_not_machine_local_naive_time():
    class ExplicitUTC(Paged):
        def copy_rates_range(self, symbol, tf, start, end):
            assert start.utcoffset() == end.utcoffset() == timedelta(0)
            return bars(0, 3000)
    assert len(add_market.pull_bars(ExplicitUTC(), 'GBPUSDm', 'M1', START, END)['time']) == 3000
