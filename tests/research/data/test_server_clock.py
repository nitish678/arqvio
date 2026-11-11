"""The server clock, measured even when the market is closed (weekends failed every launch before)."""
from __future__ import annotations

from collections import namedtuple
from datetime import UTC, datetime

import numpy as np
import pytest

from engine.research.cells import add_market
from engine.research.data import server_clock

NOW = datetime(2026, 10, 24, 18, 0, tzinfo=UTC)  # a Saturday evening, UTC
T = int(NOW.timestamp())
Tick = namedtuple("Tick", "time")
Sym = namedtuple("Sym", "name visible")


class Clocky:
    """Ticks per symbol: an int (seconds before NOW; negative = in the future) or None (no tick yet)."""

    def __init__(self, ticks: dict[str, int | None], *, offered: tuple[str, ...] | None = None) -> None:
        self.ticks = ticks
        self.offered = offered or tuple(ticks)
        self.selected: list[str] = []

    def symbol_info_tick(self, s):
        age = self.ticks.get(s)
        return Tick(0) if age is None else Tick(T - age)

    def symbols_get(self):
        return [Sym(n, n in ("EURUSD",)) for n in self.offered]

    def symbol_select(self, s, on):
        self.selected.append(s)
        return s in self.offered

    def last_error(self):
        return (0, "ok")


@pytest.fixture(autouse=True)
def _home(isolated_arqvio_home, monkeypatch):
    monkeypatch.setattr(server_clock, "FIRST_TICK_WAIT_S", 0.0)
    monkeypatch.setattr(add_market, "HISTORY_PAUSE_S", 0.0)
    del isolated_arqvio_home  # the home itself is all this fixture provides


def test_a_trading_symbol_measures_the_clock_and_it_is_remembered() -> None:
    c = server_clock.resolve(Clocky({"EURUSD": 5}), "EURUSD", server="Broker-Demo", now=NOW)
    assert c.probe.basis == "server_equals_utc" and c.source == "symbol"
    assert server_clock.recall("Broker-Demo")["basis"] == "server_equals_utc"


def test_a_closed_market_is_measured_on_a_round_the_clock_symbol() -> None:
    mt5 = Clocky({"XAUUSD": 20 * 3600, "BTCUSD": 3}, offered=("XAUUSD", "BTCUSD", "EURUSD"))
    c = server_clock.resolve(mt5, "XAUUSD", server="Broker-Demo", now=NOW)
    assert c.probe.basis == "server_equals_utc" and c.source == "always-open BTCUSD"
    assert "BTCUSD" in mt5.selected and not server_clock.closed_market(c)


def test_a_gmt_plus_2_server_is_still_named_by_its_clock_on_a_weekend() -> None:
    mt5 = Clocky({"XAUUSD": 18 * 3600, "BTCUSDm": -7200}, offered=("XAUUSD", "BTCUSDm"))
    c = server_clock.resolve(mt5, "XAUUSD", server="Other-Live", now=NOW)
    assert c.probe.basis == "broker_offset" and c.probe.measured_offset_s == 7200


def test_without_a_crypto_symbol_a_server_once_measured_utc_is_remembered() -> None:
    server_clock.resolve(Clocky({"EURUSD": 1}), "EURUSD", server="Exness-MT5Trial17", now=NOW)
    c = server_clock.resolve(Clocky({"XAUUSD": 30 * 3600}), "XAUUSD", server="Exness-MT5Trial17", now=NOW)
    assert c.probe.basis == "server_equals_utc" and c.source.startswith("remembered")


def test_nothing_measurable_stays_unresolved_and_says_the_market_is_closed() -> None:
    c = server_clock.resolve(Clocky({"XAUUSD": 30 * 3600}), "XAUUSD", server="Never-Seen", now=NOW)
    assert c.probe.basis == "unresolved" and server_clock.closed_market(c)
    msg = add_market._closed_refusal(c)
    assert "market is closed" in msg and "30.0 h old" in msg


def test_a_symbol_without_a_tick_yet_is_not_read_as_time_zero() -> None:
    c = server_clock.resolve(Clocky({"USOIL": None, "BTCUSD": 2}, offered=("USOIL", "BTCUSD")), "USOIL",
                             server="Broker-Demo", now=NOW)
    assert c.probe.basis == "server_equals_utc" and c.source == "always-open BTCUSD"


RATES = np.dtype([("time", "<i8"), ("open", "<f8"), ("high", "<f8"), ("low", "<f8"), ("close", "<f8"),
                  ("tick_volume", "<u8"), ("spread", "<i4"), ("real_volume", "<u8")])


class Capped:
    """A terminal whose 'Max bars in chart' is shorter than the window: copy_rates_range returns nothing."""

    TIMEFRAME_M5 = 5

    def terminal_info(self):
        return namedtuple("TI", "maxbars")(3000)

    def copy_rates_range(self, *a):
        return None

    def copy_rates_from(self, symbol, tf, end, count):
        out = np.zeros(count, dtype=RATES)
        out["time"] = int(end.replace(tzinfo=UTC).timestamp()) - 300 * np.arange(count, 0, -1)
        out["close"] = 1.1
        return out


def test_a_window_past_the_terminals_bars_gets_the_newest_bars_it_has() -> None:
    start, end = datetime(2016, 10, 23, tzinfo=UTC), datetime(2026, 10, 23, tzinfo=UTC)
    bars = add_market.pull_bars(Capped(), "XAUUSD", "M5", start, end)
    assert len(bars["time"]) == 3000 and bars["time"][-1] < int(end.timestamp())


def test_too_few_bars_even_from_the_newest_is_still_refused() -> None:
    class Tiny(Capped):
        def terminal_info(self):
            return namedtuple("TI", "maxbars")(500)

    with pytest.raises(add_market.AddMarketRefused, match="500 M5 bars"):
        add_market.pull_bars(Tiny(), "XAUUSD", "M5", datetime(2016, 1, 1, tzinfo=UTC), datetime(2026, 1, 1, tzinfo=UTC))


def test_history_still_downloading_is_asked_again() -> None:
    class Downloading(Capped):
        calls = 0

        def copy_rates_range(self, symbol, tf, start, end):
            Downloading.calls += 1
            if Downloading.calls < 3:
                return None  # the terminal answers empty while it downloads the history
            return Capped.copy_rates_from(self, symbol, tf, end, 5000)

    bars = add_market.pull_bars(Downloading(), "XAUUSD", "M5", datetime(2016, 1, 1, tzinfo=UTC),
                                datetime(2026, 1, 1, tzinfo=UTC))
    assert len(bars["time"]) == 5000 and Downloading.calls == 3


def test_a_weekend_reads_the_offset_from_the_weekly_close() -> None:
    """NOW is Saturday 24 Oct 18:00 UTC; Friday's close was 17:00 New York = 21:00 UTC (US summer time)."""
    close = datetime(2026, 10, 23, 21, 0, tzinfo=UTC)
    assert server_clock.last_weekly_close(NOW) == close
    utc_last = int(close.timestamp()) - 3                      # a UTC server's last tick, 20:59:57
    gmt3_last = int(close.timestamp()) + 3 * 3600 - 1          # a GMT+3 server shows 23:59:59
    c = server_clock.resolve(Clocky({"XAUUSD": T - utc_last}), "XAUUSD", server="Weekend-UTC", now=NOW)
    assert c.probe.basis == "server_equals_utc" and c.source == "weekly close" and not server_clock.closed_market(c)
    c = server_clock.resolve(Clocky({"XAUUSD": T - gmt3_last}), "XAUUSD", server="Weekend-GMT3", now=NOW)
    assert c.probe.basis == "broker_offset" and c.probe.measured_offset_s == 10800
    assert "GMT+3" in add_market._basis_refusal(c.probe)


def test_winter_close_is_an_hour_later_in_utc() -> None:
    sat = datetime(2026, 11, 7, 12, 0, tzinfo=UTC)  # after the US clocks went back on 1 Nov
    assert server_clock.last_weekly_close(sat) == datetime(2026, 11, 6, 22, 0, tzinfo=UTC)
