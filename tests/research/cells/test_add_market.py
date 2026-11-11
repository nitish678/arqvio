"""Add market: a QUICK cell from a (fake) terminal -- pinned, sealed, resolved by the research stack, refused honestly."""
from __future__ import annotations

import json
import os
import shutil
from collections import namedtuple
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from engine.research.cells import add_market
from engine.research.registry import cells as registry

NOW = datetime(2026, 10, 20, 12, 0, tzinfo=UTC)
# the developer's research artifacts (real specs and spread matrices): ARQVIO_DEV_ARTIFACTS, else ~/ARQVIO/artifacts
REAL = Path(os.environ.get("ARQVIO_DEV_ARTIFACTS", str(Path.home() / "ARQVIO" / "artifacts")))
RATES = np.dtype([("time", "<i8"), ("open", "<f8"), ("high", "<f8"), ("low", "<f8"), ("close", "<f8"),
                  ("tick_volume", "<u8"), ("spread", "<i4"), ("real_volume", "<u8")])
TICKS = np.dtype([("time", "<i8"), ("bid", "<f8"), ("ask", "<f8")])


class FakeMt5:
    """Market data only, like the real module's read calls; a server clock on UTC unless offset_s says otherwise."""

    TIMEFRAME_H4 = 16388
    TIMEFRAME_M15 = 15
    COPY_TICKS_ALL = 6

    def __init__(self, *, n_bars: int = 3000, offered: tuple[str, ...] = ("EURUSDm",), offset_s: int = 0) -> None:
        self.n_bars, self.offered, self.offset_s = n_bars, offered, offset_s
        self.calls: list[str] = []

    def initialize(self, **kw):
        self.calls.append("initialize")
        return True

    def shutdown(self):
        self.calls.append("shutdown")

    def last_error(self):
        return (0, "ok")

    def account_info(self):
        return namedtuple("AccountInfo", "server trade_mode")("Exness-MT5Trial", 0)

    def symbol_select(self, symbol, enable):
        return symbol in self.offered

    def symbol_info(self, symbol):
        return namedtuple("SymbolInfo", "point")(0.00001)

    def symbol_info_tick(self, symbol):
        return namedtuple("Tick", "time")(int(NOW.timestamp()) + self.offset_s)  # the clock the run is told

    def symbols_get(self):
        return [namedtuple("Sym", "name visible")(s, True) for s in self.offered]

    def terminal_info(self):
        return namedtuple("TerminalInfo", "maxbars")(100_000)

    def copy_rates_from(self, symbol, tf, end, count):
        return None  # the fallback for a window past 'Max bars in chart' finds nothing more here

    def copy_ticks_range(self, symbol, start, end, flags):
        t = np.arange(int(start.replace(tzinfo=UTC).timestamp()), int(end.replace(tzinfo=UTC).timestamp()), 60)
        out = np.zeros(t.size, dtype=TICKS)
        out["time"], out["bid"] = t, 1.1
        out["ask"] = 1.1 + 0.00001 * (8 + (t // 3600) % 24)  # a spread that varies by hour
        return out

    def copy_rates_range(self, symbol, tf, start, end):
        self.calls.append("copy_rates_range")
        rng = np.random.default_rng(1)
        t0 = int(start.replace(tzinfo=UTC).timestamp())
        out = np.zeros(self.n_bars, dtype=RATES)
        close = 1.1 + np.cumsum(rng.normal(0, 0.001, self.n_bars))
        out["time"] = t0 + 14400 * np.arange(self.n_bars)
        out["open"], out["close"] = close - 0.0002, close
        out["high"], out["low"] = close + 0.001, close - 0.001
        out["spread"], out["tick_volume"] = 12, 100
        return out


@pytest.fixture(autouse=True)
def _no_history_pause(monkeypatch) -> None:
    monkeypatch.setattr(add_market, "HISTORY_PAUSE_S", 0.0)


@pytest.fixture
def home(isolated_arqvio_home: Path) -> Path:
    for sub in ("specs", "spread"):
        (isolated_arqvio_home / "artifacts" / sub).mkdir(parents=True)
    return isolated_arqvio_home


def seed_symbol_files(home: Path, symbol: str = "EURUSDm") -> None:
    """The real home's specs and spread files for a symbol (an add then keeps them)."""
    for sub in ("specs", "spread"):
        src = REAL / sub / f"{symbol}.json"
        if not src.is_file():
            pytest.skip(f"the real {sub} file for {symbol} is not on this machine")
        shutil.copy2(src, home / "artifacts" / sub / f"{symbol}.json")


def test_a_market_becomes_a_sealed_quick_cell_the_research_stack_resolves(home: Path) -> None:
    from engine.research.ga.config import PINNED_CELLS, cross_cell_for
    from engine.research.search.run_campaign_cli import load_target_audit

    seed_symbol_files(home)
    fake = FakeMt5()
    rec = add_market.run("EURUSDm", "H4", "2024-01-01", "2026-01-01", "usdcadm_h1", mt5=fake, terminal=None, now=NOW)
    assert rec["cell_id"] == "eurusdm_h4" and rec["window"]["bars"] == 3000
    assert (rec["specs"]["action"], rec["spread"]["action"]) == ("kept", "kept")
    assert fake.calls[0] == "initialize" and "shutdown" in fake.calls
    e = registry.get_cell("eurusdm_h4")
    assert (e["status"], e["annualisation_ppy"], e["cross_cell"]) == ("QUICK", 1512, "usdcadm_h1")
    assert registry.load_user() == [e]  # sealed: its own hash and the chain verify on every read
    assert PINNED_CELLS["eurusdm_h4"] == ("EURUSDm", "H4", rec["dataset_hash16"])
    assert cross_cell_for("eurusdm_h4") == "usdcadm_h1"
    assert "eurusdm_h4" not in registry.validated_cells()  # the repo registry's own readers never see it
    _header, rows = load_target_audit()
    row = rows[("EURUSDm", "H4")]
    assert (row["bars_returned"], row["time_basis"]) == ("3000", "utc")
    assert float(row["years_available"]) > 1.0
    with pytest.raises(add_market.AddMarketRefused, match="already exists"):
        add_market.run("EURUSDm", "H4", "2024-01-01", "2026-01-01", "usdcadm_h1", mt5=FakeMt5(), terminal=None,
                       now=NOW)
    add_market.remove_cell("eurusdm_h4")
    assert "eurusdm_h4" not in PINNED_CELLS


def test_a_tampered_user_cells_file_is_refused(home: Path) -> None:
    seed_symbol_files(home)
    add_market.run("EURUSDm", "H4", "2024-01-01", "2026-01-01", "usdcadm_h1", mt5=FakeMt5(), terminal=None, now=NOW)
    p = registry.user_cells_path()
    doc = json.loads(p.read_text(encoding="utf-8"))
    doc["entries"][0]["dataset_window"]["bars"] = 999999
    p.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(registry.RegistryIntegrityError):
        registry.load_user()


@pytest.mark.parametrize(("kw", "run_kw", "step"), [
    ({"offset_s": 9999}, {}, "time basis"),
    ({"n_bars": 500}, {}, "bars"),
    ({"offered": ()}, {}, "symbol"),
    ({}, {"cross_cell": "nope_h1"}, "cross cell"),
    ({}, {"tf": "W1"}, "timeframe"),
    ({}, {"start": "2026-01-01", "end": "2025-01-01"}, "window"),
])
def test_each_refusal_names_its_step_and_writes_no_cell(home: Path, kw: dict, run_kw: dict, step: str) -> None:
    seed_symbol_files(home)
    a = {"tf": "H4", "start": "2024-01-01", "end": "2026-01-01", "cross_cell": "usdcadm_h1"} | run_kw
    with pytest.raises(add_market.AddMarketRefused) as exc:
        add_market.run("EURUSDm", a["tf"], a["start"], a["end"], a["cross_cell"], mt5=FakeMt5(**kw), terminal=None,
                       now=NOW)
    assert exc.value.step == step
    assert registry.load_user() == []


def test_a_missing_spread_matrix_is_measured_from_ticks(home: Path) -> None:
    from engine.research.data.time_basis import probe_basis

    fake = FakeMt5()
    matrix = add_market.spread_matrix(fake, "EURUSDm", probe_basis(fake, "EURUSDm", now=NOW), now=NOW)
    assert [round(b, 6) for b in matrix.buckets[:3]] == [8.0, 9.0, 10.0]  # the hour-of-day spread, in points
    assert matrix.sample_count == 28 * 24 * 60 and matrix.time_basis == "server_equals_utc"


def test_the_ga_preflight_accepts_a_quick_cell(home: Path) -> None:
    """prepare_cell -- the spread preflight, the audit row, the snapshot depth, the fast factory and the canary --
    runs on a QUICK cell exactly as on a registry cell."""
    from engine.research.ga.runner import prepare_cell

    seed_symbol_files(home)
    add_market.run("EURUSDm", "H4", "2024-01-01", "2026-01-01", "usdcadm_h1", mt5=FakeMt5(), terminal=None, now=NOW)
    ev = prepare_cell("eurusdm_h4")
    assert (ev.symbol, ev.timeframe, ev.engine_mode_actual) == ("EURUSDm", "H4", "fast")
    assert ev.context["bars_loaded"] == 3000


def test_a_gmt_plus_2_server_breeds_in_its_own_clock(home: Path) -> None:
    """Owner 2026-11-01: a GMT+2/+3 broker breeds. Its bars and spread hours stay in the server's own clock (the live
    EA reads TimeCurrent(), the same clock), recorded as time_basis server_time with the measured offset, and the GA
    preflight accepts the cell (spread and audit agree on the clock)."""
    from engine.research.data.dataset_snapshot import load_meta
    from engine.research.ga.runner import prepare_cell

    seed_symbol_files(home)
    fake = FakeMt5(offset_s=7200)
    rec = add_market.run("EURUSDm", "H4", "2024-01-01", "2026-01-01", "usdcadm_h1", mt5=fake, terminal=None, now=NOW)
    pulled = rec.get("pull", rec)
    assert pulled["time_basis"] == "server_time" and pulled["server_offset_s"] == 7200
    assert load_meta(pulled["dataset_hash16"]).time_basis == "server_time"
    assert prepare_cell("eurusdm_h4").context["bars_loaded"] == 3000


def test_an_unclassifiable_server_clock_is_refused(home: Path) -> None:
    seed_symbol_files(home)
    with pytest.raises(add_market.AddMarketRefused, match="time basis") as exc:
        add_market.run("EURUSDm", "H4", "2024-01-01", "2026-01-01", "usdcadm_h1",
                       mt5=FakeMt5(offset_s=9999), terminal=None, now=NOW)
    assert "neither UTC nor GMT+2/+3" in exc.value.detail
