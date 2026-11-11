"""The bounded worker: BOOT -> PROBE -> ARMED -> TRADING -> FLATTENING -> HALTED.

One worker trades one symbol under one own magic (live role ``worker``) on a DEMO account, or on a
REAL account its owner armed (``arming``). Its caps are part of its HASHED config and are the
owner's to set: volume (default 0.01), own positions (default 1; an open beyond it is refused, so
there is no averaging by accident), own orders per UTC day (default 6). Closing own exposure never
needs arming.

* PROBE: the account asserted tradable (DEMO, or REAL and armed), the magic registry loaded, the time basis probed
  (``time_basis.probe_basis``), the kill flag absent. Any failure halts with a
  named reason.
* Every TRADING cycle: the account re-asserted tradable; the basis re-probed (``unresolved``
  refuses the cycle: no reconciliation action, no order); the kill flag checked;
  then RECONCILE -- the world-model (own tickets this worker believes are open,
  with symbol, magic, volume, side) against broker truth (``positions_get`` /
  ``orders_get`` filtered to the own magic). Equal prints ``RECON OK``. Unequal
  prints ``RECON MISMATCH`` with both sides, flattens the own magic and halts
  with ``ReconciliationMismatch``. Foreign magic is counted, never touched.
* Then the intent source speaks (``LifecycleProbeSource``: a scripted
  open -> hold -> close; ``OracleSignalSource``: the oracle strategy on closed
  H1 bars) and the order goes through ``guard.send_order`` (the account re-read
  immediately before the send).

Reconciliation data never enters the hashed config. The operational arguments
(cycle period, cycle cap, log) stay outside it and are recorded.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from engine.research.data.mt5_source import BarData
from engine.research.data.time_basis import probe_basis
from engine.research.live.guard import assert_tradable, send_order
from engine.research.live.kill_switch import KillSwitch
from engine.research.live.orders import close_position_request, market_open_request
from engine.research.mql5.magic import LIVE_RANGE, live_magic, load_table
from engine.research.mql5.records import write_json_atomic

BOOT, PROBE, ARMED, TRADING, FLATTENING, HALTED = "BOOT", "PROBE", "ARMED", "TRADING", "FLATTENING", "HALTED"

MICRO_VOLUME = 0.01
MAX_OWN_POSITIONS = 1
DAILY_ORDER_CAP = 6
COMMENT_MAX = 31  # MT5 order comments are at most 31 characters


def _comment(reason: str) -> str:
    return f"arqvio-w {reason}"[:COMMENT_MAX]


class WorkerConfigError(ValueError):
    """A worker config that cannot run (non-positive caps, unknown source)."""


class ReconciliationMismatch(RuntimeError):
    """The own-magic world-model disagreed with broker truth."""


@dataclass(frozen=True)
class WorkerConfig:
    """Hashed: what the worker IS, including the owner's caps."""

    symbol: str
    source: str  # "lifecycle_probe" | "oracle_signal"
    volume: float = MICRO_VOLUME
    max_positions: int = MAX_OWN_POSITIONS
    daily_order_cap: int = DAILY_ORDER_CAP
    probe_side: str = "long"
    probe_hold_cycles: int = 3
    probe_cap_check: bool = True
    dna_hash: str = ""

    def __post_init__(self) -> None:
        if not self.volume > 0:
            raise WorkerConfigError(f"volume {self.volume} must be positive")
        if self.max_positions < 1:
            raise WorkerConfigError(f"max_positions {self.max_positions} must be at least 1")
        if self.daily_order_cap < 1:
            raise WorkerConfigError(f"daily_order_cap {self.daily_order_cap} must be at least 1")
        if self.source not in ("lifecycle_probe", "oracle_signal"):
            raise WorkerConfigError(f"unknown intent source {self.source!r}")
        if self.probe_side not in ("long", "short") or self.probe_hold_cycles < 1:
            raise WorkerConfigError("probe_side must be long|short and probe_hold_cycles >= 1")

    def config_hash(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Operational:
    """Not hashed: how this run was operated."""

    cycle_seconds: float = 15.0
    max_cycles: int = 20


@dataclass(frozen=True)
class OwnPosition:
    ticket: int
    symbol: str
    magic: int
    volume: float
    side: str


@dataclass
class Intent:
    action: str  # "open" | "close"
    side: str = "long"
    reason: str = ""


class LifecycleProbeSource:
    """One full lifecycle, counted from BROKER-CONFIRMED state only.

    It asks to open (again on every cycle while the reconciled world-model is empty -- a gate may
    refuse an open). From the first cycle whose reconciled world shows the position, it (optionally)
    asks for a second open, which the caps must refuse, holds ``probe_hold_cycles`` cycles, then asks to
    close (again while the position is still there). It is ``done`` only when a position was seen
    open and the world is empty again after the close."""

    def __init__(self, cfg: WorkerConfig) -> None:
        self.cfg = cfg
        self.open_seen_at: int | None = None
        self.cap_checked = False
        self.close_requested = False
        self.done = False

    def next(self, cycle: int, world: dict[int, OwnPosition]) -> Intent | None:
        if self.done:
            return None
        if self.open_seen_at is None:
            if not world:
                return Intent("open", self.cfg.probe_side, "LIFECYCLE_OPEN")
            self.open_seen_at = cycle
        if self.close_requested:
            if world:
                return Intent("close", reason="LIFECYCLE_CLOSE")
            self.done = True
            return None
        if self.cfg.probe_cap_check and not self.cap_checked:
            self.cap_checked = True
            return Intent("open", self.cfg.probe_side, "CAP_CHECK_SECOND_OPEN")
        if cycle >= self.open_seen_at + self.cfg.probe_hold_cycles:
            self.close_requested = True
            return Intent("close", reason="LIFECYCLE_CLOSE")
        return None


class OracleSignalSource:
    """The oracle strategy object on the symbol's CLOSED H1 bars. A new closed bar -> one on_bar call ->
    its intent, mapped to open/close (the oracle's decide-at-close, fill-at-next-open).

    The history is ANCHORED: the first call fixes the oldest bar, and every later call reads from that
    same bar to the newest closed one, so the window grows by one bar per bar exactly as the oracle's
    does (the strategies cache series by window length; a sliding fixed-length window would serve
    stale values). The forming bar is never included. The families are time-independent, so the bars'
    server timestamps need no conversion; the worker's basis probe still gates every cycle."""

    def __init__(self, mt5: Any, symbol: str, strategy: Any, *, history_bars: int = 2000) -> None:
        self.mt5, self.symbol, self.strategy, self.history = mt5, symbol, strategy, history_bars
        self.state: dict[str, object] = {}
        self.anchor: int | None = None
        self.last_bar_time: int | None = None
        self.done = False

    def _closed_rates(self) -> Any:
        if self.anchor is None:
            rates = self.mt5.copy_rates_from_pos(self.symbol, self.mt5.TIMEFRAME_H1, 1, self.history)
            if rates is not None and len(rates):
                self.anchor = int(rates[0]["time"])
            return rates
        rates = self.mt5.copy_rates_range(self.symbol, self.mt5.TIMEFRAME_H1,
                                          datetime.fromtimestamp(self.anchor, tz=UTC), datetime(2100, 1, 1, tzinfo=UTC))
        current = self.mt5.copy_rates_from_pos(self.symbol, self.mt5.TIMEFRAME_H1, 0, 1)
        if rates is None or current is None or not len(current):
            return None
        return rates[rates["time"] < current[0]["time"]]

    def next(self, cycle: int, world: dict[int, OwnPosition]) -> Intent | None:
        rates = self._closed_rates()
        if rates is None or len(rates) == 0:
            return None
        newest = int(rates[-1]["time"])
        if self.last_bar_time is not None and newest == self.last_bar_time:
            return None
        self.last_bar_time = newest
        bars = BarData(time=np.asarray(rates["time"], dtype=np.int64), open=np.asarray(rates["open"], dtype=np.float64),
                       high=np.asarray(rates["high"], dtype=np.float64), low=np.asarray(rates["low"], dtype=np.float64),
                       close=np.asarray(rates["close"], dtype=np.float64),
                       tick_volume=np.asarray(rates["tick_volume"], dtype=np.int64),
                       spread=np.asarray(rates["spread"], dtype=np.int64),
                       real_volume=np.asarray(rates["real_volume"], dtype=np.int64), symbol=self.symbol, timeframe="H1")
        intent = self.strategy.on_bar(len(rates) - 1, bars, self.state)
        if intent is None:
            return None
        if intent.volume == 0:
            return Intent("close", reason="ORACLE_CLOSE") if world else None
        return Intent("open", intent.direction, "ORACLE_ENTRY")


@dataclass
class Worker:
    mt5: Any
    cfg: WorkerConfig
    op: Operational
    source: Any
    log: Callable[[str], None] = print
    sleep: Callable[[float], None] = time.sleep
    kill: KillSwitch | None = None
    on_cycle: Callable[[int, "Worker"], None] | None = None
    max_spread_points: float | None = None
    no_trade_hours: tuple[int, ...] = ()
    status_path: Path | None = None  # opt-in status snapshot for the read-only API (records.worker_status_path)
    state: str = BOOT
    halt_reason: str = ""
    world: dict[int, OwnPosition] = field(default_factory=dict)
    orders_today: dict[str, int] = field(default_factory=dict)
    refused_cycles: int = 0
    fills: list[dict[str, Any]] = field(default_factory=list)  # broker-confirmed (DONE) orders only
    reconciled_while_open: int = 0
    last_recon: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.magic = live_magic("worker")
        if self.kill is None:
            self.kill = KillSwitch(self.mt5, log=self.log)

    # -- evidence -------------------------------------------------------
    def _say(self, msg: str) -> None:
        tick = self.mt5.symbol_info_tick(self.cfg.symbol)
        server = "?" if tick is None else datetime.fromtimestamp(int(tick.time), tz=UTC).isoformat()
        self.log(f"{datetime.now(tz=UTC).isoformat()} server={server} state={self.state} {msg}")

    def _halt(self, reason: str) -> None:
        self.state = HALTED
        self.halt_reason = reason
        self._say(f"HALTED reason={reason}")
        self._status(None)

    def _status(self, recon: dict[str, Any] | None) -> None:
        """The latest state and reconciliation, for the read-only API. Never in a hashed config."""
        if self.status_path is None:
            return
        if recon is not None:
            self.last_recon = recon
        write_json_atomic(self.status_path, {
            "schema": "arqvio.s6.worker_status", "state": self.state, "halt_reason": self.halt_reason,
            "symbol": self.cfg.symbol, "magic": self.magic, "config_hash": self.cfg.config_hash(),
            "last_reconciliation": self.last_recon, "fills": self.fills,
            "orders_today": self.orders_today, "refused_cycles": self.refused_cycles,
            "updated_utc": datetime.now(tz=UTC).isoformat(),
        })

    # -- states ---------------------------------------------------------
    def probe(self) -> bool:
        self.state = PROBE
        facts = assert_tradable(self.mt5)
        load_table()
        basis = probe_basis(self.mt5, self.cfg.symbol)
        self._say(f"PROBE account={'REAL' if facts.real else 'DEMO'} login={facts.login} server={facts.server} trade_mode={facts.trade_mode} "
                  f"magic={self.magic} config_hash={self.cfg.config_hash()} config={asdict(self.cfg)} "
                  f"operational={asdict(self.op)} basis={basis.basis} delta_s={basis.delta_s:.2f}")
        if basis.basis == "unresolved":
            self._halt(f"BasisUnresolved delta_s={basis.delta_s:.1f}")
            return False
        if self.kill.triggered():
            self._halt(f"KillFlagPresent {self.kill.flag_path()}")
            return False
        self.state = ARMED
        self._say("ARMED")
        return True

    def _truth(self) -> tuple[dict[int, OwnPosition], list[int], int]:
        own: dict[int, OwnPosition] = {}
        foreign = 0
        for p in self.mt5.positions_get(symbol=self.cfg.symbol) or ():
            if int(p.magic) == self.magic:
                side = "long" if int(p.type) == self.mt5.POSITION_TYPE_BUY else "short"
                own[int(p.ticket)] = OwnPosition(int(p.ticket), p.symbol, int(p.magic), float(p.volume), side)
            else:
                foreign += 1
        own_orders = [int(o.ticket) for o in (self.mt5.orders_get(symbol=self.cfg.symbol) or ())
                      if int(o.magic) == self.magic]
        return own, own_orders, foreign

    def reconcile(self, cycle: int) -> bool:
        truth, own_orders, foreign = self._truth()
        ok = truth == self.world and not own_orders
        recon = {"cycle": cycle, "ok": ok, "world": sorted(self.world), "truth": sorted(truth),
                 "own_orders": own_orders, "foreign": foreign, "utc": datetime.now(tz=UTC).isoformat()}
        if ok:
            self.reconciled_while_open += bool(self.world)
            self._say(f"CYCLE {cycle} RECON OK own={sorted(self.world)} foreign={foreign} "
                      f"orders_today={self.orders_today.get(self._day(), 0)}")
        else:
            self._say(f"CYCLE {cycle} RECON MISMATCH world={[asdict(v) for v in self.world.values()]} "
                      f"truth={[asdict(v) for v in truth.values()]} own_orders={own_orders} foreign={foreign}")
        self._status(recon)
        return ok

    def _day(self) -> str:
        return datetime.now(tz=UTC).date().isoformat()

    def _refuse(self, cycle: int, why: str, intent: Intent) -> None:
        self._say(f"CYCLE {cycle} CAP REFUSED {why} intent={asdict(intent)}")

    def act(self, cycle: int, intent: Intent) -> None:
        day = self._day()
        if intent.action == "open":
            if len(self.world) >= self.cfg.max_positions:
                return self._refuse(cycle, f"max_positions={self.cfg.max_positions} (no averaging)", intent)
            if self.orders_today.get(day, 0) >= self.cfg.daily_order_cap:
                return self._refuse(cycle, f"daily_order_cap={self.cfg.daily_order_cap}", intent)
            hour = datetime.now(tz=UTC).hour
            if hour in self.no_trade_hours:
                return self._refuse(cycle, f"no_trade_hour={hour}", intent)
            info, tick = self.mt5.symbol_info(self.cfg.symbol), self.mt5.symbol_info_tick(self.cfg.symbol)
            spread_pts = (tick.ask - tick.bid) / info.point
            if self.max_spread_points is not None and spread_pts > self.max_spread_points:
                return self._refuse(cycle, f"spread {spread_pts:.1f} > max_spread_points {self.max_spread_points}",
                                    intent)
            req = market_open_request(self.mt5, symbol=self.cfg.symbol, side=intent.side, volume=self.cfg.volume,
                                      magic=self.magic, comment=_comment(intent.reason))
            res = send_order(self.mt5, req, own_range=LIVE_RANGE)
            self.orders_today[day] = self.orders_today.get(day, 0) + 1
            self._say(f"CYCLE {cycle} ORDER open {intent.side} {self.cfg.volume} retcode={res.retcode} "
                      f"order={res.order} deal={res.deal} reason={intent.reason}")
            if res.retcode == self.mt5.TRADE_RETCODE_DONE:
                self.world[int(res.order)] = OwnPosition(int(res.order), self.cfg.symbol, self.magic,
                                                         float(self.cfg.volume), intent.side)
                self.fills.append({"cycle": cycle, "action": "open", "ticket": int(res.order), "deal": int(res.deal)})
            return None
        for ticket in list(self.world):
            pos = next((p for p in self.mt5.positions_get(symbol=self.cfg.symbol) or () if int(p.ticket) == ticket), None)
            if pos is None:
                continue
            res = send_order(self.mt5, close_position_request(self.mt5, pos, comment=_comment(intent.reason)),
                             own_range=LIVE_RANGE, closing=True)
            self.orders_today[day] = self.orders_today.get(day, 0) + 1
            self._say(f"CYCLE {cycle} ORDER close ticket={ticket} retcode={res.retcode} deal={res.deal} "
                      f"reason={intent.reason}")
            if res.retcode == self.mt5.TRADE_RETCODE_DONE:
                del self.world[ticket]
                self.fills.append({"cycle": cycle, "action": "close", "ticket": ticket, "deal": int(res.deal)})
        return None

    def flatten_own(self, why: str) -> None:
        """Close every position and remove every order carrying the worker's OWN magic; nothing else."""
        self.state = FLATTENING
        self._say(f"FLATTENING reason={why}")
        for p in self.mt5.positions_get(symbol=self.cfg.symbol) or ():
            if int(p.magic) != self.magic:
                continue
            res = send_order(self.mt5, close_position_request(self.mt5, p, comment=_comment("FLATTEN")),
                             own_range=LIVE_RANGE, closing=True)
            self._say(f"FLATTEN close ticket={p.ticket} magic={p.magic} retcode={res.retcode} deal={res.deal}")
        truth, own_orders, foreign = self._truth()
        self.world = {}
        self._say(f"FLATTEN done own_left={sorted(truth)} own_orders_left={own_orders} foreign={foreign}")

    def run(self) -> str:
        if not self.probe():
            return self.halt_reason
        self.state = TRADING
        for cycle in range(self.op.max_cycles):
            assert_tradable(self.mt5)
            basis = probe_basis(self.mt5, self.cfg.symbol)
            if basis.basis == "unresolved":
                self.refused_cycles += 1
                self._say(f"CYCLE {cycle} REFUSED basis unresolved delta_s={basis.delta_s:.1f}")
            elif self.kill.triggered():
                self.flatten_own("kill flag")
                self._halt("KillSwitch")
                return self.halt_reason
            elif not self.reconcile(cycle):
                world = [asdict(v) for v in self.world.values()]
                self.flatten_own("reconciliation mismatch")
                self._halt("ReconciliationMismatch")
                raise ReconciliationMismatch(f"cycle {cycle}: own-magic world-model {world} disagreed with broker "
                                             "truth; own magic flattened, worker HALTED")
            else:
                intent = self.source.next(cycle, self.world)
                if intent is not None:
                    self.act(cycle, intent)
                if getattr(self.source, "done", False):
                    self.flatten_own("lifecycle complete")
                    self._halt("Completed")
                    return self.halt_reason
            if self.on_cycle is not None:
                self.on_cycle(cycle, self)
            self.sleep(self.op.cycle_seconds)
        self.flatten_own("cycle cap reached")
        self._halt("CycleCap")
        return self.halt_reason


__all__ = [
    "ARMED",
    "BOOT",
    "DAILY_ORDER_CAP",
    "FLATTENING",
    "HALTED",
    "Intent",
    "LifecycleProbeSource",
    "MAX_OWN_POSITIONS",
    "MICRO_VOLUME",
    "Operational",
    "OracleSignalSource",
    "OwnPosition",
    "PROBE",
    "ReconciliationMismatch",
    "TRADING",
    "Worker",
    "WorkerConfig",
    "WorkerConfigError",
]
