"""The kill switch: flatten ALL own-magic positions and pending orders, never another magic.

Triggered by a flag file (``Common\\Files\\ARQVIO_S6\\KILL``, which generated EAs
also watch) or by a direct ``trigger`` call. ``flatten`` makes one
deterministic pass, not a polling loop:

1. the account is re-read (DEMO or REAL; closing own exposure never needs arming, see ``guard``);
2. every position whose magic lies in the own range is closed by ticket, and
   every own pending order is removed; anything else is listed as FOREIGN and
   never touched;
3. the account is re-read, and an own-magic position or order still present
   raises ``KillSwitchIncomplete`` naming it.

Every action and its retcode go to the evidence log verbatim.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from engine.research.live.guard import account_facts, send_order
from engine.research.live.orders import close_position_request, remove_order_request
from engine.research.mql5.magic import LIVE_RANGE, is_own
from engine.research.mql5.terminal import arqvio_files_dir

KILL_FLAG = "KILL"


class KillSwitchIncomplete(RuntimeError):
    """Own-magic exposure survived a flatten."""


@dataclass
class FlattenReport:
    closed: list[dict[str, Any]] = field(default_factory=list)
    removed: list[dict[str, Any]] = field(default_factory=list)
    foreign_positions: list[int] = field(default_factory=list)
    foreign_orders: list[int] = field(default_factory=list)
    own_left: list[int] = field(default_factory=list)


class KillSwitch:
    def __init__(self, mt5: Any, *, own_range: str = LIVE_RANGE, flag_dir: Path | None = None,
                 log: Callable[[str], None] = print) -> None:
        self.mt5 = mt5
        self.own_range = own_range
        self._flag_dir = flag_dir
        self.log = log

    def flag_path(self) -> Path:
        return (self._flag_dir if self._flag_dir is not None else arqvio_files_dir()) / KILL_FLAG

    def triggered(self) -> bool:
        return self.flag_path().is_file()

    def trigger(self, reason: str) -> Path:
        p = self.flag_path()
        p.write_text(f"{datetime.now(tz=UTC).isoformat()} {reason}\n", encoding="ascii")
        self.log(f"KILL TRIGGERED flag={p} reason={reason}")
        return p

    def clear(self) -> None:
        p = self.flag_path()
        if p.is_file():
            p.unlink()
            self.log(f"KILL FLAG CLEARED {p}")

    def flatten(self) -> FlattenReport:
        mt5 = self.mt5
        facts = account_facts(mt5)
        self.log(f"KILL FLATTEN start {datetime.now(tz=UTC).isoformat()} login={facts.login} server={facts.server} "
                 f"trade_mode={facts.trade_mode} own_range={self.own_range}")
        report = FlattenReport()
        for p in mt5.positions_get() or ():
            if not is_own(p.magic, self.own_range):
                report.foreign_positions.append(int(p.ticket))
                self.log(f"KILL skip FOREIGN position ticket={p.ticket} symbol={p.symbol} magic={p.magic}")
                continue
            res = send_order(mt5, close_position_request(mt5, p, comment="arqvio-kill"), own_range=self.own_range,
                             closing=True)
            row = {"ticket": int(p.ticket), "symbol": p.symbol, "magic": int(p.magic), "volume": float(p.volume),
                   "retcode": None if res is None else int(res.retcode), "deal": None if res is None else int(res.deal)}
            report.closed.append(row)
            self.log(f"KILL close own position {row} last_error={mt5.last_error()}")
        for o in mt5.orders_get() or ():
            if not is_own(o.magic, self.own_range):
                report.foreign_orders.append(int(o.ticket))
                self.log(f"KILL skip FOREIGN order ticket={o.ticket} symbol={o.symbol} magic={o.magic}")
                continue
            res = send_order(mt5, remove_order_request(mt5, o, comment="arqvio-kill"), own_range=self.own_range,
                             closing=True)
            row = {"ticket": int(o.ticket), "symbol": o.symbol, "magic": int(o.magic),
                   "retcode": None if res is None else int(res.retcode)}
            report.removed.append(row)
            self.log(f"KILL remove own order {row} last_error={mt5.last_error()}")
        left = [int(x.ticket) for x in (mt5.positions_get() or ()) if is_own(x.magic, self.own_range)]
        left += [int(x.ticket) for x in (mt5.orders_get() or ()) if is_own(x.magic, self.own_range)]
        report.own_left = left
        self.log(f"KILL FLATTEN end own_left={left} foreign_positions={report.foreign_positions} "
                 f"foreign_orders={report.foreign_orders}")
        if left:
            raise KillSwitchIncomplete(f"own-magic exposure survived the flatten: tickets {left}")
        return report


__all__ = ["KILL_FLAG", "FlattenReport", "KillSwitch", "KillSwitchIncomplete"]
