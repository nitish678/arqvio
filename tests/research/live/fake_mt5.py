"""An in-memory MetaTrader5 stand-in with hedging-account semantics, for offline tests.

Constants carry the real MetaTrader5 package values. Positions are per ticket;
a DEAL with ``position`` closes that ticket, a DEAL without one opens a new
position, PENDING adds an order, REMOVE deletes one. ``trade_mode`` can be
changed between calls, and ``refuse_close`` makes chosen tickets fail to close.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any


class FakeMT5:
    ACCOUNT_TRADE_MODE_DEMO, ACCOUNT_TRADE_MODE_CONTEST, ACCOUNT_TRADE_MODE_REAL = 0, 1, 2
    TRADE_ACTION_DEAL, TRADE_ACTION_PENDING, TRADE_ACTION_REMOVE = 1, 5, 8
    ORDER_TYPE_BUY, ORDER_TYPE_SELL, ORDER_TYPE_BUY_LIMIT, ORDER_TYPE_SELL_LIMIT = 0, 1, 2, 3
    POSITION_TYPE_BUY, POSITION_TYPE_SELL = 0, 1
    ORDER_FILLING_FOK, ORDER_FILLING_IOC, ORDER_FILLING_RETURN = 0, 1, 2
    ORDER_TIME_GTC = 0
    TRADE_RETCODE_DONE, TRADE_RETCODE_REJECT = 10009, 10006
    SYMBOL_TRADE_MODE_FULL = 4
    TIMEFRAME_H1 = 16385
    RATES_DTYPE = [("time", "<i8"), ("open", "<f8"), ("high", "<f8"), ("low", "<f8"), ("close", "<f8"),
                   ("tick_volume", "<u8"), ("spread", "<i4"), ("real_volume", "<u8")]

    def __init__(self, *, trade_mode: int = 0, login: int = 463965333, server: str = "Exness-MT5Trial17") -> None:
        self.trade_mode = trade_mode
        self.login = login
        self.server = server
        self.positions: dict[int, SimpleNamespace] = {}
        self.orders: dict[int, SimpleNamespace] = {}
        self.sent: list[dict[str, Any]] = []
        self.account_reads = 0
        self.refuse_close: set[int] = set()
        self.reject_opens = 0  # the next N opening deals are rejected by the "broker"
        self.tick_time = 1_790_000_000
        self.bid, self.ask = 1.37000, 1.37010
        self._next = 1000

    # -- read side -------------------------------------------------------
    def account_info(self) -> SimpleNamespace:
        self.account_reads += 1
        return SimpleNamespace(login=self.login, server=self.server, trade_mode=self.trade_mode, margin_mode=2,
                               balance=1000.0, equity=1000.0)

    def positions_get(self, symbol: str | None = None) -> tuple[SimpleNamespace, ...]:
        return tuple(p for p in self.positions.values() if symbol is None or p.symbol == symbol)

    def orders_get(self, symbol: str | None = None) -> tuple[SimpleNamespace, ...]:
        return tuple(o for o in self.orders.values() if symbol is None or o.symbol == symbol)

    def symbol_info(self, symbol: str) -> SimpleNamespace:
        return SimpleNamespace(name=symbol, filling_mode=3, digits=5, point=1e-5, volume_min=0.01, volume_step=0.01,
                               trade_mode=self.SYMBOL_TRADE_MODE_FULL, visible=True)

    def symbol_info_tick(self, symbol: str) -> SimpleNamespace:
        return SimpleNamespace(time=self.tick_time, bid=self.bid, ask=self.ask)

    def last_error(self) -> tuple[int, str]:
        return (1, "Success")

    # H1 rates: ``self.rates`` is a numpy structured array whose LAST row is the forming bar.
    rates: Any = None

    def copy_rates_from_pos(self, symbol: str, timeframe: int, start: int, count: int) -> Any:
        n = len(self.rates)
        hi = n - start
        return self.rates[max(0, hi - count):hi]

    def copy_rates_range(self, symbol: str, timeframe: int, date_from: Any, date_to: Any) -> Any:
        lo, hi = int(date_from.timestamp()), int(date_to.timestamp())
        return self.rates[(self.rates["time"] >= lo) & (self.rates["time"] <= hi)]

    # -- write side ------------------------------------------------------
    def _ticket(self) -> int:
        self._next += 1
        return self._next

    def plant_position(self, *, symbol: str, side: str, volume: float, magic: int) -> int:
        t = self._ticket()
        self.positions[t] = SimpleNamespace(
            ticket=t, symbol=symbol, type=self.POSITION_TYPE_BUY if side == "long" else self.POSITION_TYPE_SELL,
            volume=volume, price_open=self.ask if side == "long" else self.bid, sl=0.0, tp=0.0, magic=magic,
            comment="planted", time=self.tick_time)
        return t

    def plant_order(self, *, symbol: str, volume: float, price: float, magic: int) -> int:
        t = self._ticket()
        self.orders[t] = SimpleNamespace(ticket=t, symbol=symbol, type=self.ORDER_TYPE_BUY_LIMIT, volume_current=volume,
                                         price_open=price, magic=magic, comment="planted")
        return t

    def order_send(self, request: dict[str, Any]) -> SimpleNamespace:
        self.sent.append(dict(request))
        action = request["action"]
        if action == self.TRADE_ACTION_DEAL and "position" in request:
            t = int(request["position"])
            if t in self.refuse_close or t not in self.positions:
                return SimpleNamespace(retcode=self.TRADE_RETCODE_REJECT, deal=0, order=0, comment="refused")
            closing = self.ORDER_TYPE_SELL if self.positions[t].type == self.POSITION_TYPE_BUY else self.ORDER_TYPE_BUY
            if request["type"] != closing:  # the broker rejects a close sent on the position's own side
                return SimpleNamespace(retcode=self.TRADE_RETCODE_REJECT, deal=0, order=0, comment="wrong side")
            del self.positions[t]
            return SimpleNamespace(retcode=self.TRADE_RETCODE_DONE, deal=self._ticket(), order=self._ticket(), comment="")
        if action == self.TRADE_ACTION_DEAL:
            if self.reject_opens > 0:
                self.reject_opens -= 1
                return SimpleNamespace(retcode=self.TRADE_RETCODE_REJECT, deal=0, order=0, comment="rejected")
            side = "long" if request["type"] == self.ORDER_TYPE_BUY else "short"
            t = self.plant_position(symbol=request["symbol"], side=side, volume=request["volume"], magic=request["magic"])
            return SimpleNamespace(retcode=self.TRADE_RETCODE_DONE, deal=self._ticket(), order=t, comment="")
        if action == self.TRADE_ACTION_REMOVE:
            t = int(request["order"])
            if t not in self.orders:
                return SimpleNamespace(retcode=self.TRADE_RETCODE_REJECT, deal=0, order=0, comment="no order")
            del self.orders[t]
            return SimpleNamespace(retcode=self.TRADE_RETCODE_DONE, deal=0, order=t, comment="")
        if action == self.TRADE_ACTION_PENDING:
            t = self.plant_order(symbol=request["symbol"], volume=request["volume"], price=request["price"],
                                 magic=request["magic"])
            return SimpleNamespace(retcode=self.TRADE_RETCODE_DONE, deal=0, order=t, comment="")
        raise AssertionError(f"FakeMT5 does not implement action {action}")
