"""Order requests and account snapshots for the live layer (no send happens here).

Request builders return plain ``order_send`` dicts; sending is
``guard.send_order``'s job alone. The deviation (max slippage, points) is the
oracle's slippage cap for the symbol's asset class (FX 10, metals 30).
"""
from __future__ import annotations

from typing import Any

from engine.research.costs.slippage import slippage_cap_for_symbol

POSITION_FIELDS = ("ticket", "symbol", "type", "volume", "price_open", "sl", "tp", "magic", "comment", "time")
ORDER_FIELDS = ("ticket", "symbol", "type", "volume_current", "price_open", "magic", "comment")


def filling_for(mt5: Any, symbol: str) -> int:
    """The order filling type the symbol allows: IOC, else FOK, else RETURN."""
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"symbol_info({symbol!r}) is None: {mt5.last_error()}")
    mask = int(info.filling_mode)
    if mask & 2:
        return mt5.ORDER_FILLING_IOC
    if mask & 1:
        return mt5.ORDER_FILLING_FOK
    return mt5.ORDER_FILLING_RETURN


def deviation_points(symbol: str) -> int:
    return int(slippage_cap_for_symbol(symbol))


def market_open_request(mt5: Any, *, symbol: str, side: str, volume: float, magic: int, comment: str) -> dict[str, Any]:
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError(f"symbol_info_tick({symbol!r}) is None: {mt5.last_error()}")
    buy = side == "long"
    return {
        "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": float(volume),
        "type": mt5.ORDER_TYPE_BUY if buy else mt5.ORDER_TYPE_SELL,
        "price": float(tick.ask if buy else tick.bid), "deviation": deviation_points(symbol),
        "magic": int(magic), "comment": comment, "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": filling_for(mt5, symbol),
    }


def close_position_request(mt5: Any, position: Any, *, comment: str) -> dict[str, Any]:
    """Close one position by ticket (a hedging account closes per position)."""
    tick = mt5.symbol_info_tick(position.symbol)
    if tick is None:
        raise RuntimeError(f"symbol_info_tick({position.symbol!r}) is None: {mt5.last_error()}")
    was_buy = int(position.type) == mt5.POSITION_TYPE_BUY
    return {
        "action": mt5.TRADE_ACTION_DEAL, "symbol": position.symbol, "volume": float(position.volume),
        "type": mt5.ORDER_TYPE_SELL if was_buy else mt5.ORDER_TYPE_BUY, "position": int(position.ticket),
        "price": float(tick.bid if was_buy else tick.ask), "deviation": deviation_points(position.symbol),
        "magic": int(position.magic), "comment": comment, "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": filling_for(mt5, position.symbol),
    }


def remove_order_request(mt5: Any, order: Any, *, comment: str) -> dict[str, Any]:
    return {"action": mt5.TRADE_ACTION_REMOVE, "order": int(order.ticket), "magic": int(order.magic),
            "comment": comment}


def snapshot(mt5: Any) -> dict[str, list[dict[str, Any]]]:
    """Every open position and pending order on the account, field by field."""
    def rows(items: Any, fields: tuple[str, ...]) -> list[dict[str, Any]]:
        return [{f: getattr(x, f) for f in fields} for x in (items or ())]

    return {"positions": rows(mt5.positions_get(), POSITION_FIELDS),
            "orders": rows(mt5.orders_get(), ORDER_FIELDS)}


__all__ = [
    "ORDER_FIELDS",
    "POSITION_FIELDS",
    "close_position_request",
    "deviation_points",
    "filling_for",
    "market_open_request",
    "remove_order_request",
    "snapshot",
]
