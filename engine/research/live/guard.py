"""The only door to ``order_send``: the account re-read immediately before every order.

Every order-capable path in engine/research/live goes through ``send_order``, which, on EVERY call and
immediately before the send:

1. re-reads ``account_info()`` -- a cached probe is never sufficient. A DEMO account may open and close.
   A REAL account may open only if its owner armed it (``arming``); otherwise ``RealAccountNotArmed``
   names the account and how to arm it. Closing the caller's OWN positions and removing its OWN orders
   (``closing=True``) is allowed on DEMO and REAL alike, armed or not, so a disarm, a kill switch or a
   flatten can always take exposure off. A contest or unknown mode raises ``UnknownAccountMode``;
2. refuses a request whose magic is outside the caller's own range (``ForeignMagicRefused``): code that
   may act only on its own magic can never send an order under someone else's.

``assert_tradable`` is the same opening check without a send; ``account_facts`` is the read alone.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from engine.research.accounts.arming import is_armed
from engine.research.mql5.magic import is_own


class LiveGuardError(RuntimeError):
    """A live action refused before it reached the broker."""


class RealAccountNotArmed(LiveGuardError):
    """The terminal is on a real-money account its owner has not armed."""


class UnknownAccountMode(LiveGuardError):
    """The account is neither DEMO nor REAL (contest, or a mode this code does not know)."""


class ForeignMagicRefused(LiveGuardError):
    """An order request carried a magic outside the caller's own range."""


@dataclass(frozen=True)
class AccountFacts:
    login: int
    server: str
    trade_mode: int
    margin_mode: int
    real: bool


def account_facts(mt5: Any) -> AccountFacts:
    """Read the account NOW; refuse a missing read and any mode other than DEMO or REAL."""
    acc = mt5.account_info()
    if acc is None:
        raise UnknownAccountMode(f"account_info() returned None ({mt5.last_error()}); refusing to act")
    mode = int(acc.trade_mode)
    if mode not in (mt5.ACCOUNT_TRADE_MODE_DEMO, mt5.ACCOUNT_TRADE_MODE_REAL):
        raise UnknownAccountMode(f"unknown or contest account mode {mode} (login {acc.login}, server {acc.server}); "
                                 "refusing to act")
    return AccountFacts(int(acc.login), str(acc.server), mode, int(acc.margin_mode),
                        real=mode == mt5.ACCOUNT_TRADE_MODE_REAL)


def assert_tradable(mt5: Any, *, arming: Path | None = None) -> AccountFacts:
    """Read the account NOW and refuse to OPEN on a real account its owner has not armed."""
    facts = account_facts(mt5)
    if facts.real and not is_armed(facts.login, facts.server, arming):
        raise RealAccountNotArmed(
            f"real-money account login {facts.login} server {facts.server} is not armed. The owner arms it once: "
            f"python -m engine.research.accounts.arming arm --login {facts.login} --server {facts.server}"
        )
    return facts


def send_order(mt5: Any, request: dict[str, Any], *, own_range: str, closing: bool = False,
               arming: Path | None = None) -> Any:
    """``order_send`` behind the own-magic check and the account re-read (arming checked unless closing)."""
    magic = int(request.get("magic", 0))
    if not is_own(magic, own_range):
        raise ForeignMagicRefused(f"order magic {magic} is not in own range {own_range!r}; refused")
    if closing:
        account_facts(mt5)
    else:
        assert_tradable(mt5, arming=arming)
    return mt5.order_send(request)


__all__ = [
    "AccountFacts",
    "ForeignMagicRefused",
    "LiveGuardError",
    "RealAccountNotArmed",
    "UnknownAccountMode",
    "account_facts",
    "assert_tradable",
    "send_order",
]
