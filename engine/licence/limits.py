"""The ONE definition of the free caps.

Ground rule: caps are defined once, here in the engine. The licence server copies them into the tokens it signs and
the UI only reads them; neither keeps its own list. A cap that lives in two places drifts, and then the UI promises
what the engine refuses.

Free:  markets XAUUSD / GOLD / USDCAD / USDJPY, timeframes H1 / H2, power eco / medium, smart mode off, automation
       (deploy) locked, deep history off, one seat.
Paid:  unlimited.
``NO_LIMITS`` is what the app uses today: the owner decided the app stays free with no lock (docs/LICENCE.md).

Brokers decorate symbol names ("XAUUSDm", "XAUUSD.pro", "#GOLD"), so markets are matched by their base token, not by
the raw string (``base_token``). ``max_assets`` and ``max_tfs`` for the free plan are simply the sizes of its lists.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

__all__ = [
    "FREE_LIMITS",
    "FREE_SEATS",
    "NO_LIMITS",
    "PAID_LIMITS",
    "Limits",
    "base_token",
    "check_launch",
    "market_allowed",
    "power_allowed",
    "timeframe_allowed",
]

FREE_MARKETS = ("XAUUSD", "GOLD", "USDCAD", "USDJPY")
FREE_TFS = ("H1", "H2")
FREE_POWER_MODES = ("eco", "medium")
FREE_SEATS = 1
MAX_SUFFIX_LEN = 4  # "m", "c", "pro", "ecn", "raw": longer tails are a different instrument, not a decoration.
_ALNUM_RUN = re.compile(r"[0-9A-Za-z]+")


@dataclass(frozen=True, slots=True)
class Limits:
    """A plan's caps. ``None`` means unlimited / any; a bool is a feature switch."""

    max_assets: int | None = None
    max_tfs: int | None = None
    smart_mode: bool = True
    deploy: bool = True
    deep_history: bool = True
    power_modes: tuple[str, ...] | None = None
    free_tfs: tuple[str, ...] | None = None
    free_markets: tuple[str, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        """The JSON form signed into a token's ``limits``."""
        out = asdict(self)
        for name in ("power_modes", "free_tfs", "free_markets"):
            if out[name] is not None:
                out[name] = list(out[name])
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Limits:
        """Read a token's ``limits``. A missing switch is read as OFF: an old token never grants more than it said."""
        def as_tuple(value: Any) -> tuple[str, ...] | None:
            return None if value is None else tuple(str(item) for item in value)

        return cls(
            max_assets=data.get("max_assets"),
            max_tfs=data.get("max_tfs"),
            smart_mode=bool(data.get("smart_mode", False)),
            deploy=bool(data.get("deploy", False)),
            deep_history=bool(data.get("deep_history", False)),
            power_modes=as_tuple(data.get("power_modes", list(FREE_POWER_MODES))),
            free_tfs=as_tuple(data.get("free_tfs", list(FREE_TFS))),
            free_markets=as_tuple(data.get("free_markets", list(FREE_MARKETS))),
        )


FREE_LIMITS = Limits(
    max_assets=len(FREE_MARKETS),
    max_tfs=len(FREE_TFS),
    smart_mode=False,
    deploy=False,
    deep_history=False,
    power_modes=FREE_POWER_MODES,
    free_tfs=FREE_TFS,
    free_markets=FREE_MARKETS,
)
PAID_LIMITS = Limits()
NO_LIMITS = Limits()


def base_token(symbol: str, known: tuple[str, ...] = FREE_MARKETS) -> str:
    """The market a broker symbol stands for: "XAUUSDm" and "XAUUSD.pro" are both "XAUUSD".

    The first alphanumeric run, upper-cased; if it is a known market plus a short broker tail, the known market.
    """
    match = _ALNUM_RUN.search(symbol or "")
    if match is None:
        return ""
    head = match.group(0).upper()
    for market in sorted(known, key=len, reverse=True):
        if head.startswith(market) and len(head) - len(market) <= MAX_SUFFIX_LEN:
            return market
    return head


def market_allowed(symbol: str, limits: Limits) -> bool:
    """Is this broker symbol inside the plan's markets?"""
    if limits.free_markets is None:
        return True
    return base_token(symbol, limits.free_markets) in limits.free_markets


def timeframe_allowed(timeframe: str, limits: Limits) -> bool:
    """Is this timeframe ("H1", "h2", ...) inside the plan's timeframes?"""
    return limits.free_tfs is None or str(timeframe).strip().upper() in limits.free_tfs


def power_allowed(mode: str, limits: Limits) -> bool:
    """Is this power mode ("eco", "medium", ...) inside the plan's modes?"""
    return limits.power_modes is None or str(mode).strip().casefold() in limits.power_modes


def _market_reasons(markets: list[str], limits: Limits) -> list[str]:
    reasons = [f"market {symbol} is not in your plan" for symbol in markets if not market_allowed(symbol, limits)]
    known = limits.free_markets or ()
    distinct = {base_token(symbol, known) for symbol in markets}
    if limits.max_assets is not None and len(distinct) > limits.max_assets:
        reasons.append(f"{len(distinct)} markets chosen; your plan allows {limits.max_assets}")
    return reasons


def _timeframe_reasons(timeframes: list[str], limits: Limits) -> list[str]:
    reasons = [f"timeframe {tf} is not in your plan" for tf in timeframes if not timeframe_allowed(tf, limits)]
    distinct = {str(tf).strip().upper() for tf in timeframes}
    if limits.max_tfs is not None and len(distinct) > limits.max_tfs:
        reasons.append(f"{len(distinct)} timeframes chosen; your plan allows {limits.max_tfs}")
    return reasons


def check_launch(request: dict[str, Any], limits: Limits) -> list[str]:
    """Every reason this launch is outside the plan; an empty list means it may start.

    ``request`` keys (all optional): markets, timeframes, power, smart_mode, deploy, deep_history.
    """
    reasons = _market_reasons(list(request.get("markets") or []), limits)
    reasons += _timeframe_reasons(list(request.get("timeframes") or []), limits)
    power = request.get("power")
    if power is not None and not power_allowed(power, limits):
        reasons.append(f"power mode {power} is not in your plan")
    if request.get("smart_mode") and not limits.smart_mode:
        reasons.append("smart mode is not in your plan")
    if request.get("deploy") and not limits.deploy:
        reasons.append("automation (deploy) is locked on your plan")
    if request.get("deep_history") and not limits.deep_history:
        reasons.append("deep history is not in your plan")
    return reasons
