"""Typed exception hierarchy for ARQVIO.

WHAT: Root ARQVIOError + a few specialized subclasses.
WHY: Callers can catch by class; test assertions stay stable.
HOW: Plain exception inheritance; no magic.
TRADE-OFF: Tiny hierarchy; future stages may add more without changing callers.
TEST: tests/core/test_errors.py asserts inheritance and message preservation.
FAILURE: A missing import is a hard error; we want loud failures.
"""
from __future__ import annotations


class ARQVIOError(Exception):
    """Base class for all ARQVIO-raised errors."""


class ConfigError(ARQVIOError):
    """Raised when configuration is missing or invalid."""


class APIError(ARQVIOError):
    """Raised by the API layer for invalid requests or upstream failures."""


class DataError(ARQVIOError):
    """Raised when research/live data is missing, stale, or malformed."""


class RiskError(ARQVIOError):
    """Raised when a risk rule would be violated by a proposed action."""


class BrokerError(ARQVIOError):
    """Raised when MT5 returns an error or a worker is unreachable."""


__all__ = [
    "ARQVIOError",
    "ConfigError",
    "APIError",
    "DataError",
    "RiskError",
    "BrokerError",
]
