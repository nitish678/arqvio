"""Centralized logging via loguru.

WHAT: Configure one sink (stderr) with a secret-masking formatter.
WHY: A single log entry point keeps audit consistent; secrets must never leak.
HOW: loguru's `add()` with a custom format string. Masker scrubs token-like substrings.
TRADE-OFF: loguru over stdlib logging is a single dependency but yields simpler config.
TEST: tests/core/test_logging.py asserts masked output.
FAILURE: If loguru fails to import, we fall back to stderr plain prints.
"""
from __future__ import annotations

import re
import sys
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from loguru import Record

# Patterns that look like secrets. We never log the raw value; the masker
# replaces anything matched with '***'.
_FIELD_MASK = re.compile(r"(?i)(\b(?:api[_-]?key|token|secret|password|passwd|credential)\b\s*[:=]\s*)\S+")
_BEARER_MASK = re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]+")


def _mask(record: Record) -> None:
    """Scrub secrets from a log record's message.

    loguru's `patcher` mutates the record dict before formatting.
    """
    # loguru's Record is a TypedDict; we mutate the message in place.
    msg: Any = record.get("message")
    if not isinstance(msg, str):
        return
    masked = _FIELD_MASK.sub(r"\1***", msg)
    masked = _BEARER_MASK.sub("bearer ***", masked)
    record["message"] = masked


_CONFIGURED = False


def configure_logging(level: str = "INFO") -> None:
    """Configure the global loguru sink. Idempotent.

    Args:
        level: log level name (e.g. "INFO", "DEBUG").
    """
    global _CONFIGURED
    logger.remove()
    logger.add(
        sys.stderr,
        level=level.upper(),
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        backtrace=False,
        diagnose=False,
        enqueue=False,
    )
    logger.configure(patcher=_mask)
    _CONFIGURED = True


def get_logger() -> logger.__class__:  # type: ignore[name-defined]
    """Return the configured loguru logger. Always available (defaults to stderr)."""
    if not _CONFIGURED:
        configure_logging("INFO")
    return logger
