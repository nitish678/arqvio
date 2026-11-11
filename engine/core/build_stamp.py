"""Build provenance for the running engine.

WHAT: Reports the commit and UTC time this engine's source was built from.
WHY:  the build stamp exists because the installer once shipped a
      sidecar 1h40m older than the endpoints it had to serve, so the installed
      app answered 404 on /api/v1/mt5/status and /api/v1/symbols while
      /healthz reported a perfectly healthy engine. Nothing in the tree tied
      a binary to a commit, so nothing could notice.
HOW:  scripts/build_runtime.py writes engine/core/_build_stamp.py into the
      runtime's copy of the engine (never into the checkout). A checkout
      running straight from source has no such file, and reports "source"
      rather than a stale or invented commit.
TRADE-OFF: The generated file is git-ignored; it exists only in a build tree.
TEST: tests/core/test_build_stamp.py asserts the source fallback, and the
      packaged smoke test asserts a real 40-char sha in the binary.
FAILURE: Never raises. A missing or malformed stamp degrades to "unknown",
      which the caller can see, rather than breaking /healthz.
"""
from __future__ import annotations

# A checkout has no generated stamp: these are the fallbacks.
_SOURCE_COMMIT = "source"
_SOURCE_UTC = "source"

try:  # pragma: no cover - present only inside a packaged build
    from engine.core._build_stamp import BUILD_COMMIT as _BUILD_COMMIT
    from engine.core._build_stamp import BUILD_UTC as _BUILD_UTC
except ImportError:  # pragma: no cover - the normal path from a checkout
    _BUILD_COMMIT = _SOURCE_COMMIT
    _BUILD_UTC = _SOURCE_UTC


def _clean(value: object, fallback: str) -> str:
    """Return a non-empty string, or the fallback."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    return fallback


BUILD_COMMIT: str = _clean(_BUILD_COMMIT, "unknown")
BUILD_UTC: str = _clean(_BUILD_UTC, "unknown")

__all__ = ["BUILD_COMMIT", "BUILD_UTC"]