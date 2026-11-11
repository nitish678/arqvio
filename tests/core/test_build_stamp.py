"""The build stamp must never let a stale sidecar pass as a current one.

The failure this guards: the installer shipped a sidecar 1h40m older than the
endpoints it had to serve. /healthz answered "ok" with a version string that
was identical to the current source's, so the app looked healthy while
returning 404 on /api/v1/mt5/status and /api/v1/symbols. Nothing in the tree
tied a binary to a commit.

These tests assert the two halves of the fix: a checkout reports "source"
rather than inventing a commit, and a build records the commit it came from.
"""
from __future__ import annotations

import re

# engine/core/build_stamp.py falls back to "source" when the generated
# module is absent. A commit is a full 40-char sha, optionally "+dirty".
_SHA_RE = re.compile(r"^[0-9a-f]{40}(\+dirty)?$")


def test_checkout_reports_source_not_a_commit():
    """With no generated stamp, the module must not name a commit.

    Naming HEAD here would be wrong: a checkout that has been edited is not
    the commit it sits on, and a checkout is running from source, not from a
    binary.
    """
    from engine.core.build_stamp import BUILD_COMMIT, BUILD_UTC

    # The generated module is git-ignored, so a test run from a checkout
    # normally takes the fallback path.
    if BUILD_COMMIT == "source":
        assert BUILD_UTC == "source"
    else:
        # A build tree: then it must be a real sha, not a placeholder.
        assert _SHA_RE.match(BUILD_COMMIT), BUILD_COMMIT


def test_build_stamp_module_exposes_both_fields():
    """Both fields exist and are non-empty strings, whatever the source."""
    from engine.core import build_stamp

    assert isinstance(build_stamp.BUILD_COMMIT, str)
    assert isinstance(build_stamp.BUILD_UTC, str)
    assert build_stamp.BUILD_COMMIT.strip()
    assert build_stamp.BUILD_UTC.strip()


def test_clean_rejects_empty_and_non_strings():
    """A malformed generated stamp degrades to the fallback, never crashes."""
    from engine.core.build_stamp import _clean

    assert _clean("", "fallback") == "fallback"
    assert _clean("   ", "fallback") == "fallback"
    assert _clean(None, "fallback") == "fallback"
    assert _clean(123, "fallback") == "fallback"
    assert _clean(" abc123 ", "fallback") == "abc123"


def test_healthz_reports_build_provenance():
    """The two fields are on the response model, so a client can read them.

    This is what makes "is this binary current?" answerable from outside the
    process without guessing from a version string.
    """
    from engine.api.server import HealthzResponse

    fields = HealthzResponse.model_fields
    assert "build_commit" in fields
    assert "build_utc" in fields


def test_print_build_exits_without_starting_a_server():
    """--print-build must not bind a port.

    The packaged smoke test runs this against a binary while the engine under
    test already holds 8741; binding would make the check fail for a reason
    that has nothing to do with the binary being current.
    """
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-m", "engine", "--print-build"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "commit" in result.stdout
    assert "built" in result.stdout