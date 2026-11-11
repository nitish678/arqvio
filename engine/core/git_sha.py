"""The commit a run was produced by.

Trial registry rows carry a git_sha so a result can be tied back to the
code that produced it. run_campaign_cli never passed one, so every row it
wrote was stamped "unknown" -- which makes the provenance column present
and useless, and that is worse than absent because it looks filled in.

Resolution order: `git rev-parse --short HEAD`, then ARQVIO_GIT_SHA, then the
packaged build's stamp (engine/core/build_stamp.py), then "unknown" as a last
resort. The installed app has no git and no .git: without the stamp every
campaign manifest was refused as incomplete (git_sha "unknown"). Never raises: a run must not fail because the
.git directory is missing, which is exactly the case in an installed
build.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

ENV_VAR = "ARQVIO_GIT_SHA"
UNKNOWN = "unknown"
_TIMEOUT_S = 2.0

_REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_git_sha(*, repo_root: Path | None = None) -> str:
    """The short SHA of HEAD, or an env override, or "unknown"."""
    root = _REPO_ROOT if repo_root is None else repo_root
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_S,
            check=False,
        )
        sha = proc.stdout.strip()
        if proc.returncode == 0 and sha:
            return sha
    except (OSError, subprocess.SubprocessError):
        # No git binary, no .git, or it hung. Fall through; never raise.
        pass

    env_sha = os.environ.get(ENV_VAR, "").strip()
    if env_sha:
        return env_sha

    from engine.core.build_stamp import BUILD_COMMIT

    if BUILD_COMMIT not in ("source", UNKNOWN):
        return BUILD_COMMIT[:7]
    return UNKNOWN


__all__ = ["ENV_VAR", "UNKNOWN", "resolve_git_sha"]
