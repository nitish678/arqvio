"""A file replace that survives Windows readers.

``os.replace`` onto a file that another thread or process has open fails on Windows with WinError 5 (access denied)
-- and the app reads its records (vault, bots, cells, gene pool, launch state) while the engine rewrites them. The
replace is retried for a short while, then the error is raised as it was.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

ATTEMPTS = 60
PAUSE_S = 0.05


def replace(src: str | Path, dst: str | Path) -> None:
    for attempt in range(ATTEMPTS):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == ATTEMPTS - 1:
                raise
            time.sleep(PAUSE_S)


__all__ = ["replace"]
