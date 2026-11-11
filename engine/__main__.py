"""Entry point: `python -m engine` starts the ARQVIO engine.

Stage 1: starts the FastAPI/Uvicorn server with /healthz and /api/v1/ping.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on sys.path when running as `python -m engine`.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import uvicorn  # noqa: E402

from engine.config.settings import get_settings  # noqa: E402
from engine.core.build_stamp import BUILD_COMMIT, BUILD_UTC  # noqa: E402
from engine.core.logging import configure_logging  # noqa: E402

#: Modules a packaged sidecar may run in place of ``python -m <module>`` (a frozen binary has no ``-m``).
RUNNABLE_MODULES = ("engine.api.campaign_launch", "engine.research.ga.runner")


def run_module(argv: list[str]) -> int:
    """``arqvio-engine.exe --run-module <module> <args...>``: the packaged equivalent of
    ``python -m <module> <args...>``, for the whitelisted modules only."""
    import importlib

    if not argv or argv[0] not in RUNNABLE_MODULES:
        print(f"REFUSED: --run-module takes one of {RUNNABLE_MODULES}, got {argv[:1]}")
        return 2
    return int(importlib.import_module(argv[0]).main(argv[1:]) or 0)


def main() -> None:
    # A packaged sidecar re-executes itself for power-mode worker processes: let those children run.
    import multiprocessing

    multiprocessing.freeze_support()
    if sys.argv[1:2] == ["--run-module"]:
        raise SystemExit(run_module(sys.argv[2:]))
    # `arqvio-engine.exe --print-build` prints the provenance and exits. This is
    # the check a packaged sidecar has to pass before an installer is built:
    # it answers which commit the binary is, without starting a server on a
    # port that may already be in use by the engine under test.
    if "--print-build" in sys.argv[1:]:
        from engine import __version__

        print(f"version {__version__}")
        print(f"commit  {BUILD_COMMIT}")
        print(f"built   {BUILD_UTC}")
        raise SystemExit(0)

    settings = get_settings()
    configure_logging(level=settings.log_level)
    uvicorn.run(
        "engine.api.server:create_app",
        factory=True,
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower(),
        reload=False,
    )


if __name__ == "__main__":
    main()
