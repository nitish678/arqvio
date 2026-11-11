"""FastAPI server for the ARQVIO engine.

Stage 1 endpoints:
    GET  /healthz           -> {"status": "ok", "version": "...", "ts": "..."}
    GET  /api/v1/ping       -> {"pong": true, "version": "...", "ts": "..."}

S5 additions:
    GET  /api/v1/s5/audit       -> history-depth audit summary
    GET  /api/v1/s5/verdict     -> live campaign verdict
    GET  /api/v1/s5/runner      -> last search-runner result
    POST /api/v1/s5/launch      -> run the search runner synchronously

No business data, no fake counts, no fake metrics.
"""
from __future__ import annotations

import os

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from engine import __version__
from engine.api.routes_app import router as app_router
from engine.api.routes_app_campaigns import router as app_campaigns_router
from engine.api.routes_app_cells import router as app_markets_router
from engine.api.routes_app_deploy import router as app_deploy_router
from engine.api.routes_app_library import router as app_library_router
from engine.api.routes_app_keychain import router as app_keychain_router
from engine.api.routes_app_boot import router as app_boot_router
from engine.api.routes_app_terminals import router as app_terminals_router
from engine.api.routes_app_launch import log_router as app_log_router
from engine.api.routes_app_launch import router as app_launch_router
from engine.api.routes_app_support import router as app_support_router
from engine.api.routes_app_automation import portfolios_router as app_portfolios_router
from engine.api.routes_app_automation import router as app_automation_router
from engine.api.routes_app_settings import router as app_settings_router
from engine.api.routes_gene_pool import router as gene_pool_router
from engine.api.routes_s5 import router as s5_router
from engine.api.routes_s6 import router as s6_router
from engine.api.routes_s7 import router as s7_router
from engine.config.settings import get_settings
from engine.core.build_stamp import BUILD_COMMIT, BUILD_UTC
from engine.core.logging import configure_logging, get_logger
from engine.core.time import iso_utc
from engine.events.bus import get_default_bus
from engine.events.schemas import SystemHealthEvent


def _probe_mt5_live() -> bool:
    """Cheap MT5 reachability probe with a TTL cache.

    Called on every /healthz, so it must not hold the MT5 terminal open or
    re-initialize per request. Result is cached for 15s; a probe that cannot
    acquire/initialize within ~1s reports False (MT5 closed) rather than
    hanging the liveness check.
    """
    import time

    cache_ttl = 15.0
    now = time.monotonic()
    cached = _MT5_PROBE_CACHE
    if cached is not None and (now - cached[0]) < cache_ttl:
        return cached[1]

    live = False
    try:
        import MetaTrader5 as mt5

        # Best-effort: initialize against the RESOLVED terminal. This used
        # to be a second hardcoded copy of the path, escaped differently
        # from the one in real_backtest.py, so the probe and the loader
        # could disagree about where MT5 was and both report 'closed'.
        from engine.research.data.mt5_terminal import resolve_terminal

        resolution = resolve_terminal()
        if resolution.path is None:
            # Not found is not the same as closed; reported as itself.
            _set_mt5_probe_cache(now, False)
            return False
        ok = mt5.initialize(path=resolution.path)
        if ok:
            live = True
            mt5.shutdown()
    except Exception:
        live = False

    _set_mt5_probe_cache(now, live)
    return live


_MT5_PROBE_CACHE: tuple[float, bool] | None = None


def _set_mt5_probe_cache(ts: float, live: bool) -> None:
    global _MT5_PROBE_CACHE
    _MT5_PROBE_CACHE = (ts, live)


class HealthzResponse(BaseModel):
    status: str
    # The desktop shell reattaches to an engine still serving the port only when it is this app's (and this version).
    app: str = "arqvio"
    version: str
    api_port: int
    mt5_live: bool = False
    ts: str
    # Which source this engine was built from. The installed runtime reports the
    # commit and build time stamped into it by scripts/build_runtime.py; a
    # checkout running from source reports "source". Without these two fields
    # a stale sidecar is indistinguishable from a current one -- it answers
    # /healthz with the same version string either way. That is how the
    # installed app came to 404 on endpoints that existed in the repo.
    build_commit: str = "unknown"
    build_utc: str = "unknown"


class PingResponse(BaseModel):
    pong: bool
    version: str
    ts: str


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Engine startup/shutdown hooks."""
    settings = get_settings()
    configure_logging(level=settings.log_level)
    log = get_logger()
    log.info("ARQVIO engine starting (version={v}, port={p})", v=__version__, p=settings.api_port)
    get_default_bus()  # initialize default bus
    if os.environ.get("ARQVIO_APP") == "1":  # the app's engine runs saved automations (tests start it themselves)
        from engine.research.automation.runner import SCHEDULER

        SCHEDULER.start()
    yield
    log.info("ARQVIO engine stopping")


APP_ORIGINS: list[str] = [
    "http://tauri.localhost",
    "https://tauri.localhost",
    "tauri://localhost",
    "http://localhost:1420",
    "http://127.0.0.1:1420",
]


def create_app() -> FastAPI:
    """Application factory. Uvicorn's factory=True uses this."""
    app = FastAPI(
        title="ARQVIO Engine",
        version=__version__,
        lifespan=_lifespan,
        docs_url="/docs",
        redoc_url=None,
    )
    # Only the desktop app may call the engine from a browser context: the Tauri 2
    # webview origin (Windows: http://tauri.localhost; macOS/Linux: tauri://localhost)
    # and the Vite dev server. Any other web page is refused by the browser.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=APP_ORIGINS,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    log = get_logger()

    @app.get("/healthz", response_model=HealthzResponse, tags=["system"])
    async def healthz() -> HealthzResponse:
        """Liveness probe. Returns 'ok' iff the engine process is alive and reachable."""
        settings = get_settings()
        mt5_live = _probe_mt5_live()
        payload = HealthzResponse(
            status="ok",
            version=__version__,
            api_port=settings.api_port,
            mt5_live=mt5_live,
            ts=iso_utc(),
            build_commit=BUILD_COMMIT,
            build_utc=BUILD_UTC,
        )
        # Publish a health event for any subscribers (logs, future SSE bridge).
        get_default_bus().publish(
            SystemHealthEvent(
                status="ok",
                version=__version__,
                api_port=settings.api_port,
            )
        )
        return payload

    @app.get("/api/v1/mt5/status", tags=["system"])
    async def mt5_status() -> dict[str, object]:
        """Where the terminal is, and whether it answered.

        The UI shows the resolved path so a wrong one is visible instead of
        presenting as "MT5 closed". `found=False` and `live=False` are
        different failures and are reported separately.
        """
        from engine.research.data.mt5_terminal import resolve_terminal

        resolution = resolve_terminal()
        live = _probe_mt5_live()
        if not resolution.found:
            state = "DISCONNECTED"
        elif live:
            state = "CONNECTED"
        else:
            state = "DISCONNECTED"
        return {
            "state": state,
            "terminal_path": resolution.path,
            "terminal_source": resolution.source,
            "found": resolution.found,
            "live": live,
            "candidates_checked": list(resolution.candidates_checked),
            "ts": iso_utc(),
        }

    @app.get("/api/v1/symbols", tags=["data"])
    async def symbols() -> dict[str, object]:
        """The broker's catalog, via mt5.symbols_get(), plus the spread gate.

        Every symbol comes back DISABLED. Enabling one is an operator act,
        never a default, and a symbol whose spread row fails calibration is
        returned blocked with the reason attached so the UI can grey it out
        instead of silently offering it.

        When MT5 is unreachable this returns an empty catalog with a reason.
        It never returns a remembered or example list -- an empty catalog and
        a catalog of invented symbols look identical on screen, and only one
        of them is honest.
        """
        from engine.data.spread_calibration import evaluate, load, spread_dir
        from engine.research.data.mt5_source import detect_symbols
        from engine.research.data.mt5_terminal import resolve_terminal

        resolution = resolve_terminal()
        if not resolution.found:
            return {
                "symbols": [],
                "available": False,
                "reason": "MT5 terminal not found; nothing to enumerate",
                "terminal_path": None,
                "ts": iso_utc(),
            }
        try:
            catalog = detect_symbols(resolution.path or "")
        except Exception as exc:
            return {
                "symbols": [],
                "available": False,
                "reason": f"{type(exc).__name__}: {exc}",
                "terminal_path": resolution.path,
                "ts": iso_utc(),
            }

        rows: list[dict[str, object]] = []
        for info in catalog:
            matrix = load(info.name, spread_dir())
            if matrix is None:
                spread = {
                    "status": "UNAVAILABLE",
                    "blocks": False,
                    "reason": "no spread matrix calibrated for this symbol",
                }
            else:
                check = evaluate(matrix)
                spread = {
                    "status": check.status.value,
                    "blocks": check.blocks,
                    "reason": check.reason,
                }
            rows.append(
                {
                    "symbol": info.name,
                    "category": info.category,
                    "path": info.path,
                    "digits": info.digits,
                    "point": info.point,
                    "spread_points": info.spread_points,
                    # Never enabled by default. The operator turns it on.
                    "enabled": False,
                    "long": False,
                    "short": False,
                    "spread": spread,
                }
            )
        return {
            "symbols": rows,
            "available": True,
            "reason": None,
            "terminal_path": resolution.path,
            "count": len(rows),
            "ts": iso_utc(),
        }

    @app.get("/api/v1/ping", response_model=PingResponse, tags=["system"])
    async def ping() -> PingResponse:
        """Round-trip check for the Tauri shell."""
        return PingResponse(pong=True, version=__version__, ts=iso_utc())

    @app.exception_handler(HTTPException)
    async def _http_exception_handler(_request: object, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.detail, "status_code": exc.status_code},
        )

    log.debug("FastAPI app created (engine_version={v})", v=__version__)
    app.include_router(s5_router)
    app.include_router(s6_router)
    app.include_router(s7_router)
    app.include_router(gene_pool_router)
    app.include_router(app_router)
    app.include_router(app_keychain_router)
    app.include_router(app_campaigns_router)
    app.include_router(app_library_router)
    app.include_router(app_deploy_router)
    app.include_router(app_markets_router)
    app.include_router(app_terminals_router)
    app.include_router(app_boot_router)
    app.include_router(app_launch_router)
    app.include_router(app_log_router)
    app.include_router(app_support_router)
    app.include_router(app_automation_router)
    app.include_router(app_portfolios_router)
    app.include_router(app_settings_router)
    return app


# Re-exported for convenience.
__all__ = ["create_app", "HealthzResponse", "PingResponse"]


if __name__ == "__main__":
    import uvicorn

    from engine.config.settings import get_settings

    settings = get_settings()
    app = create_app()
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )

