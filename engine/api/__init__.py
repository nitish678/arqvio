"""HTTP API surface.

Stage 1: FastAPI factory in engine.api.server.
"""
from engine.api.server import create_app

__all__ = ["create_app"]
