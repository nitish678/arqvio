"""Configuration layer.

Stage 1: Pydantic Settings with env override + defaults.
"""
from .settings import Settings, get_settings

__all__ = ["Settings", "get_settings"]
