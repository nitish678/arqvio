"""Structural look-ahead-safe feature registry."""
from engine.research.features.registry import (
    BarWindow,
    DEFAULT_REGISTRY,
    FeatureFn,
    FeatureRegistrationError,
    FeatureRegistry,
    FeatureSpec,
    LookaheadError,
)
from engine.research.features.rsi import DEFAULT_RSI_PERIOD, rsi_series

__all__ = [
    "BarWindow",
    "DEFAULT_REGISTRY",
    "DEFAULT_RSI_PERIOD",
    "FeatureFn",
    "FeatureRegistrationError",
    "FeatureRegistry",
    "FeatureSpec",
    "LookaheadError",
    "rsi_series",
]
