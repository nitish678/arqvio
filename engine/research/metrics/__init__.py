"""Public API for the S4 research-metrics module.

Re-exports :func:`compute_metrics` (the orchestrator) and
:class:`ResearchMetrics` (the schema) so callers do:

    from engine.research.metrics import compute_metrics, ResearchMetrics
"""
from __future__ import annotations

from engine.research.metrics.compute import compute_metrics
from engine.research.metrics.schema import ResearchMetrics

__all__ = ["compute_metrics", "ResearchMetrics"]
