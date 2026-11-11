"""The power-check context: (cell, ppy, N) for one evaluated campaign (DR-2026-10-08-u2 §2).

The cell and its annualisation come from the U1 registry; the luck bar N is the evaluated campaign's own
``run.dsr_n_final``. Nothing is defaulted: a missing or inconsistent input is refused, because a power curve on an
invented N or ppy would be a number nobody ran.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from engine.core.errors import ConfigError
from engine.research.registry.cells import get_cell


class PowerContextError(ConfigError):
    """The power-check context cannot be resolved from the registry and the campaign manifest."""


@dataclass(frozen=True)
class PowerContext:
    cell_id: str
    symbol: str
    timeframe: str
    dataset_hash16: str
    ppy: int
    primary_n: int
    campaign_id: str

    @property
    def cell(self) -> tuple[str, str, str]:
        """(symbol, timeframe, dataset_id), as load_cell_bars_from_snapshot takes it."""
        return (self.symbol, self.timeframe, self.dataset_hash16)

    def as_dict(self) -> dict[str, Any]:
        return {"cell_id": self.cell_id, "symbol": self.symbol, "timeframe": self.timeframe,
                "dataset_hash16": self.dataset_hash16, "ppy": self.ppy, "primary_n": self.primary_n,
                "campaign_id": self.campaign_id}


def _manifest(manifest: Mapping[str, Any] | Path) -> Mapping[str, Any]:
    if isinstance(manifest, Path):
        if not manifest.is_file():
            raise PowerContextError(f"campaign manifest not found: {manifest}")
        return json.loads(manifest.read_text(encoding="utf-8"))
    return manifest


def resolve_power_context(cell_id: str, manifest: Mapping[str, Any] | Path) -> PowerContext:
    """The context for a power check of ``cell_id`` against the campaign whose manifest_v2 is ``manifest``."""
    entry = get_cell(cell_id)  # an unknown cell is refused by the registry (UnknownCellError)
    if entry["status"] != "VALIDATED":
        raise PowerContextError(f"{cell_id} is {entry['status']}, not VALIDATED")
    doc = _manifest(manifest)
    run = doc.get("run") if isinstance(doc, Mapping) else None
    if not isinstance(run, Mapping):
        raise PowerContextError("campaign manifest has no 'run' block")
    for key in ("dsr_n_final", "campaign_id", "config"):
        if key not in run:
            raise PowerContextError(f"campaign manifest run block lacks {key!r}")
    n = run["dsr_n_final"]
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise PowerContextError(f"run.dsr_n_final must be an int >= 1, got {n!r}")
    cfg = run["config"]
    searched = (cfg.get("symbol"), cfg.get("timeframe"), cfg.get("dataset")) if isinstance(cfg, Mapping) else None
    registered = (entry["symbol"], entry["timeframe"], entry["dataset_hash16"])
    if searched != registered:
        raise PowerContextError(
            f"campaign {run['campaign_id']!r} searched {searched}, not the registry's {cell_id} {registered}")
    return PowerContext(cell_id=cell_id, symbol=entry["symbol"], timeframe=entry["timeframe"],
                        dataset_hash16=entry["dataset_hash16"], ppy=entry["annualisation_ppy"], primary_n=n,
                        campaign_id=str(run["campaign_id"]))
