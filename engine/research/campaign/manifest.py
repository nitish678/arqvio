"""Campaign manifest: what produced a number (B3), and how far to trust it (C4).

A result without provenance cannot be acted on. Six months from now
"USDCADm H1 vaulted three" is unusable unless it also says which bars,
which code, which configuration and which seed produced it. Five of
those were already recorded; config_hash was a column nothing filled.

The grade is deliberately minimal. Exactly ONE rule is implemented --
a winner inside the random range is capped at WEAK -- because that is
the one the spec sanctions now. Inventing a fuller grading ladder would
put an authoritative-looking letter on evidence that does not support
it, which is worse than no grade at all.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

# The grades, worst to best. Only the WEAK cap is enforced today.
GRADE_WEAK = "WEAK"
GRADE_UNGRADED = "UNGRADED"


def config_hash(config: dict[str, Any]) -> str:
    """Stable 16-hex digest of a campaign configuration.

    Keys are sorted and the encoding is canonical, so the same
    configuration hashes the same across processes and machines. Two
    runs sharing a config_hash were configured identically; two that
    do not were not, whatever their reports claim.
    """
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


DATASET_PROVENANCE_VALUES = ("pinned_cell", "live")


def dataset_provenance_for(dataset_id: str | None) -> str:
    """The value "pinned_cell" when the run read a pinned snapshot, else "live".

    dataset_source keeps saying where bars ORIGINATE (MT5 for both; its
    vocabulary is the frozen trial registry's); this says WHICH kind of
    dataset the run consumed, so a manifest alone tells a reproducible run
    from a live pull."""
    return "pinned_cell" if dataset_id else "live"


@dataclass(frozen=True)
class Manifest:
    """Everything needed to say what produced a result."""

    campaign_id: str
    symbol: str
    timeframe: str
    dataset_hash: str
    dataset_source: str
    dataset_provenance: str
    bars_loaded: int
    date_range_utc: tuple[str, str] | None
    git_sha: str
    config_hash: str
    # One line from gene_pool.summary(): which families the sampler could
    # draw, at what weight, and which genes it held fixed.
    gene_pool: str
    seed: int
    completed_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "dataset_hash": self.dataset_hash,
            "dataset_source": self.dataset_source,
            "dataset_provenance": self.dataset_provenance,
            "bars_loaded": self.bars_loaded,
            "date_range_utc": list(self.date_range_utc)
            if self.date_range_utc else None,
            "git_sha": self.git_sha,
            "config_hash": self.config_hash,
            "gene_pool": self.gene_pool,
            "seed": self.seed,
            "completed_at_utc": self.completed_at_utc,
        }

    def missing_fields(self) -> tuple[str, ...]:
        """Which provenance fields are absent or a placeholder."""
        missing = []
        for name in (
            "campaign_id", "symbol", "timeframe", "dataset_hash",
            "dataset_source", "git_sha", "config_hash", "gene_pool",
            "completed_at_utc",
        ):
            value = getattr(self, name)
            if not value or str(value).lower() in {"unknown", "synthetic", ""}:
                missing.append(name)
        if self.dataset_provenance not in DATASET_PROVENANCE_VALUES:
            missing.append("dataset_provenance")
        if self.bars_loaded <= 0:
            missing.append("bars_loaded")
        return tuple(missing)

    def render(self) -> str:
        """The Manifest block the report carries."""
        lines = ["MANIFEST"]
        for key, value in self.to_dict().items():
            lines.append(f"  {key:18s} {value}")
        missing = self.missing_fields()
        lines.append(
            f"  {'complete':18s} "
            + ("yes" if not missing else f"NO - missing {list(missing)}")
        )
        return "\n".join(lines)


def grade_for(
    *,
    vs_random_p_max: float | None,
    current_grade: str | None = None,
    alpha: float = 0.05,
) -> str:
    """Cap a strategy at WEAK when its winner sits inside the random range.

    The ONE rule implemented (C4). A winner that does not separate from
    an unskilled control at ``alpha`` has not been shown to be
    anything, whatever its Sharpe says, and must not carry a grade that
    reads like evidence.

    An unknown p_max is also capped: "we could not test it" is not a
    reason to grade it higher.

    Everything else is UNGRADED rather than given a flattering default.
    """
    within_random = vs_random_p_max is None or float(vs_random_p_max) >= alpha
    if within_random:
        return GRADE_WEAK
    return current_grade or GRADE_UNGRADED


__all__ = [
    "DATASET_PROVENANCE_VALUES",
    "GRADE_UNGRADED",
    "GRADE_WEAK",
    "Manifest",
    "config_hash",
    "dataset_provenance_for",
    "grade_for",
]
