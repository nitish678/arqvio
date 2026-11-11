"""Immutable final-holdout manifest for validation governance.

The lockbox is identity metadata, not a data copy. It is sealed before model
selection; evaluation consumers must refuse a manifest marked opened/changed.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LockboxManifest:
    dataset_hash: str
    source_sha256: str
    config_sha256: str
    feature_pipeline_sha256: str
    sealed_utc: str
    campaign_id: str
    manifest_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {"schema": "arqvio.validation.lockbox", "version": 1,
                "dataset_hash": self.dataset_hash, "source_sha256": self.source_sha256,
                "config_sha256": self.config_sha256, "feature_pipeline_sha256": self.feature_pipeline_sha256,
                "sealed_utc": self.sealed_utc, "campaign_id": self.campaign_id,
                "manifest_sha256": self.manifest_sha256, "status": "SEALED"}


def _digest(payload: dict[str, Any]) -> str:
    body = {k: v for k, v in payload.items() if k not in ("manifest_sha256", "status", "schema", "version")}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def seal(*, dataset_hash: str, source_sha256: str, config_sha256: str,
         feature_pipeline_sha256: str, sealed_utc: str, campaign_id: str) -> LockboxManifest:
    payload = {"dataset_hash": dataset_hash, "source_sha256": source_sha256,
               "config_sha256": config_sha256, "feature_pipeline_sha256": feature_pipeline_sha256,
               "sealed_utc": sealed_utc, "campaign_id": campaign_id}
    return LockboxManifest(**payload, manifest_sha256=_digest(payload))


def verify(doc: dict[str, Any]) -> LockboxManifest:
    if doc.get("schema") != "arqvio.validation.lockbox" or doc.get("version") != 1:
        raise ValueError("unsupported lockbox manifest")
    if doc.get("status") != "SEALED":
        raise ValueError("lockbox is not sealed")
    expected = _digest(doc)
    if doc.get("manifest_sha256") != expected:
        raise ValueError("lockbox manifest digest mismatch")
    return LockboxManifest(dataset_hash=str(doc["dataset_hash"]), source_sha256=str(doc["source_sha256"]),
                           config_sha256=str(doc["config_sha256"]), feature_pipeline_sha256=str(doc["feature_pipeline_sha256"]),
                           sealed_utc=str(doc["sealed_utc"]), campaign_id=str(doc["campaign_id"]),
                           manifest_sha256=str(doc["manifest_sha256"]))


def load(path: str | Path) -> LockboxManifest:
    return verify(json.loads(Path(path).read_text(encoding="utf-8")))
