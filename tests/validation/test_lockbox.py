import dataclasses

import pytest

from engine.validation.lockbox import seal, verify


def test_lockbox_seals_and_verifies():
    m = seal(dataset_hash="d" * 16, source_sha256="a" * 64, config_sha256="b" * 64,
             feature_pipeline_sha256="c" * 64, sealed_utc="2026-11-11T00:00:00Z", campaign_id="c1")
    assert verify(m.to_dict()) == m
    assert m.to_dict()["status"] == "SEALED"


def test_lockbox_tamper_refuses():
    m = seal(dataset_hash="d" * 16, source_sha256="a" * 64, config_sha256="b" * 64,
             feature_pipeline_sha256="c" * 64, sealed_utc="2026-11-11T00:00:00Z", campaign_id="c1")
    doc = m.to_dict(); doc["dataset_hash"] = "e" * 16
    with pytest.raises(ValueError, match="digest"):
        verify(doc)


def test_lockbox_status_change_refuses():
    m = seal(dataset_hash="d" * 16, source_sha256="a" * 64, config_sha256="b" * 64,
             feature_pipeline_sha256="c" * 64, sealed_utc="2026-11-11T00:00:00Z", campaign_id="c1")
    doc = m.to_dict(); doc["status"] = "OPEN"
    with pytest.raises(ValueError, match="sealed"):
        verify(doc)
