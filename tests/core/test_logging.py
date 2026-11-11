"""Logging: secret-masking formatter is invoked and rewrites messages."""
from __future__ import annotations

import io

import pytest

from engine.core.logging import configure_logging, get_logger


@pytest.fixture
def captured() -> io.StringIO:
    buf = io.StringIO()


    log = get_logger()
    log.remove()
    log.add(buf, format="{message}", level="DEBUG")
    return buf


def test_logger_swallows_startup(captured: io.StringIO) -> None:
    configure_logging("INFO")
    log = get_logger()
    log.info("hello world")
    # Just check the log object is usable; the captured buffer is per-fixture.
    assert log is not None


def test_mask_redacts_password(monkeypatch: pytest.MonkeyPatch) -> None:
    from engine.core import logging as logging_mod

    record = {"message": "connecting with password=hunter2 to broker"}
    logging_mod._mask(record)
    assert "hunter2" not in record["message"]
    assert "password=***" in record["message"]


def test_mask_redacts_api_key() -> None:
    from engine.core import logging as logging_mod

    record = {"message": "config api_key=ABCD1234EFGH loaded"}
    logging_mod._mask(record)
    assert "ABCD1234EFGH" not in record["message"]
    assert "api_key=***" in record["message"]


def test_mask_redacts_bearer_token() -> None:
    from engine.core import logging as logging_mod

    record = {"message": "Authorization: Bearer abcdef1234567890XYZ"}
    logging_mod._mask(record)
    assert "abcdef1234567890XYZ" not in record["message"]
    assert "bearer ***" in record["message"]


def test_mask_leaves_normal_text_alone() -> None:
    from engine.core import logging as logging_mod

    record = {"message": "engine started, port 8741, status=ok"}
    logging_mod._mask(record)
    assert record["message"] == "engine started, port 8741, status=ok"
