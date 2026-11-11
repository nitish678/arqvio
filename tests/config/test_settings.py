"""Settings: defaults, env override, isolation from real user dir."""
from __future__ import annotations

from engine.config.settings import get_settings


def test_defaults(isolated_arqvio_home) -> None:  # type: ignore[no-untyped-def]
    s = get_settings()
    assert s.demo_mode is True
    assert s.live_trading_enabled is False
    assert s.api_host == "127.0.0.1"
    assert s.api_port == 8741
    assert s.log_level == "INFO"
    assert s.arqvio_home == isolated_arqvio_home


def test_env_overrides(isolated_arqvio_home, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("ARQVIO_API_PORT", "9876")
    monkeypatch.setenv("ARQVIO_LOG_LEVEL", "DEBUG")
    from engine.config import settings as settings_mod

    settings_mod.get_settings.cache_clear()
    s = get_settings()
    assert s.api_port == 9876
    assert s.log_level == "DEBUG"
