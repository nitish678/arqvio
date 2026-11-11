"""Owner arming: an explicit, recorded act per (login, server); absent file = nothing armed."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from engine.research.accounts import arming


def test_nothing_is_armed_without_a_file(tmp_path: Path) -> None:
    assert arming.load(tmp_path / "none.json") == []
    assert not arming.is_armed(1, "S", tmp_path / "none.json")


def test_arm_disarm_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "config" / "live_arming.json"
    stamp = datetime(2026, 10, 22, 12, 0, tzinfo=UTC)
    e = arming.arm(12345678, "Exness-MT5Real8", note="first live", path=path, now=lambda: stamp)
    assert e == {"login": 12345678, "server": "Exness-MT5Real8", "armed_utc": stamp.isoformat(), "note": "first live"}
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["schema"] == arming.SCHEMA and doc["accounts"] == [e]
    assert arming.is_armed(12345678, "Exness-MT5Real8", path)
    assert arming.disarm(12345678, "Exness-MT5Real8", path=path) is True
    assert not arming.is_armed(12345678, "Exness-MT5Real8", path)
    assert arming.disarm(12345678, "Exness-MT5Real8", path=path) is False


def test_re_arming_replaces_rather_than_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "a.json"
    arming.arm(1, "S", path=path)
    arming.arm(1, "S", note="again", path=path)
    arming.arm(2, "S", path=path)
    assert [(a["login"], a["note"]) for a in arming.load(path)] == [(1, "again"), (2, "")]


def test_a_foreign_file_is_refused_not_trusted(tmp_path: Path) -> None:
    path = tmp_path / "a.json"
    path.write_text(json.dumps({"schema": "something.else", "accounts": [{"login": 1, "server": "S"}]}),
                    encoding="utf-8")
    with pytest.raises(arming.ArmingFileError):
        arming.is_armed(1, "S", path)
    path.write_text(json.dumps({"schema": arming.SCHEMA, "accounts": [{"login": "1", "server": "S"}]}),
                    encoding="utf-8")
    with pytest.raises(arming.ArmingFileError):
        arming.load(path)


def test_the_cli_writes_to_arqvio_home(isolated_arqvio_home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert arming.main(["arm", "--login", "7", "--server", "Demo-1", "--note", "n"]) == 0
    assert (isolated_arqvio_home / "config" / "live_arming.json").is_file()
    assert arming.is_armed(7, "Demo-1")
    arming.main(["list"])
    assert "login=7 server=Demo-1" in capsys.readouterr().out
    assert arming.main(["disarm", "--login", "7", "--server", "Demo-1"]) == 0
    assert not arming.is_armed(7, "Demo-1")
