"""C0: measurement-driven UTC basis probe and conversion.

R-C0 replaced the unconditional broker_to_utc with a measurement-driven probe.
These tests pin the two basis classes the engine must handle:

  - server_equals_utc: tick timestamps ARE UTC; convert_times is identity.
  - broker_offset: tick timestamps are EET/EEST; convert_times subtracts the
    measured offset (7200s winter, 10800s summer).

The audit CSV (scripts/history_depth_audit_target.py) stamps TIME_BASIS per
cell; the spread calibration (scripts/calibrate_spreads.py) must now stamp the
same field on its artifact so consumers can verify alignment.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np

from engine.research.data.time_basis import (
    BasisProbe,
    EET_SUMMER_OFFSET_S,
    EET_WINTER_OFFSET_S,
    UTC_SKEW_TOLERANCE_S,
    convert_times,
    probe_basis,
    validate_measured_offset,
)


@dataclass(frozen=True)
class _FakeTick:
    time: int


class _FakeMT5:
    """Stand-in for the ``mt5`` module — only symbol_info_tick is used."""

    def __init__(self, tick_time: int) -> None:
        self._tick = _FakeTick(time=tick_time)

    def symbol_info_tick(self, _symbol: str):
        return self._tick

    def last_error(self):
        return (0, "Success")


def test_convert_times_is_identity_for_server_equals_utc() -> None:
    """server_equals_utc: convert_times must be pass-through."""
    probe = BasisProbe(
        basis="server_equals_utc",
        measured_offset_s=0,
        delta_s=5.0,
        sample_n=1,
    )
    raw = np.array([1700000000, 1700003600, 1700007200], dtype=np.int64)
    converted = convert_times(raw, probe)
    assert np.array_equal(converted, raw), (
        "server_equals_utc must leave timestamps unchanged"
    )


def test_convert_times_subtracts_7200_for_broker_offset_winter() -> None:
    """broker_offset with 7200s: every timestamp shifts −7200."""
    probe = BasisProbe(
        basis="broker_offset",
        measured_offset_s=EET_WINTER_OFFSET_S,
        delta_s=float(EET_WINTER_OFFSET_S) + 60,
        sample_n=1,
    )
    raw = np.array([1700000000, 1700003600, 1700007200], dtype=np.int64)
    converted = convert_times(raw, probe)
    expected = raw - EET_WINTER_OFFSET_S
    assert np.array_equal(converted, expected), (
        "broker_offset winter (7200s) must subtract exactly 7200 from each ts"
    )


def test_convert_times_subtracts_10800_for_broker_offset_summer() -> None:
    """broker_offset with 10800s: every timestamp shifts −10800."""
    probe = BasisProbe(
        basis="broker_offset",
        measured_offset_s=EET_SUMMER_OFFSET_S,
        delta_s=float(EET_SUMMER_OFFSET_S) + 60,
        sample_n=1,
    )
    raw = np.array([1700000000, 1700003600], dtype=np.int64)
    converted = convert_times(raw, probe)
    expected = raw - EET_SUMMER_OFFSET_S
    assert np.array_equal(converted, expected), (
        "broker_offset summer (10800s) must subtract exactly 10800 from each ts"
    )


def test_convert_times_refuses_unresolved() -> None:
    """unresolved basis: convert_times must raise, never guess."""
    probe = BasisProbe(
        basis="unresolved",
        measured_offset_s=42,
        delta_s=42.0,
        sample_n=1,
    )
    raw = np.array([1700000000], dtype=np.int64)
    try:
        convert_times(raw, probe)
    except Exception as exc:
        assert "unresolved" in str(exc)
    else:
        raise AssertionError("unresolved basis must be refused, not converted")


def test_validate_measured_offset_accepts_7200_and_10800() -> None:
    """The EU schedule check passes for both EET/EEST values."""
    winter = BasisProbe(
        basis="broker_offset", measured_offset_s=EET_WINTER_OFFSET_S,
        delta_s=7206.0, sample_n=1,
    )
    summer = BasisProbe(
        basis="broker_offset", measured_offset_s=EET_SUMMER_OFFSET_S,
        delta_s=10806.0, sample_n=1,
    )
    validate_measured_offset(winter)
    validate_measured_offset(summer)


def test_validate_measured_offset_rejects_outside_eet_eest() -> None:
    """An offset that doesn't match EET/EEST (e.g. 3600) is refused."""
    probe = BasisProbe(
        basis="broker_offset", measured_offset_s=3600,
        delta_s=3606.0, sample_n=1,
    )
    try:
        validate_measured_offset(probe)
    except Exception as exc:
        assert "EET/EEST" in str(exc)
    else:
        raise AssertionError("non-EET offset must be refused")


def test_probe_basis_classifies_server_equals_utc() -> None:
    """Tick time within UTC_SKEW_TOLERANCE of now → server_equals_utc."""
    now = datetime(2026, 9, 20, 12, 0, 0, tzinfo=UTC)
    tick_time = int(now.timestamp()) + 30  # within 120s skew
    mt5_fake = _FakeMT5(tick_time)
    probe = probe_basis(mt5_fake, "EURUSDm", now=now)
    assert probe.basis == "server_equals_utc"
    assert probe.measured_offset_s == 0


def test_probe_basis_classifies_broker_offset_winter() -> None:
    """Tick time ~7200s behind now → broker_offset, winter."""
    now = datetime(2026, 1, 10, 12, 0, 0, tzinfo=UTC)  # winter
    tick_time = int(now.timestamp()) - EET_WINTER_OFFSET_S + 60
    mt5_fake = _FakeMT5(tick_time)
    probe = probe_basis(mt5_fake, "EURUSDm", now=now)
    assert probe.basis == "broker_offset"
    assert probe.measured_offset_s == -EET_WINTER_OFFSET_S


def test_probe_basis_classifies_broker_offset_summer() -> None:
    """Tick time ~10800s behind now → broker_offset, summer."""
    now = datetime(2026, 7, 10, 12, 0, 0, tzinfo=UTC)  # summer
    tick_time = int(now.timestamp()) - EET_SUMMER_OFFSET_S + 60
    mt5_fake = _FakeMT5(tick_time)
    probe = probe_basis(mt5_fake, "EURUSDm", now=now)
    assert probe.basis == "broker_offset"
    assert probe.measured_offset_s == -EET_SUMMER_OFFSET_S


def test_probe_basis_refuses_unresolved() -> None:
    """Tick time far from any expected offset → unresolved."""
    now = datetime(2026, 9, 20, 12, 0, 0, tzinfo=UTC)
    # 3600s offset is not EET/EEST — should be unresolved.
    tick_time = int(now.timestamp()) - 3600
    mt5_fake = _FakeMT5(tick_time)
    probe = probe_basis(mt5_fake, "EURUSDm", now=now)
    assert probe.basis == "unresolved"
    assert probe.measured_offset_s != 0


def test_convert_times_preserves_hour_bucket_alignment() -> None:
    """The core invariant: after conversion, bar hours and spread matrix
    hours must agree.

    For server_equals_utc, the raw hour is already the UTC hour. For
    broker_offset 7200, a bar at raw hour 3 becomes UTC hour 1. The spread
    matrix (calibrated from converted ticks) bucket 1 must be applied.
    """
    # Simulate a broker_offset server: raw timestamps are 2h ahead of UTC.
    raw_utc = 1700000000  # 2023-11-14T22:13:20Z
    raw_broker = raw_utc + EET_WINTER_OFFSET_S  # server shows 2h ahead

    probe = BasisProbe(
        basis="broker_offset",
        measured_offset_s=EET_WINTER_OFFSET_S,
        delta_s=float(EET_WINTER_OFFSET_S),
        sample_n=1,
    )
    converted = convert_times(np.array([raw_broker], dtype=np.int64), probe)
    assert int(converted[0]) == raw_utc

    # The UTC hour of the bar:
    bar_hour = (int(converted[0]) // 3600) % 24
    # If we had bucketed by raw broker time instead, we'd get:
    wrong_hour = (raw_broker // 3600) % 24
    assert bar_hour != wrong_hour, (
        "Without conversion, bar_hour and spread-bucket hour would disagree by 2"
    )
