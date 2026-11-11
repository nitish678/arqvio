"""Fetch-time basis probe for MT5 timestamp interpretation (R-C0).

The S2 audit measurement (2c458fc) showed this server encodes UTC wall-clock
directly in its bar ``time`` field: tick 02:22:34 == system UTC 02:22:44
(server offset ≈ +10s, within clock skew). Subtracting the EET/EEST schedule
offset (broker_to_utc) from UTC-encoded timestamps SHIFTS every bar −2h/−3h
and re-introduces the mixed-basis defect G12 was about: spread artifacts
(calibrated from raw ticks, hour = (t//3600)%24) would stop agreeing with
engine hours.

The W0.2 commit applied broker_to_utc UNCONDITIONALLY, which is wrong for
this server. This module replaces that with a MEASUREMENT-DRIVEN probe:

1. ``probe_basis`` reads the newest ticks via ``symbol_info_tick``, computes
   ``delta = median(server_time - now_utc)`` and classifies the basis.

2. ``convert_times`` applies the right transform: identity for
   ``server_equals_utc`` (PASS THROUGH — reverts the −7200s shift on
   today's server), offset subtraction for ``broker_offset``.

3. ``validate_measured_offset`` sanity-checks a measured offset against the
   EU EET/EEST schedule (time_offset's real job: confirm the measurement
   matches the expected DST regime, not apply it blindly).

The three call sites (fetch_bars, load_cell_bars, audit script) all share
this single function — no more three copies.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np

from engine.core.errors import DataError

UTC_SKEW_TOLERANCE_S = 120
EET_WINTER_OFFSET_S = 2 * 3600
EET_SUMMER_OFFSET_S = 3 * 3600
EET_TOLERANCE_S = 600


@dataclass(frozen=True)
class BasisProbe:
    """Result of probing one symbol's timestamp basis against system UTC."""

    basis: str
    measured_offset_s: int
    delta_s: float
    sample_n: int


def _median_delta_seconds(tick_time: int, now_utc: datetime) -> tuple[float, int]:
    """Delta of (tick_time - now) in seconds. tick_time is UNIX-epoch int."""
    now_ts = now_utc.timestamp()
    delta = float(tick_time) - now_ts
    return delta, 1


def probe_basis(mt5_module, symbol: str, *, now: datetime | None = None) -> BasisProbe:
    """Probe one symbol's timestamp basis.

    Calls ``mt5.symbol_info_tick(symbol)`` which returns a struct with
    ``time`` (broker-server wall-clock UNIX seconds). The delta
    ``server_time - now_utc`` reveals the basis:

    * |delta| <= 120s  -> server_equals_utc (identity, pass through)
    * delta ~ 7200±600 winter / 10800±600 summer -> broker_offset
    * anything else -> unresolved -> caller must REFUSE

    Raises DataError if MT5 cannot reach the symbol.
    """
    if now is None:
        now = datetime.now(tz=UTC)
    tick = mt5_module.symbol_info_tick(symbol)
    if tick is None:
        err = mt5_module.last_error()
        raise DataError(
            f"symbol_info_tick failed for {symbol!r}: {err}"
        )
    tick_time = int(tick.time)
    delta, n = _median_delta_seconds(tick_time, now)
    abs_delta = abs(delta)

    if abs_delta <= UTC_SKEW_TOLERANCE_S:
        return BasisProbe(
            basis="server_equals_utc",
            measured_offset_s=0,
            delta_s=delta,
            sample_n=n,
        )

    for expected in (EET_WINTER_OFFSET_S, EET_SUMMER_OFFSET_S):
        if abs(abs_delta - expected) <= EET_TOLERANCE_S:
            sign = -1 if tick_time < now.timestamp() else 1
            return BasisProbe(
                basis="broker_offset",
                measured_offset_s=sign * expected,
                delta_s=delta,
                sample_n=n,
            )

    return BasisProbe(
        basis="unresolved",
        measured_offset_s=int(delta),
        delta_s=delta,
        sample_n=n,
    )


def convert_times(raw_times: np.ndarray, probe: BasisProbe) -> np.ndarray:
    """Apply the basis conversion to raw broker-server bar timestamps.

    - server_equals_utc: identity (PASS THROUGH)
    - server_time: identity -- a GMT+2/+3 server pinned in its own clock (``server_time_probe``)
    - broker_offset: subtract the measured offset
    - unresolved: refuse
    """
    raw = np.asarray(raw_times, dtype=np.int64)
    if probe.basis in ("server_equals_utc", SERVER_TIME):
        return raw
    if probe.basis == "broker_offset":
        return raw - probe.measured_offset_s
    raise DataError(
        f"refusing to convert timestamps: basis is {probe.basis!r} "
        f"(delta={probe.delta_s:.1f}s); cannot determine UTC mapping"
    )


SERVER_TIME = "server_time"


def server_time_probe(probe: BasisProbe) -> BasisProbe:
    """A broker_offset server pinned in its OWN clock instead of shifted to UTC (owner 2026-11-01: GMT+2/+3 brokers
    such as AiwaCapital must breed). The live EA reads hours with TimeCurrent(), the server's clock, so bars, spread
    hours, sessions and the Friday close all stay in one clock from backtest to live -- with no DST conversion, which
    is exactly why it is safe. The offset is kept so a deploy can refuse a broker on a different clock."""
    return BasisProbe(basis=SERVER_TIME, measured_offset_s=probe.measured_offset_s, delta_s=probe.delta_s,
                      sample_n=probe.sample_n)


def validate_measured_offset(probe: BasisProbe) -> None:
    """Confirm a broker_offset probe matches the EU DST schedule.

    For broker_offset, the magnitude must be one of the two expected EET/EEST
    values (7200 or 10800). Anything else means the broker is in an
    unexpected regime and we refuse — never guess.
    """
    if probe.basis != "broker_offset":
        return
    abs_off = abs(probe.measured_offset_s)
    if abs_off not in (EET_WINTER_OFFSET_S, EET_SUMMER_OFFSET_S):
        raise DataError(
            f"broker offset {probe.measured_offset_s}s does not match EET/EEST "
            f"(expected ±7200 or ±10800); refusing to stamp UTC"
        )


def basis_stamps(probe: BasisProbe) -> dict[str, str | int]:
    """Return the source/time_basis strings for recording on outputs."""
    return {
        "time_basis": probe.basis,
        "source": "MT5_LIVE",
        "offset_seconds": probe.measured_offset_s,
    }


__all__ = [
    "SERVER_TIME",
    "server_time_probe",
    "BasisProbe",
    "UTC_SKEW_TOLERANCE_S",
    "convert_times",
    "probe_basis",
    "validate_measured_offset",
    "basis_stamps",
]
