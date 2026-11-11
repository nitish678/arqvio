"""The spread guard must block a symbol, not merely notice it.

Before this, ``is_calibration_failure`` was called by nothing except its own
unit test. XAUUSDm reported a flat 260pt row across all 24 hours, the check
would have caught it, and nothing ever asked. A guard that is never called
is documentation.
"""
from __future__ import annotations

import pytest

from engine.data.spread_calibration import (
    SUSPICIOUS_RATIO,
    SpreadMatrix,
    SpreadStatus,
    evaluate,
)
from engine.research.campaign.preflight import PreflightError, refuse_on_spread


def _matrix(buckets, symbol: str = "EURUSDm", source: str = "MT5") -> SpreadMatrix:
    b = tuple(float(x) for x in buckets)
    return SpreadMatrix(
        symbol=symbol,
        buckets=b,
        mean_buckets=b,
        sample_count=100_000,
        source=source,
    )


def _realistic() -> list[float]:
    """A real session profile: tight in London/NY, wide over the rollover."""
    return [
        18, 20, 22, 19, 16, 14, 12, 10, 9, 8, 8, 9,
        9, 8, 8, 9, 10, 12, 15, 19, 26, 40, 30, 22,
    ]


def test_flat_spread_blocks_symbol() -> None:
    """XAUUSDm's flat 260pt row, not measured from the broker's ticks (2026-11-01: a flat row measured from the
    broker's own ticks is a fixed-spread account and passes -- test_a_measured_fixed_spread_...)."""
    flat = _matrix([260.0] * 24, symbol="XAUUSDm", source="BUNDLED")
    check = evaluate(flat)

    assert check.status is SpreadStatus.CALIBRATION_FAILURE
    assert check.blocks is True
    assert "flat 260pt" in check.reason

    with pytest.raises(PreflightError) as excinfo:
        refuse_on_spread(flat)
    assert "SPREAD_CALIBRATION_FAILURE" in str(excinfo.value)
    assert "XAUUSDm" in str(excinfo.value)


def test_zero_spread_is_reported_but_does_not_block() -> None:
    """F9: a zero-cost hour is flagged, not a refusal.

    UPDATED for F9, and this is the one existing test the S0 pass
    weakened -- deliberately, because the policy it encoded changed. It
    asserted blocks is True and that refuse_on_spread raised
    SPREAD_ZERO_SPREAD. That was correct only while the backtester could
    not act on a zero hour: refusing the whole symbol was the only way to
    avoid costing trades at zero. F1 now derives no_trade_hours from
    exactly these hours and refuses every fill inside them, so refusing
    the cell as well stops a properly calibrated symbol from running at
    all -- and made the F1 gate unreachable, since the only calibrated
    symbols with a zero hour were the ones being refused.

    A missing, unreadable or wrong-shaped matrix is still a hard block;
    that is F2 and is unchanged (see the CALIBRATION_FAILURE tests).
    """
    buckets = _realistic()
    buckets[3] = 0.0
    check = evaluate(_matrix(buckets))

    assert check.status is SpreadStatus.ZERO_SPREAD
    assert check.blocks is False
    assert "[3]" in check.reason
    assert "untradeable hour(s) in sim" in check.reason

    # It no longer raises: the cell runs with hour 3 shut.
    returned = refuse_on_spread(_matrix(buckets))
    assert returned.status is SpreadStatus.ZERO_SPREAD

    # And the hours the gate will use are exactly the ones reported.
    assert sorted(_matrix(buckets).no_trade_hours) == [3]


def test_zero_spread_symbol_is_launchable_but_missing_matrix_is_not(
    tmp_path, monkeypatch,
) -> None:
    """F9's two halves, at the launch gate the UI actually calls."""
    import json

    import engine.data.spread_calibration as sc
    from engine.api.routes_s5 import spread_blocked_reason, spread_info_line

    monkeypatch.setattr(sc, "spread_dir", lambda: tmp_path)

    # A valid matrix whose only defect is a zero hour: launchable, flagged.
    buckets = _realistic()
    buckets[21] = 0.0
    matrix = _matrix(buckets, symbol="ZEROHOURm")
    assert evaluate(matrix).blocks is False
    (tmp_path / "ZEROHOURm.json").write_text(
        json.dumps(matrix.to_dict()), encoding="utf-8",
    )

    assert spread_blocked_reason("ZEROHOURm") is None
    assert spread_info_line("ZEROHOURm") == "untradeable hour(s) in sim: [21]"

    # No matrix at all: still refused, F2 unchanged.
    assert spread_blocked_reason("NOMATRIXm") is not None
    assert "spread not calibrated" in spread_blocked_reason("NOMATRIXm")
    assert spread_info_line("NOMATRIXm") is None


def test_extreme_ratio_warns_but_does_not_block() -> None:
    """A 50x swing is suspicious, not disqualifying.

    Illiquid symbols really do gap that far, so blocking would be a false
    negative on real data. It is surfaced and the caller decides.
    """
    buckets = _realistic()
    buckets[21] = buckets[9] * (SUSPICIOUS_RATIO + 10)
    check = evaluate(_matrix(buckets))

    assert check.status is SpreadStatus.SUSPICIOUS
    assert check.blocks is False
    assert check.ratio is not None
    assert check.ratio > SUSPICIOUS_RATIO
    # Explicitly does NOT raise - the campaign runs and the line is logged.
    returned = refuse_on_spread(_matrix(buckets))
    assert returned.status is SpreadStatus.SUSPICIOUS


def test_a_real_session_profile_passes() -> None:
    check = evaluate(_matrix(_realistic()))
    assert check.status is SpreadStatus.OK
    assert check.blocks is False
    assert refuse_on_spread(_matrix(_realistic())).status is SpreadStatus.OK


def test_wrong_bucket_count_blocks() -> None:
    check = evaluate(_matrix([10.0] * 12))
    assert check.status is SpreadStatus.CALIBRATION_FAILURE
    assert "got 12" in check.reason
    with pytest.raises(PreflightError):
        refuse_on_spread(_matrix([10.0] * 12))


def test_empty_matrix_blocks() -> None:
    check = evaluate(_matrix([]))
    assert check.status is SpreadStatus.CALIBRATION_FAILURE
    assert check.blocks is True


def test_severity_order_flat_beats_zero() -> None:
    """An all-zero row is flat AND zero; it is reported as the flat failure.

    Both block, so the ordering only affects the message, but the message is
    what tells an operator which calibration to re-run.
    """
    check = evaluate(_matrix([0.0] * 24))
    assert check.status is SpreadStatus.CALIBRATION_FAILURE
    assert check.blocks is True


# --- the spec's entry point, and the reason reaching the REPORT ------------
def test_validate_spread_matrix_accepts_a_bare_row() -> None:
    """The spec's signature: a 24-element row in, a validation out."""
    from engine.data.spread_calibration import (
        SpreadValidation,
        validate_spread_matrix,
    )

    out = validate_spread_matrix([260.0] * 24, symbol="XAUUSDm")
    assert isinstance(out, SpreadValidation)
    assert out.status is SpreadStatus.CALIBRATION_FAILURE
    assert out.blocks is True
    assert out.symbol == "XAUUSDm"

    # A SpreadMatrix works through the same door.
    assert validate_spread_matrix(_matrix(_realistic())).status is SpreadStatus.OK


def test_usdjpy_reference_row_passes() -> None:
    """The reference profile from the spec, including the 21h spike.

    0h:10 ... 19h:10 | 20h:18 | 21h:76 | 22h:18 | 23h:10 -- the rollover
    spike is 7.6x baseline, which a flat row silently deletes.
    """
    from engine.data.spread_calibration import validate_spread_matrix

    row = [10.0] * 20 + [18.0, 76.0, 18.0, 10.0]
    out = validate_spread_matrix(row, symbol="USDJPYm")
    assert out.status is SpreadStatus.OK
    assert out.blocks is False
    assert out.ratio is not None
    assert 7.5 < out.ratio < 7.7, out.ratio


def test_block_reason_reaches_the_campaign_report(tmp_path) -> None:
    """Not just the log. A reason inside an exception message is a log line.

    spread_preflight GRADES and returns the fragment the report carries, so
    the caller can record it and then refuse.
    """
    from engine.data.spread_calibration import persist
    from engine.research.campaign.preflight import spread_preflight

    persist(_matrix([260.0] * 24, symbol="XAUUSDm", source="BUNDLED"), tmp_path)
    fragment = spread_preflight("XAUUSDm", tmp_path)

    assert fragment["blocks"] is True
    assert fragment["status"] == "CALIBRATION_FAILURE"
    assert "flat 260pt" in str(fragment["reason"])

    # And the campaign turns that fragment into a report entry.
    from scripts.search_runner_real import SpreadBlockedError

    exc = SpreadBlockedError("XAUUSDm", "D1", fragment)
    entry = {
        "symbol": "XAUUSDm",
        "timeframe": "D1",
        "error": f"{type(exc).__name__}: {exc}",
        "spread_check": exc.fragment,
        "blocked_before_launch": True,
    }
    assert entry["blocked_before_launch"] is True
    assert "flat 260pt" in entry["spread_check"]["reason"]
    assert "SPREAD_CALIBRATION_FAILURE" in entry["error"]


def test_an_uncalibrated_symbol_is_unavailable_not_blocked(tmp_path) -> None:
    """"Not calibrated yet" and "calibrated and unusable" differ.

    Refusing every uncalibrated symbol would stop every campaign rather
    than protect one, so the absence is reported and does not block.
    """
    from engine.research.campaign.preflight import spread_preflight

    fragment = spread_preflight("EURUSDm", tmp_path)
    assert fragment["status"] == "UNAVAILABLE"
    assert fragment["blocks"] is False
    assert "no spread matrix persisted" in str(fragment["reason"])


def test_a_good_symbol_passes_preflight(tmp_path) -> None:
    from engine.data.spread_calibration import persist
    from engine.research.campaign.preflight import spread_preflight

    persist(_matrix(_realistic(), symbol="EURUSDm"), tmp_path)
    fragment = spread_preflight("EURUSDm", tmp_path)
    assert fragment["status"] == "OK"
    assert fragment["blocks"] is False


def test_a_measured_fixed_spread_is_information_not_a_failure() -> None:
    """2026-11-01: a broker's own ticks with one spread at every hour (a fixed-spread account) do not block; a flat
    row without that measurement still does."""
    from engine.data.spread_calibration import FIXED_SPREAD_MIN_TICKS, SpreadMatrix, SpreadStatus, evaluate

    row = (21.0,) * 24
    measured = SpreadMatrix(symbol="XAGUSD", buckets=row, mean_buckets=row, sample_count=FIXED_SPREAD_MIN_TICKS,
                            source="MT5")
    c = evaluate(measured)
    assert c.status == SpreadStatus.FIXED_SPREAD and not c.blocks
    for unmeasured in (SpreadMatrix(symbol="XAGUSD", buckets=row, mean_buckets=row, sample_count=50, source="MT5"),
                       SpreadMatrix(symbol="XAGUSD", buckets=row, mean_buckets=row, sample_count=10**6,
                                    source="BUNDLED")):
        assert evaluate(unmeasured).status == SpreadStatus.CALIBRATION_FAILURE and evaluate(unmeasured).blocks
