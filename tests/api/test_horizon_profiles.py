import pytest

from engine.api.routes_app_launch import LaunchRequest, SCALPING_REFUSAL, horizon_profile, plan, supported_horizons


def test_supported_horizons_are_explicit_about_ohlc_limits():
    profiles = supported_horizons()
    assert profiles["intraday"]["execution"] == "MARKET2_OHLC"
    assert profiles["swing"]["timeframes"] == ("H1", "H4", "D1")
    assert profiles["scalping"]["scalping_validated"] is False
    assert profiles["scalping"]["execution"] == "BID_ASK_TICKS_REQUIRED"


def test_intraday_and_swing_reject_wrong_timeframes():
    assert horizon_profile("intraday", ("M15",))["horizon"] == "intraday"
    assert horizon_profile("swing", ("H4",))["horizon"] == "swing"
    with pytest.raises(ValueError, match="does not support"):
        horizon_profile("swing", ("M15",))


def test_scalping_refuses_ohlc_launch_before_task_creation():
    req = LaunchRequest(terminal="t", symbols=["XAUUSD"], timeframes=["M5"], horizon="scalping")
    with pytest.raises(ValueError, match="bid/ask tick data"):
        plan(req, [{"name": "XAUUSD", "visible": True}])
    assert SCALPING_REFUSAL in str(next(iter([SCALPING_REFUSAL])))


def test_swing_launch_plans_tasks_without_changing_legacy_defaults():
    req = LaunchRequest(terminal="t", symbols=["XAUUSD"], timeframes=["H4"], horizon="swing", depth="QUICK")
    tasks = plan(req, [{"name": "XAUUSD", "visible": True}, {"name": "XAGUSD", "visible": True}])
    assert any(t["kind"] == "breed" and t["timeframe"] == "H4" for t in tasks)
