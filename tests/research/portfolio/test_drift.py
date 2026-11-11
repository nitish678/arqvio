"""The drift monitor (DR §5): the planted-degraded fixture FIRES, the healthy control stays inside the budget,
the BY cut, the window geometry and its refusals."""
from __future__ import annotations

import numpy as np
import pytest

from engine.research.portfolio.drift import (CONTROL_DRAWS, DEGRADE_FROM, DEGRADED_MEMBER, Q, DriftError,
                                             by_threshold, check_times, degraded_fixture, healthy_control,
                                             healthy_draw, ks_checks, monitor)


def test_the_planted_degraded_fixture_fires_before_the_degraded_segment_ends() -> None:
    r = monitor(degraded_fixture())
    ff = r.first_fire(DEGRADED_MEMBER)
    assert len(r.checks) == 35 and ff is not None and ff.t <= 1000
    print(f"\nDEGRADED FIXTURE fires on {DEGRADED_MEMBER} at t={ff.t}, delay {ff.t - DEGRADE_FROM} bars, p={ff.p:.3e}, "
          f"BY cut {r.cut:.3e}; rejected {[(c.member, c.t) for c in r.rejected]}")
    assert ff.t - DEGRADE_FROM == 100  # the first check whose recent window lies inside the degraded segment
    assert {c.member for c in r.rejected} == {DEGRADED_MEMBER}


def test_the_healthy_control_false_alarm_rate_is_inside_the_budget() -> None:
    hc = healthy_control()
    print(f"\nHEALTHY CONTROL {hc['draws']} draws: false-alarm draws {hc['false_alarm_draws']}, rate {hc['rate']:.4f} "
          f"(budget {Q})")
    assert hc["draws"] == CONTROL_DRAWS == 200 and hc["rate"] <= Q


def test_a_single_member_without_drift_does_not_fire() -> None:
    assert not monitor({"m0": healthy_draw(3)["m0"]}).fired


def test_the_window_geometry_and_its_boundaries() -> None:
    assert check_times(400) == [400] and check_times(1000) == list(range(400, 1001, 100))
    assert check_times(1050)[-1] == 1000 and check_times(999)[-1] == 900
    with pytest.raises(DriftError, match="shorter than reference 300 \\+ recent 100"):
        check_times(399)
    x = np.arange(1000, dtype=np.float64)
    last = ks_checks("x", x)[-1]
    assert last.t == 1000 and last.d == 1.0  # recent [900, 1000) vs reference [0, 300): disjoint supports


def test_the_monitor_refuses_bad_series() -> None:
    with pytest.raises(DriftError, match="finite"):
        ks_checks("x", np.array([0.0] * 399 + [np.nan]))
    with pytest.raises(DriftError, match="different lengths"):
        monitor({"a": np.zeros(400), "b": np.zeros(500)})


def test_the_benjamini_yekutieli_cut() -> None:
    m = 4
    cm = 1 + 1 / 2 + 1 / 3 + 1 / 4
    ps = [0.001, 0.2, 0.3, 0.9]
    assert by_threshold(ps, 0.05) == 0.001 and 0.001 <= 1 * 0.05 / (m * cm)
    assert by_threshold([0.02, 0.5], 0.05) == 0.0  # 0.02 > 0.05 / (2 * 1.5)
    assert by_threshold([], 0.05) == 0.0
    assert by_threshold([0.004, 0.0059], 0.05) == 0.0059  # step-up: rank 2 passes 2 * 0.05 / (2 * 1.5)
