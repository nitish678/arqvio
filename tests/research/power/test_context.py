"""The power-check resolver: the registry and the campaign manifest give (cell, ppy, N), or it refuses."""
from __future__ import annotations

import copy
import json

import pytest

from engine.research.power.context import PowerContext, PowerContextError, resolve_power_context
from engine.research.registry.cells import UnknownCellError

# The recorded values of the two pinned campaigns' manifest_v2 run blocks (part0_gate.log).
MANIFESTS = {
    "usdcadm_h1": {"run": {"campaign_id": "ga-usdcadm_h1-s7-cbcb2ae061e70bef", "dsr_n_final": 4760,
                           "config": {"symbol": "USDCADm", "timeframe": "H1", "dataset": "aed3bf314539ec5a"}}},
    "xauusdm_h1": {"run": {"campaign_id": "ga-xauusdm_h1-s7-f32e0d4defd9cf07", "dsr_n_final": 4760,
                           "config": {"symbol": "XAUUSDm", "timeframe": "H1", "dataset": "447b843f87438a9a"}}},
}

# DR §2: the retired literals of engine/research/ga/power_check.py at 4a85481. For usdcadm the resolved context MUST
# equal them (P2); they are data here, not prose.
RETIRED = {"CELL": ("USDCADm", "H1", "aed3bf314539ec5a"), "PPY_H1": 6048, "PRIMARY_N": 4760}


def test_usdcadm_resolves_to_exactly_the_retired_literals() -> None:
    ctx = resolve_power_context("usdcadm_h1", MANIFESTS["usdcadm_h1"])
    assert (ctx.cell, ctx.ppy, ctx.primary_n) == (RETIRED["CELL"], RETIRED["PPY_H1"], RETIRED["PRIMARY_N"])
    assert ctx.campaign_id == "ga-usdcadm_h1-s7-cbcb2ae061e70bef"


def test_xauusdm_resolves_from_its_registry_entry_and_its_own_campaign() -> None:
    ctx = resolve_power_context("xauusdm_h1", MANIFESTS["xauusdm_h1"])
    assert ctx == PowerContext("xauusdm_h1", "XAUUSDm", "H1", "447b843f87438a9a", 6048, 4760,
                               "ga-xauusdm_h1-s7-f32e0d4defd9cf07")
    assert ctx.as_dict()["primary_n"] == 4760


def test_a_manifest_path_resolves_like_its_dict(tmp_path) -> None:
    p = tmp_path / "manifest_v2.json"
    p.write_text(json.dumps(MANIFESTS["xauusdm_h1"]), encoding="utf-8")
    assert resolve_power_context("xauusdm_h1", p) == resolve_power_context("xauusdm_h1", MANIFESTS["xauusdm_h1"])
    with pytest.raises(PowerContextError, match="not found"):
        resolve_power_context("xauusdm_h1", tmp_path / "absent.json")


def test_an_unknown_cell_is_refused() -> None:
    with pytest.raises(UnknownCellError):
        resolve_power_context("eurusdm_h1", MANIFESTS["usdcadm_h1"])


def _without(path: tuple[str, ...]):
    doc = copy.deepcopy(MANIFESTS["usdcadm_h1"])
    node = doc
    for k in path[:-1]:
        node = node[k]
    del node[path[-1]]
    return doc


@pytest.mark.parametrize("path", [("run",), ("run", "dsr_n_final"), ("run", "campaign_id"), ("run", "config")])
def test_a_missing_manifest_field_is_refused_never_defaulted(path) -> None:
    with pytest.raises(PowerContextError):
        resolve_power_context("usdcadm_h1", _without(path))


@pytest.mark.parametrize("n", [0, -1, 4760.0, "4760", True, None])
def test_a_bad_n_is_refused(n) -> None:
    doc = copy.deepcopy(MANIFESTS["usdcadm_h1"])
    doc["run"]["dsr_n_final"] = n
    with pytest.raises(PowerContextError, match="dsr_n_final"):
        resolve_power_context("usdcadm_h1", doc)


def test_a_campaign_on_another_cell_is_refused() -> None:
    with pytest.raises(PowerContextError, match="searched"):
        resolve_power_context("usdcadm_h1", MANIFESTS["xauusdm_h1"])
    doc = copy.deepcopy(MANIFESTS["xauusdm_h1"])
    doc["run"]["config"]["dataset"] = "0123456789abcdef"
    with pytest.raises(PowerContextError, match="searched"):
        resolve_power_context("xauusdm_h1", doc)


def test_a_retired_cell_is_refused(monkeypatch) -> None:
    from engine.research.power import context

    real = context.get_cell
    monkeypatch.setattr(context, "get_cell", lambda c: dict(real(c), status="RETIRED"))
    with pytest.raises(PowerContextError, match="not VALIDATED"):
        resolve_power_context("usdcadm_h1", MANIFESTS["usdcadm_h1"])
