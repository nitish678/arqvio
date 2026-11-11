"""Manifest dataset_provenance: a pinned run says so beside dataset_source.

dataset_source stays "MT5_LIVE" for both a live pull and a pinned snapshot --
it answers where bars ORIGINATE, and its vocabulary is the frozen trial
registry's. The sibling dataset_provenance answers which kind of dataset the
run consumed: "pinned_cell" when a dataset_id was given, else "live".
"""
from __future__ import annotations

import ast
import inspect
import textwrap

import pytest

from engine.research.campaign.manifest import (
    DATASET_PROVENANCE_VALUES,
    Manifest,
    dataset_provenance_for,
)
from engine.research.search import run_campaign_cli

# The two pinned cells (tests/research/data/test_c0_pinned_hashes_survive.py).
PINNED = {("USDCADm", "H1"): "aed3bf314539ec5a", ("XAUUSDm", "H1"): "447b843f87438a9a"}


def _manifest(**kw) -> Manifest:
    base = dict(
        campaign_id="c", symbol="USDCADm", timeframe="H1", dataset_hash="aed3bf314539ec5a",
        dataset_source="MT5_LIVE", dataset_provenance="pinned_cell", bars_loaded=59957,
        date_range_utc=None, git_sha="abc1234", config_hash="0123456789abcdef",
        gene_pool="g", seed=0, completed_at_utc="2026-09-30T00:00:00+00:00",
    )
    base.update(kw)
    return Manifest(**base)


@pytest.mark.parametrize("cell", sorted(PINNED))
def test_a_pinned_cell_is_pinned_cell(cell: tuple[str, str]) -> None:
    assert dataset_provenance_for(PINNED[cell]) == "pinned_cell"


@pytest.mark.parametrize("dataset_id", [None, ""])
def test_no_dataset_id_is_live(dataset_id: str | None) -> None:
    assert dataset_provenance_for(dataset_id) == "live"


def test_the_manifest_carries_it_beside_an_unchanged_dataset_source() -> None:
    d = _manifest().to_dict()
    assert d["dataset_source"] == "MT5_LIVE"
    assert d["dataset_provenance"] == "pinned_cell"
    keys = list(d)
    assert keys.index("dataset_provenance") == keys.index("dataset_source") + 1
    assert f"  {'dataset_provenance':18s} pinned_cell" in _manifest().render().splitlines()


@pytest.mark.parametrize("value", ["", "MT5_LIVE", "pinned", "unknown"])
def test_anything_else_counts_as_missing(value: str) -> None:
    assert value not in DATASET_PROVENANCE_VALUES
    assert "dataset_provenance" in _manifest(dataset_provenance=value).missing_fields()


def test_run_cell_derives_it_from_its_dataset_id() -> None:
    """run_cell's one Manifest(...) passes dataset_provenance_for(dataset_id),
    and run_cell never rebinds its dataset_id parameter -- the value that also
    sets dataset_reproducible. Checked on the syntax tree, so formatting and
    comments cannot fool it. (A full run_cell needs MT5, the audit, the
    snapshot and the canary.)"""
    fn = ast.parse(textwrap.dedent(inspect.getsource(run_campaign_cli.run_cell))).body[0]
    assert "dataset_id" in [a.arg for a in fn.args.kwonlyargs]
    manifests = [
        n for n in ast.walk(fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "Manifest"
    ]
    assert len(manifests) == 1
    value = {k.arg: k.value for k in manifests[0].keywords}["dataset_provenance"]
    assert isinstance(value, ast.Call) and not value.keywords
    assert isinstance(value.func, ast.Name) and value.func.id == "dataset_provenance_for"
    assert [(type(a), getattr(a, "id", None)) for a in value.args] == [(ast.Name, "dataset_id")]
    rebinds = [
        n.lineno for n in ast.walk(fn)
        if isinstance(n, ast.Name) and n.id == "dataset_id" and not isinstance(n.ctx, ast.Load)
    ]
    assert rebinds == []
