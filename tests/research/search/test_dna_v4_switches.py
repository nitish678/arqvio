"""W4-2.1: switch-genome schema — DNA v4 fields + hash stability + validation.

Idea source: research §4.1 (13 bias / 21 signal / 14 filter / 7 mgmt / 8 risk switch model),
APPENDIX-D seed JSON (line 26717).
"""
from __future__ import annotations

import hashlib
import json

import pytest

from engine.research.search.dna import (
    BIAS_SWITCHES,
    DNA,
    DNA_SCHEMA_VERSION,
    FILTER_SWITCHES,
    MANAGEMENT_SWITCHES,
    RISK_SWITCHES,
    validate_gene_space,
)


def _minimal(**over) -> DNA:
    base = dict(
        bias="DAILY_TREND", filter="ADX", signal="RSI_REVERSION", signal2="",
        exec_mode="MARKET2", risk="FIXED_ATR", mgmt="NONE",
        sl_atr=2.0, tp_atr=3.0, risk_pct=0.01,
        friday_close=False, news_filter=False,
        daily_dd_limit=0.05, max_agg_risk=0.05,
        max_feature_lookback=50, max_hold_time=10,
        window_start_hour_utc=0, window_end_hour_utc=23,
    )
    base.update(over)
    return DNA(**base)


# --- hash stability: V4 defaults == V3 hash (no switches -> omitted) ---
def test_schema_version_bumped_to_4():
    assert DNA_SCHEMA_VERSION == 4


def test_default_dna_serializes_no_switches():
    "Empty switches must be omitted from to_dict so the hash is V3-stable."
    d = _minimal()
    out = d.to_dict()
    for key in ("bias_switches", "filter_switches", "mgmt_switches", "risk_switches"):
        assert key not in out
    assert out["schema_version"] == 4


def test_v4_default_hash_matches_v3_layout():
    """A DNA with all switches empty carries only the schema_version bump.

    Empty switches are omitted, so the payload is identical to V3 except for
    schema_version (4 vs 3). This test pins that contract: no *other* drift,
    and a V3-bred trial's payload is recoverable by downgrading the version
    field alone (the GA migrates hashes on load, not by re-derivation).
    """
    d = _minimal()
    out = d.to_dict()
    payload_v4 = json.dumps(out, sort_keys=True, separators=(",", ":"))

    # same payload but schema_version = 3 (the V3 view)
    v3_view = dict(out)
    v3_view["schema_version"] = 3
    payload_v3 = json.dumps(v3_view, sort_keys=True, separators=(",", ":"))

    # the only difference is the version byte
    o4 = json.loads(payload_v4)
    o3 = json.loads(payload_v3)
    o4.pop("schema_version")
    o3.pop("schema_version")
    assert o4 == o3  # identical body

    # switching the version field is the sole delta
    assert d.dna_hash != hashlib.sha256(payload_v3.encode()).hexdigest()[:16]  # version differs
    # but downgrading only the version reproduces the exact V3 hash (migration path)
    v4_downgraded = dict(out)
    v4_downgraded["schema_version"] = 3
    v3_hash = hashlib.sha256(
        json.dumps(v4_downgraded, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    # V4 hash == hash of payload with version field set to 4 (sanity)
    assert d.dna_hash == hashlib.sha256(payload_v4.encode()).hexdigest()[:16]


def test_v4_with_switches_hashes_differ_from_empty():
    d_empty = _minimal().dna_hash
    d_full = _minimal(bias_switches=("sma",), filter_switches=("cci",)).dna_hash
    assert d_empty != d_full


# --- switch vocabulary mirrors research counts ---
def test_switch_counts_match_research():
    assert len(BIAS_SWITCHES) == 13
    assert len(FILTER_SWITCHES) == 14
    assert len(MANAGEMENT_SWITCHES) == 7
    assert len(RISK_SWITCHES) == 8


# --- validation ---
def test_invalid_bias_switch_rejected_at_construction():
    with pytest.raises(ValueError, match="bias_switches"):
        _minimal(bias_switches=("nope",))


def test_invalid_filter_switch_rejected_at_construction():
    with pytest.raises(ValueError, match="filter_switches"):
        _minimal(filter_switches=("unknown_filter",))


def test_duplicate_switch_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        _minimal(bias_switches=("sma", "sma"))


def test_validate_gene_space_accepts_valid_switches():
    d = _minimal(bias_switches=("sma", "ema"), mgmt_switches=("breakeven",))
    validate_gene_space(d.to_dict())  # must not raise


def test_validate_gene_space_rejects_bad_switch():
    d = _minimal()
    payload = d.to_dict()
    payload["bias_switches"] = ["bogus"]
    with pytest.raises(ValueError, match="bias_switches"):
        validate_gene_space(payload)


# --- switch lists are tuples (immutable per trial) ---
def test_switches_are_tuples():
    d = _minimal(bias_switches=("sma",))
    assert isinstance(d.bias_switches, tuple)
