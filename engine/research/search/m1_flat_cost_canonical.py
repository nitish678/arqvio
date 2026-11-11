"""M1: flat-spread canonical (S2 cost model) for delta vs hourly-mean.

Runs the EXACT same production path as run_campaign_cli.run_cell, but passes
flat_cost=True so that the per-hour cost vector is forced to None — prep_cell
charges the flat spec spread instead of mean_buckets — while the no-trade gate
(F1) and max-spread gate (A5) stay live from the calibrated SpreadMatrix.

This isolates the spread cost as the single variable: hourly-mean cost
(mean_buckets) vs flat cost (spec.spread_in_price). The canary and the rest
of the pipeline are unchanged.

Usage:
    python -m engine.research.search.m1_flat_cost_canonical
"""
from __future__ import annotations

import json
from engine.core.paths import get_artifacts_dir

ARTIFACTS = get_artifacts_dir()

# Pinned datasets — the same ones used by the canonical fast/slow runs.
# A pinned dataset is reproducible (G14); a live pull is not.
_DATASETS = {
    ("XAUUSDm", "H1"): "447b843f87438a9a",
    ("USDCADm", "H1"): "aed3bf314539ec5a",
}

_CELLS = [
    ("XAUUSDm", "H1", 250, 1.5),
    ("USDCADm", "H1", 500, 1.5),
]

_SEED = 20260917


def main() -> int:
    from engine.research.search.run_campaign_cli import run_cell

    reports = []
    for symbol, timeframe, pop, target_sr in _CELLS:
        report = run_cell(
            symbol=symbol,
            timeframe=timeframe,
            population=pop,
            target_sharpe=target_sr,
            seed=_SEED,
            dataset_id=_DATASETS[(symbol, timeframe)],
            engine_mode="fast",
            flat_cost=True,
        )
        report["spread_cost_source_expected"] = "flat"
        reports.append(report)

    out = ARTIFACTS / "canonical" / "step_c_canonical_flat.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(reports, indent=2, default=str), encoding="utf-8")
    print(f"\nWROTE {out}")
    for r in reports:
        print(
            f"  {r['symbol']} {r['timeframe']}: "
            f"cost_source={r['spread_cost_source']}  "
            f"blocked_zero_spread={r['blocked_zero_spread_events']}  "
            f"fallbacks={r['fast_fallback_count']}  "
            f"best_sr={r['best_sharpe']:.6f}  "
            f"dsr_within={r['dsr_within']:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
