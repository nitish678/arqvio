"""M1: canonical slow (oracle) reproduction (mean_buckets cost model).

Runs run_campaign_cli.run_cell with engine_mode='slow' on the same pinned
datasets as the fast canonical. This is the oracle reference for parity.

Usage:
    python -m engine.research.search.m1_canonical_slow
"""
from __future__ import annotations

import json

from engine.core.paths import get_artifacts_dir

ARTIFACTS = get_artifacts_dir()

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
            engine_mode="slow",
            flat_cost=False,
        )
        report["spread_cost_source_expected"] = "mean_buckets"
        reports.append(report)

    out = ARTIFACTS / "canonical" / "step_c_canonical_slow.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(reports, indent=2, default=str), encoding="utf-8")
    print(f"\nWROTE {out}")
    for r in reports:
        print(
            f"  {r['symbol']} {r['timeframe']}: "
            f"cost_source={r['spread_cost_source']}  "
            f"blocked_zero_spread={r['blocked_zero_spread_events']}  "
            f"best_sr={r['best_sharpe']:.6f}  "
            f"dsr_within={r['dsr_within']:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
