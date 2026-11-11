# Phase 5 — leakage-safe ML baseline

2026-11-11. Status: partial, shadow-only baseline.

## Implemented

- Added `engine/research/ml/baseline.py`: deterministic NumPy logistic scorer, chronological train/test split, train-only mean/std normalization, purge and embargo metadata, model hash and manifest. No new dependency.
- Added ML tests: chronological disjointness, reproducible model hash, test-only perturbation invariance, invalid labels/split refusal.
- Model output is a research score only. It does not create orders, change risk, choose deployment, replace strategy validation or retrain automatically.

## Verification

- ML baseline tests: **4 passed**.
- No MT5 calls in ML tests. No market performance claim.

## Evidence discipline

ML must include failed candidates/experiments, not only vault survivors; model/feature selection must be performed on development folds; final lockbox remains untouched. `scikit-learn`-style train-only preprocessing guidance and DSR/PBO abstracts motivate this boundary. The baseline is not evidence of an edge until compared against equal-budget non-ML controls on unseen data.

## Remaining

- Feature pipeline tied to actual bar/tick timestamps and publication availability.
- Nested selection/model tuning with purged label horizons.
- Calibration/precision-recall and net-of-cost comparison against non-ML baseline.
- Drift/shadow monitoring and model artifact provenance.
- ONNX/MT5 inference parity only after operator/type/shape/preprocessing tests; no live model path now.
