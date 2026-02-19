# Step 06 — Synthetic validation harness (precision/recall/FPR)

Status: DONE
Date: 2026-02-11

## Goal
Add measurable quality gates using labeled synthetic data.

## What was implemented

### Synthetic data generator
File: `src/wash_detector/synthetic.py`

- Generates deterministic source DB with required schema:
  - `trades`
  - `candles`
  - `orderbooks`
- Injects known suspicious mirror-pair patterns.
- Records labeled positive source rowids.

### Validation engine
File: `src/wash_detector/validation.py`

- Runs synthetic fixture creation -> ingest -> detect.
- Maps source labels to normalized trade IDs.
- Computes trade-level metrics:
  - precision
  - recall
  - false positive rate
  - TP / FP / FN / TN
- Applies thresholds:
  - min precision = 0.60
  - min recall = 0.60
  - max FPR = 0.20

### CLI command
- `wash-detector validate-synthetic [--artifacts-dir ...] [--run-name ...] [--overwrite] [--normal-trades N] [--suspicious-pairs N] [--seed N]`

## Validation artifacts
Under validation run directory:
- `synthetic_source.db`
- `normalized.db`
- `detection_report.json`
- `validation_summary.json`
- `validation_summary.txt`

## Next step
Step 07: final v1 readiness check + freeze baseline for user testing.
