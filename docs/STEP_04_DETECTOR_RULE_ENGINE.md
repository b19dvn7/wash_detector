# Step 04 — Detector rule engine (explainable alerts)

Status: DONE
Date: 2026-02-11

## Goal
Run explainable detection rules over `normalized_trades` and produce transparent risk scoring.

## What was implemented

### Module
File: `src/wash_detector/detect.py`

Implemented:
- `detect_suspicious_patterns()`
- `run_detection()`
- `DetectionReport` and `Alert`

### Implemented detectors (v1)
1. `mirror_reversal`
   - Opposite-side near-equal trade pair in short interval at near-equal price.
2. `layering_cluster`
   - Dense short-window cluster with low price drift and small average size.
3. `balanced_churn`
   - Large balanced two-sided notional with minimal net price movement.

### Explainability guarantees
Each alert carries:
- detector name
- reason text
- risk points
- evidence payload (window stats / deltas / IDs)

### Risk scoring
- Score = sum(alert risk_points), capped at 100.
- Levels:
  - LOW: 0–19
  - MODERATE: 20–49
  - HIGH: 50–74
  - CRITICAL: 75–100

### CLI command
- `wash-detector detect --normalized-db <path> [--lookback-minutes N] [--output-json report.json]`

## Next step
Step 05: end-to-end pipeline command + validation harness and metrics report generation.
