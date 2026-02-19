# Step 08 — Real-data compatibility sweep (no synthetic)

Status: DONE
Date: 2026-02-11

## Goal
Validate the standalone pipeline directly against real tablet DBs and confirm failure modes are safe/explicit.

## What was run

### 1) Schema sweep across all available day DBs
- Path scanned: `/home/bigdan7/Documents/TRADING/DATA/TABLET_TRADING_DATA/btcusdt_*.db`
- Result summary:
  - PASS: all days except one
  - FAIL: `btcusdt_20260206.db` (sqlite malformed)

### 2) Real pipeline run on valid day: `20260205`
Command:
```bash
PYTHONPATH=src python3 -m wash_detector.cli run \
  --source-db /home/bigdan7/Documents/TRADING/DATA/TABLET_TRADING_DATA/btcusdt_20260205.db \
  --artifacts-dir /home/bigdan7/Projects/wash_detector/artifacts \
  --run-name real_20260205 \
  --overwrite
```
Outcome:
- status: PASS
- ingest rows: 625,923 in / 625,923 out / 0 skipped
- alerts: 4,681
- risk: 100 (CRITICAL)
- detector split: mirror_reversal=4,510, balanced_churn=171

### 3) Real pipeline run on valid day: `20260204`
Outcome:
- status: PASS
- ingest rows: 498,426 in / 498,426 out / 0 skipped
- alerts: 3,859
- risk: 100 (CRITICAL)
- detector split: mirror_reversal=3,671, balanced_churn=188

### 4) Real pipeline run on malformed day: `20260206`
Outcome:
- status: FAIL (expected)
- ingest schema: FAIL
- detection: not run
- graceful error path confirmed (no traceback crash in CLI output)

## Implementation hardening added
- Updated schema validation to catch sqlite read errors (e.g., malformed DB) and return structured FAIL instead of crashing with traceback.

## Artifacts created
- `/home/bigdan7/Projects/wash_detector/artifacts/real_20260204/*`
- `/home/bigdan7/Projects/wash_detector/artifacts/real_20260205/*`
- `/home/bigdan7/Projects/wash_detector/artifacts/real_20260206_malformed/*`
