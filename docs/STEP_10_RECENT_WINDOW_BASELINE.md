# Step 10 — Recent-window real-data baseline (7-day)

Status: DONE
Date: 2026-02-11

## Goal
Run a larger recent real-data window and establish a baseline profile for:
- ingest volume
- alert counts
- normalized risk score behavior

## Run executed
Command:
```bash
PYTHONPATH=src python3 -m wash_detector.cli batch-run-real \
  --data-dir /home/bigdan7/Documents/TRADING/DATA/TABLET_TRADING_DATA \
  --artifacts-dir /home/bigdan7/Projects/wash_detector/artifacts \
  --run-name batch_recent7 \
  --max-days 7 \
  --overwrite
```

Summary:
- processed_files: 7
- pass_count: 6
- fail_count: 1
- fail day: `btcusdt_20260206.db` (malformed sqlite)

## PASS day results
| day | input_rows | alerts | alert_rate_per_10k | risk_score | risk_level |
|---|---:|---:|---:|---:|---|
| 20260131 | 436,348 | 3,137 | 71.89 | 86 | CRITICAL |
| 20260201 | 459,394 | 3,183 | 69.29 | 82 | CRITICAL |
| 20260202 | 466,869 | 3,051 | 65.35 | 79 | CRITICAL |
| 20260203 | 422,777 | 2,813 | 66.54 | 80 | CRITICAL |
| 20260204 | 498,426 | 3,859 | 77.42 | 84 | CRITICAL |
| 20260205 | 625,923 | 4,681 | 74.79 | 80 | CRITICAL |

## Aggregate baseline stats (PASS days)
- mean risk_score: **81.83**
- min/max risk_score: **79 / 86**
- mean alert_rate_per_10k: **70.88**
- min/max alert_rate_per_10k: **65.35 / 77.42**

## Notes
- New risk scoring calibration avoids blanket `100` scores on large days while preserving high-risk classification for high-density suspicious patterns.
- Batch mode dropped `normalized.db` by default for PASS days to reduce disk usage.
