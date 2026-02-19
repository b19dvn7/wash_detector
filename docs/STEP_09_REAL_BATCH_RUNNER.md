# Step 09 — Real-data batch runner (no synthetic)

Status: DONE
Date: 2026-02-11

## Goal
Move from one-off day runs to a repeatable real-data batch workflow against:
`/home/bigdan7/Documents/TRADING/DATA/TABLET_TRADING_DATA`

## What was implemented

### New module
- `src/wash_detector/real_batch.py`
  - `run_real_batch(...)`
  - per-day result tracking (`DayRunResult`)
  - aggregate report (`BatchRunReport`)

### New CLI command
- `batch-run-real`
- Default behavior drops per-day `normalized.db` files to reduce disk pressure.
- Use `--keep-normalized` to preserve those files when needed.

Example:
```bash
PYTHONPATH=src python3 -m wash_detector.cli batch-run-real \
  --data-dir /home/bigdan7/Documents/TRADING/DATA/TABLET_TRADING_DATA \
  --artifacts-dir /home/bigdan7/Projects/wash_detector/artifacts \
  --run-name batch_recent \
  --max-days 5 \
  --overwrite
```

### Artifacts per batch run
Under `<artifacts-dir>/<run-name>/`:
- `batch_summary.json`
- `batch_summary.csv`
- `batch_summary.txt`
- `runs/<db_stem>/...` (normal pipeline artifacts per day)

## Behavior
- Scans `btcusdt_*.db`
- Processes newest first by default (or `--oldest-first`)
- Runs schema validation first per day
- Skips malformed/failing days with structured FAIL entries
- Continues remaining days instead of aborting batch

## Additional alignment change
- Removed `validate-synthetic` from CLI surface to keep workflow real-data-only.
