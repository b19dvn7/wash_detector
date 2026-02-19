# Step 11 — Recent-14 baseline push

Status: DONE
Date: 2026-02-11

## Goal
Extend baseline from 7 recent days to 14 recent days and summarize distribution.

## Run
- batch: `batch_recent14`
- source: `/home/bigdan7/Documents/TRADING/DATA/TABLET_TRADING_DATA`
- total_files: 14
- pass_count: 13
- fail_count: 1 (`20260206` malformed)

## Aggregate stats (PASS days)
- total_rows: 4,977,532
- total_alerts: 37,398
- overall alert rate: 75.13 alerts / 10k rows
- risk mean/min/max: 84.92 / 78 / 100

## Detector totals across PASS days
- `mirror_reversal`: 34,935
- `balanced_churn`: 2,421
- `layering_cluster`: 42

## Notes
- Disk-safe batch mode kept artifacts compact by dropping per-day normalized DB files.
- Next push launched: `batch_recent28` (background), same real-data-only flow.
