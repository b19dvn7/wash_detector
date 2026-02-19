# Step 03 — Read-only ingestion + normalization

Status: DONE
Date: 2026-02-11

## Goal
Build a deterministic ingest stage that:
1. validates source schema first,
2. reads source DB in read-only mode,
3. writes a normalized detector-input DB under this standalone project.

## What was implemented

### Module
File: `src/wash_detector/ingest.py`

Implemented:
- `ingest_to_normalized_db(source_db_path, output_db_path, overwrite=False)`
- `IngestReport` (structured run summary)
- Timestamp normalization to unix ms (`parse_timestamp_ms`)
- Side normalization (`BUY`/`SELL`)
- Context alignment:
  - nearest candle at-or-before trade timestamp
  - nearest orderbook row at-or-before trade timestamp (if available)

### Output tables
1. `normalized_trades`
   - canonical fields for detection
   - includes contextual candle/orderbook columns
2. `ingest_meta`
   - source path, row counts, warning counts, timestamp

### CLI command
- `wash-detector ingest --source-db <src.db> --output-db <out.db> [--overwrite]`

## Safety and isolation
- Source DB is opened using sqlite read-only URI mode.
- Output DB is created separately and only in the requested output path.
- No writes to source DB.

## Current limitations (explicit)
- If source timestamps are malformed, rows are skipped with warnings.
- First implementation uses in-memory context timelines; optimization can be added later for very large datasets.

## Next step
Step 04: detector rule engine over `normalized_trades` + explainable risk contributions.
