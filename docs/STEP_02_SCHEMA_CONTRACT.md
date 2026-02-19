# Step 02 — Canonical Schema Contract

Status: DONE
Date: 2026-02-11

## Goal
Define and enforce a clear input schema contract for source SQLite datasets before implementing ingestion/detection logic.

## What was implemented

### 1) Contract module
File: `src/wash_detector/schema.py`

Implemented:
- `TableContract`
- `SourceSchemaContract`
- `ValidationReport`
- `DEFAULT_SOURCE_CONTRACT`
- `contract_to_text()`
- `validate_source_db()`

### 2) CLI wiring
File: `src/wash_detector/cli.py`

New commands:
- `wash-detector schema-contract`
- `wash-detector schema-check --db-path <path>`

## Canonical source contract (v1)

### Required tables
1. `trades`
   - required columns: `timestamp`, `side`, `price`, `amount`
   - optional columns: `trade_id`, `exchange`, `symbol`

2. `candles`
   - required columns: `timestamp`, `open`, `high`, `low`, `close`, `volume`
   - optional columns: `vwap`, `quote_volume`, `trade_count`

### Optional table
1. `orderbooks`
   - required columns (if table present): `timestamp`
   - optional columns: `spread`, `mid_price`, `imbalance`, `best_bid`, `best_ask`, `bids`, `asks`

## Validation behavior
- Missing required table: **error**
- Missing required column in required table: **error**
- Missing optional table: **warning**
- Missing recommended columns in optional table: **warning**

## Why this matters
This prevents hidden assumptions and placeholder-driven behavior. The detector pipeline can now fail fast when expected source structures are absent.

## Next step
Step 03: implement ingestion module that reads the validated source DB into a normalized internal dataset (read-only source access).
