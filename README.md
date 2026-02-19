# wash_detector

Standalone rebuild of wash-trading detection.

## Scope
- Lives fully outside the existing `wash_trading` project.
- No code changes to `/home/bigdan7/Projects/wash_trading`.
- Build toward a reliable, testable detector pipeline.

## Initial Goals
1. Define stable schema/contracts.
2. Build deterministic ingestion + feature generation.
3. Implement explainable detector rules with calibrated risk scoring.
4. Add reproducible real-data compatibility checks and reports.

## Quick start
```bash
cd /home/bigdan7/Projects/wash_detector
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
wash-detector --help
```

## Current implemented commands
```bash
# Print required/optional source schema
PYTHONPATH=src python3 -m wash_detector.cli schema-contract

# Validate a source DB
PYTHONPATH=src python3 -m wash_detector.cli schema-check --db-path /path/to/source.db

# Build normalized detector-input DB (read-only source)
PYTHONPATH=src python3 -m wash_detector.cli ingest \
  --source-db /path/to/source.db \
  --output-db /path/to/normalized.db \
  --overwrite

# Run detectors and optionally write JSON report
PYTHONPATH=src python3 -m wash_detector.cli detect \
  --normalized-db /path/to/normalized.db \
  --output-json /path/to/report.json

# Run end-to-end pipeline (ingest + detect + artifacts)
PYTHONPATH=src python3 -m wash_detector.cli run \
  --source-db /path/to/source.db \
  --artifacts-dir /home/bigdan7/Projects/wash_detector/artifacts \
  --run-name first_run \
  --overwrite

# Batch-run real day DBs (newest first by default)
# Note: normalized.db files are dropped by default in batch mode to save disk.
PYTHONPATH=src python3 -m wash_detector.cli batch-run-real \
  --data-dir /home/bigdan7/Documents/TRADING/DATA/TABLET_TRADING_DATA \
  --artifacts-dir /home/bigdan7/Projects/wash_detector/artifacts \
  --run-name batch_recent \
  --max-days 5 \
  --overwrite
```
