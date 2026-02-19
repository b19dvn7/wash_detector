# Step 05 — End-to-end pipeline runner

Status: DONE
Date: 2026-02-11

## Goal
Provide a single command that executes:
1. schema-gated ingest,
2. normalized output creation,
3. detector execution,
4. artifact/report writing.

## What was implemented

### Module
File: `src/wash_detector/pipeline.py`

Implemented:
- `run_pipeline(...)`
- `PipelineReport`

### CLI command
- `wash-detector run --source-db <src.db> [--artifacts-dir artifacts] [--run-name name] [--overwrite] [--lookback-minutes N]`

## Artifacts written per run
Under `<artifacts-dir>/<run-name>/`:
- `normalized.db`
- `detection_report.json`
- `pipeline_summary.json`
- `pipeline_summary.txt`

## Safety
- Source DB remains read-only during ingest.
- All outputs stay inside the standalone project artifacts path unless user explicitly points elsewhere.

## Next step
Step 06: synthetic validation harness with precision/recall/FPR metrics.
