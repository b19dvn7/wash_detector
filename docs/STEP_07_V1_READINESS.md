# Step 07 — V1 readiness checkpoint

Status: DONE
Date: 2026-02-11

## Goal
Confirm the standalone v1 baseline is coherent and testable end-to-end.

## Acceptance criteria status

1. Standalone execution: ✅
   - Project runs from `/home/bigdan7/Projects/wash_detector`.
   - No dependency on `/home/bigdan7/Projects/wash_trading` runtime code.

2. Deterministic pipeline: ✅
   - Stages are explicit: schema-check -> ingest -> detect -> reports.
   - Single command `run` executes ingest+detect and writes artifacts.

3. Data contract clarity: ✅
   - Canonical contract documented and enforceable with `schema-check`.

4. Explainable alerts: ✅
   - Alerts include detector, reason, risk points, and evidence payload.

5. Real-data compatibility gates: ✅
   - Schema checks pass across real day DBs except malformed inputs.
   - Valid days run end-to-end with 0 skipped rows in tested runs.
   - Malformed DBs fail safely with explicit error and no crash.

6. Operational safety: ✅
   - Source DB opened read-only during ingest.
   - Artifacts isolated in configured run directory.

## Commands for user testing (real-data workflow)

```bash
cd /home/bigdan7/Projects/wash_detector

# 1) Core unit tests (non-synthetic)
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_schema_contract.py' -v
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_ingest.py' -v
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_detect.py' -v
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_pipeline.py' -v

# 2) Real source DB pipeline run (replace path)
PYTHONPATH=src python3 -m wash_detector.cli run \
  --source-db /path/to/source.db \
  --artifacts-dir /home/bigdan7/Projects/wash_detector/artifacts \
  --run-name real_data_trial \
  --overwrite
```

## Notes
- This is a robust standalone anomaly-detection baseline, not final compliance attribution.
- Next iteration should add deeper market-structure features and calibration against real labeled datasets.
