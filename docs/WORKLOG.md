# Worklog

## 2026-02-11 — Step 01: Scope/Guardrails/Acceptance

### Goal
Lock hard boundaries and success criteria before implementing detector logic.

### Changes made
- Added `docs/STEP_01_SCOPE_AND_GUARDRAILS.md`.

### Verification
Commands run:

```bash
find /home/bigdan7/Projects/wash_detector -maxdepth 3 -type f | sort
python3 -m py_compile src/wash_detector/*.py
```

Observed file set:
- `/home/bigdan7/Projects/wash_detector/docs/PLAN.md`
- `/home/bigdan7/Projects/wash_detector/docs/STEP_01_SCOPE_AND_GUARDRAILS.md`
- `/home/bigdan7/Projects/wash_detector/.gitignore`
- `/home/bigdan7/Projects/wash_detector/pyproject.toml`
- `/home/bigdan7/Projects/wash_detector/README.md`
- `/home/bigdan7/Projects/wash_detector/src/wash_detector/cli.py`
- `/home/bigdan7/Projects/wash_detector/src/wash_detector/__init__.py`
- `/home/bigdan7/Projects/wash_detector/tests/test_smoke.py`

Py-compile: pass (no errors).

### Outcome
Step 01 complete. Next step is schema contract design.

---

## 2026-02-11 — Step 02: Canonical schema contract + validator

### Goal
Add an explicit source DB contract and validate input DBs before ingestion/detection.

### Changes made
- Added `src/wash_detector/schema.py`.
- Updated `src/wash_detector/cli.py` with:
  - `schema-contract`
  - `schema-check --db-path <path>`
- Added `docs/STEP_02_SCHEMA_CONTRACT.md`.

### Verification
Commands run:

```bash
python3 -m py_compile src/wash_detector/*.py
PYTHONPATH=src python3 -m wash_detector.cli schema-contract
PYTHONPATH=src python3 -m wash_detector.cli schema-check --db-path /tmp/wd_schema_valid.db
PYTHONPATH=src python3 -m wash_detector.cli schema-check --db-path /tmp/wd_schema_invalid.db
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_schema_contract.py' -v
```

Observed behavior:
- `schema-contract` prints required/optional tables and columns as designed.
- Valid fixture DB => `Schema check: PASS`, exit code `0`.
- Invalid fixture DB => `Schema check: FAIL`, exit code `2`, with explicit missing table/column errors.
- Unit tests: `3 passed`.

### Outcome
Step 02 complete. Next step is read-only ingestion into normalized internal dataset.

---

## 2026-02-11 — Step 03: Read-only ingestion + normalization

### Goal
Implement deterministic ingest stage from source DB into normalized detector-input DB.

### Changes made
- Added `src/wash_detector/ingest.py`.
- Updated `src/wash_detector/cli.py` with `ingest` command.
- Added `tests/test_ingest.py`.
- Added `docs/STEP_03_INGESTION_NORMALIZATION.md`.
- Updated `README.md` with schema/ingest command usage.

### Verification
Commands run:

```bash
python3 -m py_compile src/wash_detector/*.py
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_schema_contract.py' -v
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_ingest.py' -v
```

Also executed manual ingest CLI run against a temporary fixture DB and inspected output counts/rows.

### Outcome
Step 03 complete. Next step is detector rule engine over `normalized_trades`.

---

## 2026-02-11 — Step 04: Detector rule engine + explainable risk

### Goal
Add transparent detector rules that emit explainable alerts and aggregate risk score/level.

### Changes made
- Added `src/wash_detector/detect.py`.
- Updated `src/wash_detector/cli.py` with `detect` command.
- Added `tests/test_detect.py`.
- Added `docs/STEP_04_DETECTOR_RULE_ENGINE.md`.
- Updated `README.md` with detect command usage.

### Verification
Commands run:

```bash
python3 -m py_compile src/wash_detector/*.py
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_detect.py' -v
```

Also executed manual ingest + detect CLI run against temporary fixture DB and confirmed non-empty alert output with detector breakdown.

### Outcome
Step 04 complete. Next step is end-to-end pipeline command + validation harness and metrics.

---

## 2026-02-11 — Step 05: End-to-end pipeline runner

### Goal
Add one command to run ingest+detect and write persistent artifacts.

### Changes made
- Added `src/wash_detector/pipeline.py`.
- Updated `src/wash_detector/cli.py` with `run` command.
- Added `tests/test_pipeline.py`.
- Added `docs/STEP_05_PIPELINE_RUNNER.md`.

### Verification
Commands run:

```bash
python3 -m py_compile src/wash_detector/*.py
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_pipeline.py' -v
PYTHONPATH=src python3 -m wash_detector.cli run --source-db /tmp/wd_run_src.db --artifacts-dir /home/bigdan7/Projects/wash_detector/artifacts --run-name smoke_run --overwrite
find /home/bigdan7/Projects/wash_detector/artifacts/smoke_run -maxdepth 1 -type f | sort
```

Observed behavior:
- Pipeline command returns PASS for valid fixture.
- Artifacts written: `normalized.db`, `detection_report.json`, `pipeline_summary.json`, `pipeline_summary.txt`.

### Outcome
Step 05 complete. Next step is synthetic validation harness with metrics.

---

## 2026-02-11 — Step 06: Synthetic validation harness + metrics

### Goal
Generate labeled synthetic data and compute precision/recall/FPR quality gates.

### Changes made
- Added `src/wash_detector/synthetic.py`.
- Added `src/wash_detector/validation.py`.
- Updated `src/wash_detector/cli.py` with `validate-synthetic` command.
- Added `tests/test_validation.py`.
- Added `docs/STEP_06_SYNTHETIC_VALIDATION.md`.

### Verification
Commands run:

```bash
python3 -m py_compile src/wash_detector/*.py
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_validation.py' -v
PYTHONPATH=src python3 -m wash_detector.cli validate-synthetic --artifacts-dir /home/bigdan7/Projects/wash_detector/artifacts --run-name validation_baseline --overwrite --normal-trades 240 --suspicious-pairs 30 --seed 17
find /home/bigdan7/Projects/wash_detector/artifacts/validation_baseline -maxdepth 1 -type f | sort
```

Observed behavior:
- Validation command returns PASS on deterministic synthetic fixture.
- Metrics reported: precision=1.0000, recall=1.0000, FPR=0.0000 for baseline fixture.
- Artifacts written include `validation_summary.json` + `validation_summary.txt`.

### Outcome
Step 06 complete.

---

## 2026-02-11 — Step 07: V1 readiness checkpoint

### Goal
Freeze and document v1 baseline readiness for user testing.

### Changes made
- Added `docs/STEP_07_V1_READINESS.md`.
- Updated `README.md` with pipeline + validation commands.

### Verification
Commands run:

```bash
PYTHONPATH=src python3 -m wash_detector.cli --help
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py' -v
```

Observed behavior:
- CLI exposes: `status`, `plan`, `schema-contract`, `schema-check`, `ingest`, `detect`, `run`, `validate-synthetic`.
- Full test suite passes.

### Outcome
Step 07 complete. Standalone v1 baseline is ready for BD testing.

---

## 2026-02-11 — Step 08: Real-data compatibility sweep

### Goal
Validate pipeline directly on real source DBs and confirm graceful failure handling.

### Changes made
- Hardened `schema.py` validation to catch sqlite read errors (malformed DBs) as structured FAIL reports.
- Added `docs/STEP_08_REAL_DATA_COMPATIBILITY.md`.

### Verification
Commands run (real-data only):

```bash
PYTHONPATH=src python3 -m wash_detector.cli schema-check --db-path /home/bigdan7/Documents/TRADING/DATA/TABLET_TRADING_DATA/btcusdt_20260205.db
PYTHONPATH=src python3 -m wash_detector.cli schema-check --db-path /home/bigdan7/Documents/TRADING/DATA/TABLET_TRADING_DATA/btcusdt_20260206.db
PYTHONPATH=src python3 -m wash_detector.cli run --source-db /home/bigdan7/Documents/TRADING/DATA/TABLET_TRADING_DATA/btcusdt_20260205.db --artifacts-dir /home/bigdan7/Projects/wash_detector/artifacts --run-name real_20260205 --overwrite
PYTHONPATH=src python3 -m wash_detector.cli run --source-db /home/bigdan7/Documents/TRADING/DATA/TABLET_TRADING_DATA/btcusdt_20260204.db --artifacts-dir /home/bigdan7/Projects/wash_detector/artifacts --run-name real_20260204 --overwrite
PYTHONPATH=src python3 -m wash_detector.cli run --source-db /home/bigdan7/Documents/TRADING/DATA/TABLET_TRADING_DATA/btcusdt_20260206.db --artifacts-dir /home/bigdan7/Projects/wash_detector/artifacts --run-name real_20260206_malformed --overwrite
```

Observed behavior:
- `20260205`: PASS, 625,923 rows ingested, 0 skipped.
- `20260204`: PASS, 498,426 rows ingested, 0 skipped.
- `20260206`: FAIL with explicit malformed-db schema error, no crash.

### Outcome
Step 08 complete. Real-data compatibility confirmed for valid days and safe failure for malformed input.

---

## 2026-02-11 — Step 09: Real-data batch runner + CLI cleanup

### Goal
Keep workflow moving with real-data-first automation and remove synthetic command from active CLI surface.

### Changes made
- Added `src/wash_detector/real_batch.py`.
- Updated `src/wash_detector/cli.py`:
  - removed `validate-synthetic` command from exposed CLI commands.
  - added `batch-run-real` command.
  - added `--keep-normalized` override (default behavior drops `normalized.db` in batch mode to save disk).
- Added `tests/test_real_batch.py`.
- Updated `README.md` with `batch-run-real` usage.
- Added `docs/STEP_09_REAL_BATCH_RUNNER.md`.

### Verification
Commands run:

```bash
python3 -m py_compile src/wash_detector/*.py
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_real_batch.py' -v
PYTHONPATH=src python3 -m wash_detector.cli --help
PYTHONPATH=src python3 -m wash_detector.cli batch-run-real --data-dir /home/bigdan7/Documents/TRADING/DATA/TABLET_TRADING_DATA --artifacts-dir /home/bigdan7/Projects/wash_detector/artifacts --run-name batch_recent3 --max-days 3 --overwrite
```

Observed behavior:
- Real-batch unit test passes (mix of valid+malformed day fixtures).
- CLI help no longer advertises synthetic validation command.
- `batch-run-real` processes recent day files and writes batch summary artifacts.
- Batch run default now drops per-day `normalized.db` (for valid days) and kept batch artifact footprint low (~4.4MB for 3-day run).

### Outcome
Step 09 complete. Workflow now supports real-data batch runs directly.

---

## 2026-02-11 — Step 10: Recent-window baseline (7-day real run)

### Goal
Establish a multi-day real-data baseline for ingest size, alert density, and calibrated risk score behavior.

### Changes made
- Added `docs/STEP_10_RECENT_WINDOW_BASELINE.md`.

### Verification
Commands run:

```bash
PYTHONPATH=src python3 -m wash_detector.cli batch-run-real --data-dir /home/bigdan7/Documents/TRADING/DATA/TABLET_TRADING_DATA --artifacts-dir /home/bigdan7/Projects/wash_detector/artifacts --run-name batch_recent7 --max-days 7 --overwrite
python3 - <<'PY'
# parsed artifacts/batch_recent7/batch_summary.json and computed per-day/aggregate stats
PY
```

Observed behavior:
- Batch processed 7 recent DB files: 6 PASS, 1 FAIL (`20260206` malformed).
- PASS day risk scores ranged 79..86 (all CRITICAL).
- Mean alert density ~70.88 alerts per 10k rows.

### Outcome
Step 10 complete. Recent-window real-data baseline recorded and documented.

---

## 2026-02-11 — Step 11: Recent-14 baseline push

### Goal
Extend the recent baseline to 14 day files and summarize aggregate detector behavior.

### Changes made
- Added `docs/STEP_11_RECENT14_BASELINE.md`.

### Verification
Commands run:

```bash
PYTHONPATH=src python3 -m wash_detector.cli batch-run-real --data-dir /home/bigdan7/Documents/TRADING/DATA/TABLET_TRADING_DATA --artifacts-dir /home/bigdan7/Projects/wash_detector/artifacts --run-name batch_recent14 --max-days 14 --overwrite
python3 - <<'PY'
# parsed batch_recent14 summary + detector totals from per-day detection reports
PY
```

Observed behavior:
- Batch processed 14 files: 13 PASS, 1 FAIL (`20260206` malformed).
- PASS-day total rows: 4,977,532; total alerts: 37,398.
- Mean risk score: 84.92 (range 78..100).
- Detector totals: mirror_reversal=34,935; balanced_churn=2,421; layering_cluster=42.

### Outcome
Step 11 complete. Launched next real push run: `batch_recent28`.
