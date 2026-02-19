# AI Handoff Summary (wash_detector)

Last updated: 2026-02-13 (local)
Project root: `/home/bigdan7/Projects/wash_detector`

## 1) Mission and Hard Boundaries

- Mission: standalone wash-trading anomaly detector with deterministic ingest -> detect -> report flow.
- Hard boundary: do not modify `/home/bigdan7/Projects/wash_trading`.
- Current stack is Python stdlib only (no external runtime deps in `pyproject.toml`).

## 2) Current Status (What Is Actually Working)

- Steps 01-11 documented as complete in `docs/STEP_*.md`.
- Full unit suite currently passes: 9/9 tests.
- Real-data batch runs exist and produce stable artifacts.
- One known malformed source DB (`btcusdt_20260206.db`) fails safely at schema validation stage.

## 3) Core Architecture (Code Map)

- CLI entrypoint: `src/wash_detector/cli.py`
  - Commands: `status`, `plan`, `schema-contract`, `schema-check`, `ingest`, `detect`, `run`, `batch-run-real`
  - Exit codes:
    - `schema-check`: `0` pass, `2` fail
    - `ingest`: `0` pass, `3` fail
    - `run`: `0` pass, `4` fail
    - `batch-run-real`: `0` if no failures, `6` if any day failed

- Schema contract: `src/wash_detector/schema.py`
  - Required tables:
    - `trades(timestamp, side, price, amount)`
    - `candles(timestamp, open, high, low, close, volume)`
  - Optional table:
    - `orderbooks(timestamp, ...optional microstructure columns...)`
  - Produces structured validation report with `errors`, `warnings`, and discovered table map.

- Ingestion/normalization: `src/wash_detector/ingest.py`
  - Reads source DB read-only, writes normalized output DB.
  - Output tables:
    - `normalized_trades`
    - `ingest_meta`
  - Timestamp normalization:
    - numeric >= `1e12` treated as ms
    - numeric >= `1e9` treated as seconds -> converted to ms
    - ISO supported (`Z` normalized to UTC)
  - Side normalization: only `BUY`/`SELL` accepted.
  - Context join rule: nearest candle/orderbook **at or before** trade timestamp (no forward look).

- Detector engine: `src/wash_detector/detect.py`
  - Reads `normalized_trades`, optional `lookback_minutes`.
  - Alert detectors:
    1. `mirror_reversal`
    2. `layering_cluster`
    3. `balanced_churn`
  - Outputs explainable alerts with `reason`, `risk_points`, and structured `evidence`.

- Pipeline wrapper: `src/wash_detector/pipeline.py`
  - Runs ingest + detect
  - Writes:
    - `normalized.db`
    - `detection_report.json`
    - `pipeline_summary.json`
    - `pipeline_summary.txt`

- Real batch orchestration: `src/wash_detector/real_batch.py`
  - Scans `btcusdt_*.db`, validates each, runs pipeline per day.
  - Continues on failures (does not abort entire batch).
  - Batch artifacts:
    - `batch_summary.json`
    - `batch_summary.csv`
    - `batch_summary.txt`
    - `runs/<db_stem>/...`
  - Important default: drops per-day `normalized.db` to save disk (`--keep-normalized` disables drop).

- Synthetic validation harness (still in code, not in active CLI):
  - `src/wash_detector/synthetic.py`
  - `src/wash_detector/validation.py`
  - Tests still cover it (`tests/test_validation.py`).

## 4) Detector Logic (Exact v1 Rules)

### A) `mirror_reversal`
- Evaluates adjacent trades.
- Conditions:
  - opposite side
  - time gap <= 120,000 ms
  - amount gap ratio <= 0.20
  - price diff <= 8 bps
- Risk points:
  - base `12`
  - +`3` if candle high/low volatility < 6 bps

### B) `layering_cluster`
- Sliding 60-second window.
- Requires window size >= 8 trades.
- Conditions:
  - price range <= 6 bps
  - avg trade size <= 60% of global median amount
- Risk points: `9`
- Cooldown: 60 seconds after trigger.

### C) `balanced_churn`
- Sliding 300-second window.
- Requires window size >= 20 trades.
- Conditions:
  - side balance ratio <= 0.20
  - price move <= 10 bps
  - total window notional >= 20 * median notional
- Risk points: `14`
- Cooldown: 300 seconds after trigger.

## 5) Risk Score Model (Current Calibration)

In `DetectionReport.risk_score`:

- Convert detector counts to rates per 10k rows:
  - `mirror_rate`, `layering_rate`, `churn_rate`
- Score components:
  - `min(45, mirror_rate * 0.60)`
  - `min(20, layering_rate * 6.00)`
  - `min(35, churn_rate * 8.00)`
- Escalation by raw risk points:
  - +15 if `>= 50,000`
  - +10 if `>= 10,000`
  - +5 if `>= 2,000`
- Final score capped at 100 and rounded to int.
- Risk levels:
  - `LOW` < 20
  - `MODERATE` 20-49
  - `HIGH` 50-74
  - `CRITICAL` >= 75

## 6) Real-Data Baselines (From Artifacts)

- `artifacts/batch_recent3`: 2 PASS / 1 FAIL
  - PASS rows: 1,124,349
  - alerts: 8,540
  - alert rate: 75.96 per 10k
  - mean risk: 82.00

- `artifacts/batch_recent7`: 6 PASS / 1 FAIL
  - PASS rows: 2,909,737
  - alerts: 20,724
  - alert rate: 71.22 per 10k
  - mean risk: 81.83

- `artifacts/batch_recent14`: 13 PASS / 1 FAIL
  - PASS rows: 4,977,532
  - alerts: 37,398
  - alert rate: 75.13 per 10k
  - mean risk: 84.92 (range 78-100)
  - detector totals:
    - `mirror_reversal`: 34,935
    - `balanced_churn`: 2,421
    - `layering_cluster`: 42

- `artifacts/batch_recent28`: 27 PASS / 1 FAIL
  - PASS rows: 9,728,304
  - alerts: 74,153
  - alert rate: 76.22 per 10k
  - mean risk: 85.59

- `artifacts/batch_oldest6_tail`: 6 PASS / 0 FAIL
  - PASS rows: 2,450,475
  - alerts: 14,222
  - alert rate: 58.04 per 10k
  - mean risk: 75.67

## 7) Important Caveats / Drift You Must Know

1. Risk-score drift across artifact generations:
   - Early single-day runs in `artifacts/real_20260204` and `artifacts/real_20260205` show risk `100`.
   - Later batch runs for same days show lower calibrated scores (e.g., 84 / 80).
   - Treat older one-off artifact scores as pre-calibration reference.

2. Synthetic validation command mismatch:
   - `docs/STEP_06` and parts of `WORKLOG` show CLI `validate-synthetic`.
   - Current `cli.py` does **not** expose `validate-synthetic`.
   - Synthetic modules/tests still exist and work via Python imports.

3. `batch_full65` is referenced in file tree/history, but no top-level `batch_full65/batch_summary.*` currently present in `artifacts/`.

4. Ingest aligns to prior context only (at-or-before); if no prior candle/orderbook, context fields are null.

5. Path handling note:
   - Read-only open uses URI `file:{db_path}?mode=ro` without explicit URL encoding.
   - Likely fine for current paths; could break on unusual path chars.

## 8) How To Run (Fast Start for Next AI)

From repo root:

```bash
cd /home/bigdan7/Projects/wash_detector
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py' -v
```

Single DB pipeline:

```bash
PYTHONPATH=src python3 -m wash_detector.cli run \
  --source-db /home/bigdan7/Documents/TRADING/DATA/TABLET_TRADING_DATA/btcusdt_20260205.db \
  --artifacts-dir /home/bigdan7/Projects/wash_detector/artifacts \
  --run-name ai_takeoff_trial \
  --overwrite
```

Batch run:

```bash
PYTHONPATH=src python3 -m wash_detector.cli batch-run-real \
  --data-dir /home/bigdan7/Documents/TRADING/DATA/TABLET_TRADING_DATA \
  --artifacts-dir /home/bigdan7/Projects/wash_detector/artifacts \
  --run-name ai_takeoff_batch \
  --max-days 14 \
  --overwrite
```

## 9) Recommended Next Moves (If You’re Taking Over)

1. Decide target operating regime:
   - Is high alert density desired (surveillance mode), or do you need higher precision with lower alert volume?
2. Add config file support for all detector thresholds (currently hard-coded in `detect.py`).
3. Re-expose synthetic validation in CLI (or document it as code-only) to remove docs/code drift.
4. Add regression snapshot tests for detector counts/risk score on fixed fixtures.
5. Add explicit versioning for risk calibration so artifact comparisons are unambiguous.

## 10) Minimum File Set to Read If Time-Capped

If another AI truly cannot read everything, these files are enough to continue safely:

- `docs/AI_HANDOFF_SUMMARY.md` (this file)
- `src/wash_detector/cli.py`
- `src/wash_detector/schema.py`
- `src/wash_detector/ingest.py`
- `src/wash_detector/detect.py`
- `src/wash_detector/real_batch.py`
- `artifacts/batch_recent14/batch_summary.txt`

