# Wash Detector Changelog

## Session Work Summary (2026-02-17)

### Features Added

#### 1. Configurable Detection Thresholds (`config.py`)
External JSON configuration support for all detector thresholds.

**Files:** `src/wash_detector/config.py`

**What it does:**
- Dataclass-based configuration for mirror reversal, layering cluster, balanced churn detectors
- Risk scoring weights and escalation thresholds
- Load from JSON file or use sensible defaults
- Export/save configuration

**Usage:**
```python
from wash_detector.config import DetectorConfig

# Load from file
config = DetectorConfig.load("my_config.json")

# Or use defaults
config = DetectorConfig()

# Save current config
config.save("exported_config.json")
```

---

#### 2. Mirror Reversal ±N Window Check
Improved mirror reversal detection to check against multiple previous trades, not just the immediately adjacent one.

**Files:** `src/wash_detector/detect.py`

**What changed:**
- Now checks `lookback_trades` (default: 5) previous trades for reversals
- More robust detection of wash patterns where trades aren't strictly adjacent

**Configuration:**
```json
{
  "mirror_reversal": {
    "lookback_trades": 5,
    "window_ms": 120000,
    "amount_gap_ratio": 0.20,
    "price_diff_bps": 8.0
  }
}
```

---

#### 3. Orderbook Context in Detection Scoring
Detection scoring now incorporates orderbook spread and imbalance data when available.

**Files:** `src/wash_detector/detect.py`, `src/wash_detector/config.py`

**What it does:**
- Mirror reversal: bonus points when spread is tight (`spread_threshold_bps`)
- Layering cluster: bonus points when orderbook is imbalanced (`imbalance_threshold`)
- Low volatility detection using candle data

**Configuration:**
```json
{
  "mirror_reversal": {
    "tight_spread_bonus": 2,
    "spread_threshold_bps": 5.0,
    "low_vol_bonus": 3,
    "low_volatility_bps": 6.0
  },
  "layering_cluster": {
    "high_imbalance_bonus": 2,
    "imbalance_threshold": 0.3
  }
}
```

---

#### 4. Logging Module
Proper Python logging instead of print statements.

**Files:**
- `src/wash_detector/logging_config.py` (new)
- `src/wash_detector/cli.py` (modified)
- `src/wash_detector/detect.py` (modified)
- `src/wash_detector/ingest.py` (modified)
- `src/wash_detector/pipeline.py` (modified)
- `src/wash_detector/real_batch.py` (modified)

**CLI flags:**
```bash
# Normal (WARNING level - quiet)
wash-detector run --source-db data.db

# Verbose (INFO level - progress info)
wash-detector -v run --source-db data.db

# Debug (DEBUG level - everything)
wash-detector --debug run --source-db data.db

# Log to file
wash-detector -v --log-file run.log run --source-db data.db
```

**Programmatic:**
```python
from wash_detector.logging_config import configure_logging
import logging

configure_logging(level=logging.DEBUG, log_file="debug.log")
```

---

#### 5. Progress Indicators (tqdm)
Colorful progress bars for batch processing.

**Files:**
- `src/wash_detector/real_batch.py` (modified)
- `pyproject.toml` (added tqdm dependency)

**What it looks like:**
```
Processing DBs:  60%|████████████▊        | 3/5 [00:12<00:08, btcusdt_20240210.db]
```

---

#### 6. Timestamp Parsing Edge Case Tests
Comprehensive test coverage for timestamp normalization.

**Files:** `tests/test_timestamp_parsing.py` (new)

**Formats tested:**
- Milliseconds (13 digits) - passthrough
- Seconds (10 digits) - ×1000
- Microseconds (16 digits) - ÷1000
- ISO strings with timezone (`+00:00`)
- ISO strings with Z suffix
- ISO strings without timezone (treated as UTC)
- Numeric strings
- Error cases: null, empty, NaN, infinity, too large, too small

---

### What Wasn't Great / Limitations

#### 1. **No streaming/incremental detection**
The detector loads all trades into memory. For very large datasets (millions of trades), this could be problematic. Would need chunked processing for production scale.

#### 2. **Orderbook context is point-in-time only**
Uses the most recent orderbook snapshot before each trade. Doesn't track orderbook changes or depth evolution, which could provide richer signal.

#### 3. **tqdm dependency added**
Added external dependency for progress bars. Could have used a zero-dependency approach with simple text progress, but tqdm is standard and widely available.

#### 4. **Logging goes to stderr**
This is intentional (keeps stdout clean for CLI output) but might confuse users expecting logs on stdout.

#### 5. **Config changes require code changes to add new fields**
Adding new configuration options requires modifying the dataclass. A more flexible approach would be a dict-based config, but that loses type safety.

#### 6. **Mirror reversal window is count-based, not time-based**
Checks N previous trades regardless of time gap. Could miss patterns if there's a gap in trading activity.

---

### Files Changed Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `src/wash_detector/config.py` | Modified | Added orderbook/volatility bonus configs |
| `src/wash_detector/detect.py` | Modified | ±N window, orderbook scoring, logging |
| `src/wash_detector/ingest.py` | Modified | Added logging |
| `src/wash_detector/pipeline.py` | Modified | Added logging + config_path parameter |
| `src/wash_detector/real_batch.py` | Modified | Added logging + tqdm progress + config_path |
| `src/wash_detector/cli.py` | Modified | Added -v/--debug/--log-file/--config flags |
| `src/wash_detector/logging_config.py` | **New** | Centralized logging configuration |
| `src/wash_detector/calibrate.py` | **New** | Auto-calibration from data distributions |
| `src/wash_detector/feedback.py` | **New** | Feedback storage and precision learning |
| `src/wash_detector/learner.py` | **New** | Adaptive learning orchestrator |
| `tests/test_timestamp_parsing.py` | **New** | 15 timestamp edge case tests |
| `tests/test_learning.py` | **New** | 6 learning system tests |
| `pyproject.toml` | Modified | Added tqdm>=4.66 dependency |
| `CHANGELOG.md` | **New** | This documentation file |

---

### How to Run

#### Install
```bash
cd /home/bigdan7/Projects/wash_detector
pip install -e .
```

#### Run Tests
```bash
PYTHONPATH=src python3 -m pytest tests/ -v
```

#### CLI Commands

**Check schema contract:**
```bash
wash-detector schema-contract
```

**Validate a source DB:**
```bash
wash-detector schema-check --db-path /path/to/data.db
```

**Run full pipeline (ingest + detect):**
```bash
wash-detector -v run --source-db /path/to/data.db --artifacts-dir ./artifacts
```

**Batch process multiple days:**
```bash
wash-detector -v batch-run-real \
  --data-dir /home/bigdan7/Documents/TRADING/DATA/TABLET_TRADING_DATA \
  --artifacts-dir ./artifacts \
  --max-days 5
```

**With custom config:**
```bash
# First export default config
python3 -c "from wash_detector.config import DetectorConfig; DetectorConfig().save('config.json')"

# Edit config.json as needed, then run with --config flag
wash-detector -v run --source-db data.db --config config.json
wash-detector -v batch-run-real --data-dir /path/to/data --config config.json
```

#### Programmatic Usage

```python
from wash_detector.pipeline import run_pipeline
from wash_detector.logging_config import configure_logging
import logging

# Enable verbose logging
configure_logging(level=logging.INFO)

# Run detection
report = run_pipeline(
    source_db="/path/to/data.db",
    artifacts_dir="./artifacts",
    lookback_minutes=60,  # Optional: only analyze last 60 min
)

print(f"Risk score: {report.detection_report.risk_score}")
print(f"Risk level: {report.detection_report.risk_level}")
print(f"Alerts: {len(report.detection_report.alerts)}")
```

---

---

### Tuning Status

**Current state: DEFAULT THRESHOLDS - NOT TUNED TO YOUR DATA**

The detector uses conservative defaults designed to minimize false positives. You likely need to tune thresholds based on your specific trading data characteristics.

#### How to Tune

**Step 1: Run on your data and examine results**
```bash
wash-detector -v batch-run-real \
  --data-dir /home/bigdan7/Documents/TRADING/DATA/TABLET_TRADING_DATA \
  --max-days 5 \
  --artifacts-dir ./tuning_baseline
```

**Step 2: Review the detection reports**
```bash
# Check summary
cat ./tuning_baseline/batch_summary.txt

# Look at individual day alerts
cat ./tuning_baseline/runs/btcusdt_20240210/detection_report.json | python3 -m json.tool
```

**Step 3: Export and modify config**
```python
from wash_detector.config import DetectorConfig

# Export current defaults
config = DetectorConfig()
config.save("tuning_config.json")
```

**Step 4: Adjust thresholds in `tuning_config.json`**

Key parameters to tune:

| Parameter | Default | Tune If... |
|-----------|---------|------------|
| `mirror_reversal.window_ms` | 120000 (2 min) | Too many/few mirror alerts |
| `mirror_reversal.amount_gap_ratio` | 0.20 | Trades vary in size |
| `mirror_reversal.price_diff_bps` | 8.0 | Market is volatile/stable |
| `layering_cluster.min_trades` | 8 | Clusters are larger/smaller |
| `layering_cluster.price_range_bps` | 6.0 | Price ranges differ |
| `balanced_churn.balance_ratio` | 0.20 | Side imbalance tolerance |
| `balanced_churn.min_trades` | 20 | Churn windows are different |

**Step 5: Run with modified config**
```bash
# Via CLI (all commands support --config)
wash-detector -v detect --normalized-db normalized.db --config tuning_config.json
wash-detector -v run --source-db data.db --config tuning_config.json
wash-detector -v batch-run-real --data-dir /path/to/data --config tuning_config.json

# Or programmatically
from wash_detector.pipeline import run_pipeline
report = run_pipeline(source_db="data.db", config_path="tuning_config.json")
```

#### Tuning Strategy

1. **Start conservative** (current defaults) - minimize false positives
2. **Run on known-clean data** - establish baseline alert rate
3. **Run on suspected-dirty data** - see if alerts increase
4. **Adjust iteratively**:
   - Too many alerts → tighten thresholds (lower bps, shorter windows)
   - Too few alerts → loosen thresholds (higher bps, longer windows)

#### What "Tuned" Means

The detector is "tuned" when:
- False positive rate is acceptable for your use case
- Known wash patterns are detected
- Alert volume is manageable for review
- Risk scores correlate with actual suspicious activity

**Recommended: Run on 1 week of data, review 50+ alerts manually, adjust based on false positive rate.**

---

---

### Adaptive Learning System (NEW)

The detector now includes a **self-learning system** that combines:
1. **Auto-calibration** - learns from your data distributions
2. **Feedback learning** - improves from your corrections

#### New Files
| File | Purpose |
|------|---------|
| `src/wash_detector/calibrate.py` | Analyzes data, learns distributions, generates thresholds |
| `src/wash_detector/feedback.py` | Stores user feedback, computes precision, learns weights |
| `src/wash_detector/learner.py` | Combines calibration + feedback into adaptive configs |
| `tests/test_learning.py` | 6 tests for learning system |

#### New CLI Commands

**Auto-calibrate from your data:**
```bash
# Quick calibration - generates config from data distributions
wash-detector calibrate \
  --normalized-db normalized.db \
  --output-config calibrated_config.json \
  --sensitivity 0.95

# Sensitivity: 0.95 = flag top 5%, 0.99 = flag top 1%
```

**Interactive feedback session:**
```bash
# Review alerts and mark them as TP (true positive) or FP (false positive)
wash-detector feedback \
  --detection-json artifacts/run_xyz/detection_report.json \
  --feedback-db my_feedback.db
```

**Adaptive learning (full workflow):**
```bash
# Initialize learning state directory
wash-detector learn --state-dir .wash_learning --summary

# Calibrate from your data
wash-detector learn --state-dir .wash_learning \
  --calibrate-from normalized.db

# Export current best config
wash-detector learn --state-dir .wash_learning \
  --export-config adaptive_config.json
```

#### How the Learning Works

**Phase 1: Auto-Calibration**
- Analyzes your normalized trades database
- Computes distributions for: time gaps, price differences, amount ratios, etc.
- Sets thresholds at percentiles (e.g., "flag trades in the 1st percentile of price difference")

**Phase 2: Feedback Learning**
- You review alerts and mark them TP/FP
- System tracks precision per detector
- Adjusts risk weights based on false positive rate
- High FP rate → tighter thresholds; High precision → can loosen slightly

**Phase 3: Hybrid Config**
- Combines calibration (data-driven thresholds) with feedback (precision-adjusted weights)
- Confidence score indicates reliability (more data + feedback = higher confidence)

#### Complete Learning Workflow

```bash
# 1. Run pipeline to get initial detection
wash-detector run --source-db data.db --artifacts-dir ./artifacts

# 2. Calibrate from the normalized data
wash-detector learn --state-dir .wash_learning \
  --calibrate-from ./artifacts/run_*/normalized.db

# 3. Review alerts and provide feedback
wash-detector feedback \
  --detection-json ./artifacts/run_*/detection_report.json \
  --feedback-db .wash_learning/feedback.db

# 4. Export improved config
wash-detector learn --state-dir .wash_learning \
  --export-config improved_config.json

# 5. Re-run with improved config
wash-detector run --source-db data.db --config improved_config.json

# 6. Repeat steps 3-5 as the system learns
```

#### Programmatic Usage

```python
from wash_detector.learner import AdaptiveLearner

# Create learner with persistent state
learner = AdaptiveLearner(state_dir=".wash_learning")

# Calibrate from data
learner.calibrate_from_data("normalized.db", sensitivity=0.95)

# Add feedback programmatically
learner.add_feedback(
    alert_id="mirror_1707600000",
    detector="mirror_reversal",
    verdict="FP",  # or "TP" or "UNCERTAIN"
    features={"risk_points": 12, "price_diff_bps": 5.0}
)

# Get adaptive config (combines calibration + feedback)
adaptive = learner.get_adaptive_config()
print(f"Config source: {adaptive.source}")  # "hybrid"
print(f"Confidence: {adaptive.confidence:.1%}")

# Export for CLI use
learner.export_config("adaptive_config.json")

# Check learning progress
print(learner.get_learning_summary())
```

---

### Test Results

```
31 passed in 0.97s

tests/test_detect.py - 1 test
tests/test_ingest.py - 2 tests
tests/test_learning.py - 6 tests (NEW)
tests/test_pipeline.py - 1 test
tests/test_real_batch.py - 1 test
tests/test_schema_contract.py - 3 tests
tests/test_smoke.py - 1 test
tests/test_timestamp_parsing.py - 15 tests
tests/test_validation.py - 1 test
```
