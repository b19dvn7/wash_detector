# Claude Chat Ledger (2025-02-17)

Source analyzed in full: `/home/bigdan7/Documents/claude_chat_2025-0217.txt` (5413 lines).

Purpose: preserve the actionable history so the raw transcript can be removed.

---

## 1) Chronological work ledger (from transcript)

## Phase A — Initial validation + code review
- Project explored and test suite run early.
- Reported: initial tests passing.
- A code-review pass identified **7 issues** and then started patching.

Files repeatedly touched in this phase:
- `src/wash_detector/detect.py`
- `src/wash_detector/real_batch.py`
- `src/wash_detector/cli.py`
- `src/wash_detector/ingest.py`
- `src/wash_detector/pipeline.py`
- `src/wash_detector/config.py`

## Phase B — Logging / CLI / runtime hardening
- Added structured logging support and CLI improvements.
- Added/updated:
  - `src/wash_detector/logging_config.py`
  - `src/wash_detector/cli.py`
  - pipeline/real-batch flow
- Ran tests and CLI help checks repeatedly.
- `CHANGELOG.md` updated multiple times.

## Phase C — Timestamp parsing and test expansion
- Added dedicated timestamp parsing test file.
- Transcript shows test count growing and re-runs.

File added:
- `tests/test_timestamp_parsing.py`

## Phase D — Learning pipeline build-out
- Added learning/calibration components and CLI commands.
- Introduced feedback + learner workflow.

Files added:
- `src/wash_detector/calibrate.py`
- `src/wash_detector/feedback.py`
- `src/wash_detector/learner.py`

## Phase E — Auto-labeling and “smart” workflow
- Created and iteratively tuned auto-labeling logic.
- Added “smart” modules and integrated with CLI.

Files added:
- `src/wash_detector/auto_label.py`
- `src/wash_detector/smart_learner.py`
- `src/wash_detector/smart_workflow.py`

## Phase F — Batch runs + export tooling + dataset generation
- Large batch runs over trading DBs.
- Added export labels command/path and fixed export formatting.

Files added:
- `src/wash_detector/export_csv.py`

Outputs generated (learning folder):
- many weekly/monthly labeled CSV artifacts
- `month_final.csv`
- `tuned_config.json`
- DB artifacts (`feedback.db`, `smart.db`)

## Phase G — Threshold tuning + ML check
- Transcript claims threshold tuning reduced uncertain rows substantially.
- ML experiment run (scikit-learn); conclusion in transcript: model mostly learned a single dominant feature (`delta_ms`) rather than new signal.

---

## 2) What is verified now (filesystem/runtime check)

As of current check:
- Project exists: `/home/bigdan7/Projects/wash_detector`
- Key added modules exist:
  - `auto_label.py`, `calibrate.py`, `feedback.py`, `learner.py`, `smart_learner.py`, `smart_workflow.py`, `export_csv.py`, `logging_config.py`
- Learning artifacts exist (including):
  - `learning/month_final.csv`
  - `learning/tuned_config.json`
  - `learning/all_feedback.csv`
  - `learning/feedback.db`, `learning/smart.db`
- Current tests pass:
  - `PYTHONPATH=src python3 -m pytest -q`
  - result: **31 passed**

---

## 3) Claimed in transcript but not independently recomputed here

These were stated in chat output and should be treated as claims until rerun:
- exact metric deltas/tables (e.g., uncertain-rate reductions, detector precision percentages)
- exact class-distribution percentages from intermediate runs
- specific “100% F1” ML claim details

If needed, regenerate from current artifacts with a reproducible script before relying on numbers for decisions.

---

## 4) Practical TODO (clean + high-value)

1. **Freeze reproducible metrics script**
   - one script that recomputes all headline metrics from `learning/month_final.csv`
   - output to `learning/RESULTS_RECOMPUTED.md`

2. **Lock timestamp contract in this project**
   - enforce UTC-aware parsing/writing for ingest/label pipeline
   - fail fast on malformed/naive timestamps

3. **Promote stable configs**
   - keep `tuned_config.json` + `user_learned_config.json` with explicit version/date

4. **Data lineage note**
   - map each major output CSV to source run command and date window

5. **Optional precision/recall modes**
   - keep two explicit profiles (precision-first vs recall-first) to avoid mixing thresholds silently

---

## 5) Safe deletion note

If you remove the raw chat transcript, keep this ledger file plus the project artifacts above. That preserves actionable history without storing the full conversational dump.
