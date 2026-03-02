# wash_detector Codebase Scan Report
Generated: 2026-03-02

---

## Summary
- **Python Syntax**: ✅ PASSED (no errors)
- **Tests**: ✅ 31/31 PASSED
- **Package Install**: ✅ Installs successfully

---

## Issues Found

### 1. CRITICAL: 99 Deleted Files in Git
**Location**: `learning/` directory

Many CSV and JSON files have been deleted and are showing as deleted in git:
- `learning/month_final.csv`
- `learning/month_labeled*.csv` (multiple versions)
- `learning/month_test/daily_runs/20260101-20260131/` (detection reports, pipeline summaries)

These appear to be training data or test artifacts that were tracked in git but are now deleted.

**Suggestion**: 
- If these are intentionally deleted, run `git add -A` to stage the deletions
- If accidental, restore with `git restore learning/`

---

### 2. Artifacts Directory
**Location**: `artifacts/`

Contains generated output files. Should likely be in `.gitignore`.

**Suggestion**: Check if `artifacts/` should be ignored

---

### 3. Permissions Issue
Some files have restrictive permissions (`-r-------`):
- `CHANGELOG.md`
- `tests/test_learning.py`

**Suggestion**: Fix permissions with `chmod 644`

---

## Recommendations

1. **Git**: Decide whether to commit the deletions or restore the files
2. **.gitignore**: Consider adding `artifacts/` if not already ignored
3. **Permissions**: Fix chmod on restricted files

---

## Commands to Check/Fix

```bash
# Stage deletions (if intentional)
git add -A

# Fix permissions
chmod 644 CHANGELOG.md tests/test_learning.py

# Check .gitignore
cat .gitignore
```
