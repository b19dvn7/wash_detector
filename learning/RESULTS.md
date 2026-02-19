# Wash Detector — Training Results (January 2026)

## Dataset
- **Source**: 31 days of BTC/USDT trades, January 2026
- **Final file**: `learning/month_final.csv` — 111,964 rows
- **Labels**: TP=101,612 (90.75%) | FP=10,351 (9.24%) | UNCERTAIN=1 (0.00%)

---

## Per-Detector Breakdown

| Detector | Total | TP | FP | TP% | FP% |
|---|---|---|---|---|---|
| mirror_reversal | 105,920 | 100,971 | 4,949 | 95.3% | 4.7% |
| balanced_churn | 5,826 | 450 | 5,375 | 7.7% | 92.3% |
| layering_cluster | 218 | 191 | 27 | 87.6% | 12.4% |

---

## Learned Thresholds (from user feedback + data analysis)

### mirror_reversal
- **TP**: delta_ms < 2933ms (user-labeled max TP = 2861ms)
- **FP**: delta_ms >= 3000ms (user-labeled min FP = 3004ms)
- **Gap zone** 2934–2999ms → labeled TP (closer to TP boundary)
- Config: `learning/tuned_config.json` → `user_tp_threshold_ms: 2933`, `user_fp_threshold_ms: 3000`

### balanced_churn
- **TP**: imbalance_ratio < 0.08 AND price_move < 3.0 bps AND notional > $100k
- **FP**: imbalance_ratio > 0.15 OR price_move > 5 bps OR notional < $50k
- Data stats: TP median imbalance=0.042, FP median=0.157 | TP median price_move=1.32bps, FP=4.35bps

### layering_cluster
- **TP**: price_range < 0.1 bps + 30+ trades (sub-tick clustering)
- **TP**: price_range < 0.5 bps + 25+ trades + window ≤ 60s
- **FP**: price_range > 1.0 bps OR < 10 trades
- Key field: `window_trade_count` (NOT `trade_count`)

---

## QC Results (from `learning/qc_out/`)

| Check | Result |
|---|---|
| Contradiction rows | **0** |
| Duplicate keys | **0** |
| layering_cluster precision (all 191 TP) | **96.3%** (184/191) |
| balanced_churn TP price_move max | 2.999 bps (no trend contamination) |

**Questionable layering rows (7 total)**: 10-20 trades, 0.11–0.41 bps, confidence 0.65-0.70 — borderline rule firing at minimum threshold.

---

## ML Finding
- GradientBoostingClassifier trained on 105K mirror_reversal rows → 100% accuracy, 100% delta_ms importance
- **Conclusion**: ML just re-learns the hard rule. Useful only when account-level features added (cross-pair patterns, velocity over time).

---

## Key Bug Fixes (permanent)
1. **Falsy zero**: `if ev.get("amount_gap_ratio")` → `if amount_gap_ratio is not None` (0 is valid)
2. **Rule ordering**: user-learned threshold must be FIRST in `auto_label_mirror_reversal()`
3. **Field name**: layering uses `window_trade_count` not `trade_count`
4. **Timestamps**: always export both `timestamp_iso_utc` (+00:00) and `timestamp_ms` (epoch)

---

## Files
| File | Purpose |
|---|---|
| `learning/month_final.csv` | 111,964 labeled alerts, January 2026 |
| `learning/tuned_config.json` | Data-tuned thresholds (use with `--config`) |
| `learning/qc_out/` | QC packs: boundaries, contradictions, review sample |
| `src/wash_detector/auto_label.py` | Auto-labeling rules |
| `src/wash_detector/export_csv.py` | CSV export with proper timezone + pair_id |
