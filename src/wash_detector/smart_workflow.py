"""Smart workflow: minimal human input, maximum learning.

The optimal workflow:
1. Auto-label obvious cases (no human needed)
2. Sample 50 truly ambiguous cases for human review
3. Learn thresholds from human decisions
4. Apply learned thresholds to everything
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .feedback import FeedbackStore, _format_alert


@dataclass
class WorkflowStats:
    """Stats from the smart workflow."""
    total_alerts: int
    auto_tp: int
    auto_fp: int
    human_reviewed: int
    human_tp: int
    human_fp: int
    learned_threshold_ms: Optional[float]


def _is_obviously_tp(evidence: Dict) -> bool:
    """Check if alert is obviously a wash trade (no human needed)."""
    delta_ms = evidence.get("delta_ms", 9999)
    amount_gap = evidence.get("amount_gap_ratio", 1.0) * 100

    # Ultra-fast + similar amount = obvious wash
    if delta_ms < 100 and amount_gap < 1:
        return True
    # Very fast + identical = obvious wash
    if delta_ms < 200 and amount_gap < 0.1:
        return True
    return False


def _is_obviously_fp(evidence: Dict) -> bool:
    """Check if alert is obviously normal (no human needed)."""
    delta_ms = evidence.get("delta_ms", 9999)
    amount_gap = evidence.get("amount_gap_ratio", 1.0) * 100

    # Slow = obvious normal
    if delta_ms > 3000:
        return True
    # Different amounts + not super fast = normal
    if delta_ms > 1000 and amount_gap > 1:
        return True
    return False


def run_smart_workflow(
    detection_json_path: str | Path,
    feedback_db_path: str | Path,
    sample_size: int = 50,
) -> WorkflowStats:
    """Run the smart workflow with minimal human input.

    1. Auto-labels obvious cases
    2. Samples ambiguous cases for human review
    3. Learns from human decisions
    4. Returns stats and saves learned thresholds
    """
    report = json.loads(Path(detection_json_path).read_text(encoding="utf-8"))
    alerts = report.get("alerts", [])

    store = FeedbackStore(feedback_db_path)

    obvious_tp: List[Dict] = []
    obvious_fp: List[Dict] = []
    ambiguous: List[Dict] = []

    # Step 1: Categorize all alerts
    for alert in alerts:
        evidence = alert.get("evidence", {})
        if alert.get("detector") == "mirror_reversal":
            if _is_obviously_tp(evidence):
                obvious_tp.append(alert)
            elif _is_obviously_fp(evidence):
                obvious_fp.append(alert)
            else:
                ambiguous.append(alert)
        else:
            # For other detectors, be more conservative
            ambiguous.append(alert)

    print("\n" + "=" * 60)
    print("  SMART WORKFLOW")
    print("=" * 60)
    print(f"""
  Total alerts: {len(alerts)}

  Step 1: Auto-categorized
  ────────────────────────
  🚨 Obviously WASH:     {len(obvious_tp):>5}  (auto-saved as TP)
  ✅ Obviously NORMAL:   {len(obvious_fp):>5}  (auto-saved as FP)
  🤔 Ambiguous:          {len(ambiguous):>5}  (need your input)
""")

    # Save obvious ones
    for alert in obvious_tp:
        features = {k: float(v) for k, v in alert.get("evidence", {}).items() if isinstance(v, (int, float))}
        alert_id = f"{alert['detector']}_{alert.get('timestamp_ms', 0)}_auto"
        store.add_feedback(alert_id, alert["detector"], "TP", features, notes="auto:obvious")

    for alert in obvious_fp:
        features = {k: float(v) for k, v in alert.get("evidence", {}).items() if isinstance(v, (int, float))}
        alert_id = f"{alert['detector']}_{alert.get('timestamp_ms', 0)}_auto"
        store.add_feedback(alert_id, alert["detector"], "FP", features, notes="auto:obvious")

    # Step 2: Sample ambiguous cases
    if len(ambiguous) > sample_size:
        # Sample diverse cases (spread across delta_ms range)
        ambiguous.sort(key=lambda x: x.get("evidence", {}).get("delta_ms", 0))
        step = len(ambiguous) // sample_size
        sampled = [ambiguous[i * step] for i in range(sample_size)]
    else:
        sampled = ambiguous

    print(f"""  Step 2: Human review
  ────────────────────────
  Sampled {len(sampled)} ambiguous cases for your review.
  (This teaches the system YOUR judgment)

  t = wash trade | f = normal | u = skip | q = quit
""")

    human_tp = 0
    human_fp = 0
    human_reviewed = 0
    delta_ms_tp: List[float] = []
    delta_ms_fp: List[float] = []

    for i, alert in enumerate(sampled, 1):
        evidence = alert.get("evidence", {})
        print(f"\n[{i}/{len(sampled)}] {alert.get('timestamp_iso', '')[:19]}")
        print(_format_alert(alert))

        while True:
            try:
                choice = input("\n  Your verdict (t/f/u/q): ").strip().lower()
            except EOFError:
                choice = "q"

            if choice == "q":
                break
            elif choice == "t":
                features = {k: float(v) for k, v in evidence.items() if isinstance(v, (int, float))}
                alert_id = f"{alert['detector']}_{alert.get('timestamp_ms', 0)}_human"
                store.add_feedback(alert_id, alert["detector"], "TP", features, notes="human")
                human_tp += 1
                human_reviewed += 1
                if "delta_ms" in evidence:
                    delta_ms_tp.append(evidence["delta_ms"])
                print("  → Recorded: WASH TRADE")
                break
            elif choice == "f":
                features = {k: float(v) for k, v in evidence.items() if isinstance(v, (int, float))}
                alert_id = f"{alert['detector']}_{alert.get('timestamp_ms', 0)}_human"
                store.add_feedback(alert_id, alert["detector"], "FP", features, notes="human")
                human_fp += 1
                human_reviewed += 1
                if "delta_ms" in evidence:
                    delta_ms_fp.append(evidence["delta_ms"])
                print("  → Recorded: NORMAL")
                break
            elif choice == "u":
                print("  → Skipped")
                break
            else:
                print("  Invalid. Use: t, f, u, or q")

        if choice == "q":
            break

    # Step 3: Learn from human decisions
    learned_threshold = None
    if delta_ms_tp and delta_ms_fp:
        tp_sorted = sorted(delta_ms_tp)
        n = len(tp_sorted)
        tp_median = (tp_sorted[n // 2 - 1] + tp_sorted[n // 2]) / 2 if n % 2 == 0 else tp_sorted[n // 2]
        fp_sorted = sorted(delta_ms_fp)
        n = len(fp_sorted)
        fp_median = (fp_sorted[n // 2 - 1] + fp_sorted[n // 2]) / 2 if n % 2 == 0 else fp_sorted[n // 2]
        learned_threshold = (tp_median + fp_median) / 2

    print(f"""
  ════════════════════════════════════════════════════════════
  RESULTS
  ════════════════════════════════════════════════════════════

  Auto-labeled:
    🚨 Wash trades (TP):   {len(obvious_tp)}
    ✅ Normal (FP):        {len(obvious_fp)}

  Human reviewed:
    🚨 You marked as wash: {human_tp}
    ✅ You marked normal:  {human_fp}
    Total reviewed:        {human_reviewed}
""")

    if learned_threshold:
        print(f"""  LEARNED FROM YOUR DECISIONS:
  ────────────────────────────
  Your TP median delta_ms: {sorted(delta_ms_tp)[len(delta_ms_tp)//2] if delta_ms_tp else 'N/A'}ms
  Your FP median delta_ms: {sorted(delta_ms_fp)[len(delta_ms_fp)//2] if delta_ms_fp else 'N/A'}ms

  → New threshold: {learned_threshold:.0f}ms
    (Below = wash trade, Above = normal)
""")

        # Save learned threshold
        learned_path = Path(feedback_db_path).parent / "learned_threshold.json"
        learned_path.write_text(json.dumps({
            "delta_ms_threshold": learned_threshold,
            "human_tp_count": human_tp,
            "human_fp_count": human_fp,
            "tp_median": sorted(delta_ms_tp)[len(delta_ms_tp)//2] if delta_ms_tp else None,
            "fp_median": sorted(delta_ms_fp)[len(delta_ms_fp)//2] if delta_ms_fp else None,
        }, indent=2))
        print(f"  Saved to: {learned_path}")

    return WorkflowStats(
        total_alerts=len(alerts),
        auto_tp=len(obvious_tp),
        auto_fp=len(obvious_fp),
        human_reviewed=human_reviewed,
        human_tp=human_tp,
        human_fp=human_fp,
        learned_threshold_ms=learned_threshold,
    )
