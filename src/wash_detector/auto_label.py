"""Auto-labeling for obvious cases.

Automatically classifies clear-cut alerts, only asks human about borderline cases.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Tuple

Label = Literal["TP", "FP", "UNCERTAIN"]


@dataclass
class AutoLabelResult:
    """Result of auto-labeling an alert."""
    label: Label
    confidence: float  # 0-1
    reason: str


def auto_label_mirror_reversal(evidence: Dict) -> AutoLabelResult:
    """Auto-label a mirror reversal alert."""
    delta_ms = evidence.get("delta_ms", 9999)
    amount_gap = evidence.get("amount_gap_ratio", 1.0) * 100  # as percentage
    price_diff = evidence.get("price_diff_bps", 100)
    trades_apart = evidence.get("trades_apart", 99)

    # === USER LEARNED THRESHOLD (PRIORITY 1) ===
    # From user feedback: Max TP = 2861ms, Min FP = 3004ms
    # This MUST be checked first - user's judgment overrides heuristics
    if delta_ms >= 3000:
        return AutoLabelResult("FP", 0.90, f"Slow ({delta_ms/1000:.1f}s) = normal (user-learned)")

    if delta_ms > 2933:
        # Gap zone (2934-2999ms): between TP max (2861ms) and FP min (3004ms)
        # Policy: label as TP with lower confidence since it's in the ambiguous range
        return AutoLabelResult("TP", 0.70, f"Gap zone ({delta_ms}ms) - between TP/FP boundary, labeled wash")

    # === RULE 1: Ultra-fast is always suspicious ===
    if delta_ms < 100:
        if amount_gap < 1:
            return AutoLabelResult("TP", 0.95, f"Ultra-fast ({delta_ms}ms) + similar amount = wash")
        else:
            return AutoLabelResult("TP", 0.90, f"Ultra-fast ({delta_ms}ms) = wash")

    # === RULE 2: Under user threshold = wash (user-learned) ===
    # User labeled everything under 2933ms as TP (covers all remaining delta_ms <= 2933)
    return AutoLabelResult("TP", 0.85, f"Under 2.9s ({delta_ms}ms) = wash (user-learned)")


def auto_label_balanced_churn(evidence: Dict) -> AutoLabelResult:
    """Auto-label a balanced churn alert.

    Thresholds tuned from month of data analysis (5826 alerts):
    - balance_ratio median: 0.15, Q1: 0.08
    - price_move median: 3.92 bps, Q1: 1.59 bps
    - notional median: $402k, Q1: $225k
    """
    total_notional = evidence.get("total_notional", 0)
    balance_ratio = evidence.get("side_balance_ratio", 1.0)
    price_move = evidence.get("price_move_bps", 100)
    window_seconds = evidence.get("window_seconds", 999)

    # === RULE 1: Suspicious if well-balanced + minimal price move (targets top 5%) ===
    # balance < 0.08 (most balanced quarter)
    # price_move < 3 bps (minimal price impact)
    # notional > 100k (meaningful volume)
    if balance_ratio < 0.08 and price_move < 3.0 and total_notional > 100000:
        return AutoLabelResult("TP", 0.75, f"Suspicious: balanced ({balance_ratio:.1%}) + no price move + ${total_notional/1000:.0f}k volume")

    # === RULE 2: Very imbalanced or low volume = normal ===
    if balance_ratio > 0.15:
        return AutoLabelResult("FP", 0.85, "Normal: buy/sell clearly imbalanced (not wash)")

    if total_notional < 50000:
        return AutoLabelResult("FP", 0.80, "Normal: small volume")

    if price_move > 5.0:
        return AutoLabelResult("FP", 0.80, "Normal: significant price move (real trading)")

    if window_seconds > 30:
        return AutoLabelResult("FP", 0.75, "Normal: long time window")

    # === RULE 3: Grey zone (medium balance + medium price move) = uncertain ===
    return AutoLabelResult("UNCERTAIN", 0.50, f"Borderline: balance={balance_ratio:.2f}, price_move={price_move:.1f}bps")


def auto_label_layering_cluster(evidence: Dict) -> AutoLabelResult:
    """Auto-label a layering cluster alert.

    Layering = multiple orders clustered at same price level.
    High trade count + ultra-tight price range = suspicious pattern.
    Evidence: window_trade_count, price_range_bps, window_seconds
    """
    window_seconds = evidence.get("window_seconds", 60)
    trade_count = evidence.get("window_trade_count", 0)
    price_range_bps = evidence.get("price_range_bps", 100)

    # === RULE 1: Very tight clustering + many trades = high suspicion ===
    # trade_count >= 30 AND price_range < 0.1 bps (sub-tick clustering)
    if trade_count >= 30 and price_range_bps < 0.1:
        return AutoLabelResult("TP", 0.80, f"Suspicious: {trade_count} trades at sub-tick level (layering)")

    # === RULE 2: Significant clustering + tight range ===
    # trade_count >= 25 AND price_range < 0.5 bps AND fast
    if trade_count >= 25 and price_range_bps < 0.5 and window_seconds <= 60:
        return AutoLabelResult("TP", 0.75, f"Suspicious: {trade_count} trades in {price_range_bps:.3f}bps range")

    # === RULE 3: Medium clustering at very tight range ===
    if trade_count >= 15 and price_range_bps < 0.2:
        return AutoLabelResult("TP", 0.70, f"Suspicious: dense cluster of {trade_count} trades")

    # === RULE 4: Spread out = normal ===
    if price_range_bps > 1.0:
        return AutoLabelResult("FP", 0.85, f"Normal: trades spread over {price_range_bps:.1f}bps (not clustered)")

    if trade_count < 10:
        return AutoLabelResult("FP", 0.75, "Normal: too few trades to form suspicious layer")

    if window_seconds > 120:
        return AutoLabelResult("FP", 0.70, "Normal: orders spread over 2+ minutes")

    # === RULE 5: Data-tuned: UNCERTAIN splits by price_range ===
    # From backtest: UNCERTAIN with range < 0.05 cluster near TP range (0.01-0.5)
    # While > 0.5 cluster near FP (spread out)
    if price_range_bps < 0.05:
        return AutoLabelResult("TP", 0.70, f"Borderline but tight range ({price_range_bps:.4f}bps) = likely layering")
    if price_range_bps > 0.5:
        return AutoLabelResult("FP", 0.70, f"Wide price range ({price_range_bps:.2f}bps) = not clustered")

    # Genuinely uncertain: moderate clustering (10-20 trades, 0.2-0.5bps)
    # From data: these tend to cluster near TP, lean towards wash
    if trade_count >= 10:
        return AutoLabelResult("TP", 0.65, f"Borderline but significant clustering: {trade_count} trades at {price_range_bps:.3f}bps")
    return AutoLabelResult("UNCERTAIN", 0.50, f"Borderline: {trade_count} trades, {price_range_bps:.3f}bps range")


def auto_label_alert(alert: Dict) -> AutoLabelResult:
    """Auto-label any alert type."""
    detector = alert.get("detector", "unknown")
    evidence = alert.get("evidence", {})

    if detector == "mirror_reversal":
        return auto_label_mirror_reversal(evidence)
    elif detector == "balanced_churn":
        return auto_label_balanced_churn(evidence)
    elif detector == "layering_cluster":
        return auto_label_layering_cluster(evidence)
    else:
        return AutoLabelResult("UNCERTAIN", 0.0, f"Unknown detector: {detector}")


def auto_label_all(
    detection_json_path: str | Path,
    confidence_threshold: float = 0.70,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Auto-label all alerts, separating by confidence.

    Returns:
        (high_confidence_tp, high_confidence_fp, needs_review)
    """
    report = json.loads(Path(detection_json_path).read_text(encoding="utf-8"))
    alerts = report.get("alerts", [])

    auto_tp: List[Dict] = []
    auto_fp: List[Dict] = []
    needs_review: List[Dict] = []

    for alert in alerts:
        result = auto_label_alert(alert)
        alert_with_label = {**alert, "_auto_label": result.label, "_auto_confidence": result.confidence, "_auto_reason": result.reason}

        if result.confidence >= confidence_threshold:
            if result.label == "TP":
                auto_tp.append(alert_with_label)
            elif result.label == "FP":
                auto_fp.append(alert_with_label)
            else:
                needs_review.append(alert_with_label)
        else:
            needs_review.append(alert_with_label)

    return auto_tp, auto_fp, needs_review


def run_auto_feedback(
    detection_json_path: str | Path,
    feedback_db_path: str | Path,
    confidence_threshold: float = 0.75,
    review_uncertain: bool = True,
) -> Dict[str, int]:
    """Run automatic feedback with optional human review of uncertain cases.

    Args:
        detection_json_path: Path to detection report
        feedback_db_path: Path to feedback database
        confidence_threshold: Min confidence for auto-labeling (0.70-0.95)
        review_uncertain: If True, prompts for human review of uncertain cases

    Returns:
        Dict with counts of auto_tp, auto_fp, human_reviewed
    """
    from .feedback import FeedbackStore, interactive_feedback_session, _format_alert

    auto_tp, auto_fp, needs_review = auto_label_all(detection_json_path, confidence_threshold)

    store = FeedbackStore(feedback_db_path)

    print("\n" + "=" * 60)
    print("  AUTO-LABELING RESULTS")
    print("=" * 60)
    print(f"""
  📊 Total alerts analyzed: {len(auto_tp) + len(auto_fp) + len(needs_review)}

  ✅ Auto-labeled as NORMAL (FP):     {len(auto_fp):>5}
  🚨 Auto-labeled as WASH (TP):       {len(auto_tp):>5}
  🤔 Needs human review:              {len(needs_review):>5}

  Confidence threshold: {confidence_threshold:.0%}
""")

    # Record auto-labeled ones
    for alert in auto_tp:
        features = {"risk_points": float(alert.get("risk_points", 0))}
        for k, v in alert.get("evidence", {}).items():
            if isinstance(v, (int, float)):
                features[k] = float(v)
        alert_id = f"{alert['detector']}_{alert.get('timestamp_ms', 0)}"
        store.add_feedback(alert_id, alert["detector"], "TP", features, notes="auto-labeled")

    for alert in auto_fp:
        features = {"risk_points": float(alert.get("risk_points", 0))}
        for k, v in alert.get("evidence", {}).items():
            if isinstance(v, (int, float)):
                features[k] = float(v)
        alert_id = f"{alert['detector']}_{alert.get('timestamp_ms', 0)}"
        store.add_feedback(alert_id, alert["detector"], "FP", features, notes="auto-labeled")

    print(f"  ✓ Saved {len(auto_tp)} TP and {len(auto_fp)} FP auto-labels")

    human_reviewed = 0

    if review_uncertain and needs_review:
        print(f"\n  Now reviewing {len(needs_review)} uncertain cases...")
        print("  (These are the borderline ones that need your judgment)\n")
        print("  t = wash trade | f = normal | u = skip | q = quit\n")

        for i, alert in enumerate(needs_review, 1):
            print(f"\n[{i}/{len(needs_review)}] {alert.get('timestamp_iso', '')[:19]}")
            print(f"  Auto-label guess: {alert['_auto_label']} ({alert['_auto_confidence']:.0%})")
            print(f"  Reason: {alert['_auto_reason']}")
            print(_format_alert(alert))

            while True:
                try:
                    choice = input("\n  Your verdict (t/f/u/q): ").strip().lower()
                except EOFError:
                    choice = "q"

                if choice == "q":
                    break
                elif choice in ("t", "f", "u"):
                    verdict = {"t": "TP", "f": "FP", "u": "UNCERTAIN"}[choice]
                    features = {"risk_points": float(alert.get("risk_points", 0))}
                    for k, v in alert.get("evidence", {}).items():
                        if isinstance(v, (int, float)):
                            features[k] = float(v)
                    alert_id = f"{alert['detector']}_{alert.get('timestamp_ms', 0)}"
                    store.add_feedback(alert_id, alert["detector"], verdict, features, notes="human-reviewed")
                    human_reviewed += 1
                    print(f"  → Recorded: {verdict}")
                    break
                else:
                    print("  Invalid. Use: t, f, u, or q")

            if choice == "q":
                break

    print(f"\n" + "=" * 60)
    print(f"  SUMMARY")
    print(f"=" * 60)
    print(f"  Auto-labeled TP (wash):    {len(auto_tp)}")
    print(f"  Auto-labeled FP (normal):  {len(auto_fp)}")
    print(f"  Human reviewed:            {human_reviewed}")
    print(f"  Skipped:                   {len(needs_review) - human_reviewed}")
    print(f"\n  Total feedback entries:    {len(auto_tp) + len(auto_fp) + human_reviewed}")
    print(f"=" * 60)

    return {
        "auto_tp": len(auto_tp),
        "auto_fp": len(auto_fp),
        "human_reviewed": human_reviewed,
        "skipped": len(needs_review) - human_reviewed,
    }
