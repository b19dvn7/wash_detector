"""Feedback-based learning for wash_detector.

Stores user feedback on alerts and uses it to improve detection over time.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

logger = logging.getLogger(__name__)

Verdict = Literal["TP", "FP", "UNCERTAIN"]


@dataclass
class FeedbackEntry:
    """A single feedback entry."""
    alert_id: str
    detector: str
    verdict: Verdict
    timestamp_utc: str
    features: Dict[str, float]
    notes: Optional[str] = None


@dataclass
class FeedbackStats:
    """Statistics about collected feedback."""
    total_entries: int
    by_detector: Dict[str, Dict[str, int]]  # detector -> {TP: n, FP: n, UNCERTAIN: n}
    precision_by_detector: Dict[str, float]  # detector -> precision (TP / (TP + FP))
    feature_importance: Dict[str, float]  # feature -> learned importance weight


@dataclass
class LearnedAdjustments:
    """Adjustments learned from feedback."""
    feature_weights: Dict[str, float]  # feature -> weight multiplier
    threshold_adjustments: Dict[str, float]  # config_key -> adjustment factor
    confidence: float  # 0-1, based on feedback volume


class FeedbackStore:
    """SQLite-backed storage for feedback."""

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the feedback database schema."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT NOT NULL,
                    detector TEXT NOT NULL,
                    verdict TEXT NOT NULL CHECK(verdict IN ('TP', 'FP', 'UNCERTAIN')),
                    timestamp_utc TEXT NOT NULL,
                    features_json TEXT NOT NULL,
                    notes TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_detector ON feedback(detector)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_verdict ON feedback(verdict)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_alert ON feedback(alert_id)")
            conn.commit()
        finally:
            conn.close()

    def add_feedback(
        self,
        alert_id: str,
        detector: str,
        verdict: Verdict,
        features: Dict[str, float],
        notes: Optional[str] = None,
    ) -> None:
        """Record feedback for an alert."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO feedback (alert_id, detector, verdict, timestamp_utc, features_json, notes)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    alert_id,
                    detector,
                    verdict,
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps(features),
                    notes,
                )
            )
            conn.commit()
        finally:
            conn.close()
        logger.info(f"Recorded {verdict} feedback for alert {alert_id}")

    def get_all_feedback(self) -> List[FeedbackEntry]:
        """Retrieve all feedback entries."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                "SELECT alert_id, detector, verdict, timestamp_utc, features_json, notes FROM feedback"
            )
            entries = []
            for row in cursor:
                entries.append(FeedbackEntry(
                    alert_id=row[0],
                    detector=row[1],
                    verdict=row[2],
                    timestamp_utc=row[3],
                    features=json.loads(row[4]),
                    notes=row[5],
                ))
        finally:
            conn.close()
        return entries

    def get_stats(self) -> FeedbackStats:
        """Compute statistics from collected feedback."""
        conn = sqlite3.connect(self.db_path)
        try:
            # Count by detector and verdict
            cursor = conn.execute("""
                SELECT detector, verdict, COUNT(*) as cnt
                FROM feedback
                GROUP BY detector, verdict
            """)

            by_detector: Dict[str, Dict[str, int]] = {}
            for detector, verdict, count in cursor:
                if detector not in by_detector:
                    by_detector[detector] = {"TP": 0, "FP": 0, "UNCERTAIN": 0}
                by_detector[detector][verdict] = count

            # Compute precision
            precision_by_detector: Dict[str, float] = {}
            for detector, counts in by_detector.items():
                tp = counts.get("TP", 0)
                fp = counts.get("FP", 0)
                if tp + fp > 0:
                    precision_by_detector[detector] = tp / (tp + fp)
                else:
                    precision_by_detector[detector] = 0.0

            # Total count
            cursor = conn.execute("SELECT COUNT(*) FROM feedback")
            total = cursor.fetchone()[0]
        finally:
            conn.close()

        # Feature importance (simple: average feature values for TP vs FP)
        feature_importance = self._compute_feature_importance()

        return FeedbackStats(
            total_entries=total,
            by_detector=by_detector,
            precision_by_detector=precision_by_detector,
            feature_importance=feature_importance,
        )

    def _compute_feature_importance(self) -> Dict[str, float]:
        """Compute feature importance based on TP vs FP distributions.

        Higher values = feature is more predictive of true positives.
        """
        entries = self.get_all_feedback()

        if len(entries) < 10:
            return {}  # Not enough data

        # Separate TP and FP
        tp_features: Dict[str, List[float]] = {}
        fp_features: Dict[str, List[float]] = {}

        for entry in entries:
            target = tp_features if entry.verdict == "TP" else fp_features if entry.verdict == "FP" else None
            if target is None:
                continue
            for feat, val in entry.features.items():
                if feat not in target:
                    target[feat] = []
                target[feat].append(val)

        # Compute importance: mean(TP) / mean(FP) for each feature
        importance: Dict[str, float] = {}
        all_features = set(tp_features.keys()) | set(fp_features.keys())

        for feat in all_features:
            tp_vals = tp_features.get(feat, [])
            fp_vals = fp_features.get(feat, [])

            if len(tp_vals) >= 3 and len(fp_vals) >= 3:
                tp_mean = sum(tp_vals) / len(tp_vals)
                fp_mean = sum(fp_vals) / len(fp_vals)

                if fp_mean > 0:
                    # Ratio > 1 means feature is higher for TPs
                    importance[feat] = tp_mean / fp_mean
                elif tp_mean > 0:
                    importance[feat] = 2.0  # TP has signal, FP doesn't
                else:
                    importance[feat] = 1.0

        return importance

    def learn_adjustments(self) -> LearnedAdjustments:
        """Learn threshold adjustments from feedback.

        Returns adjustments that can be applied to detection config.
        """
        stats = self.get_stats()

        # Base confidence on feedback volume
        confidence = min(1.0, stats.total_entries / 100)  # Max confidence at 100 samples

        # Feature weights from importance
        feature_weights = stats.feature_importance.copy()

        # Threshold adjustments based on precision
        threshold_adjustments: Dict[str, float] = {}

        for detector, precision in stats.precision_by_detector.items():
            if stats.by_detector[detector]["TP"] + stats.by_detector[detector]["FP"] < 5:
                continue  # Not enough data

            # If precision is low, make thresholds tighter (multiply by < 1)
            # If precision is high, can loosen slightly (multiply by > 1)
            if precision < 0.5:
                # Too many false positives - tighten by 20%
                adjustment = 0.8
            elif precision < 0.7:
                # Moderate FP rate - tighten by 10%
                adjustment = 0.9
            elif precision > 0.9:
                # Very high precision - could loosen by 10%
                adjustment = 1.1
            else:
                adjustment = 1.0

            threshold_adjustments[f"{detector}_sensitivity"] = adjustment

        return LearnedAdjustments(
            feature_weights=feature_weights,
            threshold_adjustments=threshold_adjustments,
            confidence=confidence,
        )


def apply_learned_adjustments(
    base_config: Dict,
    adjustments: LearnedAdjustments,
) -> Dict:
    """Apply learned adjustments to a base configuration.

    Args:
        base_config: Base configuration dict
        adjustments: Learned adjustments from feedback

    Returns:
        Adjusted configuration
    """
    if adjustments.confidence < 0.1:
        logger.warning("Low confidence in learned adjustments - using base config")
        return base_config

    config = json.loads(json.dumps(base_config))  # Deep copy

    # Apply threshold adjustments
    for key, factor in adjustments.threshold_adjustments.items():
        if "mirror_reversal" in key:
            section = "mirror_reversal"
        elif "layering_cluster" in key:
            section = "layering_cluster"
        elif "balanced_churn" in key:
            section = "balanced_churn"
        else:
            continue

        # Adjust risk points based on precision
        if section in config:
            if "base_risk_points" in config[section]:
                original = config[section]["base_risk_points"]
                config[section]["base_risk_points"] = int(original * factor)
                logger.debug(f"Adjusted {section}.base_risk_points: {original} -> {config[section]['base_risk_points']}")

    # Log adjustments applied
    logger.info(f"Applied learned adjustments (confidence={adjustments.confidence:.2f})")

    return config


def _format_mirror_reversal(evidence: Dict) -> str:
    """Format mirror reversal alert in human-readable way."""
    delta_ms = evidence.get("delta_ms", 0)
    amount_gap = evidence.get("amount_gap_ratio", 0) * 100
    price_diff = evidence.get("price_diff_bps", 0)
    trades_apart = evidence.get("trades_apart", 0)

    # Time formatting
    if delta_ms < 1000:
        time_str = f"{delta_ms}ms"
        time_flag = "⚡ VERY FAST" if delta_ms < 100 else "fast"
    else:
        time_str = f"{delta_ms/1000:.1f}s"
        time_flag = "normal"

    # Amount similarity
    if amount_gap < 1:
        amount_str = f"{amount_gap:.2f}% different"
        amount_flag = "🎯 IDENTICAL" if amount_gap < 0.1 else "nearly same"
    else:
        amount_str = f"{amount_gap:.1f}% different"
        amount_flag = "different" if amount_gap > 5 else "similar"

    # Price difference
    if price_diff < 0.5:
        price_flag = "🎯 SAME PRICE"
    elif price_diff < 2:
        price_flag = "very close"
    else:
        price_flag = "different"

    lines = [
        "┌─────────────────────────────────────────────────────────┐",
        "│  MIRROR REVERSAL: BUY ↔ SELL pair detected             │",
        "├─────────────────────────────────────────────────────────┤",
        f"│  ⏱️  Time between:   {time_str:>10}  ({time_flag:>12})   │",
        f"│  📊 Amount match:   {amount_str:>10}  ({amount_flag:>12})   │",
        f"│  💰 Price diff:     {price_diff:>10.2f} bps ({price_flag:>12})   │",
        f"│  📍 Trades apart:   {trades_apart:>10}                       │",
        "├─────────────────────────────────────────────────────────┤",
    ]

    # Recommendation
    suspicious_count = sum([
        delta_ms < 500,
        amount_gap < 1,
        price_diff < 1,
        trades_apart <= 3
    ])

    if suspicious_count >= 3:
        lines.append("│  🚨 LOOKS SUSPICIOUS → probably 't' (wash trade)        │")
    elif suspicious_count <= 1:
        lines.append("│  ✅ LOOKS NORMAL → probably 'f' (legitimate)            │")
    else:
        lines.append("│  🤔 BORDERLINE → use your judgment or 'u' (skip)        │")

    lines.append("└─────────────────────────────────────────────────────────┘")
    return "\n".join(lines)


def _format_balanced_churn(evidence: Dict) -> str:
    """Format balanced churn alert in human-readable way."""
    window_sec = evidence.get("window_seconds", 0)
    trade_count = evidence.get("window_trade_count", 0)
    buy_notional = evidence.get("buy_notional", 0)
    sell_notional = evidence.get("sell_notional", 0)
    total = evidence.get("total_notional", 0)
    price_move = evidence.get("price_move_bps", 0)
    balance_ratio = evidence.get("side_balance_ratio", 0)

    trades_per_sec = trade_count / window_sec if window_sec > 0 else 0

    lines = [
        "┌─────────────────────────────────────────────────────────┐",
        "│  BALANCED CHURN: High volume, price didn't move        │",
        "├─────────────────────────────────────────────────────────┤",
        f"│  ⏱️  Window:        {window_sec:>8.1f} seconds                  │",
        f"│  📈 Trades:        {trade_count:>8} ({trades_per_sec:.0f}/sec)               │",
        f"│  💵 Buy total:     ${buy_notional:>10,.0f}                    │",
        f"│  💵 Sell total:    ${sell_notional:>10,.0f}                    │",
        f"│  📊 Balance:       {balance_ratio*100:>8.1f}% imbalance              │",
        f"│  💰 Price moved:   {price_move:>8.2f} bps                      │",
        "├─────────────────────────────────────────────────────────┤",
    ]

    # For BTC which is very liquid, high volume is normal
    if total > 100000 and price_move < 2 and balance_ratio < 0.1:
        lines.append("│  🚨 SUSPICIOUS: Huge volume, no price move, balanced    │")
    elif total < 50000 or price_move > 5:
        lines.append("│  ✅ PROBABLY NORMAL: Typical market activity            │")
    else:
        lines.append("│  🤔 BORDERLINE: Could be either, use 'u' if unsure      │")

    lines.append("└─────────────────────────────────────────────────────────┘")
    return "\n".join(lines)


def _format_layering_cluster(evidence: Dict) -> str:
    """Format layering cluster alert in human-readable way."""
    trade_count = evidence.get("window_trade_count", 0)
    price_range = evidence.get("price_range_bps", 0)
    avg_amount = evidence.get("avg_amount", 0)

    lines = [
        "┌─────────────────────────────────────────────────────────┐",
        "│  LAYERING CLUSTER: Many small trades, tight price      │",
        "├─────────────────────────────────────────────────────────┤",
        f"│  📈 Trade count:   {trade_count:>10}                       │",
        f"│  💰 Price range:   {price_range:>10.2f} bps                   │",
        f"│  📊 Avg amount:    {avg_amount:>10.4f}                       │",
        "├─────────────────────────────────────────────────────────┤",
        "│  🤔 Check if trades look artificially uniform          │",
        "└─────────────────────────────────────────────────────────┘",
    ]
    return "\n".join(lines)


def _format_alert(alert: Dict) -> str:
    """Format any alert type in human-readable way."""
    detector = alert.get("detector", "unknown")
    evidence = alert.get("evidence", {})

    if detector == "mirror_reversal":
        return _format_mirror_reversal(evidence)
    elif detector == "balanced_churn":
        return _format_balanced_churn(evidence)
    elif detector == "layering_cluster":
        return _format_layering_cluster(evidence)
    else:
        return f"Unknown detector: {detector}\n{json.dumps(evidence, indent=2)}"


def interactive_feedback_session(
    detection_json_path: str | Path,
    feedback_db_path: str | Path,
) -> int:
    """Run interactive feedback session for alerts in a detection report.

    Returns number of feedback entries recorded.
    """
    # Load detection report
    report = json.loads(Path(detection_json_path).read_text(encoding="utf-8"))
    alerts = report.get("alerts", [])

    if not alerts:
        print("No alerts to review.")
        return 0

    store = FeedbackStore(feedback_db_path)
    recorded = 0

    print("\n" + "=" * 60)
    print("  ALERT REVIEW SESSION")
    print("=" * 60)
    print("""
  Your responses:
    t = Wash trade (TRUE positive - detector was RIGHT)
    f = Not wash   (FALSE positive - detector was WRONG)
    u = Unsure     (skip this one)
    q = Quit
""")

    for i, alert in enumerate(alerts, 1):
        detector = alert.get("detector", "unknown")
        evidence = alert.get("evidence", {})
        risk_points = alert.get("risk_points", 0)
        timestamp = alert.get("timestamp_iso", "")

        print(f"\n[{i}/{len(alerts)}] {timestamp[:19]}")
        print(_format_alert(alert))

        while True:
            try:
                choice = input("\n  Your verdict (t/f/u/q): ").strip().lower()
            except EOFError:
                choice = "q"

            if choice == "q":
                print(f"\n✓ Saved {recorded} reviews. Run 'learn --summary' to see progress.")
                return recorded
            elif choice == "t":
                verdict: Verdict = "TP"
                print("  → Marked as WASH TRADE ✓")
                break
            elif choice == "f":
                verdict = "FP"
                print("  → Marked as NORMAL (not wash) ✓")
                break
            elif choice == "u":
                verdict = "UNCERTAIN"
                print("  → Skipped ✓")
                break
            else:
                print("  Invalid. Use: t, f, u, or q")

        # Extract features for learning
        features = {
            "risk_points": float(risk_points),
        }
        for k, v in evidence.items():
            if isinstance(v, (int, float)):
                features[k] = float(v)

        alert_id = f"{detector}_{alert.get('timestamp_ms', i)}"
        store.add_feedback(alert_id, detector, verdict, features)
        recorded += 1

    print(f"\n✓ Review complete! Saved {recorded} reviews.")
    print("  Next: Run 'wash-detector learn --summary' to see what it learned.")
    return recorded
