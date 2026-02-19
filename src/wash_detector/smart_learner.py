"""Smart learner that actually learns from feedback data.

Uses the feedback you provide to build a model that improves over time.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

from .feedback import FeedbackStore, FeedbackEntry

Label = Literal["TP", "FP", "UNCERTAIN"]


@dataclass
class LearnedThresholds:
    """Thresholds learned from feedback data."""
    # For each feature: (tp_median, fp_median, best_threshold, direction)
    # direction: "below" means values below threshold are TP, "above" means values above are TP
    thresholds: Dict[str, Tuple[float, float, float, str]]
    tp_count: int
    fp_count: int

    def to_dict(self) -> Dict:
        return {
            "thresholds": {k: {"tp_median": v[0], "fp_median": v[1], "threshold": v[2], "direction": v[3]}
                          for k, v in self.thresholds.items()},
            "tp_count": self.tp_count,
            "fp_count": self.fp_count,
        }


def _median(values: List[float]) -> float:
    """Compute median of a list."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n % 2 == 0:
        return (sorted_vals[n//2 - 1] + sorted_vals[n//2]) / 2
    return sorted_vals[n//2]


def _compute_thresholds_from_feedback(entries: List[FeedbackEntry]) -> LearnedThresholds:
    """Learn thresholds from feedback data."""
    tp_features: Dict[str, List[float]] = {}
    fp_features: Dict[str, List[float]] = {}

    tp_count = 0
    fp_count = 0

    for entry in entries:
        if entry.verdict == "TP":
            target = tp_features
            tp_count += 1
        elif entry.verdict == "FP":
            target = fp_features
            fp_count += 1
        else:
            continue

        for feat, val in entry.features.items():
            if feat not in target:
                target[feat] = []
            target[feat].append(val)

    thresholds: Dict[str, Tuple[float, float, float, str]] = {}

    # For each feature, find the best threshold
    all_features = set(tp_features.keys()) | set(fp_features.keys())

    for feat in all_features:
        tp_vals = tp_features.get(feat, [])
        fp_vals = fp_features.get(feat, [])

        if len(tp_vals) < 5 or len(fp_vals) < 5:
            continue  # Not enough data

        tp_med = _median(tp_vals)
        fp_med = _median(fp_vals)

        # Determine direction: which way indicates TP?
        if tp_med < fp_med:
            # Lower values = more likely TP (e.g., delta_ms)
            direction = "below"
            threshold = (tp_med + fp_med) / 2
        else:
            # Higher values = more likely TP
            direction = "above"
            threshold = (tp_med + fp_med) / 2

        thresholds[feat] = (tp_med, fp_med, threshold, direction)

    return LearnedThresholds(thresholds=thresholds, tp_count=tp_count, fp_count=fp_count)


@dataclass
class SmartPrediction:
    """Prediction from the smart learner."""
    label: Label
    confidence: float
    reason: str
    feature_scores: Dict[str, float]  # How each feature voted


class SmartLearner:
    """A learner that actually learns from your feedback."""

    def __init__(self, feedback_db_path: str | Path):
        self.store = FeedbackStore(feedback_db_path)
        self._thresholds: Optional[LearnedThresholds] = None

    def learn(self) -> LearnedThresholds:
        """Learn thresholds from all feedback data."""
        entries = self.store.get_all_feedback()
        self._thresholds = _compute_thresholds_from_feedback(entries)
        return self._thresholds

    def predict(self, evidence: Dict[str, float], detector: str = "mirror_reversal") -> SmartPrediction:
        """Predict TP/FP based on learned thresholds."""
        if self._thresholds is None:
            self.learn()

        if self._thresholds.tp_count < 10 or self._thresholds.fp_count < 10:
            return SmartPrediction("UNCERTAIN", 0.0, "Not enough feedback data yet", {})

        feature_scores: Dict[str, float] = {}
        tp_votes = 0
        fp_votes = 0
        total_weight = 0

        for feat, (tp_med, fp_med, threshold, direction) in self._thresholds.thresholds.items():
            if feat not in evidence:
                continue

            val = evidence[feat]

            # How far is this value from the threshold?
            if direction == "below":
                # Lower = TP
                if val < threshold:
                    # Leans TP
                    distance = (threshold - val) / (threshold - tp_med + 0.001)
                    score = min(1.0, distance)
                    tp_votes += score
                    feature_scores[feat] = score
                else:
                    # Leans FP
                    distance = (val - threshold) / (fp_med - threshold + 0.001)
                    score = min(1.0, distance)
                    fp_votes += score
                    feature_scores[feat] = -score
            else:
                # Higher = TP
                if val > threshold:
                    distance = (val - threshold) / (tp_med - threshold + 0.001)
                    score = min(1.0, distance)
                    tp_votes += score
                    feature_scores[feat] = score
                else:
                    distance = (threshold - val) / (threshold - fp_med + 0.001)
                    score = min(1.0, distance)
                    fp_votes += score
                    feature_scores[feat] = -score

            total_weight += 1

        if total_weight == 0:
            return SmartPrediction("UNCERTAIN", 0.0, "No matching features", feature_scores)

        # Compute final prediction
        tp_score = tp_votes / total_weight
        fp_score = fp_votes / total_weight

        if tp_score > fp_score:
            confidence = min(0.95, 0.5 + (tp_score - fp_score) * 0.5)
            label: Label = "TP"
            reason = f"Learned: {tp_score:.2f} TP vs {fp_score:.2f} FP"
        elif fp_score > tp_score:
            confidence = min(0.95, 0.5 + (fp_score - tp_score) * 0.5)
            label = "FP"
            reason = f"Learned: {fp_score:.2f} FP vs {tp_score:.2f} TP"
        else:
            confidence = 0.5
            label = "UNCERTAIN"
            reason = "Evenly split"

        return SmartPrediction(label, confidence, reason, feature_scores)

    def get_learned_rules(self) -> str:
        """Get human-readable learned rules."""
        if self._thresholds is None:
            self.learn()

        lines = [
            "LEARNED RULES (from your feedback)",
            "=" * 50,
            f"Based on {self._thresholds.tp_count} TP and {self._thresholds.fp_count} FP examples",
            ""
        ]

        for feat, (tp_med, fp_med, threshold, direction) in self._thresholds.thresholds.items():
            if direction == "below":
                lines.append(f"  {feat}:")
                lines.append(f"    TP median: {tp_med:.2f}")
                lines.append(f"    FP median: {fp_med:.2f}")
                lines.append(f"    Rule: if {feat} < {threshold:.2f} → likely TP")
            else:
                lines.append(f"  {feat}:")
                lines.append(f"    TP median: {tp_med:.2f}")
                lines.append(f"    FP median: {fp_med:.2f}")
                lines.append(f"    Rule: if {feat} > {threshold:.2f} → likely TP")
            lines.append("")

        return "\n".join(lines)


def smart_auto_label(
    detection_json_path: str | Path,
    feedback_db_path: str | Path,
    confidence_threshold: float = 0.75,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Auto-label using the smart learner trained on feedback.

    Returns:
        (auto_tp, auto_fp, needs_review)
    """
    learner = SmartLearner(feedback_db_path)
    learner.learn()

    report = json.loads(Path(detection_json_path).read_text(encoding="utf-8"))
    alerts = report.get("alerts", [])

    auto_tp: List[Dict] = []
    auto_fp: List[Dict] = []
    needs_review: List[Dict] = []

    for alert in alerts:
        evidence = alert.get("evidence", {})
        # Convert evidence to flat dict of floats
        features = {}
        for k, v in evidence.items():
            if isinstance(v, (int, float)):
                features[k] = float(v)

        prediction = learner.predict(features, alert.get("detector", "unknown"))

        alert_with_pred = {
            **alert,
            "_smart_label": prediction.label,
            "_smart_confidence": prediction.confidence,
            "_smart_reason": prediction.reason,
        }

        if prediction.confidence >= confidence_threshold:
            if prediction.label == "TP":
                auto_tp.append(alert_with_pred)
            elif prediction.label == "FP":
                auto_fp.append(alert_with_pred)
            else:
                needs_review.append(alert_with_pred)
        else:
            needs_review.append(alert_with_pred)

    return auto_tp, auto_fp, needs_review
