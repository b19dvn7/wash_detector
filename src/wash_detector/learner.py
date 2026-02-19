"""Adaptive learning system for wash_detector.

Combines auto-calibration from data with feedback-based learning
to produce optimized detection configurations.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .calibrate import CalibrationReport, calibrate, generate_calibrated_config
from .config import DetectorConfig
from .feedback import FeedbackStore, LearnedAdjustments, apply_learned_adjustments

logger = logging.getLogger(__name__)


@dataclass
class LearningState:
    """Persistent state for the learning system."""
    calibration_report: Optional[CalibrationReport] = None
    feedback_count: int = 0
    last_calibrated_utc: Optional[str] = None
    last_feedback_utc: Optional[str] = None
    current_config: Optional[Dict] = None
    learning_iterations: int = 0


@dataclass
class AdaptiveConfig:
    """An adaptively-generated configuration with provenance."""
    config: Dict
    source: str  # "default", "calibrated", "feedback_adjusted", "hybrid"
    confidence: float  # 0-1
    calibration_trades: int
    feedback_entries: int
    generated_at_utc: str

    def to_dict(self) -> Dict:
        return {
            "config": self.config,
            "meta": {
                "source": self.source,
                "confidence": self.confidence,
                "calibration_trades": self.calibration_trades,
                "feedback_entries": self.feedback_entries,
                "generated_at_utc": self.generated_at_utc,
            }
        }


class AdaptiveLearner:
    """Main adaptive learning orchestrator.

    Usage:
        learner = AdaptiveLearner(state_dir="./learning_state")

        # Phase 1: Calibrate from data
        learner.calibrate_from_data("normalized.db")

        # Phase 2: Generate initial config
        config = learner.get_adaptive_config()

        # Phase 3: After user reviews alerts, record feedback
        learner.record_feedback_session("detection_report.json")

        # Phase 4: Get improved config incorporating feedback
        improved_config = learner.get_adaptive_config()
    """

    def __init__(self, state_dir: str | Path):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.state_file = self.state_dir / "learning_state.json"
        self.calibration_file = self.state_dir / "calibration.json"
        self.feedback_db = self.state_dir / "feedback.db"
        self.current_config_file = self.state_dir / "adaptive_config.json"

        self._feedback_store: Optional[FeedbackStore] = None
        self._calibration: Optional[CalibrationReport] = None

    @property
    def feedback_store(self) -> FeedbackStore:
        if self._feedback_store is None:
            self._feedback_store = FeedbackStore(self.feedback_db)
        return self._feedback_store

    def _load_calibration(self) -> Optional[CalibrationReport]:
        """Load cached calibration report."""
        if self._calibration is not None:
            return self._calibration

        if not self.calibration_file.exists():
            return None

        try:
            data = json.loads(self.calibration_file.read_text(encoding="utf-8"))
            # Reconstruct CalibrationReport from dict
            from .calibrate import FeatureStats
            features = {}
            for name, fdata in data.get("features", {}).items():
                features[name] = FeatureStats(**fdata)

            self._calibration = CalibrationReport(
                source_db=data["source_db"],
                trade_count=data["trade_count"],
                generated_at_utc=data["generated_at_utc"],
                features=features,
                recommended_config=data.get("recommended_config", {}),
            )
            return self._calibration
        except Exception as e:
            logger.warning(f"Failed to load calibration: {e}")
            return None

    def calibrate_from_data(
        self,
        normalized_db_path: str | Path,
        sensitivity: float = 0.95,
    ) -> CalibrationReport:
        """Run calibration on trade data.

        Args:
            normalized_db_path: Path to normalized trades database
            sensitivity: Detection sensitivity (0.95 = flag top 5%)

        Returns:
            CalibrationReport with learned distributions
        """
        logger.info(f"Calibrating from {normalized_db_path}")

        report = calibrate(
            normalized_db_path,
            output_json_path=self.calibration_file,
            sensitivity=sensitivity,
        )

        self._calibration = report
        logger.info(f"Calibration complete: {report.trade_count} trades analyzed")

        return report

    def record_feedback_session(
        self,
        detection_json_path: str | Path,
    ) -> int:
        """Run interactive feedback session.

        Returns number of feedback entries recorded.
        """
        from .feedback import interactive_feedback_session
        return interactive_feedback_session(detection_json_path, self.feedback_db)

    def add_feedback(
        self,
        alert_id: str,
        detector: str,
        verdict: str,
        features: Dict[str, float],
        notes: Optional[str] = None,
    ) -> None:
        """Programmatically add feedback for an alert."""
        self.feedback_store.add_feedback(alert_id, detector, verdict, features, notes)

    def get_adaptive_config(self) -> AdaptiveConfig:
        """Generate the best configuration based on all available learning.

        Combines:
        1. Default config as base
        2. Calibration adjustments (if available)
        3. Feedback-based adjustments (if available)
        """
        # Start with defaults
        base_config = DetectorConfig().to_dict()
        source = "default"
        confidence = 0.5
        calibration_trades = 0
        feedback_entries = 0

        # Apply calibration if available
        calibration = self._load_calibration()
        if calibration is not None:
            calibration_trades = calibration.trade_count
            rec = calibration.recommended_config

            # Merge calibrated values
            for section in ["mirror_reversal", "layering_cluster", "balanced_churn"]:
                if section in rec:
                    for key, value in rec[section].items():
                        if not key.startswith("_") and key in base_config.get(section, {}):
                            base_config[section][key] = value

            source = "calibrated"
            # Confidence based on data volume
            confidence = min(0.8, 0.5 + calibration_trades / 100000)
            logger.info(f"Applied calibration from {calibration_trades} trades")

        # Apply feedback adjustments if available
        stats = self.feedback_store.get_stats()
        feedback_entries = stats.total_entries

        if feedback_entries >= 10:
            adjustments = self.feedback_store.learn_adjustments()
            base_config = apply_learned_adjustments(base_config, adjustments)

            source = "hybrid" if calibration is not None else "feedback_adjusted"
            # Boost confidence with feedback
            confidence = min(0.95, confidence + adjustments.confidence * 0.2)
            logger.info(f"Applied feedback learning from {feedback_entries} entries")

        # Save current config
        adaptive = AdaptiveConfig(
            config=base_config,
            source=source,
            confidence=confidence,
            calibration_trades=calibration_trades,
            feedback_entries=feedback_entries,
            generated_at_utc=datetime.now(timezone.utc).isoformat(),
        )

        self.current_config_file.write_text(
            json.dumps(adaptive.to_dict(), indent=2),
            encoding="utf-8"
        )

        return adaptive

    def export_config(self, output_path: str | Path) -> None:
        """Export current adaptive config as a standard config.json."""
        adaptive = self.get_adaptive_config()
        Path(output_path).write_text(
            json.dumps(adaptive.config, indent=2),
            encoding="utf-8"
        )
        logger.info(f"Exported adaptive config to {output_path}")

    def get_learning_summary(self) -> str:
        """Get a human-readable summary of learning state."""
        lines = ["Adaptive Learning Summary", "=" * 40]

        # Calibration status
        calibration = self._load_calibration()
        if calibration:
            lines.append(f"Calibration: {calibration.trade_count} trades analyzed")
            lines.append(f"  Last calibrated: {calibration.generated_at_utc}")
        else:
            lines.append("Calibration: Not yet calibrated")

        # Feedback status
        stats = self.feedback_store.get_stats()
        lines.append(f"\nFeedback: {stats.total_entries} entries")

        if stats.by_detector:
            lines.append("  By detector:")
            for detector, counts in stats.by_detector.items():
                tp = counts.get("TP", 0)
                fp = counts.get("FP", 0)
                precision = stats.precision_by_detector.get(detector, 0)
                lines.append(f"    {detector}: {tp} TP, {fp} FP (precision={precision:.1%})")

        # Current config source
        if self.current_config_file.exists():
            data = json.loads(self.current_config_file.read_text())
            meta = data.get("meta", {})
            lines.append(f"\nCurrent config: {meta.get('source', 'unknown')}")
            lines.append(f"  Confidence: {meta.get('confidence', 0):.1%}")

        return "\n".join(lines)


def quick_calibrate(
    normalized_db_path: str | Path,
    output_config_path: str | Path,
    sensitivity: float = 0.95,
) -> CalibrationReport:
    """Quick one-shot calibration without persistent state.

    Use this for simple calibration without the full learning system.
    """
    report = calibrate(normalized_db_path, sensitivity=sensitivity)
    generate_calibrated_config(report, output_config_path)
    return report
