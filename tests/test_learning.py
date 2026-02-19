"""Tests for adaptive learning system."""
from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from wash_detector.calibrate import calibrate
from wash_detector.feedback import FeedbackStore
from wash_detector.learner import AdaptiveLearner


class CalibrationTests(unittest.TestCase):
    """Tests for auto-calibration."""

    def _tmp_path(self, suffix: str = ".db") -> Path:
        fd, raw_path = tempfile.mkstemp(prefix="wash_learning_", suffix=suffix)
        path = Path(raw_path)
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        return path

    def _create_normalized_db(self, trade_count: int = 100) -> Path:
        """Create a normalized DB with synthetic trades."""
        db_path = self._tmp_path()
        conn = sqlite3.connect(str(db_path))

        conn.execute("""
            CREATE TABLE normalized_trades (
                id INTEGER PRIMARY KEY,
                timestamp_ms INTEGER,
                side TEXT,
                price REAL,
                amount REAL,
                notional REAL,
                ob_spread REAL,
                ob_mid_price REAL,
                ob_imbalance REAL,
                candle_high REAL,
                candle_low REAL
            )
        """)

        # Generate trades with some variation
        base_ts = 1707600000000
        base_price = 100.0

        for i in range(trade_count):
            ts = base_ts + i * 1000  # 1 second apart
            side = "BUY" if i % 2 == 0 else "SELL"
            price = base_price + (i % 10) * 0.1  # Price oscillates
            amount = 0.1 + (i % 5) * 0.05
            notional = price * amount

            conn.execute(
                """INSERT INTO normalized_trades
                   (timestamp_ms, side, price, amount, notional,
                    ob_spread, ob_mid_price, ob_imbalance, candle_high, candle_low)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (ts, side, price, amount, notional, 0.5, price, 0.1, price + 0.5, price - 0.5)
            )

        conn.commit()
        conn.close()
        return db_path

    def test_calibrate_returns_report(self):
        """Calibration should return a report with feature stats."""
        db_path = self._create_normalized_db(200)

        report = calibrate(db_path, sensitivity=0.95)

        self.assertEqual(200, report.trade_count)
        self.assertIn("time_gap_ms", report.features)
        self.assertIn("price_diff_bps", report.features)
        self.assertIn("amount_ratio", report.features)
        self.assertIn("recommended_config", report.to_dict())

    def test_calibrate_generates_config(self):
        """Calibration should generate recommended config values."""
        db_path = self._create_normalized_db(200)

        report = calibrate(db_path, sensitivity=0.95)
        config = report.recommended_config

        self.assertIn("mirror_reversal", config)
        self.assertIn("calibration_meta", config)


class FeedbackTests(unittest.TestCase):
    """Tests for feedback storage."""

    def _tmp_path(self, suffix: str = ".db") -> Path:
        fd, raw_path = tempfile.mkstemp(prefix="wash_feedback_", suffix=suffix)
        path = Path(raw_path)
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        return path

    def test_feedback_store_roundtrip(self):
        """Feedback can be stored and retrieved."""
        db_path = self._tmp_path()
        store = FeedbackStore(db_path)

        store.add_feedback(
            alert_id="test_1",
            detector="mirror_reversal",
            verdict="TP",
            features={"risk_points": 12.0, "price_diff_bps": 5.0},
        )
        store.add_feedback(
            alert_id="test_2",
            detector="mirror_reversal",
            verdict="FP",
            features={"risk_points": 8.0, "price_diff_bps": 15.0},
        )

        entries = store.get_all_feedback()
        self.assertEqual(2, len(entries))

        stats = store.get_stats()
        self.assertEqual(2, stats.total_entries)
        self.assertIn("mirror_reversal", stats.by_detector)
        self.assertEqual(1, stats.by_detector["mirror_reversal"]["TP"])
        self.assertEqual(1, stats.by_detector["mirror_reversal"]["FP"])

    def test_feedback_precision_calculation(self):
        """Precision is calculated correctly."""
        db_path = self._tmp_path()
        store = FeedbackStore(db_path)

        # Add 3 TP and 1 FP
        for i in range(3):
            store.add_feedback(f"tp_{i}", "test_detector", "TP", {"x": 1.0})
        store.add_feedback("fp_1", "test_detector", "FP", {"x": 0.5})

        stats = store.get_stats()
        # Precision = 3 / (3 + 1) = 0.75
        self.assertAlmostEqual(0.75, stats.precision_by_detector["test_detector"])


class AdaptiveLearnerTests(unittest.TestCase):
    """Tests for adaptive learning orchestrator."""

    def _tmp_dir(self) -> Path:
        import tempfile
        path = Path(tempfile.mkdtemp(prefix="wash_learner_"))
        self.addCleanup(lambda: __import__("shutil").rmtree(path, ignore_errors=True))
        return path

    def _create_normalized_db(self) -> Path:
        fd, raw_path = tempfile.mkstemp(prefix="wash_norm_", suffix=".db")
        path = Path(raw_path)
        self.addCleanup(lambda: path.unlink(missing_ok=True))

        conn = sqlite3.connect(str(path))
        conn.execute("""
            CREATE TABLE normalized_trades (
                id INTEGER PRIMARY KEY, timestamp_ms INTEGER, side TEXT,
                price REAL, amount REAL, notional REAL,
                ob_spread REAL, ob_mid_price REAL, ob_imbalance REAL,
                candle_high REAL, candle_low REAL
            )
        """)
        for i in range(100):
            conn.execute(
                "INSERT INTO normalized_trades VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (i, 1707600000000 + i * 1000, "BUY" if i % 2 == 0 else "SELL",
                 100 + i * 0.1, 0.1, 10, 0.5, 100, 0.1, 100.5, 99.5)
            )
        conn.commit()
        conn.close()
        return path

    def test_learner_calibrate_and_export(self):
        """Learner can calibrate and export config."""
        state_dir = self._tmp_dir()
        db_path = self._create_normalized_db()

        learner = AdaptiveLearner(state_dir)
        learner.calibrate_from_data(db_path)

        config = learner.get_adaptive_config()
        self.assertEqual("calibrated", config.source)
        self.assertGreater(config.confidence, 0.5)

    def test_learner_summary(self):
        """Learner can generate summary."""
        state_dir = self._tmp_dir()
        learner = AdaptiveLearner(state_dir)

        summary = learner.get_learning_summary()
        self.assertIn("Adaptive Learning Summary", summary)
        self.assertIn("Calibration:", summary)


if __name__ == "__main__":
    unittest.main()
