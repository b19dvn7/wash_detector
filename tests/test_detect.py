from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from wash_detector.detect import detect_suspicious_patterns


class DetectTests(unittest.TestCase):
    def _tmp_path(self) -> Path:
        fd, raw_path = tempfile.mkstemp(prefix="wash_detector_detect_", suffix=".db")
        Path(raw_path).unlink(missing_ok=True)
        path = Path(raw_path)
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        return path

    def _build_normalized_db_with_mirror_pair(self) -> Path:
        db = self._tmp_path()
        conn = sqlite3.connect(str(db))
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE normalized_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_rowid INTEGER,
                timestamp_ms INTEGER NOT NULL,
                timestamp_iso TEXT NOT NULL,
                side TEXT NOT NULL,
                price REAL NOT NULL,
                amount REAL NOT NULL,
                notional REAL NOT NULL,
                candle_open REAL,
                candle_high REAL,
                candle_low REAL,
                candle_close REAL,
                candle_volume REAL,
                candle_vwap REAL,
                ob_spread REAL,
                ob_mid_price REAL,
                ob_imbalance REAL
            )
            """
        )

        rows = [
            (1, 1707600000000, "2024-02-11T00:00:00+00:00", "BUY", 100.00, 1.00, 100.00, 99.9, 100.1, 99.9, 100.0, 1000, 100.0, 0.5, 100.0, 0.1),
            (2, 1707600003000, "2024-02-11T00:00:03+00:00", "SELL", 100.02, 0.98, 98.0196, 99.9, 100.1, 99.9, 100.0, 1000, 100.0, 0.5, 100.0, 0.1),
        ]

        cur.executemany(
            """
            INSERT INTO normalized_trades (
                source_rowid,
                timestamp_ms,
                timestamp_iso,
                side,
                price,
                amount,
                notional,
                candle_open,
                candle_high,
                candle_low,
                candle_close,
                candle_volume,
                candle_vwap,
                ob_spread,
                ob_mid_price,
                ob_imbalance
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

        conn.commit()
        conn.close()
        return db

    def test_detects_mirror_reversal_pair(self) -> None:
        db = self._build_normalized_db_with_mirror_pair()
        report = detect_suspicious_patterns(db)

        detectors = [alert.detector for alert in report.alerts]
        self.assertIn("mirror_reversal", detectors)
        self.assertGreaterEqual(report.risk_score, 1)


if __name__ == "__main__":
    unittest.main()
