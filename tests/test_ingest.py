from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from wash_detector.ingest import ingest_to_normalized_db


class IngestTests(unittest.TestCase):
    def _tmp_path(self, suffix: str = ".db") -> Path:
        fd, raw_path = tempfile.mkstemp(prefix="wash_detector_ingest_", suffix=suffix)
        Path(raw_path).unlink(missing_ok=True)
        path = Path(raw_path)
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        return path

    def _build_source_db(self, include_invalid_trade: bool = False) -> Path:
        db_path = self._tmp_path()
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()

        cur.execute(
            "CREATE TABLE trades (timestamp INTEGER, side TEXT, price REAL, amount REAL)"
        )
        cur.execute(
            "CREATE TABLE candles (timestamp INTEGER, open REAL, high REAL, low REAL, close REAL, volume REAL, vwap REAL)"
        )
        cur.execute(
            "CREATE TABLE orderbooks (timestamp INTEGER, spread REAL, mid_price REAL, imbalance REAL)"
        )

        # baseline context rows
        cur.execute(
            "INSERT INTO candles VALUES (1707600000000, 100, 101, 99, 100.5, 1000, 100.2)"
        )
        cur.execute(
            "INSERT INTO orderbooks VALUES (1707600000000, 0.5, 100.4, 0.1)"
        )

        # two valid trades
        cur.execute("INSERT INTO trades VALUES (1707600001000, 'buy', 100.5, 0.2)")
        cur.execute("INSERT INTO trades VALUES (1707600002000, 'SELL', 100.7, 0.3)")

        if include_invalid_trade:
            cur.execute("INSERT INTO trades VALUES (1707600003000, 'HOLD', 100.8, 0.1)")

        conn.commit()
        conn.close()
        return db_path

    def test_ingest_happy_path(self) -> None:
        source = self._build_source_db()
        output = self._tmp_path()

        report = ingest_to_normalized_db(source, output, overwrite=True)

        self.assertTrue(report.schema_report.passed)
        self.assertEqual(2, report.input_rows)
        self.assertEqual(2, report.output_rows)
        self.assertEqual(0, report.skipped_rows)

        conn = sqlite3.connect(str(output))
        cur = conn.cursor()
        cur.execute("SELECT side, price, amount, notional FROM normalized_trades ORDER BY id ASC")
        rows = cur.fetchall()
        conn.close()

        self.assertEqual([("BUY", 100.5, 0.2, 20.1), ("SELL", 100.7, 0.3, 30.21)], rows)

    def test_ingest_skips_invalid_rows(self) -> None:
        source = self._build_source_db(include_invalid_trade=True)
        output = self._tmp_path()

        report = ingest_to_normalized_db(source, output, overwrite=True)

        self.assertEqual(3, report.input_rows)
        self.assertEqual(2, report.output_rows)
        self.assertEqual(1, report.skipped_rows)
        self.assertGreaterEqual(report.warning_count, 1)


if __name__ == "__main__":
    unittest.main()
