from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from wash_detector.schema import validate_source_db


class SchemaContractTests(unittest.TestCase):
    def _make_db(self, ddl: list[str]) -> Path:
        fd, raw_path = tempfile.mkstemp(prefix="wash_detector_schema_", suffix=".db")
        Path(raw_path).unlink(missing_ok=True)
        db_path = Path(raw_path)
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        for stmt in ddl:
            cur.execute(stmt)
        conn.commit()
        conn.close()
        self.addCleanup(lambda: db_path.unlink(missing_ok=True))
        return db_path

    def test_valid_schema_passes(self) -> None:
        db = self._make_db(
            [
                "CREATE TABLE trades (timestamp INTEGER, side TEXT, price REAL, amount REAL)",
                "CREATE TABLE candles (timestamp INTEGER, open REAL, high REAL, low REAL, close REAL, volume REAL)",
            ]
        )

        report = validate_source_db(db)
        self.assertTrue(report.passed)
        self.assertEqual([], report.errors)

    def test_missing_required_table_fails(self) -> None:
        db = self._make_db(
            [
                "CREATE TABLE trades (timestamp INTEGER, side TEXT, price REAL, amount REAL)",
            ]
        )

        report = validate_source_db(db)
        self.assertFalse(report.passed)
        self.assertTrue(any("Missing required table: candles" in err for err in report.errors))

    def test_missing_required_column_fails(self) -> None:
        db = self._make_db(
            [
                "CREATE TABLE trades (timestamp INTEGER, side TEXT, price REAL)",
                "CREATE TABLE candles (timestamp INTEGER, open REAL, high REAL, low REAL, close REAL, volume REAL)",
            ]
        )

        report = validate_source_db(db)
        self.assertFalse(report.passed)
        self.assertTrue(any("missing required columns: amount" in err.lower() for err in report.errors))


if __name__ == "__main__":
    unittest.main()
