from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from wash_detector.real_batch import run_real_batch


class RealBatchTests(unittest.TestCase):
    def _create_valid_day_db(self, path: Path) -> None:
        conn = sqlite3.connect(str(path))
        cur = conn.cursor()
        cur.execute("CREATE TABLE trades (timestamp INTEGER, side TEXT, price REAL, amount REAL)")
        cur.execute(
            "CREATE TABLE candles (timestamp INTEGER, open REAL, high REAL, low REAL, close REAL, volume REAL, vwap REAL)"
        )
        cur.execute(
            "CREATE TABLE orderbooks (timestamp INTEGER, spread REAL, mid_price REAL, imbalance REAL)"
        )

        cur.execute("INSERT INTO candles VALUES (1707600000000, 100, 100.2, 99.8, 100.0, 1000, 100.0)")
        cur.execute("INSERT INTO orderbooks VALUES (1707600000000, 0.4, 100.0, 0.0)")
        cur.execute("INSERT INTO trades VALUES (1707600001000, 'BUY', 100.0, 0.2)")
        cur.execute("INSERT INTO trades VALUES (1707600002000, 'SELL', 100.01, 0.2)")

        conn.commit()
        conn.close()

    def test_batch_run_handles_pass_and_fail_days(self) -> None:
        with tempfile.TemporaryDirectory(prefix="wash_detector_real_batch_") as td:
            root = Path(td)
            data_dir = root / "data"
            data_dir.mkdir(parents=True, exist_ok=True)

            valid = data_dir / "btcusdt_20260101.db"
            self._create_valid_day_db(valid)

            malformed = data_dir / "btcusdt_20260102.db"
            malformed.write_text("not a sqlite database", encoding="utf-8")

            report = run_real_batch(
                data_dir=data_dir,
                artifacts_dir=root,
                run_name="batch_test",
                overwrite=True,
                max_days=10,
                newest_first=False,
            )

            self.assertEqual(2, report.total_files)
            self.assertEqual(2, report.processed_files)
            self.assertEqual(1, report.pass_count)
            self.assertEqual(1, report.fail_count)

            run_dir = root / "batch_test"
            self.assertTrue((run_dir / "batch_summary.json").exists())
            self.assertTrue((run_dir / "batch_summary.txt").exists())
            self.assertTrue((run_dir / "batch_summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
