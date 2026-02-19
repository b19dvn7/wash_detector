from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from wash_detector.pipeline import run_pipeline


class PipelineTests(unittest.TestCase):
    def _build_source_db(self, db_path: Path) -> None:
        conn = sqlite3.connect(str(db_path))
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

    def test_run_pipeline_outputs_artifacts(self) -> None:
        with tempfile.TemporaryDirectory(prefix="wash_detector_pipeline_") as td:
            tmpdir = Path(td)
            source = tmpdir / "source.db"
            self._build_source_db(source)

            report = run_pipeline(
                source_db=source,
                artifacts_dir=tmpdir,
                run_name="run_test",
                overwrite=True,
            )

            run_dir = tmpdir / "run_test"
            self.assertTrue(report.passed)
            self.assertTrue((run_dir / "normalized.db").exists())
            self.assertTrue((run_dir / "detection_report.json").exists())
            self.assertTrue((run_dir / "pipeline_summary.json").exists())
            self.assertTrue((run_dir / "pipeline_summary.txt").exists())


if __name__ == "__main__":
    unittest.main()
