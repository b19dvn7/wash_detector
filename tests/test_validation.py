from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from wash_detector.validation import run_synthetic_validation


class ValidationTests(unittest.TestCase):
    def test_synthetic_validation_generates_metrics_and_artifacts(self) -> None:
        with tempfile.TemporaryDirectory(prefix="wash_detector_validation_") as td:
            tmpdir = Path(td)

            report = run_synthetic_validation(
                artifacts_dir=tmpdir,
                run_name="validation_test",
                overwrite=True,
                normal_trades=120,
                suspicious_pairs=20,
                seed=123,
            )

            self.assertIsNotNone(report.metrics)
            self.assertGreater(report.metrics.total_rows, 0)
            self.assertGreater(report.metrics.positive_labels, 0)

            run_dir = tmpdir / "validation_test"
            self.assertTrue((run_dir / "synthetic_source.db").exists())
            self.assertTrue((run_dir / "normalized.db").exists())
            self.assertTrue((run_dir / "detection_report.json").exists())
            self.assertTrue((run_dir / "validation_summary.json").exists())
            self.assertTrue((run_dir / "validation_summary.txt").exists())


if __name__ == "__main__":
    unittest.main()
