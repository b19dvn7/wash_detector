from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional
import json
import shutil

from .detect import DetectionReport, run_detection
from .ingest import IngestReport, ingest_to_normalized_db

logger = logging.getLogger(__name__)


@dataclass
class PipelineReport:
    source_db: str
    run_dir: str
    normalized_db: str
    detection_json: str
    ingest_report: IngestReport
    detection_report: Optional[DetectionReport]
    generated_at_utc: str

    @property
    def passed(self) -> bool:
        if not self.ingest_report.schema_report.passed:
            return False
        if self.detection_report is None:
            return False
        return True

    def to_dict(self) -> Dict[str, object]:
        return {
            "source_db": self.source_db,
            "run_dir": self.run_dir,
            "normalized_db": self.normalized_db,
            "detection_json": self.detection_json,
            "generated_at_utc": self.generated_at_utc,
            "passed": self.passed,
            "ingest": {
                "schema_passed": self.ingest_report.schema_report.passed,
                "input_rows": self.ingest_report.input_rows,
                "output_rows": self.ingest_report.output_rows,
                "skipped_rows": self.ingest_report.skipped_rows,
                "warning_count": self.ingest_report.warning_count,
            },
            "detection": None
            if self.detection_report is None
            else {
                "analyzed_rows": self.detection_report.analyzed_rows,
                "alert_count": len(self.detection_report.alerts),
                "risk_score": self.detection_report.risk_score,
                "risk_level": self.detection_report.risk_level,
            },
        }

    def to_text(self) -> str:
        lines = [
            "Pipeline report",
            f"  source_db: {self.source_db}",
            f"  run_dir: {self.run_dir}",
            f"  normalized_db: {self.normalized_db}",
            f"  detection_json: {self.detection_json}",
            f"  generated_at_utc: {self.generated_at_utc}",
            f"  status: {'PASS' if self.passed else 'FAIL'}",
            "  ingest:",
            f"    - schema: {'PASS' if self.ingest_report.schema_report.passed else 'FAIL'}",
            f"    - input_rows: {self.ingest_report.input_rows}",
            f"    - output_rows: {self.ingest_report.output_rows}",
            f"    - skipped_rows: {self.ingest_report.skipped_rows}",
            f"    - warning_count: {self.ingest_report.warning_count}",
        ]

        if self.detection_report is None:
            lines.append("  detection: not run")
        else:
            lines.extend(
                [
                    "  detection:",
                    f"    - analyzed_rows: {self.detection_report.analyzed_rows}",
                    f"    - alert_count: {len(self.detection_report.alerts)}",
                    f"    - risk_score: {self.detection_report.risk_score}",
                    f"    - risk_level: {self.detection_report.risk_level}",
                ]
            )

        return "\n".join(lines)


def run_pipeline(
    source_db: str | Path,
    *,
    artifacts_dir: str | Path = "artifacts",
    run_name: str | None = None,
    overwrite: bool = False,
    lookback_minutes: int | None = None,
    config_path: str | Path | None = None,
) -> PipelineReport:
    source_db_str = str(source_db)

    if run_name is None:
        run_name = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")

    root = Path(artifacts_dir)
    run_dir = root / run_name

    if run_dir.exists():
        if overwrite:
            shutil.rmtree(run_dir)
        else:
            raise FileExistsError(
                f"run directory already exists: {run_dir} (use overwrite=True to replace)"
            )

    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Pipeline run directory: {run_dir}")

    normalized_db = run_dir / "normalized.db"
    detection_json = run_dir / "detection_report.json"

    logger.info(f"Starting ingest from {source_db_str}")
    ingest_report = ingest_to_normalized_db(
        source_db_path=source_db_str,
        output_db_path=normalized_db,
        overwrite=True,
    )
    logger.info(f"Ingest complete: {ingest_report.output_rows} rows (skipped {ingest_report.skipped_rows})")

    detection_report: DetectionReport | None = None
    if ingest_report.schema_report.passed and ingest_report.output_rows > 0:
        logger.info("Starting detection phase")
        detection_report = run_detection(
            normalized_db_path=normalized_db,
            lookback_minutes=lookback_minutes,
            output_json_path=detection_json,
            config_path=config_path,
        )
        logger.info(f"Detection complete: {len(detection_report.alerts)} alerts")
    else:
        logger.warning("Skipping detection: ingest failed or no output rows")

    report = PipelineReport(
        source_db=source_db_str,
        run_dir=str(run_dir),
        normalized_db=str(normalized_db),
        detection_json=str(detection_json),
        ingest_report=ingest_report,
        detection_report=detection_report,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
    )

    summary_json_path = run_dir / "pipeline_summary.json"
    summary_json_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    summary_txt_path = run_dir / "pipeline_summary.txt"
    summary_txt_path.write_text(report.to_text() + "\n", encoding="utf-8")

    return report
