from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional, Set
import json
import shutil
import sqlite3

from .detect import DetectionReport, run_detection
from .ingest import IngestReport, ingest_to_normalized_db
from .synthetic import SyntheticFixture, create_synthetic_source_db


@dataclass
class ValidationMetrics:
    total_rows: int
    positive_labels: int
    predicted_positive: int
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int
    precision: float
    recall: float
    false_positive_rate: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "total_rows": self.total_rows,
            "positive_labels": self.positive_labels,
            "predicted_positive": self.predicted_positive,
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "true_negative": self.true_negative,
            "precision": self.precision,
            "recall": self.recall,
            "false_positive_rate": self.false_positive_rate,
        }


@dataclass
class ValidationThresholds:
    min_precision: float = 0.60
    min_recall: float = 0.60
    max_false_positive_rate: float = 0.20


@dataclass
class SyntheticValidationReport:
    run_dir: str
    fixture: SyntheticFixture
    ingest_report: IngestReport
    detection_report: Optional[DetectionReport]
    metrics: Optional[ValidationMetrics]
    thresholds: ValidationThresholds
    generated_at_utc: str

    @property
    def passed(self) -> bool:
        if not self.ingest_report.schema_report.passed:
            return False
        if self.metrics is None:
            return False

        return (
            self.metrics.precision >= self.thresholds.min_precision
            and self.metrics.recall >= self.thresholds.min_recall
            and self.metrics.false_positive_rate <= self.thresholds.max_false_positive_rate
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "run_dir": self.run_dir,
            "generated_at_utc": self.generated_at_utc,
            "passed": self.passed,
            "fixture": {
                "source_db": self.fixture.source_db,
                "normal_trades": self.fixture.normal_trades,
                "suspicious_pairs": self.fixture.suspicious_pairs,
                "positive_source_rowids": self.fixture.positive_source_rowids,
            },
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
            "metrics": None if self.metrics is None else self.metrics.to_dict(),
            "thresholds": {
                "min_precision": self.thresholds.min_precision,
                "min_recall": self.thresholds.min_recall,
                "max_false_positive_rate": self.thresholds.max_false_positive_rate,
            },
        }

    def to_text(self) -> str:
        lines = [
            "Synthetic validation report",
            f"  run_dir: {self.run_dir}",
            f"  generated_at_utc: {self.generated_at_utc}",
            f"  status: {'PASS' if self.passed else 'FAIL'}",
            "  fixture:",
            f"    - normal_trades: {self.fixture.normal_trades}",
            f"    - suspicious_pairs: {self.fixture.suspicious_pairs}",
            f"    - labeled_positives: {len(self.fixture.positive_source_rowids)}",
            "  ingest:",
            f"    - schema: {'PASS' if self.ingest_report.schema_report.passed else 'FAIL'}",
            f"    - input_rows: {self.ingest_report.input_rows}",
            f"    - output_rows: {self.ingest_report.output_rows}",
            f"    - skipped_rows: {self.ingest_report.skipped_rows}",
        ]

        if self.detection_report is not None:
            lines.extend(
                [
                    "  detection:",
                    f"    - analyzed_rows: {self.detection_report.analyzed_rows}",
                    f"    - alert_count: {len(self.detection_report.alerts)}",
                    f"    - risk_score: {self.detection_report.risk_score}",
                    f"    - risk_level: {self.detection_report.risk_level}",
                ]
            )

        if self.metrics is not None:
            lines.extend(
                [
                    "  metrics:",
                    f"    - precision: {self.metrics.precision:.4f}",
                    f"    - recall: {self.metrics.recall:.4f}",
                    f"    - false_positive_rate: {self.metrics.false_positive_rate:.4f}",
                    f"    - TP/FP/FN/TN: {self.metrics.true_positive}/{self.metrics.false_positive}/{self.metrics.false_negative}/{self.metrics.true_negative}",
                    "  thresholds:",
                    f"    - min_precision: {self.thresholds.min_precision:.2f}",
                    f"    - min_recall: {self.thresholds.min_recall:.2f}",
                    f"    - max_false_positive_rate: {self.thresholds.max_false_positive_rate:.2f}",
                ]
            )

        return "\n".join(lines)


def _map_source_rowids_to_normalized_ids(
    normalized_db: str | Path,
    source_rowids: Iterable[int],
) -> Set[int]:
    source_set = {int(v) for v in source_rowids}
    if not source_set:
        return set()

    conn = sqlite3.connect(str(normalized_db))
    try:
        placeholders = ",".join(["?"] * len(source_set))
        sql = f"SELECT id FROM normalized_trades WHERE source_rowid IN ({placeholders})"
        rows = conn.execute(sql, list(source_set)).fetchall()
    finally:
        conn.close()

    return {int(r[0]) for r in rows}


def _predicted_anchor_ids(detection_report: DetectionReport) -> Set[int]:
    out: Set[int] = set()
    for alert in detection_report.alerts:
        anchor = alert.evidence.get("anchor_trade_id")
        if isinstance(anchor, int):
            out.add(anchor)
    return out


def evaluate_trade_level_metrics(
    detection_report: DetectionReport,
    *,
    positive_trade_ids: Set[int],
    total_rows: int,
) -> ValidationMetrics:
    predicted = _predicted_anchor_ids(detection_report)

    tp = len(predicted & positive_trade_ids)
    fp = len(predicted - positive_trade_ids)
    fn = len(positive_trade_ids - predicted)

    negatives = max(0, total_rows - len(positive_trade_ids))
    tn = max(0, negatives - fp)

    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(positive_trade_ids) if positive_trade_ids else 0.0
    fpr = fp / negatives if negatives > 0 else 0.0

    return ValidationMetrics(
        total_rows=total_rows,
        positive_labels=len(positive_trade_ids),
        predicted_positive=len(predicted),
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
        true_negative=tn,
        precision=precision,
        recall=recall,
        false_positive_rate=fpr,
    )


def run_synthetic_validation(
    *,
    artifacts_dir: str | Path = "artifacts",
    run_name: str | None = None,
    overwrite: bool = False,
    normal_trades: int = 300,
    suspicious_pairs: int = 40,
    seed: int = 17,
    thresholds: ValidationThresholds | None = None,
) -> SyntheticValidationReport:
    if thresholds is None:
        thresholds = ValidationThresholds()

    if run_name is None:
        run_name = datetime.now(timezone.utc).strftime("validation_%Y%m%dT%H%M%SZ")

    run_dir = Path(artifacts_dir) / run_name
    if run_dir.exists():
        if overwrite:
            shutil.rmtree(run_dir)
        else:
            raise FileExistsError(
                f"run directory already exists: {run_dir} (use overwrite=True to replace)"
            )

    run_dir.mkdir(parents=True, exist_ok=True)

    source_db = run_dir / "synthetic_source.db"
    normalized_db = run_dir / "normalized.db"
    detection_json = run_dir / "detection_report.json"

    fixture = create_synthetic_source_db(
        source_db,
        normal_trades=normal_trades,
        suspicious_pairs=suspicious_pairs,
        seed=seed,
    )

    ingest_report = ingest_to_normalized_db(source_db, normalized_db, overwrite=True)

    detection_report: DetectionReport | None = None
    metrics: ValidationMetrics | None = None

    if ingest_report.schema_report.passed and ingest_report.output_rows > 0:
        detection_report = run_detection(
            normalized_db,
            output_json_path=detection_json,
        )

        positive_trade_ids = _map_source_rowids_to_normalized_ids(
            normalized_db,
            fixture.positive_source_rowids,
        )

        metrics = evaluate_trade_level_metrics(
            detection_report,
            positive_trade_ids=positive_trade_ids,
            total_rows=detection_report.analyzed_rows,
        )

    report = SyntheticValidationReport(
        run_dir=str(run_dir),
        fixture=fixture,
        ingest_report=ingest_report,
        detection_report=detection_report,
        metrics=metrics,
        thresholds=thresholds,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
    )

    (run_dir / "validation_summary.json").write_text(
        json.dumps(report.to_dict(), indent=2),
        encoding="utf-8",
    )
    (run_dir / "validation_summary.txt").write_text(
        report.to_text() + "\n",
        encoding="utf-8",
    )

    return report
