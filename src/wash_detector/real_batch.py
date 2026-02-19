from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
import csv
import json
import re
import shutil

from tqdm import tqdm

from .pipeline import run_pipeline
from .schema import validate_source_db

logger = logging.getLogger(__name__)


_DAY_RE = re.compile(r"btcusdt_(\d{8})\.db$", re.IGNORECASE)


@dataclass
class DayRunResult:
    db_file: str
    day: Optional[str]
    status: str
    schema_passed: bool
    reason: str
    run_dir: Optional[str] = None
    input_rows: int = 0
    output_rows: int = 0
    skipped_rows: int = 0
    alert_count: int = 0
    risk_score: int = 0
    risk_level: str = "UNKNOWN"
    normalized_kept: bool = True


@dataclass
class BatchRunReport:
    data_dir: str
    run_dir: str
    generated_at_utc: str
    total_files: int
    processed_files: int
    pass_count: int
    fail_count: int
    results: List[DayRunResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "data_dir": self.data_dir,
            "run_dir": self.run_dir,
            "generated_at_utc": self.generated_at_utc,
            "total_files": self.total_files,
            "processed_files": self.processed_files,
            "pass_count": self.pass_count,
            "fail_count": self.fail_count,
            "results": [asdict(r) for r in self.results],
        }

    def to_text(self) -> str:
        lines: List[str] = []
        lines.append("Real batch run report")
        lines.append(f"  data_dir: {self.data_dir}")
        lines.append(f"  run_dir: {self.run_dir}")
        lines.append(f"  generated_at_utc: {self.generated_at_utc}")
        lines.append(f"  total_files: {self.total_files}")
        lines.append(f"  processed_files: {self.processed_files}")
        lines.append(f"  pass_count: {self.pass_count}")
        lines.append(f"  fail_count: {self.fail_count}")

        if self.results:
            lines.append("  day_results:")
            for r in self.results:
                norm_note = "norm:kept" if r.normalized_kept else "norm:dropped"
                lines.append(
                    f"    - {r.db_file}: {r.status} | rows {r.input_rows}->{r.output_rows} | "
                    f"alerts {r.alert_count} | risk {r.risk_score} {r.risk_level} | {norm_note}"
                )
                if r.status != "PASS" and r.reason:
                    lines.append(f"      reason: {r.reason}")

        return "\n".join(lines)


def _extract_day(db_name: str) -> Optional[str]:
    m = _DAY_RE.match(db_name)
    if not m:
        return None
    return m.group(1)


def _write_csv(path: Path, rows: List[DayRunResult]) -> None:
    fields = [
        "db_file",
        "day",
        "status",
        "schema_passed",
        "reason",
        "run_dir",
        "input_rows",
        "output_rows",
        "skipped_rows",
        "alert_count",
        "risk_score",
        "risk_level",
        "normalized_kept",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def run_real_batch(
    *,
    data_dir: str | Path,
    artifacts_dir: str | Path = "artifacts",
    run_name: str | None = None,
    overwrite: bool = False,
    max_days: int = 5,
    newest_first: bool = True,
    lookback_minutes: int | None = None,
    drop_normalized: bool = True,
    config_path: str | Path | None = None,
) -> BatchRunReport:
    data_path = Path(data_dir)
    if not data_path.exists() or not data_path.is_dir():
        raise FileNotFoundError(f"data-dir not found or not a directory: {data_path}")

    all_db_files = sorted(data_path.glob("btcusdt_*.db"))
    total_files_in_dir = len(all_db_files)

    db_files = list(reversed(all_db_files)) if newest_first else list(all_db_files)

    if max_days > 0:
        db_files = db_files[:max_days]

    if run_name is None:
        run_name = datetime.now(timezone.utc).strftime("batch_real_%Y%m%dT%H%M%SZ")

    run_dir = Path(artifacts_dir) / run_name
    if run_dir.exists():
        if overwrite:
            shutil.rmtree(run_dir)
        else:
            raise FileExistsError(
                f"run directory already exists: {run_dir} (use overwrite=True to replace)"
            )

    run_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = run_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Batch run directory: {run_dir}")
    logger.info(f"Processing {len(db_files)} DB files (of {total_files_in_dir} total in directory)")

    results: List[DayRunResult] = []

    progress_bar = tqdm(
        db_files,
        desc="Processing DBs",
        unit="day",
        colour="green",
        dynamic_ncols=True,
    )

    for db_path in progress_bar:
        db_file = db_path.name
        day = _extract_day(db_file)
        progress_bar.set_postfix_str(db_file)
        logger.debug(f"Processing {db_file}")

        schema_report = validate_source_db(db_path)
        if not schema_report.passed:
            reason = "; ".join(schema_report.errors) if schema_report.errors else "schema validation failed"
            logger.warning(f"  Schema failed for {db_file}: {reason}")
            results.append(
                DayRunResult(
                    db_file=db_file,
                    day=day,
                    status="FAIL",
                    schema_passed=False,
                    reason=reason,
                )
            )
            continue

        try:
            day_run_name = db_path.stem
            pipeline_report = run_pipeline(
                source_db=db_path,
                artifacts_dir=runs_dir,
                run_name=day_run_name,
                overwrite=True,
                lookback_minutes=lookback_minutes,
                config_path=config_path,
            )

            detection = pipeline_report.detection_report
            normalized_kept = True
            if drop_normalized:
                normalized_path = Path(pipeline_report.normalized_db)
                if normalized_path.exists():
                    normalized_path.unlink()
                    normalized_kept = False

            results.append(
                DayRunResult(
                    db_file=db_file,
                    day=day,
                    status="PASS" if pipeline_report.passed else "FAIL",
                    schema_passed=True,
                    reason="" if pipeline_report.passed else "pipeline failed",
                    run_dir=pipeline_report.run_dir,
                    input_rows=pipeline_report.ingest_report.input_rows,
                    output_rows=pipeline_report.ingest_report.output_rows,
                    skipped_rows=pipeline_report.ingest_report.skipped_rows,
                    alert_count=0 if detection is None else len(detection.alerts),
                    risk_score=0 if detection is None else detection.risk_score,
                    risk_level="UNKNOWN" if detection is None else detection.risk_level,
                    normalized_kept=normalized_kept,
                )
            )

        except Exception as exc:  # noqa: BLE001
            results.append(
                DayRunResult(
                    db_file=db_file,
                    day=day,
                    status="FAIL",
                    schema_passed=True,
                    reason=f"exception during pipeline: {exc}",
                )
            )

    pass_count = sum(1 for r in results if r.status == "PASS")
    fail_count = sum(1 for r in results if r.status != "PASS")

    report = BatchRunReport(
        data_dir=str(data_path),
        run_dir=str(run_dir),
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        total_files=total_files_in_dir,
        processed_files=len(results),
        pass_count=pass_count,
        fail_count=fail_count,
        results=results,
    )

    (run_dir / "batch_summary.json").write_text(
        json.dumps(report.to_dict(), indent=2),
        encoding="utf-8",
    )
    (run_dir / "batch_summary.txt").write_text(report.to_text() + "\n", encoding="utf-8")
    _write_csv(run_dir / "batch_summary.csv", results)

    return report
