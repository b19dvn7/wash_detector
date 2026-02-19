from __future__ import annotations

import argparse
import logging
from typing import Sequence

from .auto_label import run_auto_feedback
from .smart_workflow import run_smart_workflow
from .calibrate import calibrate
from .detect import run_detection
from .feedback import interactive_feedback_session
from .ingest import ingest_to_normalized_db
from .learner import AdaptiveLearner, quick_calibrate
from .logging_config import configure_logging
from .pipeline import run_pipeline
from .real_batch import run_real_batch
from .schema import contract_to_text, validate_source_db


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wash-detector",
        description="Standalone wash-trading detector CLI",
    )

    # Global logging flags
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output (INFO level logging)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output (DEBUG level logging)",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional path to write logs to file",
    )

    sub = parser.add_subparsers(dest="command", required=False)

    sub.add_parser("status", help="Show scaffold status")
    sub.add_parser("plan", help="Show current build plan")
    sub.add_parser("schema-contract", help="Print canonical source DB contract")

    schema_check = sub.add_parser(
        "schema-check",
        help="Validate a source SQLite DB against the canonical contract",
    )
    schema_check.add_argument(
        "--db-path",
        required=True,
        help="Path to source sqlite DB",
    )

    ingest = sub.add_parser(
        "ingest",
        help="Build normalized detector-input DB from a validated source DB",
    )
    ingest.add_argument(
        "--source-db",
        required=True,
        help="Path to source sqlite DB",
    )
    ingest.add_argument(
        "--output-db",
        required=True,
        help="Path to normalized output sqlite DB",
    )
    ingest.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output DB if it already exists",
    )

    detect = sub.add_parser(
        "detect",
        help="Run detector rules against normalized_trades DB",
    )
    detect.add_argument(
        "--normalized-db",
        required=True,
        help="Path to normalized sqlite DB",
    )
    detect.add_argument(
        "--lookback-minutes",
        type=int,
        default=None,
        help="Optional lookback window from latest timestamp",
    )
    detect.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write full JSON report",
    )
    detect.add_argument(
        "--config",
        default=None,
        help="Path to JSON config file for detection thresholds",
    )

    run = sub.add_parser(
        "run",
        help="Run full pipeline (ingest + detect) and write run artifacts",
    )
    run.add_argument(
        "--source-db",
        required=True,
        help="Path to source sqlite DB",
    )
    run.add_argument(
        "--artifacts-dir",
        default="artifacts",
        help="Root directory for pipeline run artifacts",
    )
    run.add_argument(
        "--run-name",
        default=None,
        help="Optional run folder name under artifacts-dir",
    )
    run.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing run directory if it already exists",
    )
    run.add_argument(
        "--lookback-minutes",
        type=int,
        default=None,
        help="Optional lookback window for detection stage",
    )
    run.add_argument(
        "--config",
        default=None,
        help="Path to JSON config file for detection thresholds",
    )

    batch = sub.add_parser(
        "batch-run-real",
        help="Run schema+pipeline across real day DBs in a directory",
    )
    batch.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing btcusdt_YYYYMMDD.db files (required)",
    )
    batch.add_argument(
        "--artifacts-dir",
        default="artifacts",
        help="Root directory for batch artifacts",
    )
    batch.add_argument(
        "--run-name",
        default=None,
        help="Optional batch folder name under artifacts-dir",
    )
    batch.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing batch run directory if it already exists",
    )
    batch.add_argument(
        "--max-days",
        type=int,
        default=5,
        help="Max number of DB days to process (<=0 means all)",
    )
    batch.add_argument(
        "--oldest-first",
        action="store_true",
        help="Process oldest files first (default is newest first)",
    )
    batch.add_argument(
        "--lookback-minutes",
        type=int,
        default=None,
        help="Optional detection lookback window per day run",
    )
    batch.add_argument(
        "--keep-normalized",
        action="store_true",
        help="Keep normalized.db files for each processed day (default: drop to save disk)",
    )
    batch.add_argument(
        "--config",
        default=None,
        help="Path to JSON config file for detection thresholds",
    )

    # === Learning commands ===
    cal = sub.add_parser(
        "calibrate",
        help="Auto-calibrate detection thresholds from your data",
    )
    cal.add_argument(
        "--normalized-db",
        required=True,
        help="Path to normalized trades DB to learn from",
    )
    cal.add_argument(
        "--output-config",
        default=None,
        help="Path to write calibrated config.json",
    )
    cal.add_argument(
        "--output-report",
        default=None,
        help="Path to write calibration report JSON",
    )
    cal.add_argument(
        "--sensitivity",
        type=float,
        default=0.95,
        help="Detection sensitivity 0-1 (0.95 = flag top 5%%, 0.99 = flag top 1%%)",
    )

    fb = sub.add_parser(
        "feedback",
        help="Interactive feedback session to improve detection",
    )
    fb.add_argument(
        "--detection-json",
        required=True,
        help="Path to detection_report.json to review",
    )
    fb.add_argument(
        "--feedback-db",
        default="feedback.db",
        help="Path to feedback database (created if not exists)",
    )

    smart = sub.add_parser(
        "smart-learn",
        help="Best approach: auto-label obvious, review 50 ambiguous, learn your thresholds",
    )
    smart.add_argument(
        "--detection-json",
        required=True,
        help="Path to detection_report.json",
    )
    smart.add_argument(
        "--feedback-db",
        default="smart_feedback.db",
        help="Path to feedback database",
    )
    smart.add_argument(
        "--sample-size",
        type=int,
        default=50,
        help="Number of ambiguous cases to review (default: 50)",
    )

    autofb = sub.add_parser(
        "auto-feedback",
        help="Smart auto-labeling: auto-sorts obvious cases, only asks about borderline",
    )
    autofb.add_argument(
        "--detection-json",
        required=True,
        help="Path to detection_report.json",
    )
    autofb.add_argument(
        "--feedback-db",
        default="feedback.db",
        help="Path to feedback database",
    )
    autofb.add_argument(
        "--confidence",
        type=float,
        default=0.75,
        help="Confidence threshold for auto-labeling (0.70-0.95, default 0.75)",
    )
    autofb.add_argument(
        "--no-review",
        action="store_true",
        help="Skip human review of uncertain cases (just auto-label what's clear)",
    )

    learn = sub.add_parser(
        "learn",
        help="Manage adaptive learning state",
    )
    learn.add_argument(
        "--state-dir",
        default=".wash_learning",
        help="Directory to store learning state",
    )
    learn.add_argument(
        "--calibrate-from",
        default=None,
        help="Path to normalized DB to calibrate from",
    )
    learn.add_argument(
        "--export-config",
        default=None,
        help="Export current adaptive config to this path",
    )
    learn.add_argument(
        "--summary",
        action="store_true",
        help="Show learning summary",
    )
    learn.add_argument(
        "--sensitivity",
        type=float,
        default=0.95,
        help="Calibration sensitivity (default: 0.95)",
    )

    export = sub.add_parser(
        "export-labels",
        help="Export auto-labeled alerts to CSV (with proper timezone handling)",
    )
    export.add_argument(
        "--detection-json",
        required=True,
        help="Path to detection_report.json",
    )
    export.add_argument(
        "--output-csv",
        required=True,
        help="Path to write CSV file",
    )
    export.add_argument(
        "--day-name",
        default="",
        help="Optional day identifier (YYYYMMDD or description)",
    )
    export.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.75,
        help="Minimum confidence for auto-labeling (0.70-0.95)",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Configure logging based on verbosity flags
    if args.debug:
        log_level = logging.DEBUG
    elif args.verbose:
        log_level = logging.INFO
    else:
        log_level = logging.WARNING

    configure_logging(level=log_level, log_file=args.log_file)

    if args.command == "status":
        print("wash_detector scaffold: ready")
        print("next: schema -> ingest -> features -> detect -> validate")
        return 0

    if args.command == "plan":
        print("Phase 1: schema contracts")
        print("Phase 2: ingest + feature pipeline")
        print("Phase 3: detector engine + risk scoring")
        print("Phase 4: validation + reports")
        return 0

    if args.command == "schema-contract":
        print(contract_to_text())
        return 0

    if args.command == "schema-check":
        report = validate_source_db(args.db_path)
        print(report.to_text())
        return 0 if report.passed else 2

    if args.command == "ingest":
        report = ingest_to_normalized_db(
            source_db_path=args.source_db,
            output_db_path=args.output_db,
            overwrite=args.overwrite,
        )
        print(report.to_text())
        return 0 if report.passed else 3

    if args.command == "detect":
        report = run_detection(
            normalized_db_path=args.normalized_db,
            lookback_minutes=args.lookback_minutes,
            output_json_path=args.output_json,
            config_path=args.config,
        )
        print(report.to_text())
        # Exit code 5 if no rows analyzed (empty DB or bad window)
        return 0 if report.analyzed_rows > 0 else 5

    if args.command == "run":
        report = run_pipeline(
            source_db=args.source_db,
            artifacts_dir=args.artifacts_dir,
            run_name=args.run_name,
            overwrite=args.overwrite,
            lookback_minutes=args.lookback_minutes,
            config_path=args.config,
        )
        print(report.to_text())
        return 0 if report.passed else 4

    if args.command == "batch-run-real":
        report = run_real_batch(
            data_dir=args.data_dir,
            artifacts_dir=args.artifacts_dir,
            run_name=args.run_name,
            overwrite=args.overwrite,
            max_days=args.max_days,
            newest_first=not args.oldest_first,
            lookback_minutes=args.lookback_minutes,
            drop_normalized=not args.keep_normalized,
            config_path=args.config,
        )
        print(report.to_text())
        return 0 if report.fail_count == 0 else 6

    if args.command == "calibrate":
        report = calibrate(
            normalized_db_path=args.normalized_db,
            output_json_path=args.output_report,
            sensitivity=args.sensitivity,
        )
        print(report.to_text())

        if args.output_config:
            from .calibrate import generate_calibrated_config
            generate_calibrated_config(report, args.output_config)
            print(f"\nCalibrated config written to: {args.output_config}")

        return 0

    if args.command == "smart-learn":
        run_smart_workflow(
            detection_json_path=args.detection_json,
            feedback_db_path=args.feedback_db,
            sample_size=args.sample_size,
        )
        return 0

    if args.command == "feedback":
        count = interactive_feedback_session(
            detection_json_path=args.detection_json,
            feedback_db_path=args.feedback_db,
        )
        print(f"\nFeedback session complete. Recorded {count} entries.")
        return 0

    if args.command == "auto-feedback":
        results = run_auto_feedback(
            detection_json_path=args.detection_json,
            feedback_db_path=args.feedback_db,
            confidence_threshold=args.confidence,
            review_uncertain=not args.no_review,
        )
        return 0

    if args.command == "learn":
        learner = AdaptiveLearner(state_dir=args.state_dir)

        if args.calibrate_from:
            print(f"Calibrating from {args.calibrate_from}...")
            report = learner.calibrate_from_data(
                args.calibrate_from,
                sensitivity=args.sensitivity,
            )
            print(f"Calibration complete: {report.trade_count} trades analyzed")

        if args.export_config:
            learner.export_config(args.export_config)
            print(f"Adaptive config exported to: {args.export_config}")

        if args.summary or (not args.calibrate_from and not args.export_config):
            print(learner.get_learning_summary())

        return 0

    if args.command == "export-labels":
        from .auto_label import auto_label_all
        from .export_csv import export_alerts_csv

        tp, fp, review = auto_label_all(
            args.detection_json,
            confidence_threshold=args.confidence_threshold,
        )

        count = export_alerts_csv(
            tp + fp + review,
            args.output_csv,
            day_name=args.day_name,
        )

        print(f"\n✓ Exported {count} alerts to {args.output_csv}")
        print(f"  Timestamp columns: timestamp_iso_utc (ISO+TZ) and timestamp_ms (epoch)")
        print(f"  Auto-labeled: {len(tp)} TP + {len(fp)} FP = {len(tp)+len(fp)}")
        print(f"  Needs review: {len(review)}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
