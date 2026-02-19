"""
Export labeled alerts to CSV with proper timezone handling and deduplication support.

Timestamp columns:
  - timestamp_iso_utc: ISO 8601 with explicit +00:00 timezone (human-readable)
  - timestamp_ms: Epoch milliseconds since 1970 (machine-readable, timezone-free)

Pair ID column:
  - pair_id: Stable unique key for deduplication (buy_trade_id_sell_trade_id or anchor_id)

Usage:
  from wash_detector.auto_label import auto_label_all
  from wash_detector.export_csv import export_alerts_csv

  tp, fp, review = auto_label_all("detection_report.json")
  count = export_alerts_csv(tp + fp + review, "output.csv", day_name="20260101")
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any


@dataclass
class ExportConfig:
    """CSV export configuration."""
    include_reason: bool = True  # Include 'reason' column
    include_evidence: bool = True  # Include detector-specific evidence columns
    confidence_decimals: int = 2


def export_alerts_csv(
    alerts: List[Dict[str, Any]],
    output_path: str | Path,
    day_name: str = "",
    config: ExportConfig = None,
) -> int:
    """Export alerts to CSV with proper timestamp handling, pair IDs, and detector-specific metrics.

    Args:
        alerts: List of alert dicts (from auto_label_all)
        output_path: Path to write CSV
        day_name: Optional day identifier (YYYYMMDD or description)
        config: Export configuration (uses defaults if None)

    Returns:
        Number of rows written
    """
    if config is None:
        config = ExportConfig()

    rows = []
    for alert in alerts:
        ev = alert.get("evidence", {})
        detector = alert.get("detector", "unknown")

        # Create stable pair ID for deduplication
        pair_id = None
        if "trade_id_prev" in ev and "trade_id_curr" in ev:
            pair_id = f"{int(ev['trade_id_prev'])}_{int(ev['trade_id_curr'])}"
        elif "anchor_trade_id" in ev:
            pair_id = str(int(ev["anchor_trade_id"]))

        row = {
            "timestamp_iso_utc": alert.get("timestamp_iso"),  # Full ISO with +00:00
            "timestamp_ms": alert.get("timestamp_ms"),  # Epoch ms
            "day": day_name,
            "detector": detector,
            "pair_id": pair_id,  # For deduplication
            "auto_label": alert.get("_auto_label"),
            "confidence": round(alert.get("_auto_confidence", 0), config.confidence_decimals),
        }

        if config.include_reason:
            row["reason"] = alert.get("_auto_reason", "")

        if config.include_evidence:
            # Mirror reversal evidence
            if detector == "mirror_reversal":
                # BUG FIX: Check presence, not truthiness (0 is falsy but valid amount)
                amount_gap_ratio = ev.get("amount_gap_ratio")
                amount_gap_pct = round(amount_gap_ratio * 100, 4) \
                    if amount_gap_ratio is not None else None

                row.update({
                    "delta_ms": ev.get("delta_ms"),
                    "amount_gap_pct": amount_gap_pct,  # Now includes 0%
                    "price_diff_bps": ev.get("price_diff_bps"),
                    "trades_apart": ev.get("trades_apart"),
                })
            # Balanced churn evidence
            elif detector == "balanced_churn":
                row.update({
                    "delta_ms": None,  # N/A for balanced churn
                    "window_seconds": ev.get("window_seconds"),
                    "balance_ratio": ev.get("side_balance_ratio"),
                    "notional_usd": ev.get("total_notional"),
                    "price_move_bps": ev.get("price_move_bps"),
                })
            # Layering cluster evidence
            elif detector == "layering_cluster":
                row.update({
                    "delta_ms": None,  # N/A for layering
                    "window_seconds": ev.get("window_seconds"),
                    "price_range_bps": ev.get("price_range_bps"),
                    "trade_count": ev.get("trade_count"),
                    "imbalance_ratio": ev.get("imbalance_ratio"),
                })

        rows.append(row)

    # Determine fieldnames based on detectors present
    all_detectors = set(r["detector"] for r in rows)
    fieldnames = [
        "timestamp_iso_utc", "timestamp_ms", "day", "detector", "pair_id",
        "auto_label", "confidence"
    ]
    if config.include_reason:
        fieldnames.append("reason")

    if config.include_evidence:
        # Always include delta_ms and amount_gap_pct as common columns
        fieldnames.extend(["delta_ms", "amount_gap_pct"])
        # Add detector-specific columns if present
        if "mirror_reversal" in all_detectors:
            fieldnames.extend(["price_diff_bps", "trades_apart"])
        if "balanced_churn" in all_detectors:
            fieldnames.extend(["window_seconds", "balance_ratio", "notional_usd", "price_move_bps"])
        if "layering_cluster" in all_detectors:
            fieldnames.extend(["price_range_bps", "trade_count", "imbalance_ratio"])

    # Write CSV
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval=None)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)
