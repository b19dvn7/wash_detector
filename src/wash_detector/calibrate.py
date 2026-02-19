"""Auto-calibration module for wash_detector.

Analyzes trade data to learn feature distributions and generate
data-driven thresholds instead of arbitrary defaults.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, stdev
from typing import Dict, List, Optional, Tuple

import math

logger = logging.getLogger(__name__)


@dataclass
class FeatureStats:
    """Statistics for a single feature."""
    name: str
    count: int
    min_val: float
    max_val: float
    mean_val: float
    median_val: float
    std_val: float
    percentiles: Dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class CalibrationReport:
    """Results of calibration analysis."""
    source_db: str
    trade_count: int
    generated_at_utc: str
    features: Dict[str, FeatureStats] = field(default_factory=dict)
    recommended_config: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "source_db": self.source_db,
            "trade_count": self.trade_count,
            "generated_at_utc": self.generated_at_utc,
            "features": {k: v.to_dict() for k, v in self.features.items()},
            "recommended_config": self.recommended_config,
        }

    def to_text(self) -> str:
        lines = [
            "Calibration Report",
            f"  source_db: {self.source_db}",
            f"  trade_count: {self.trade_count}",
            f"  generated_at_utc: {self.generated_at_utc}",
            "",
            "Feature Statistics:",
        ]
        for name, stats in self.features.items():
            lines.append(f"  {name}:")
            lines.append(f"    count: {stats.count}")
            lines.append(f"    range: [{stats.min_val:.4f}, {stats.max_val:.4f}]")
            lines.append(f"    mean: {stats.mean_val:.4f}, median: {stats.median_val:.4f}")
            lines.append(f"    std: {stats.std_val:.4f}")
            if stats.percentiles:
                pct_str = ", ".join(f"p{k}={v:.4f}" for k, v in sorted(stats.percentiles.items()))
                lines.append(f"    percentiles: {pct_str}")
        return "\n".join(lines)


def _percentile(sorted_values: List[float], p: int) -> float:
    """Calculate percentile from sorted list."""
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * p / 100
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_values) else f
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


def _compute_stats(values: List[float], name: str, percentiles: List[int]) -> FeatureStats:
    """Compute statistics for a list of values."""
    if not values:
        return FeatureStats(
            name=name, count=0, min_val=0, max_val=0,
            mean_val=0, median_val=0, std_val=0, percentiles={}
        )

    sorted_vals = sorted(values)
    std = stdev(values) if len(values) > 1 else 0.0

    pct_dict = {p: _percentile(sorted_vals, p) for p in percentiles}

    return FeatureStats(
        name=name,
        count=len(values),
        min_val=min(values),
        max_val=max(values),
        mean_val=mean(values),
        median_val=median(values),
        std_val=std,
        percentiles=pct_dict,
    )


def _load_trades(db_path: str) -> List[Tuple]:
    """Load trades from normalized database."""
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute("""
            SELECT id, timestamp_ms, side, price, amount, notional,
                   ob_spread, ob_mid_price, ob_imbalance,
                   candle_high, candle_low
            FROM normalized_trades
            ORDER BY timestamp_ms ASC
        """)
        trades = cursor.fetchall()
    finally:
        conn.close()
    return trades


def _compute_pairwise_features(trades: List[Tuple]) -> Dict[str, List[float]]:
    """Compute features between consecutive trade pairs."""
    time_gaps_ms: List[float] = []
    price_diffs_bps: List[float] = []
    amount_ratios: List[float] = []
    reversal_time_gaps: List[float] = []  # Time gaps for opposite-side pairs

    for i in range(1, len(trades)):
        prev = trades[i - 1]
        curr = trades[i]

        # Unpack: id, ts, side, price, amount, notional, spread, mid, imb, high, low
        _, prev_ts, prev_side, prev_price, prev_amount, _, _, _, _, _, _ = prev
        _, curr_ts, curr_side, curr_price, curr_amount, _, _, _, _, _, _ = curr

        # Time gap
        time_gap = curr_ts - prev_ts
        if time_gap >= 0:
            time_gaps_ms.append(time_gap)

        # Price difference in basis points
        if prev_price > 0:
            price_diff_bps = abs(curr_price - prev_price) / prev_price * 10000
            price_diffs_bps.append(price_diff_bps)

        # Amount ratio (smaller / larger)
        if prev_amount > 0 and curr_amount > 0:
            ratio = min(prev_amount, curr_amount) / max(prev_amount, curr_amount)
            amount_ratios.append(ratio)

        # Track reversal-specific gaps
        if prev_side != curr_side and time_gap >= 0:
            reversal_time_gaps.append(time_gap)

    return {
        "time_gap_ms": time_gaps_ms,
        "price_diff_bps": price_diffs_bps,
        "amount_ratio": amount_ratios,
        "reversal_time_gap_ms": reversal_time_gaps,
    }


def _compute_trade_features(trades: List[Tuple]) -> Dict[str, List[float]]:
    """Compute per-trade features."""
    notionals: List[float] = []
    spreads: List[float] = []
    imbalances: List[float] = []
    volatilities: List[float] = []  # candle range as % of mid

    for trade in trades:
        _, _, _, price, _, notional, spread, mid, imbalance, high, low = trade

        notionals.append(notional)

        if spread is not None and spread > 0:
            spreads.append(spread)

        if imbalance is not None:
            imbalances.append(abs(imbalance))

        if high is not None and low is not None and mid is not None and mid > 0:
            vol_bps = (high - low) / mid * 10000
            if vol_bps >= 0:
                volatilities.append(vol_bps)

    return {
        "notional": notionals,
        "spread": spreads,
        "imbalance": imbalances,
        "volatility_bps": volatilities,
    }


def _generate_recommended_config(features: Dict[str, FeatureStats], sensitivity: float = 0.95) -> Dict:
    """Generate recommended config based on learned distributions.

    Args:
        features: Computed feature statistics
        sensitivity: Percentile for thresholds (0.95 = flag top 5% most suspicious)
    """
    pct = int(sensitivity * 100)
    inv_pct = 100 - pct  # For "low is suspicious" features

    config = {
        "mirror_reversal": {},
        "layering_cluster": {},
        "balanced_churn": {},
        "calibration_meta": {
            "sensitivity": sensitivity,
            "percentile_used": pct,
        }
    }

    # Mirror reversal thresholds
    if "reversal_time_gap_ms" in features:
        stats = features["reversal_time_gap_ms"]
        if pct in stats.percentiles:
            # Use 95th percentile of reversal gaps as window
            config["mirror_reversal"]["window_ms"] = int(stats.percentiles[pct])

    if "price_diff_bps" in features:
        stats = features["price_diff_bps"]
        if inv_pct in stats.percentiles:
            # Flag when price diff is in bottom 5% (very close prices)
            config["mirror_reversal"]["price_diff_bps"] = max(1.0, stats.percentiles[inv_pct])

    if "amount_ratio" in features:
        stats = features["amount_ratio"]
        if pct in stats.percentiles:
            # Flag when amount ratio is in top 5% (very similar amounts)
            # Convert to gap ratio: gap = 1 - ratio
            config["mirror_reversal"]["amount_gap_ratio"] = max(0.01, 1.0 - stats.percentiles[pct])

    # Volatility for low-vol bonus
    if "volatility_bps" in features:
        stats = features["volatility_bps"]
        if inv_pct in stats.percentiles:
            config["mirror_reversal"]["low_volatility_bps"] = stats.percentiles[inv_pct]

    # Spread threshold
    if "spread" in features:
        stats = features["spread"]
        if inv_pct in stats.percentiles:
            config["mirror_reversal"]["spread_threshold_bps"] = stats.percentiles[inv_pct]

    # Layering cluster - price range threshold
    if "price_diff_bps" in features:
        stats = features["price_diff_bps"]
        if 25 in stats.percentiles:
            # Tight clustering = 25th percentile price movement
            config["layering_cluster"]["price_range_bps"] = max(1.0, stats.percentiles[25])

    # Imbalance threshold for layering
    if "imbalance" in features:
        stats = features["imbalance"]
        if pct in stats.percentiles:
            config["layering_cluster"]["imbalance_threshold"] = stats.percentiles[pct]

    # Balanced churn - notional multiplier
    if "notional" in features:
        stats = features["notional"]
        if stats.median_val > 0:
            config["balanced_churn"]["_median_notional"] = stats.median_val
            # Churn = 20x median notional in window
            config["balanced_churn"]["notional_multiplier"] = 20.0

    return config


def calibrate(
    normalized_db_path: str | Path,
    *,
    output_json_path: str | Path | None = None,
    sensitivity: float = 0.95,
) -> CalibrationReport:
    """Analyze trade data and generate calibrated configuration.

    Args:
        normalized_db_path: Path to normalized trades database
        output_json_path: Optional path to write calibration report
        sensitivity: Detection sensitivity (0.90 = flag top 10%, 0.99 = flag top 1%)

    Returns:
        CalibrationReport with statistics and recommended config
    """
    db_path = str(normalized_db_path)
    logger.info(f"Loading trades from {db_path}")

    trades = _load_trades(db_path)
    logger.info(f"Loaded {len(trades)} trades")

    if len(trades) < 100:
        logger.warning("Very small dataset - calibration may not be reliable")

    # Compute features
    pairwise = _compute_pairwise_features(trades)
    per_trade = _compute_trade_features(trades)

    # Percentiles to compute
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]

    # Compute statistics
    features: Dict[str, FeatureStats] = {}

    for name, values in pairwise.items():
        features[name] = _compute_stats(values, name, percentiles)
        logger.debug(f"Computed {name}: n={len(values)}, median={features[name].median_val:.2f}")

    for name, values in per_trade.items():
        features[name] = _compute_stats(values, name, percentiles)
        logger.debug(f"Computed {name}: n={len(values)}, median={features[name].median_val:.2f}")

    # Generate recommended config
    recommended = _generate_recommended_config(features, sensitivity)

    report = CalibrationReport(
        source_db=db_path,
        trade_count=len(trades),
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        features=features,
        recommended_config=recommended,
    )

    if output_json_path:
        Path(output_json_path).write_text(
            json.dumps(report.to_dict(), indent=2),
            encoding="utf-8"
        )
        logger.info(f"Wrote calibration report to {output_json_path}")

    return report


def generate_calibrated_config(
    calibration_report: CalibrationReport,
    output_path: str | Path,
) -> None:
    """Generate a config.json file from calibration results.

    This merges the recommended config with default values for
    fields that weren't calibrated.
    """
    from .config import DetectorConfig

    # Start with defaults
    config = DetectorConfig()
    base_dict = config.to_dict()

    # Merge in calibrated values
    rec = calibration_report.recommended_config

    for section in ["mirror_reversal", "layering_cluster", "balanced_churn"]:
        if section in rec:
            for key, value in rec[section].items():
                if not key.startswith("_") and key in base_dict.get(section, {}):
                    base_dict[section][key] = value

    # Write merged config
    Path(output_path).write_text(
        json.dumps(base_dict, indent=2),
        encoding="utf-8"
    )
    logger.info(f"Wrote calibrated config to {output_path}")
