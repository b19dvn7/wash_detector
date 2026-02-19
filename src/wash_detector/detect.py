"""Wash trading pattern detection engine.

Detects suspicious patterns:
- Mirror reversals: opposite-side trades at similar price/amount
- Layering clusters: dense small trades in tight price range
- Balanced churn: large balanced notional with minimal price movement
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional, Sequence, Set
import json
import logging
import math
import sqlite3

from .config import DetectorConfig, get_default_config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TradeRow:
    id: int
    timestamp_ms: int
    timestamp_iso: str
    side: str
    price: float
    amount: float
    notional: float
    candle_open: Optional[float]
    candle_high: Optional[float]
    candle_low: Optional[float]
    candle_close: Optional[float]
    candle_volume: Optional[float]
    candle_vwap: Optional[float]
    ob_spread: Optional[float]
    ob_mid_price: Optional[float]
    ob_imbalance: Optional[float]
    micro_orderbook_pressure: Optional[float]
    micro_volume_imbalance: Optional[float]
    micro_price_momentum: Optional[float]
    micro_liquidity_score: Optional[float]
    micro_trade_intensity: Optional[float]
    micro_spread_compression: Optional[float]
    micro_depth_ratio_5: Optional[float]
    micro_depth_ratio_20: Optional[float]
    micro_price_acceleration: Optional[float]
    micro_realized_volatility: Optional[float]
    micro_large_trade_ratio: Optional[float]
    micro_effective_spread: Optional[float]
    micro_liquidity_depth: Optional[float]
    micro_depth_weighted: Optional[float]
    micro_liquidity_quality: Optional[float]


@dataclass
class Alert:
    detector: str
    timestamp_ms: int
    timestamp_iso: str
    risk_points: int
    reason: str
    evidence: Dict[str, object]


@dataclass
class DetectionReport:
    normalized_db: str
    analyzed_rows: int
    config_used: Optional[Dict[str, object]] = None
    alerts: List[Alert] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    generated_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    _config: Optional[DetectorConfig] = field(default=None, repr=False)

    @property
    def detector_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for alert in self.alerts:
            counts[alert.detector] = counts.get(alert.detector, 0) + 1
        return counts

    @property
    def raw_risk_points(self) -> int:
        return sum(alert.risk_points for alert in self.alerts)

    @property
    def risk_score(self) -> int:
        if self.analyzed_rows <= 0:
            return 0

        cfg = self._config.risk_scoring if self._config else get_default_config().risk_scoring
        counts = self.detector_counts
        analyzed = float(self.analyzed_rows)

        mirror_rate = counts.get("mirror_reversal", 0) / analyzed * 10_000.0
        layering_rate = counts.get("layering_cluster", 0) / analyzed * 10_000.0
        churn_rate = counts.get("balanced_churn", 0) / analyzed * 10_000.0

        score = 0.0
        score += min(cfg.mirror_rate_cap, mirror_rate * cfg.mirror_rate_weight)
        score += min(cfg.layering_rate_cap, layering_rate * cfg.layering_rate_weight)
        score += min(cfg.churn_rate_cap, churn_rate * cfg.churn_rate_weight)

        # Absolute activity escalation (prevents under-scoring very large datasets)
        raw = self.raw_risk_points
        if raw >= cfg.escalation_tier3_points:
            score += cfg.escalation_tier3_bonus
        elif raw >= cfg.escalation_tier2_points:
            score += cfg.escalation_tier2_bonus
        elif raw >= cfg.escalation_tier1_points:
            score += cfg.escalation_tier1_bonus

        return min(100, int(round(score)))

    @property
    def risk_level(self) -> str:
        score = self.risk_score
        if score >= 75:
            return "CRITICAL"
        if score >= 50:
            return "HIGH"
        if score >= 20:
            return "MODERATE"
        return "LOW"

    def to_dict(self) -> Dict[str, object]:
        return {
            "normalized_db": self.normalized_db,
            "generated_at_utc": self.generated_at_utc,
            "analyzed_rows": self.analyzed_rows,
            "alert_count": len(self.alerts),
            "detector_counts": self.detector_counts,
            "raw_risk_points": self.raw_risk_points,
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "warnings": self.warnings,
            "config_used": self.config_used,
            "alerts": [asdict(alert) for alert in self.alerts],
        }

    def to_text(self) -> str:
        lines: List[str] = []
        lines.append("Detection report")
        lines.append(f"  normalized_db: {self.normalized_db}")
        lines.append(f"  generated_at_utc: {self.generated_at_utc}")
        lines.append(f"  analyzed_rows: {self.analyzed_rows}")
        lines.append(f"  alert_count: {len(self.alerts)}")
        lines.append(f"  raw_risk_points: {self.raw_risk_points}")
        lines.append(f"  risk_score: {self.risk_score}")
        lines.append(f"  risk_level: {self.risk_level}")

        by_detector = self.detector_counts
        if by_detector:
            lines.append("  detectors:")
            for detector in sorted(by_detector):
                lines.append(f"    - {detector}: {by_detector[detector]}")

        if self.warnings:
            lines.append("Warnings:")
            for warn in self.warnings:
                lines.append(f"  - {warn}")

        return "\n".join(lines)


def _bps_diff(a: float, b: float) -> float:
    """Calculate basis points difference between two values."""
    ref = (abs(a) + abs(b)) / 2.0
    if ref <= 0:
        return 0.0
    return abs(a - b) / ref * 10_000.0


def _load_trades(
    normalized_db_path: str | Path,
    lookback_minutes: Optional[int] = None,
) -> List[TradeRow]:
    """Load normalized trades from database."""
    db_path = str(normalized_db_path)
    conn = sqlite3.connect(db_path)

    try:
        table_exists = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='normalized_trades'"
        ).fetchone()[0]
        if table_exists == 0:
            raise ValueError("normalized_trades table not found")

        where_clause = ""
        params: List[object] = []

        if lookback_minutes is not None:
            if lookback_minutes <= 0:
                raise ValueError("lookback_minutes must be > 0")

            max_ts = conn.execute("SELECT MAX(timestamp_ms) FROM normalized_trades").fetchone()[0]
            if max_ts is not None:
                cutoff = int(max_ts) - int(lookback_minutes * 60_000)
                where_clause = "WHERE timestamp_ms >= ?"
                params.append(cutoff)

        sql = f"""
            SELECT
                id,
                timestamp_ms,
                timestamp_iso,
                side,
                price,
                amount,
                notional,
                candle_open,
                candle_high,
                candle_low,
                candle_close,
                candle_volume,
                candle_vwap,
                ob_spread,
                ob_mid_price,
                ob_imbalance,
                micro_orderbook_pressure,
                micro_volume_imbalance,
                micro_price_momentum,
                micro_liquidity_score,
                micro_trade_intensity,
                micro_spread_compression,
                micro_depth_ratio_5,
                micro_depth_ratio_20,
                micro_price_acceleration,
                micro_realized_volatility,
                micro_large_trade_ratio,
                micro_effective_spread,
                micro_liquidity_depth,
                micro_depth_weighted,
                micro_liquidity_quality
            FROM normalized_trades
            {where_clause}
            ORDER BY timestamp_ms ASC, id ASC
        """

        rows = conn.execute(sql, params).fetchall()

    finally:
        conn.close()

    result: List[TradeRow] = []
    for row in rows:
        result.append(
            TradeRow(
                id=int(row[0]),
                timestamp_ms=int(row[1]),
                timestamp_iso=str(row[2]),
                side=str(row[3]),
                price=float(row[4]),
                amount=float(row[5]),
                notional=float(row[6]),
                candle_open=_opt_float(row[7]),
                candle_high=_opt_float(row[8]),
                candle_low=_opt_float(row[9]),
                candle_close=_opt_float(row[10]),
                candle_volume=_opt_float(row[11]),
                candle_vwap=_opt_float(row[12]),
                ob_spread=_opt_float(row[13]),
                ob_mid_price=_opt_float(row[14]),
                ob_imbalance=_opt_float(row[15]),
                micro_orderbook_pressure=_opt_float(row[16]),
                micro_volume_imbalance=_opt_float(row[17]),
                micro_price_momentum=_opt_float(row[18]),
                micro_liquidity_score=_opt_float(row[19]),
                micro_trade_intensity=_opt_float(row[20]),
                micro_spread_compression=_opt_float(row[21]),
                micro_depth_ratio_5=_opt_float(row[22]),
                micro_depth_ratio_20=_opt_float(row[23]),
                micro_price_acceleration=_opt_float(row[24]),
                micro_realized_volatility=_opt_float(row[25]),
                micro_large_trade_ratio=_opt_float(row[26]),
                micro_effective_spread=_opt_float(row[27]),
                micro_liquidity_depth=_opt_float(row[28]),
                micro_depth_weighted=_opt_float(row[29]),
                micro_liquidity_quality=_opt_float(row[30]),
            )
        )

    return result


def _opt_float(value: object) -> Optional[float]:
    """Convert to float, returning None for invalid values."""
    if value is None:
        return None
    out = float(value)
    if not math.isfinite(out):
        return None
    return out


def _mirror_reversal_alerts(
    trades: Sequence[TradeRow],
    cfg: DetectorConfig,
) -> List[Alert]:
    """
    Detect mirror reversal patterns.

    Improved: checks ±N trades window instead of just adjacent,
    uses orderbook spread for additional risk scoring.
    """
    alerts: List[Alert] = []
    mcfg = cfg.mirror_reversal
    seen_pairs: Set[tuple] = set()  # Avoid duplicate alerts for same pair

    for i in range(1, len(trades)):
        curr_t = trades[i]

        # Look back up to lookback_trades previous trades
        lookback_start = max(0, i - mcfg.lookback_trades)

        for j in range(lookback_start, i):
            prev_t = trades[j]

            # Must be opposite sides
            if prev_t.side == curr_t.side:
                continue

            # Check time window
            dt_ms = curr_t.timestamp_ms - prev_t.timestamp_ms
            if dt_ms < 0 or dt_ms > mcfg.window_ms:
                continue

            # Check amount similarity
            max_amount = max(prev_t.amount, curr_t.amount)
            if max_amount <= 0:
                continue
            amount_gap_ratio = abs(prev_t.amount - curr_t.amount) / max_amount
            if amount_gap_ratio > mcfg.amount_gap_ratio:
                continue

            # Check price similarity
            price_diff_bps = _bps_diff(prev_t.price, curr_t.price)
            if price_diff_bps > mcfg.price_diff_bps:
                continue

            # Avoid duplicate alerts for same pair
            pair_key = (min(prev_t.id, curr_t.id), max(prev_t.id, curr_t.id))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            # Calculate risk points
            risk = mcfg.base_risk_points

            # Volatility bonus (low volatility = more suspicious)
            volatility_bps = None
            if curr_t.candle_high is not None and curr_t.candle_low is not None and curr_t.candle_low > 0:
                volatility_bps = _bps_diff(curr_t.candle_high, curr_t.candle_low)
                if volatility_bps < mcfg.low_volatility_bps:
                    risk += mcfg.low_vol_bonus

            # Spread bonus (tight spread = more suspicious - market maker behavior)
            spread_bps = None
            if curr_t.ob_spread is not None and curr_t.ob_mid_price is not None and curr_t.ob_mid_price > 0:
                spread_bps = (curr_t.ob_spread / curr_t.ob_mid_price) * 10_000.0
                if spread_bps < mcfg.spread_threshold_bps:
                    risk += mcfg.tight_spread_bonus

            # Microstructure signals for enhanced context
            liquidity_score = curr_t.micro_liquidity_score
            orderbook_pressure = curr_t.micro_orderbook_pressure
            spread_compression = curr_t.micro_spread_compression
            effective_spread = curr_t.micro_effective_spread

            alerts.append(
                Alert(
                    detector="mirror_reversal",
                    timestamp_ms=curr_t.timestamp_ms,
                    timestamp_iso=curr_t.timestamp_iso,
                    risk_points=risk,
                    reason="Opposite-side near-equal trade pair in short interval at near-equal price.",
                    evidence={
                        "anchor_trade_id": curr_t.id,
                        "trade_id_prev": prev_t.id,
                        "trade_id_curr": curr_t.id,
                        "trades_apart": i - j,
                        "delta_ms": dt_ms,
                        "amount_gap_ratio": round(amount_gap_ratio, 6),
                        "price_diff_bps": round(price_diff_bps, 6),
                        "volatility_bps": None if volatility_bps is None else round(volatility_bps, 6),
                        "spread_bps": None if spread_bps is None else round(spread_bps, 6),
                        "liquidity_score": None if liquidity_score is None else round(liquidity_score, 6),
                        "orderbook_pressure": None if orderbook_pressure is None else round(orderbook_pressure, 6),
                        "spread_compression": None if spread_compression is None else round(spread_compression, 6),
                        "effective_spread": None if effective_spread is None else round(effective_spread, 6),
                    },
                )
            )

    return alerts


def _layering_cluster_alerts(
    trades: Sequence[TradeRow],
    cfg: DetectorConfig,
) -> List[Alert]:
    """
    Detect layering cluster patterns.

    Uses orderbook imbalance for additional risk scoring.
    """
    alerts: List[Alert] = []
    lcfg = cfg.layering_cluster

    if len(trades) < lcfg.min_trades:
        return alerts

    amounts = [t.amount for t in trades if t.amount > 0]
    if not amounts:
        return alerts
    global_median_amount = median(amounts)

    left = 0
    cooldown_until = -1

    # Incremental window tracking for O(n) performance
    sum_amount = 0.0
    window_size = 0

    for right, t in enumerate(trades):
        # Add current trade to window
        sum_amount += t.amount
        window_size += 1

        # Slide window left edge forward
        while left < right and trades[right].timestamp_ms - trades[left].timestamp_ms > lcfg.window_ms:
            removed = trades[left]
            sum_amount -= removed.amount
            window_size -= 1
            left += 1

        if window_size < lcfg.min_trades:
            continue

        if t.timestamp_ms < cooldown_until:
            continue

        window = trades[left : right + 1]
        prices = [x.price for x in window]
        price_range_bps = _bps_diff(max(prices), min(prices))
        avg_amount = sum_amount / window_size

        if price_range_bps <= lcfg.price_range_bps and avg_amount <= global_median_amount * lcfg.amount_ratio:
            risk = lcfg.base_risk_points

            # Imbalance bonus (high imbalance = one-sided pressure = more suspicious)
            avg_imbalance = None
            imbalances = [x.ob_imbalance for x in window if x.ob_imbalance is not None]
            if imbalances:
                avg_imbalance = sum(abs(x) for x in imbalances) / len(imbalances)
                if avg_imbalance > lcfg.imbalance_threshold:
                    risk += lcfg.high_imbalance_bonus

            # Microstructure signals for enhanced context
            depth_ratio_5_vals = [x.micro_depth_ratio_5 for x in window if x.micro_depth_ratio_5 is not None]
            depth_ratio_20_vals = [x.micro_depth_ratio_20 for x in window if x.micro_depth_ratio_20 is not None]
            trade_intensity_vals = [x.micro_trade_intensity for x in window if x.micro_trade_intensity is not None]
            liquidity_depth_vals = [x.micro_liquidity_depth for x in window if x.micro_liquidity_depth is not None]

            avg_depth_ratio_5 = sum(depth_ratio_5_vals) / len(depth_ratio_5_vals) if depth_ratio_5_vals else None
            avg_depth_ratio_20 = sum(depth_ratio_20_vals) / len(depth_ratio_20_vals) if depth_ratio_20_vals else None
            avg_trade_intensity = sum(trade_intensity_vals) / len(trade_intensity_vals) if trade_intensity_vals else None
            avg_liquidity_depth = sum(liquidity_depth_vals) / len(liquidity_depth_vals) if liquidity_depth_vals else None

            alerts.append(
                Alert(
                    detector="layering_cluster",
                    timestamp_ms=t.timestamp_ms,
                    timestamp_iso=t.timestamp_iso,
                    risk_points=risk,
                    reason="Dense short-window trade cluster with low price drift and small average size.",
                    evidence={
                        "anchor_trade_id": t.id,
                        "window_trade_count": len(window),
                        "window_seconds": (window[-1].timestamp_ms - window[0].timestamp_ms) / 1000.0,
                        "price_range_bps": round(price_range_bps, 6),
                        "avg_amount": round(avg_amount, 8),
                        "global_median_amount": round(global_median_amount, 8),
                        "avg_imbalance": None if avg_imbalance is None else round(avg_imbalance, 6),
                        "avg_depth_ratio_5": None if avg_depth_ratio_5 is None else round(avg_depth_ratio_5, 6),
                        "avg_depth_ratio_20": None if avg_depth_ratio_20 is None else round(avg_depth_ratio_20, 6),
                        "avg_trade_intensity": None if avg_trade_intensity is None else round(avg_trade_intensity, 6),
                        "avg_liquidity_depth": None if avg_liquidity_depth is None else round(avg_liquidity_depth, 6),
                    },
                )
            )
            cooldown_until = t.timestamp_ms + lcfg.cooldown_ms

    return alerts


def _balanced_churn_alerts(
    trades: Sequence[TradeRow],
    cfg: DetectorConfig,
) -> List[Alert]:
    """Detect balanced churn patterns."""
    alerts: List[Alert] = []
    bcfg = cfg.balanced_churn

    if len(trades) < bcfg.min_trades:
        return alerts

    notionals = [t.notional for t in trades if t.notional > 0]
    if not notionals:
        return alerts

    baseline_notional = median(notionals)
    min_window_notional = baseline_notional * bcfg.notional_multiplier

    left = 0
    cooldown_until = -1

    # Incremental window tracking for O(n) performance
    buy_notional = 0.0
    sell_notional = 0.0

    for right, t in enumerate(trades):
        # Add current trade to window
        if t.side == "BUY":
            buy_notional += t.notional
        else:
            sell_notional += t.notional

        # Slide window left edge forward
        while left < right and trades[right].timestamp_ms - trades[left].timestamp_ms > bcfg.window_ms:
            removed = trades[left]
            if removed.side == "BUY":
                buy_notional -= removed.notional
            else:
                sell_notional -= removed.notional
            left += 1

        window = trades[left : right + 1]
        if len(window) < bcfg.min_trades:
            continue

        if t.timestamp_ms < cooldown_until:
            continue

        total_notional = buy_notional + sell_notional
        if total_notional <= 0:
            continue

        side_balance_ratio = abs(buy_notional - sell_notional) / total_notional
        price_move_bps = _bps_diff(window[0].price, window[-1].price)

        if (
            side_balance_ratio <= bcfg.balance_ratio
            and price_move_bps <= bcfg.price_move_bps
            and total_notional >= min_window_notional
            and total_notional >= bcfg.min_notional_usd
        ):
            # Microstructure signals for enhanced context
            volume_imbalance_vals = [x.micro_volume_imbalance for x in window if x.micro_volume_imbalance is not None]
            large_trade_ratio_vals = [x.micro_large_trade_ratio for x in window if x.micro_large_trade_ratio is not None]
            realized_volatility_vals = [x.micro_realized_volatility for x in window if x.micro_realized_volatility is not None]

            avg_volume_imbalance = sum(abs(x) for x in volume_imbalance_vals) / len(volume_imbalance_vals) if volume_imbalance_vals else None
            avg_large_trade_ratio = sum(large_trade_ratio_vals) / len(large_trade_ratio_vals) if large_trade_ratio_vals else None
            avg_realized_volatility = sum(realized_volatility_vals) / len(realized_volatility_vals) if realized_volatility_vals else None

            alerts.append(
                Alert(
                    detector="balanced_churn",
                    timestamp_ms=t.timestamp_ms,
                    timestamp_iso=t.timestamp_iso,
                    risk_points=bcfg.base_risk_points,
                    reason="Large balanced two-sided notional with minimal net price movement.",
                    evidence={
                        "anchor_trade_id": t.id,
                        "window_trade_count": len(window),
                        "window_seconds": (window[-1].timestamp_ms - window[0].timestamp_ms) / 1000.0,
                        "buy_notional": round(buy_notional, 6),
                        "sell_notional": round(sell_notional, 6),
                        "side_balance_ratio": round(side_balance_ratio, 6),
                        "price_move_bps": round(price_move_bps, 6),
                        "total_notional": round(total_notional, 6),
                        "min_window_notional": round(min_window_notional, 6),
                        "avg_volume_imbalance": None if avg_volume_imbalance is None else round(avg_volume_imbalance, 6),
                        "avg_large_trade_ratio": None if avg_large_trade_ratio is None else round(avg_large_trade_ratio, 6),
                        "avg_realized_volatility": None if avg_realized_volatility is None else round(avg_realized_volatility, 6),
                    },
                )
            )
            cooldown_until = t.timestamp_ms + bcfg.cooldown_ms

    return alerts


def _compute_near_touch_depth(
    bids: List[List[float]],
    asks: List[List[float]],
    bps_window: float = 5.0,
) -> tuple[float, float, float]:
    """
    Compute depth within X bps of best bid/ask.

    Args:
        bids: [[price, size], ...] sorted descending
        asks: [[price, size], ...] sorted ascending
        bps_window: basis points window (default 5 bps = 0.05%)

    Returns:
        (bid_depth_usd, ask_depth_usd, near_touch_imbalance)
    """
    if not bids or not asks:
        return 0.0, 0.0, 0.0

    best_bid = bids[0][0]
    best_ask = asks[0][0]

    # Depth within bps_window of best
    bid_depth = sum(
        price * size
        for price, size in bids
        if price >= best_bid * (1 - bps_window / 10000)
    )

    ask_depth = sum(
        price * size
        for price, size in asks
        if price <= best_ask * (1 + bps_window / 10000)
    )

    # Imbalance: -1 (all ask) to +1 (all bid)
    total_depth = bid_depth + ask_depth
    if total_depth > 0:
        imbalance = (bid_depth - ask_depth) / total_depth
    else:
        imbalance = 0.0

    return bid_depth, ask_depth, imbalance


def _load_orderbook_snapshots(
    db_path: str | Path,
    start_ms: int,
    end_ms: int,
) -> List[tuple[int, List[List[float]], List[List[float]]]]:
    """
    Load orderbook snapshots within time window.

    Returns:
        [(timestamp_ms, bids, asks), ...]
    """
    import json
    import sqlite3

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT timestamp, bids, asks
            FROM orderbooks
            WHERE timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp
            """,
            (start_ms, end_ms),
        ).fetchall()

        snapshots = []
        for ts, bids_json, asks_json in rows:
            try:
                bids = json.loads(bids_json)
                asks = json.loads(asks_json)
                snapshots.append((ts, bids, asks))
            except (json.JSONDecodeError, ValueError):
                continue

        return snapshots
    finally:
        conn.close()


def _detect_ramp_yank(
    snapshots: List[tuple[int, float, float]],  # [(ts, bid_depth, ask_depth), ...]
    window_trades_notional: float,
    min_ramp_ratio: float = 2.0,  # Depth must grow by 2x
    min_yank_ratio: float = 0.4,  # Depth must drop to <40% of peak
) -> tuple[bool, float, float]:
    """
    Detect depth ramp → yank pattern without matching trade volume.

    Returns:
        (is_spoofing, peak_depth, final_depth)
    """
    if len(snapshots) < 3:
        return False, 0.0, 0.0

    # Extract max depth per side
    depths = [(ts, max(bid, ask)) for ts, bid, ask in snapshots]

    if not depths:
        return False, 0.0, 0.0

    # Find peak depth
    peak_depth = max(d for _, d in depths)
    initial_depth = depths[0][1]
    final_depth = depths[-1][1]

    # Ramp: depth grows significantly
    ramp = peak_depth >= initial_depth * min_ramp_ratio

    # Yank: depth collapses from peak
    yank = final_depth <= peak_depth * min_yank_ratio

    # No matching volume: traded notional << peak depth
    # (If traders consumed the depth, notional would be ~peak_depth)
    no_matched_volume = window_trades_notional < peak_depth * 0.3

    is_spoofing = ramp and yank and no_matched_volume

    return is_spoofing, peak_depth, final_depth


def _spoofing_alerts(
    trades: Sequence[TradeRow],
    cfg: DetectorConfig,
    normalized_db_path: str | Path,
) -> List[Alert]:
    """
    Detect spoofing patterns via orderbook manipulation.

    Monitors for rapid asymmetric depth changes and spread compression
    indicating fake liquidity that disappears before execution.

    Enhanced with near-touch depth ramp/yank detection from raw orderbook ladders.
    """
    alerts: List[Alert] = []
    scfg = cfg.spoofing

    if len(trades) < 2:
        return alerts

    # Get source tablet DB path for orderbook data
    import sqlite3
    conn = sqlite3.connect(normalized_db_path)
    try:
        source_db_row = conn.execute("SELECT value FROM ingest_meta WHERE key = 'source_db'").fetchone()
        if not source_db_row:
            # No source DB metadata - skip orderbook-based checks
            source_db_path = None
        else:
            source_db_path = source_db_row[0]
    finally:
        conn.close()

    # Pre-load all orderbook snapshots for efficiency
    orderbook_snapshots = {}  # timestamp_ms -> (bids, asks)
    if source_db_path and Path(source_db_path).exists():
        min_ts = trades[0].timestamp_ms
        max_ts = trades[-1].timestamp_ms
        all_snapshots = _load_orderbook_snapshots(source_db_path, min_ts, max_ts)
        orderbook_snapshots = {ts: (bids, asks) for ts, bids, asks in all_snapshots}

    left = 0
    cooldown_until = -1

    for right, t in enumerate(trades):
        # Slide window left edge forward
        while left < right and trades[right].timestamp_ms - trades[left].timestamp_ms > scfg.window_ms:
            left += 1

        if right - left < 2:
            continue

        if t.timestamp_ms < cooldown_until:
            continue

        window = trades[left : right + 1]

        # Extract microstructure signals
        depth_ratio_5_vals = [x.micro_depth_ratio_5 for x in window if x.micro_depth_ratio_5 is not None]
        depth_ratio_20_vals = [x.micro_depth_ratio_20 for x in window if x.micro_depth_ratio_20 is not None]
        orderbook_pressure_vals = [x.micro_orderbook_pressure for x in window if x.micro_orderbook_pressure is not None]
        spread_compression_vals = [x.micro_spread_compression for x in window if x.micro_spread_compression is not None]

        if not depth_ratio_5_vals or not orderbook_pressure_vals:
            continue

        # Detect rapid asymmetric depth changes
        # Note: depth_ratio signals can cross zero (-0.1 to +0.1), so we use range (max-min), not ratio
        depth_5_min = min(depth_ratio_5_vals)
        depth_5_max = max(depth_ratio_5_vals)
        depth_5_range = depth_5_max - depth_5_min  # Renamed from swing to range (it's a difference)

        # Detect strong directional pressure
        avg_pressure = sum(abs(x) for x in orderbook_pressure_vals) / len(orderbook_pressure_vals)

        # Detect spread compression spikes
        max_compression = max(spread_compression_vals) if spread_compression_vals else 0.0

        # Spoofing fingerprint: large depth changes + directional pressure
        if (depth_5_range >= scfg.min_depth_ratio_change and
            avg_pressure >= scfg.min_orderbook_pressure):

            # Enhanced gate: check for ramp/yank pattern in raw orderbook ladders
            is_ramp_yank = False
            peak_depth = 0.0
            final_depth = 0.0

            if orderbook_snapshots:
                # Get orderbook snapshots within window
                window_start_ms = window[0].timestamp_ms
                window_end_ms = window[-1].timestamp_ms

                # Compute near-touch depth for each snapshot in window
                depth_timeline = []
                for ts in sorted(orderbook_snapshots.keys()):
                    if window_start_ms <= ts <= window_end_ms:
                        bids, asks = orderbook_snapshots[ts]
                        bid_depth, ask_depth, imbalance = _compute_near_touch_depth(bids, asks, bps_window=5.0)
                        depth_timeline.append((ts, bid_depth, ask_depth))

                # Check for ramp→yank pattern
                if depth_timeline:
                    window_notional = sum(tr.notional for tr in window)
                    is_ramp_yank, peak_depth, final_depth = _detect_ramp_yank(
                        depth_timeline,
                        window_notional,
                        min_ramp_ratio=2.0,
                        min_yank_ratio=0.4,
                    )

            # Require either:
            # 1. Ramp/yank pattern confirmed (high confidence)
            # 2. Very strong microstructure signals (pressure >=0.85 + range >=1.75)
            strong_signals = avg_pressure >= 0.85 and depth_5_range >= 1.75

            if not (is_ramp_yank or strong_signals):
                continue  # Skip this alert

            risk = scfg.base_risk_points

            # Bonus for confirmed ramp/yank
            if is_ramp_yank:
                risk += 5

            # Bonus for very high one-sided pressure (strong manipulation signal)
            if avg_pressure >= 0.85:
                risk += 5
            elif avg_pressure >= 0.75:
                risk += 3

            # Bonus for extreme depth ranges (large fake liquidity)
            if depth_5_range >= 1.75:
                risk += 3
            elif depth_5_range >= 1.5:
                risk += 2

            # Bonus for spread compression (fake tightening)
            if max_compression >= scfg.spread_compression_threshold:
                risk += scfg.compression_bonus

            depth_20_range = None
            if depth_ratio_20_vals:
                depth_20_range = max(depth_ratio_20_vals) - min(depth_ratio_20_vals)

            alerts.append(
                Alert(
                    detector="spoofing",
                    timestamp_ms=t.timestamp_ms,
                    timestamp_iso=t.timestamp_iso,
                    risk_points=risk,
                    reason="Rapid asymmetric orderbook depth changes with directional pressure.",
                    evidence={
                        "anchor_trade_id": t.id,
                        "window_trade_count": len(window),
                        "window_seconds": (window[-1].timestamp_ms - window[0].timestamp_ms) / 1000.0,
                        "depth_5_range": round(depth_5_range, 6),
                        "depth_20_range": None if depth_20_range is None else round(depth_20_range, 6),
                        "avg_orderbook_pressure": round(avg_pressure, 6),
                        "max_spread_compression": round(max_compression, 6),
                        "is_ramp_yank": is_ramp_yank,
                        "peak_depth_usd": round(peak_depth, 2) if peak_depth > 0 else None,
                        "final_depth_usd": round(final_depth, 2) if final_depth > 0 else None,
                    },
                )
            )
            cooldown_until = t.timestamp_ms + scfg.cooldown_ms

    return alerts


def _quote_stuffing_alerts(
    trades: Sequence[TradeRow],
    cfg: DetectorConfig,
) -> List[Alert]:
    """
    Detect quote stuffing patterns via high-frequency churning.

    Monitors for abnormal trade clustering with minimal price movement,
    indicating market manipulation through latency/confusion.
    """
    alerts: List[Alert] = []
    qscfg = cfg.quote_stuffing

    if len(trades) < 2:
        return alerts

    left = 0
    cooldown_until = -1

    for right, t in enumerate(trades):
        # Slide window left edge forward
        while left < right and trades[right].timestamp_ms - trades[left].timestamp_ms > qscfg.window_ms:
            left += 1

        if right - left < 2:
            continue

        if t.timestamp_ms < cooldown_until:
            continue

        window = trades[left : right + 1]
        window_duration_sec = (window[-1].timestamp_ms - window[0].timestamp_ms) / 1000.0

        # Use fixed 5s window for consistent rate calculation
        # (avoids extrapolating microsecond bursts)
        if window_duration_sec < 1.0:
            continue

        # Calculate actual trades per second over the window
        trade_intensity = len(window) / window_duration_sec

        if trade_intensity < qscfg.min_trade_intensity:
            continue

        # Extract microstructure signals
        realized_volatility_vals = [x.micro_realized_volatility for x in window if x.micro_realized_volatility is not None]
        liquidity_quality_vals = [x.micro_liquidity_quality for x in window if x.micro_liquidity_quality is not None]
        trade_intensity_vals = [x.micro_trade_intensity for x in window if x.micro_trade_intensity is not None]

        # Stuffing fingerprint: high intensity + low volatility + degraded liquidity
        avg_realized_vol = sum(realized_volatility_vals) / len(realized_volatility_vals) if realized_volatility_vals else None
        avg_liquidity_quality = sum(liquidity_quality_vals) / len(liquidity_quality_vals) if liquidity_quality_vals else None
        avg_micro_intensity = sum(trade_intensity_vals) / len(trade_intensity_vals) if trade_intensity_vals else None

        if avg_realized_vol is not None and avg_realized_vol > qscfg.max_realized_volatility:
            continue

        # Check liquidity degradation
        if avg_liquidity_quality is not None:
            liquidity_drop = 1.0 - avg_liquidity_quality
            if liquidity_drop < qscfg.min_liquidity_drop:
                continue
        else:
            continue  # Need liquidity quality signal

        risk = qscfg.base_risk_points

        # Bonus for extremely low volatility (churning without movement)
        if avg_realized_vol is not None and avg_realized_vol < qscfg.max_realized_volatility / 2:
            risk += qscfg.low_volatility_bonus

        alerts.append(
            Alert(
                detector="quote_stuffing",
                timestamp_ms=t.timestamp_ms,
                timestamp_iso=t.timestamp_iso,
                risk_points=risk,
                reason="High-frequency trade clustering with minimal price movement and liquidity degradation.",
                evidence={
                    "anchor_trade_id": t.id,
                    "window_trade_count": len(window),
                    "window_seconds": round(window_duration_sec, 3),
                    "trade_intensity": round(trade_intensity, 2),
                    "avg_realized_volatility": None if avg_realized_vol is None else round(avg_realized_vol, 8),
                    "avg_liquidity_quality": None if avg_liquidity_quality is None else round(avg_liquidity_quality, 6),
                    "avg_micro_trade_intensity": None if avg_micro_intensity is None else round(avg_micro_intensity, 2),
                },
            )
        )
        cooldown_until = t.timestamp_ms + qscfg.cooldown_ms

    return alerts


def detect_suspicious_patterns(
    normalized_db_path: str | Path,
    *,
    lookback_minutes: Optional[int] = None,
    config: Optional[DetectorConfig] = None,
) -> DetectionReport:
    """Run all detectors against normalized trades."""
    if config is None:
        config = get_default_config()

    trades = _load_trades(normalized_db_path, lookback_minutes=lookback_minutes)

    report = DetectionReport(
        normalized_db=str(normalized_db_path),
        analyzed_rows=len(trades),
        config_used={"lookback_minutes": lookback_minutes},
        _config=config,
    )

    if not trades:
        report.warnings.append("No rows found in normalized_trades for selected window.")
        return report

    logger.info(f"Running detection on {len(trades)} trades")

    report.alerts.extend(_mirror_reversal_alerts(trades, config))
    report.alerts.extend(_layering_cluster_alerts(trades, config))
    report.alerts.extend(_spoofing_alerts(trades, config, normalized_db_path))
    report.alerts.extend(_quote_stuffing_alerts(trades, config))
    report.alerts.extend(_balanced_churn_alerts(trades, config))

    report.alerts.sort(key=lambda a: (a.timestamp_ms, a.detector))

    logger.info(f"Detection complete: {len(report.alerts)} alerts, risk_score={report.risk_score}")
    return report


def run_detection(
    normalized_db_path: str | Path,
    *,
    lookback_minutes: Optional[int] = None,
    output_json_path: str | Path | None = None,
    config: Optional[DetectorConfig] = None,
    config_path: Optional[str | Path] = None,
) -> DetectionReport:
    """
    Run detection with optional config file support.

    Args:
        normalized_db_path: Path to normalized trades database
        lookback_minutes: Optional time window from latest trade
        output_json_path: Optional path to write JSON report
        config: Optional DetectorConfig instance (takes precedence)
        config_path: Optional path to JSON config file
    """
    if config is None and config_path is not None:
        config = DetectorConfig.load(config_path)

    report = detect_suspicious_patterns(
        normalized_db_path,
        lookback_minutes=lookback_minutes,
        config=config,
    )

    if output_json_path is not None:
        output_path = Path(output_json_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    return report
