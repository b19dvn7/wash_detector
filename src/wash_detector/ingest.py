from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence
import bisect
import math
import sqlite3

from .schema import ValidationReport, validate_source_db

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CandlePoint:
    timestamp_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: Optional[float]


@dataclass(frozen=True)
class OrderbookPoint:
    timestamp_ms: int
    spread: Optional[float]
    mid_price: Optional[float]
    imbalance: Optional[float]


@dataclass(frozen=True)
class MicrostructurePoint:
    timestamp_ms: int
    orderbook_pressure: Optional[float]
    volume_imbalance: Optional[float]
    price_momentum: Optional[float]
    liquidity_score: Optional[float]
    trade_intensity: Optional[float]
    spread_compression: Optional[float]
    depth_ratio_5: Optional[float]
    depth_ratio_20: Optional[float]
    price_acceleration: Optional[float]
    realized_volatility: Optional[float]
    large_trade_ratio: Optional[float]
    effective_spread: Optional[float]
    liquidity_depth: Optional[float]
    depth_weighted: Optional[float]
    liquidity_quality: Optional[float]


@dataclass
class IngestReport:
    source_db: str
    output_db: str
    schema_report: ValidationReport
    has_orderbooks: bool
    candle_points: int
    orderbook_points: int
    microstructure_points: int = 0
    input_rows: int = 0
    output_rows: int = 0
    skipped_rows: int = 0
    warning_count: int = 0
    warnings: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        # Tolerates skipped rows - real exchange data often has zero-amount rows
        return self.schema_report.passed and self.output_rows > 0

    def to_text(self) -> str:
        lines: List[str] = []
        lines.append("Ingest report")
        lines.append(f"  source_db: {self.source_db}")
        lines.append(f"  output_db: {self.output_db}")
        lines.append(f"  schema: {'PASS' if self.schema_report.passed else 'FAIL'}")
        lines.append(f"  orderbook_context: {'present' if self.has_orderbooks else 'absent'}")
        lines.append(f"  candle_points: {self.candle_points}")
        lines.append(f"  orderbook_points: {self.orderbook_points}")
        lines.append(f"  microstructure_points: {self.microstructure_points}")
        lines.append(f"  input_rows: {self.input_rows}")
        lines.append(f"  output_rows: {self.output_rows}")
        lines.append(f"  skipped_rows: {self.skipped_rows}")
        lines.append(f"  warning_count: {self.warning_count}")

        if self.schema_report.errors:
            lines.append("Schema errors:")
            lines.extend([f"  - {err}" for err in self.schema_report.errors])

        if self.schema_report.warnings:
            lines.append("Schema warnings:")
            lines.extend([f"  - {warn}" for warn in self.schema_report.warnings])

        if self.warnings:
            lines.append("Runtime warnings (sample):")
            lines.extend([f"  - {warn}" for warn in self.warnings])
            if self.warning_count > len(self.warnings):
                lines.append(
                    f"  - ... {self.warning_count - len(self.warnings)} additional warning(s) not shown"
                )

        return "\n".join(lines)


def parse_timestamp_ms(raw_value: object) -> int:
    """Normalize timestamps to unix epoch milliseconds."""
    if raw_value is None:
        raise ValueError("timestamp is null")

    # Fast path for numeric inputs
    if isinstance(raw_value, (int, float)):
        return _normalize_numeric_timestamp(float(raw_value))

    raw_text = str(raw_value).strip()
    if raw_text == "":
        raise ValueError("timestamp is empty")

    # Numeric timestamp encoded as text
    try:
        num = float(raw_text)
        return _normalize_numeric_timestamp(num)
    except ValueError:
        pass

    # ISO timestamp path
    iso_text = raw_text
    if iso_text.endswith("Z"):
        iso_text = iso_text[:-1] + "+00:00"

    dt = datetime.fromisoformat(iso_text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return int(dt.timestamp() * 1000)


def _normalize_numeric_timestamp(value: float) -> int:
    if not math.isfinite(value):
        raise ValueError("timestamp is not finite")

    absolute = abs(value)

    # Upper bound check - reject clearly invalid timestamps
    if absolute >= 1e16:
        raise ValueError(f"timestamp too large to be valid: {value}")

    if absolute >= 1e15:
        # Microseconds -> ms
        return int(value / 1000)

    if absolute >= 1e12:
        # Already in ms
        return int(value)

    if absolute >= 1e9:
        # seconds -> ms
        return int(value * 1000)

    raise ValueError(f"numeric timestamp too small to classify: {value}")


def normalize_side(raw_side: object) -> str:
    if raw_side is None:
        raise ValueError("side is null")

    side = str(raw_side).strip().upper()
    if side not in ("BUY", "SELL"):
        raise ValueError(f"unsupported side value: {raw_side}")

    return side


def _as_float(raw_value: object, field_name: str) -> float:
    value = float(raw_value)
    if not math.isfinite(value):
        raise ValueError(f"{field_name} is not finite")
    return value


def _as_optional_float(raw_value: object) -> Optional[float]:
    if raw_value is None:
        return None

    value = float(raw_value)
    if not math.isfinite(value):
        return None
    return value


def _add_warning(report: IngestReport, message: str, max_samples: int = 25) -> None:
    report.warning_count += 1
    if len(report.warnings) < max_samples:
        report.warnings.append(message)


def _open_readonly_sqlite(db_path: str) -> sqlite3.Connection:
    uri = f"file:{db_path}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def _create_output_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE normalized_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_rowid INTEGER,
            timestamp_ms INTEGER NOT NULL,
            timestamp_iso TEXT NOT NULL,
            side TEXT NOT NULL,
            price REAL NOT NULL,
            amount REAL NOT NULL,
            notional REAL NOT NULL,
            candle_open REAL,
            candle_high REAL,
            candle_low REAL,
            candle_close REAL,
            candle_volume REAL,
            candle_vwap REAL,
            ob_spread REAL,
            ob_mid_price REAL,
            ob_imbalance REAL,
            micro_orderbook_pressure REAL,
            micro_volume_imbalance REAL,
            micro_price_momentum REAL,
            micro_liquidity_score REAL,
            micro_trade_intensity REAL,
            micro_spread_compression REAL,
            micro_depth_ratio_5 REAL,
            micro_depth_ratio_20 REAL,
            micro_price_acceleration REAL,
            micro_realized_volatility REAL,
            micro_large_trade_ratio REAL,
            micro_effective_spread REAL,
            micro_liquidity_depth REAL,
            micro_depth_weighted REAL,
            micro_liquidity_quality REAL
        )
        """
    )
    conn.execute("CREATE INDEX idx_norm_trades_ts ON normalized_trades(timestamp_ms)")
    conn.execute("CREATE INDEX idx_norm_trades_side ON normalized_trades(side)")

    conn.execute(
        """
        CREATE TABLE ingest_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )


def _load_candle_points(
    source_conn: sqlite3.Connection,
    report: IngestReport,
) -> tuple[List[int], List[CandlePoint]]:
    candles_cols = set(report.schema_report.discovered_tables.get("candles", ()))
    vwap_expr = "vwap" if "vwap" in candles_cols else "NULL AS vwap"

    query = f"""
        SELECT rowid, timestamp, open, high, low, close, volume, {vwap_expr}
        FROM candles
        ORDER BY timestamp ASC, rowid ASC
    """

    time_index: List[int] = []
    points: List[CandlePoint] = []

    for row in source_conn.execute(query):
        rowid, raw_ts, raw_open, raw_high, raw_low, raw_close, raw_volume, raw_vwap = row
        try:
            ts_ms = parse_timestamp_ms(raw_ts)
            point = CandlePoint(
                timestamp_ms=ts_ms,
                open=_as_float(raw_open, "candle.open"),
                high=_as_float(raw_high, "candle.high"),
                low=_as_float(raw_low, "candle.low"),
                close=_as_float(raw_close, "candle.close"),
                volume=_as_float(raw_volume, "candle.volume"),
                vwap=_as_optional_float(raw_vwap),
            )
        except Exception as exc:  # noqa: BLE001 - we record warning and skip bad rows
            _add_warning(report, f"candles rowid={rowid} skipped: {exc}")
            continue

        time_index.append(ts_ms)
        points.append(point)

    return time_index, points


def _load_orderbook_points(
    source_conn: sqlite3.Connection,
    report: IngestReport,
) -> tuple[List[int], List[OrderbookPoint]]:
    if "orderbooks" not in report.schema_report.discovered_tables:
        return [], []

    ob_cols = set(report.schema_report.discovered_tables.get("orderbooks", ()))

    spread_expr = "spread" if "spread" in ob_cols else "NULL AS spread"
    mid_expr = "mid_price" if "mid_price" in ob_cols else "NULL AS mid_price"
    imb_expr = "imbalance" if "imbalance" in ob_cols else "NULL AS imbalance"

    query = f"""
        SELECT rowid, timestamp, {spread_expr}, {mid_expr}, {imb_expr}
        FROM orderbooks
        ORDER BY timestamp ASC, rowid ASC
    """

    time_index: List[int] = []
    points: List[OrderbookPoint] = []

    for row in source_conn.execute(query):
        rowid, raw_ts, raw_spread, raw_mid, raw_imb = row
        try:
            ts_ms = parse_timestamp_ms(raw_ts)
            point = OrderbookPoint(
                timestamp_ms=ts_ms,
                spread=_as_optional_float(raw_spread),
                mid_price=_as_optional_float(raw_mid),
                imbalance=_as_optional_float(raw_imb),
            )
        except Exception as exc:  # noqa: BLE001 - we record warning and skip bad rows
            _add_warning(report, f"orderbooks rowid={rowid} skipped: {exc}")
            continue

        time_index.append(ts_ms)
        points.append(point)

    return time_index, points


def _load_microstructure_points(
    source_conn: sqlite3.Connection,
    report: IngestReport,
) -> tuple[List[int], List[MicrostructurePoint]]:
    if "microstructure" not in report.schema_report.discovered_tables:
        return [], []

    micro_cols = set(report.schema_report.discovered_tables.get("microstructure", ()))

    # Build SELECT with NULL fallbacks for any missing columns
    fields = [
        "orderbook_pressure", "volume_imbalance", "price_momentum", "liquidity_score",
        "trade_intensity", "spread_compression", "depth_ratio_5", "depth_ratio_20",
        "price_acceleration", "realized_volatility", "large_trade_ratio", "effective_spread",
        "liquidity_depth", "depth_weighted", "liquidity_quality"
    ]

    select_exprs = [f"{field}" if field in micro_cols else f"NULL AS {field}" for field in fields]

    query = f"""
        SELECT rowid, timestamp, {', '.join(select_exprs)}
        FROM microstructure
        ORDER BY timestamp ASC, rowid ASC
    """

    time_index: List[int] = []
    points: List[MicrostructurePoint] = []

    for row in source_conn.execute(query):
        rowid = row[0]
        raw_ts = row[1]
        raw_values = row[2:]  # 15 microstructure values

        try:
            ts_ms = parse_timestamp_ms(raw_ts)
            point = MicrostructurePoint(
                timestamp_ms=ts_ms,
                orderbook_pressure=_as_optional_float(raw_values[0]),
                volume_imbalance=_as_optional_float(raw_values[1]),
                price_momentum=_as_optional_float(raw_values[2]),
                liquidity_score=_as_optional_float(raw_values[3]),
                trade_intensity=_as_optional_float(raw_values[4]),
                spread_compression=_as_optional_float(raw_values[5]),
                depth_ratio_5=_as_optional_float(raw_values[6]),
                depth_ratio_20=_as_optional_float(raw_values[7]),
                price_acceleration=_as_optional_float(raw_values[8]),
                realized_volatility=_as_optional_float(raw_values[9]),
                large_trade_ratio=_as_optional_float(raw_values[10]),
                effective_spread=_as_optional_float(raw_values[11]),
                liquidity_depth=_as_optional_float(raw_values[12]),
                depth_weighted=_as_optional_float(raw_values[13]),
                liquidity_quality=_as_optional_float(raw_values[14]),
            )
        except Exception as exc:  # noqa: BLE001 - we record warning and skip bad rows
            _add_warning(report, f"microstructure rowid={rowid} skipped: {exc}")
            continue

        time_index.append(ts_ms)
        points.append(point)

    return time_index, points


def _latest_at_or_before(ts_ms: int, time_index: Sequence[int], points: Sequence[object]) -> object | None:
    if not time_index:
        return None

    pos = bisect.bisect_right(time_index, ts_ms) - 1
    if pos < 0:
        return None

    return points[pos]


def ingest_to_normalized_db(
    source_db_path: str | Path,
    output_db_path: str | Path,
    *,
    overwrite: bool = False,
) -> IngestReport:
    """
    Read source DB (read-only), normalize trades + context, and write output DB.
    """

    source_db = str(source_db_path)
    output_db = str(output_db_path)

    schema_report = validate_source_db(source_db)
    has_orderbooks = "orderbooks" in schema_report.discovered_tables

    report = IngestReport(
        source_db=source_db,
        output_db=output_db,
        schema_report=schema_report,
        has_orderbooks=has_orderbooks,
        candle_points=0,
        orderbook_points=0,
        microstructure_points=0,
    )

    if not schema_report.passed:
        return report

    output_path = Path(output_db)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        if overwrite:
            output_path.unlink()
        else:
            report.warning_count += 1
            report.warnings.append(
                f"output already exists: {output_db} (use overwrite=True to replace)"
            )
            return report

    logger.debug(f"Opening source DB: {source_db}")
    source_conn = _open_readonly_sqlite(source_db)
    output_conn = sqlite3.connect(output_db)

    try:
        _create_output_schema(output_conn)

        logger.debug("Loading candle context data")
        candle_times, candle_points = _load_candle_points(source_conn, report)
        logger.debug("Loading orderbook context data")
        ob_times, ob_points = _load_orderbook_points(source_conn, report)
        logger.debug("Loading microstructure context data")
        micro_times, micro_points = _load_microstructure_points(source_conn, report)
        report.candle_points = len(candle_points)
        report.orderbook_points = len(ob_points)
        report.microstructure_points = len(micro_points)
        logger.info(f"Context loaded: {len(candle_points)} candles, {len(ob_points)} orderbook snapshots, {len(micro_points)} microstructure signals")

        insert_sql = """
            INSERT INTO normalized_trades (
                source_rowid,
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
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        batch: List[tuple] = []
        batch_size = 5000

        trade_query = """
            SELECT rowid, timestamp, side, price, amount
            FROM trades
            ORDER BY timestamp ASC, rowid ASC
        """

        for row in source_conn.execute(trade_query):
            rowid, raw_ts, raw_side, raw_price, raw_amount = row
            report.input_rows += 1

            try:
                ts_ms = parse_timestamp_ms(raw_ts)
                side = normalize_side(raw_side)
                price = _as_float(raw_price, "trade.price")
                amount = _as_float(raw_amount, "trade.amount")

                if price <= 0:
                    raise ValueError("trade.price must be > 0")
                if amount <= 0:
                    raise ValueError("trade.amount must be > 0")

            except Exception as exc:  # noqa: BLE001
                report.skipped_rows += 1
                _add_warning(report, f"trades rowid={rowid} skipped: {exc}")
                continue

            candle = _latest_at_or_before(ts_ms, candle_times, candle_points)
            orderbook = _latest_at_or_before(ts_ms, ob_times, ob_points)
            microstructure = _latest_at_or_before(ts_ms, micro_times, micro_points)

            ts_iso = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).isoformat()
            notional = price * amount

            batch.append(
                (
                    rowid,
                    ts_ms,
                    ts_iso,
                    side,
                    price,
                    amount,
                    notional,
                    candle.open if candle else None,
                    candle.high if candle else None,
                    candle.low if candle else None,
                    candle.close if candle else None,
                    candle.volume if candle else None,
                    candle.vwap if candle else None,
                    orderbook.spread if orderbook else None,
                    orderbook.mid_price if orderbook else None,
                    orderbook.imbalance if orderbook else None,
                    microstructure.orderbook_pressure if microstructure else None,
                    microstructure.volume_imbalance if microstructure else None,
                    microstructure.price_momentum if microstructure else None,
                    microstructure.liquidity_score if microstructure else None,
                    microstructure.trade_intensity if microstructure else None,
                    microstructure.spread_compression if microstructure else None,
                    microstructure.depth_ratio_5 if microstructure else None,
                    microstructure.depth_ratio_20 if microstructure else None,
                    microstructure.price_acceleration if microstructure else None,
                    microstructure.realized_volatility if microstructure else None,
                    microstructure.large_trade_ratio if microstructure else None,
                    microstructure.effective_spread if microstructure else None,
                    microstructure.liquidity_depth if microstructure else None,
                    microstructure.depth_weighted if microstructure else None,
                    microstructure.liquidity_quality if microstructure else None,
                )
            )

            if len(batch) >= batch_size:
                output_conn.executemany(insert_sql, batch)
                output_conn.commit()
                report.output_rows += len(batch)
                batch.clear()

        if batch:
            output_conn.executemany(insert_sql, batch)
            output_conn.commit()
            report.output_rows += len(batch)
            batch.clear()

        ingest_meta = {
            "source_db": source_db,
            "ingested_at_utc": datetime.now(timezone.utc).isoformat(),
            "input_rows": str(report.input_rows),
            "output_rows": str(report.output_rows),
            "skipped_rows": str(report.skipped_rows),
            "warning_count": str(report.warning_count),
            "has_orderbooks": str(report.has_orderbooks),
        }

        output_conn.executemany(
            "INSERT INTO ingest_meta(key, value) VALUES (?, ?)",
            list(ingest_meta.items()),
        )
        output_conn.commit()

    finally:
        source_conn.close()
        output_conn.close()

    return report
