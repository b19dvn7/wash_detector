from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import List
import sqlite3


@dataclass
class SyntheticFixture:
    source_db: str
    normal_trades: int
    suspicious_pairs: int
    positive_source_rowids: List[int]


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE trades (
            timestamp INTEGER,
            side TEXT,
            price REAL,
            amount REAL
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE candles (
            timestamp INTEGER,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            vwap REAL
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE orderbooks (
            timestamp INTEGER,
            spread REAL,
            mid_price REAL,
            imbalance REAL
        )
        """
    )


def _insert_context(conn: sqlite3.Connection, ts_ms: int, price: float, volume: float = 1000.0) -> None:
    high = price * 1.0002
    low = price * 0.9998
    vwap = price

    conn.execute(
        "INSERT INTO candles(timestamp, open, high, low, close, volume, vwap) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (ts_ms, price, high, low, price, volume, vwap),
    )

    conn.execute(
        "INSERT INTO orderbooks(timestamp, spread, mid_price, imbalance) VALUES (?, ?, ?, ?)",
        (ts_ms, 0.5, price, 0.0),
    )


def create_synthetic_source_db(
    output_db_path: str | Path,
    *,
    normal_trades: int = 300,
    suspicious_pairs: int = 40,
    seed: int = 17,
) -> SyntheticFixture:
    """Create deterministic synthetic source DB with known positive labels."""

    output_path = Path(output_db_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    rng = Random(seed)
    conn = sqlite3.connect(str(output_path))

    positive_source_rowids: List[int] = []

    try:
        _create_schema(conn)

        ts_ms = 1_707_600_000_000
        price = 100.0

        # Normal background trades: sparse in time to avoid cluster false positives.
        for i in range(normal_trades):
            ts_ms += 30_000  # 30s intervals
            price += rng.uniform(-0.8, 0.8)
            side = "BUY" if rng.random() < 0.5 else "SELL"
            amount = rng.uniform(0.05, 0.50)

            _insert_context(conn, ts_ms, price, volume=900.0 + rng.uniform(-100.0, 100.0))
            cur = conn.execute(
                "INSERT INTO trades(timestamp, side, price, amount) VALUES (?, ?, ?, ?)",
                (ts_ms, side, round(price, 6), round(amount, 8)),
            )
            _ = cur.lastrowid

        # Inject suspicious mirror pairs with low drift and near-equal amount.
        for i in range(suspicious_pairs):
            ts_ms += 300_000  # 5m gap between suspicious pair blocks
            base_price = price + rng.uniform(-0.3, 0.3)
            amount = rng.uniform(0.15, 0.45)
            first_side = "BUY" if (i % 2 == 0) else "SELL"
            second_side = "SELL" if first_side == "BUY" else "BUY"

            _insert_context(conn, ts_ms, base_price, volume=2200.0)
            conn.execute(
                "INSERT INTO trades(timestamp, side, price, amount) VALUES (?, ?, ?, ?)",
                (ts_ms, first_side, round(base_price, 6), round(amount, 8)),
            )

            ts2 = ts_ms + 5_000
            price2 = base_price * (1.0 + rng.uniform(-0.0002, 0.0002))  # <=2 bps approx
            amount2 = amount * (1.0 + rng.uniform(-0.03, 0.03))
            _insert_context(conn, ts2, price2, volume=2100.0)
            cur = conn.execute(
                "INSERT INTO trades(timestamp, side, price, amount) VALUES (?, ?, ?, ?)",
                (ts2, second_side, round(price2, 6), round(amount2, 8)),
            )

            positive_source_rowids.append(int(cur.lastrowid))
            ts_ms = ts2
            price = price2

        conn.commit()

    finally:
        conn.close()

    return SyntheticFixture(
        source_db=str(output_path),
        normal_trades=normal_trades,
        suspicious_pairs=suspicious_pairs,
        positive_source_rowids=positive_source_rowids,
    )
