from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple
import sqlite3


@dataclass(frozen=True)
class TableContract:
    """Contract for one SQLite table."""

    name: str
    required_columns: Tuple[str, ...]
    optional_columns: Tuple[str, ...] = ()
    notes: str = ""


@dataclass(frozen=True)
class SourceSchemaContract:
    """Contract for source databases consumed by the detector pipeline."""

    required_tables: Tuple[TableContract, ...]
    optional_tables: Tuple[TableContract, ...] = ()


@dataclass
class ValidationReport:
    """Schema-validation result for one SQLite source DB."""

    db_path: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    discovered_tables: Dict[str, Tuple[str, ...]] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0

    def to_text(self) -> str:
        lines: List[str] = []
        lines.append(f"DB: {self.db_path}")
        lines.append(f"Schema check: {'PASS' if self.passed else 'FAIL'}")

        if self.errors:
            lines.append("Errors:")
            lines.extend([f"  - {err}" for err in self.errors])

        if self.warnings:
            lines.append("Warnings:")
            lines.extend([f"  - {warn}" for warn in self.warnings])

        lines.append("Discovered tables:")
        if not self.discovered_tables:
            lines.append("  - (none)")
        else:
            for table_name in sorted(self.discovered_tables.keys()):
                cols = ", ".join(self.discovered_tables[table_name])
                lines.append(f"  - {table_name}: {cols}")

        return "\n".join(lines)


DEFAULT_SOURCE_CONTRACT = SourceSchemaContract(
    required_tables=(
        TableContract(
            name="trades",
            required_columns=("timestamp", "side", "price", "amount"),
            optional_columns=("trade_id", "exchange", "symbol"),
            notes="Raw executed trades from venue/API.",
        ),
        TableContract(
            name="candles",
            required_columns=("timestamp", "open", "high", "low", "close", "volume"),
            optional_columns=("vwap", "quote_volume", "trade_count"),
            notes="Market context candles, typically 1m bars.",
        ),
    ),
    optional_tables=(
        TableContract(
            name="orderbooks",
            required_columns=("timestamp",),
            optional_columns=(
                "spread",
                "mid_price",
                "imbalance",
                "best_bid",
                "best_ask",
                "bids",
                "asks",
            ),
            notes="Optional but useful for microstructure context.",
        ),
    ),
)


def contract_to_text(contract: SourceSchemaContract = DEFAULT_SOURCE_CONTRACT) -> str:
    lines: List[str] = []
    lines.append("Source schema contract")
    lines.append("Required tables:")
    for table in contract.required_tables:
        lines.append(f"  - {table.name}")
        lines.append(f"    required: {', '.join(table.required_columns)}")
        if table.optional_columns:
            lines.append(f"    optional: {', '.join(table.optional_columns)}")
        if table.notes:
            lines.append(f"    notes: {table.notes}")

    lines.append("Optional tables:")
    if not contract.optional_tables:
        lines.append("  - (none)")
    else:
        for table in contract.optional_tables:
            lines.append(f"  - {table.name}")
            lines.append(f"    required: {', '.join(table.required_columns)}")
            if table.optional_columns:
                lines.append(f"    optional: {', '.join(table.optional_columns)}")
            if table.notes:
                lines.append(f"    notes: {table.notes}")

    return "\n".join(lines)


def _normalize_name(name: str) -> str:
    return name.strip().lower()


def _quote_ident(identifier: str) -> str:
    # Safe for known/internal identifiers; escape embedded quotes defensively.
    escaped = identifier.replace('"', '""')
    return f'"{escaped}"'


def _discover_tables(conn: sqlite3.Connection) -> Dict[str, Tuple[str, ...]]:
    tables: Dict[str, Tuple[str, ...]] = {}

    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name ASC"
    ).fetchall()

    for (table_name_raw,) in rows:
        table_name = _normalize_name(str(table_name_raw))
        pragma_sql = f"PRAGMA table_info({_quote_ident(table_name_raw)})"
        pragma_rows = conn.execute(pragma_sql).fetchall()
        cols = tuple(_normalize_name(str(row[1])) for row in pragma_rows)
        tables[table_name] = cols

    return tables


def validate_source_db(
    db_path: str | Path,
    contract: SourceSchemaContract = DEFAULT_SOURCE_CONTRACT,
) -> ValidationReport:
    """Validate source DB tables/columns against the canonical contract."""

    db_path_str = str(db_path)
    report = ValidationReport(db_path=db_path_str)
    path_obj = Path(db_path)

    if not path_obj.exists():
        report.errors.append(f"Database file does not exist: {db_path_str}")
        return report

    if not path_obj.is_file():
        report.errors.append(f"Path is not a file: {db_path_str}")
        return report

    try:
        conn = sqlite3.connect(db_path_str)
    except sqlite3.Error as exc:
        report.errors.append(f"Failed to open sqlite database: {exc}")
        return report

    try:
        with conn:
            discovered = _discover_tables(conn)
            report.discovered_tables = discovered

            for table in contract.required_tables:
                table_key = _normalize_name(table.name)
                if table_key not in discovered:
                    report.errors.append(f"Missing required table: {table.name}")
                    continue

                discovered_cols = set(discovered[table_key])
                missing_cols = [
                    col for col in table.required_columns if _normalize_name(col) not in discovered_cols
                ]
                if missing_cols:
                    report.errors.append(
                        f"Table '{table.name}' missing required columns: {', '.join(missing_cols)}"
                    )

            for table in contract.optional_tables:
                table_key = _normalize_name(table.name)
                if table_key not in discovered:
                    report.warnings.append(
                        f"Optional table not present: {table.name} (context features may be reduced)"
                    )
                    continue

                discovered_cols = set(discovered[table_key])
                missing_cols = [
                    col for col in table.required_columns if _normalize_name(col) not in discovered_cols
                ]
                if missing_cols:
                    report.warnings.append(
                        f"Optional table '{table.name}' missing recommended columns: {', '.join(missing_cols)}"
                    )
    except sqlite3.Error as exc:
        report.errors.append(f"SQLite read error while validating schema: {exc}")
    finally:
        conn.close()

    return report
