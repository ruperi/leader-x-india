"""Offline persistence helpers for Parquet files and DuckDB."""

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq


def write_records_to_parquet(records: Iterable[dict[str, Any]], path: Path) -> None:
    rows = list(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def create_duckdb_from_parquet(database_path: Path, parquet_dir: Path) -> None:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(database_path)) as connection:
        for parquet_path in sorted(parquet_dir.glob("*.parquet")):
            table_name = parquet_path.stem.replace("-", "_")
            escaped_path = str(parquet_path).replace("'", "''")
            connection.execute(
                f"CREATE OR REPLACE TABLE {table_name} AS "
                f"SELECT * FROM read_parquet('{escaped_path}')"
            )
