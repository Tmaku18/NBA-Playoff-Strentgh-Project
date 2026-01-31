"""DuckDB connection helper."""
from __future__ import annotations

from pathlib import Path

import duckdb


def get_connection(db_path: str | Path, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    path = Path(db_path)
    if read_only and path.exists():
        return duckdb.connect(str(path), read_only=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path))
