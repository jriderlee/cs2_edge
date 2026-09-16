from __future__ import annotations

import sys
from pathlib import Path

import duckdb

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "cs2_edge.duckdb"


def init_db(db_path: str | Path = DEFAULT_DB_PATH) -> duckdb.DuckDBPyConnection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    con.execute(SCHEMA_PATH.read_text())
    return con


def main() -> None:
    db_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DB_PATH
    con = init_db(db_path)
    tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
    print(f"Initialized {db_path} — tables: {tables}")
    con.close()


if __name__ == "__main__":
    main()
