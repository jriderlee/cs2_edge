"""One-shot migration: populate match_results.tier from event_name.

Run once manually after schema changes (idempotent):

    python -m cs2_edge.db.migrate_backfill_tier [db_path]
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

from cs2_edge.db.db_init import DEFAULT_DB_PATH, init_db
from cs2_edge.models.win_probability import infer_tier


def backfill_tier(con) -> int:
    """Add match_results.tier and populate it from event_name (idempotent)."""
    con.execute("ALTER TABLE match_results ADD COLUMN IF NOT EXISTS tier VARCHAR")
    df = con.execute("SELECT match_id, event_name FROM match_results").pl().with_columns(
        tier=pl.col("event_name").map_elements(infer_tier, return_dtype=pl.Utf8)
    )
    con.register("_tierdf", df)
    con.execute(
        "UPDATE match_results SET tier = _tierdf.tier "
        "FROM _tierdf WHERE match_results.match_id = _tierdf.match_id"
    )
    return df.height


def main() -> None:
    db_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DB_PATH
    con = init_db(db_path)
    n = backfill_tier(con)
    con.close()
    print(f"backfilled tier for {n} matches in {db_path}")


if __name__ == "__main__":
    main()
