"""Trade frequency of 90-100% opening contracts.

How often a "near-certain" contract opens, its share of all contracts, and
which tournaments it clusters in. Time is keyed on resolution_date (settlement
happens ~1 day after open; open_time isn't stored in kalshi_contracts).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from cs2_edge.db.db_init import DEFAULT_DB_PATH, init_db

CUTOFF = 0.90


def monthly(contracts: pl.DataFrame) -> pl.DataFrame:
    df = contracts.with_columns(
        is_90100=pl.col("open_price") >= CUTOFF,
        month=pl.col("resolution_date").dt.strftime("%Y-%m"),
    )
    return (
        df.group_by("month")
        .agg(total=pl.len(), n90100=pl.col("is_90100").sum())
        .sort("month")
        .with_columns(pct=pl.col("n90100") / pl.col("total") * 100)
    )


def opening_gaps(contracts: pl.DataFrame) -> pl.DataFrame:
    dates = (
        contracts.filter(pl.col("open_price") >= CUTOFF)
        .sort("resolution_date")
        .select("resolution_date")
    )
    return dates.with_columns(
        gap_days=(
            pl.col("resolution_date") - pl.col("resolution_date").shift(1)
        ).dt.total_days()
    )


def tournament_breakdown(contracts: pl.DataFrame, events: pl.DataFrame) -> pl.DataFrame:
    return (
        contracts.filter(pl.col("open_price") >= CUTOFF)
        .join(events, on="match_id", how="inner")
        .group_by("event_name")
        .agg(n90100=pl.len())
        .sort("n90100", descending=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="90-100% contract trade frequency")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Path to DuckDB file")
    parser.add_argument("--selftest", action="store_true", help="Run analysis self-check and exit")
    args = parser.parse_args()

    if args.selftest:
        _selftest()
        return

    con = init_db(args.db)
    contracts = con.execute(
        "SELECT contract_id, match_id, open_price, resolution_date FROM kalshi_contracts "
        "WHERE resolved IN ('yes', 'no') AND open_price IS NOT NULL AND resolution_date IS NOT NULL"
    ).pl()
    events = con.execute("SELECT match_id, event_name FROM match_results").pl()
    con.close()

    total = contracts.height
    n90100 = contracts.filter(pl.col("open_price") >= CUTOFF).height

    print("=== Monthly 90-100% openings ===")
    print(f"{'month':<9}{'total':>7}{'n90-100':>9}{'%ofmonth':>10}")
    for month, tot, n, pct in monthly(contracts).iter_rows():
        print(f"{month:<9}{tot:>7}{n:>9}{pct:>9.1f}%")

    print(f"\n=== Overall ===\n90-100% contracts: {n90100} / {total} = {n90100 / total * 100:.2f}%")

    gaps = opening_gaps(contracts)
    gaps = gaps.filter(pl.col("gap_days").is_not_null())["gap_days"]
    print(
        "\n=== Time between 90-100% openings ===\n"
        f"mean gap: {gaps.mean():.2f} days, median gap: {gaps.median():.1f} days"
    )

    print("\n=== Top tournaments (90-100% contracts) ===")
    for event, n in tournament_breakdown(contracts, events).head(20).iter_rows():
        print(f"{event:<55}{n:>4}")


def _selftest() -> None:
    from datetime import date

    contracts = pl.DataFrame(
        {
            "contract_id": ["a", "b", "c", "d"],
            "match_id": [1, 2, 3, 4],
            "open_price": [0.95, 0.50, 0.92, 0.10],
            "resolution_date": [
                date(2026, 1, 1),
                date(2026, 1, 2),
                date(2026, 1, 3),
                date(2026, 2, 1),
            ],
        }
    )
    m = monthly(contracts)
    assert m.height == 2, m
    jan = m.filter(pl.col("month") == "2026-01").row(0)
    assert jan[1] == 3 and jan[2] == 2, jan

    gaps = opening_gaps(contracts)
    g = gaps.filter(pl.col("gap_days").is_not_null())["gap_days"].to_list()
    assert g == [2.0], g  # Jan 1 -> Jan 3
    print("selftest OK")


if __name__ == "__main__":
    main()
