"""Longshot bias test: Kalshi open price vs realized win rate by price decile."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import polars as pl

from cs2_edge.db.db_init import DEFAULT_DB_PATH, init_db

BIN_LABELS = [
    "0-10%",
    "10-20%",
    "20-30%",
    "30-40%",
    "40-50%",
    "50-60%",
    "60-70%",
    "70-80%",
    "80-90%",
    "90-100%",
]


def analyze(df: pl.DataFrame) -> pl.DataFrame:
    """Bin resolved contracts by open_price decile and return per-bin stats."""
    return (
        df.with_columns(
            win=(pl.col("resolved") == "yes").cast(pl.Float64),
            bin=(pl.col("open_price") * 10).floor().clip(0, 9).cast(pl.Int8),
        )
        .group_by("bin")
        .agg(
            n=pl.len(),
            avg_open_price=pl.col("open_price").mean(),
            win_rate=pl.col("win").mean(),
        )
        .with_columns(diff=pl.col("avg_open_price") - pl.col("win_rate"))
        .sort("bin")
    )


def print_table(agg: pl.DataFrame) -> None:
    print(f"{'bin':<10}{'n':>7}{'avg price':>11}{'win rate':>11}{'diff':>9}")
    print("-" * 48)
    total_n = total_win = 0
    for bin_, n, avg, win, diff in agg.iter_rows():
        print(f"{BIN_LABELS[bin_]:<10}{n:>7}{avg:>11.4f}{win:>11.4f}{diff:>+9.4f}")
        total_n += n
        total_win += win * n
    print("-" * 48)
    print(f"{'total':<10}{total_n:>7}{'':>11}{total_win / total_n:>11.4f}")


def plot_bias(agg: pl.DataFrame, out_path: Path) -> None:
    x = agg["avg_open_price"].to_list()
    y = agg["win_rate"].to_list()
    n = agg["n"].to_list()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="calibrated")
    ax.plot(x, y, "o-", color="tab:red", label="actual")
    for xi, yi, ni in zip(x, y, n):
        ax.annotate(str(ni), (xi, yi), textcoords="offset points", xytext=(5, 4), fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("market open price")
    ax.set_ylabel("actual win rate")
    ax.set_title("Longshot bias: open price vs realized win rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"plot saved to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Longshot bias test")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Path to DuckDB file")
    parser.add_argument("--out", default=None, help="Output plot path (default: data/bias_curve.png)")
    parser.add_argument("--selftest", action="store_true", help="Run analysis self-check and exit")
    args = parser.parse_args()

    if args.selftest:
        _selftest()
        return

    db_path = Path(args.db)
    con = init_db(db_path)
    df = con.execute(
        "SELECT open_price, resolved FROM kalshi_contracts "
        "WHERE resolved IN ('yes', 'no') AND open_price IS NOT NULL"
    ).pl()
    con.close()

    agg = analyze(df)
    print_table(agg)

    out = Path(args.out) if args.out else db_path.parent / "bias_curve.png"
    plot_bias(agg, out)


def _selftest() -> None:
    df = pl.DataFrame(
        {
            "open_price": [0.05, 0.05, 0.05, 0.15, 0.15, 0.95, 0.95],
            "resolved": ["yes", "no", "no", "yes", "no", "yes", "yes"],
        }
    )
    agg = analyze(df)
    rows = {r[0]: r for r in agg.iter_rows()}

    def close(a: float, b: float, tol: float = 1e-9) -> bool:
        return abs(a - b) < tol

    assert rows[0][0] == 0 and rows[0][1] == 3, rows[0]
    assert close(rows[0][2], 0.05) and close(rows[0][3], 1 / 3) and close(rows[0][4], 0.05 - 1 / 3), rows[0]
    assert close(rows[1][2], 0.15) and close(rows[1][3], 0.5) and close(rows[1][4], -0.35), rows[1]
    assert close(rows[9][2], 0.95) and close(rows[9][3], 1.0) and close(rows[9][4], -0.05), rows[9]
    print("selftest OK")


if __name__ == "__main__":
    main()
