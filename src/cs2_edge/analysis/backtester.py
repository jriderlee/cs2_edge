"""Backtester for the overpriced-favorite short strategy.

For each flagged contract: buy 100 NO contracts at the NO price (1 - open_price),
pay the maker fee at entry, hold to resolution. NO wins when the team loses.
Flat position sizing only.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import polars as pl

from cs2_edge.analysis.divergence_scorer import FLAG_MIN_DIVERGENCE, flags, score
from cs2_edge.db.db_init import DEFAULT_DB_PATH, init_db

POSITION = 100
MAKER_FEE = 0.0175
TRADING_DAYS = 252


def apply_pnl(flagged: pl.DataFrame) -> pl.DataFrame:
    open_p = flagged["open_price"]
    entry = POSITION * (1.0 - open_p)
    fee = POSITION * MAKER_FEE * open_p * (1.0 - open_p)
    won = flagged["resolved"] == "no"
    pnl = pl.when(won).then(POSITION - entry - fee).otherwise(-entry - fee)
    return flagged.with_columns(entry=entry, fee=fee, pnl=pnl)


def divergence_bucket(divergence: pl.Expr) -> pl.Expr:
    return (
        pl.when(divergence < 0.20)
        .then(pl.lit("0.10-0.20"))
        .when(divergence < 0.30)
        .then(pl.lit("0.20-0.30"))
        .otherwise(pl.lit("0.30+"))
    )


def sharpe(daily_pnl: pl.Series) -> float:
    if daily_pnl.len() < 2:
        return float("nan")
    s = daily_pnl.std(ddof=1)
    return daily_pnl.mean() / s * math.sqrt(TRADING_DAYS) if s and s > 0 else float("nan")


def max_drawdown(cum: pl.Series) -> float:
    return float((cum.cum_max() - cum).max())


def main() -> None:
    parser = argparse.ArgumentParser(description="Overpriced-favorite short backtester")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Path to DuckDB file")
    parser.add_argument(
        "--pred", default="data/win_prob_predictions_all.csv", help="Model predictions CSV"
    )
    parser.add_argument("--plot", default="data/equity_curve.png", help="Equity curve output path")
    parser.add_argument("--selftest", action="store_true", help="Run self-check and exit")
    args = parser.parse_args()

    if args.selftest:
        _selftest()
        return

    con = init_db(args.db)
    contracts = con.execute(
        "SELECT contract_id, match_id, team, open_price, resolved, resolution_date "
        "FROM kalshi_contracts "
        "WHERE resolved IN ('yes', 'no') AND open_price IS NOT NULL AND match_id IS NOT NULL"
    ).pl()
    con.close()

    predictions = pl.read_csv(args.pred)
    flagged = apply_pnl(flags(score(contracts, predictions))).sort("resolution_date")
    flagged = flagged.with_columns(div_bucket=divergence_bucket(pl.col("divergence")))

    n = flagged.height
    if n == 0:
        print("no flagged trades")
        return

    total_pnl = float(flagged["pnl"].sum())
    win_rate = float((flagged["resolved"] == "no").mean())
    avg_pnl = float(flagged["pnl"].mean())

    daily = flagged.group_by("resolution_date").agg(pl.col("pnl").sum()).sort("resolution_date")
    cum = flagged["pnl"].cum_sum()

    print("=== overall ===")
    print(f"trades: {n}")
    print(f"total PnL: ${total_pnl:,.2f}")
    print(f"win rate: {win_rate * 100:.1f}%")
    print(f"avg PnL/trade: ${avg_pnl:,.2f}")
    print(f"Sharpe (daily): {sharpe(daily['pnl']):.2f}")
    print(f"max drawdown: ${max_drawdown(cum):,.2f}")

    print("\n=== by tier ===")
    print(f"{'tier':<5}{'n':>5}{'pnl':>12}{'win%':>7}{'avg/trade':>11}")
    for tier in ("T1", "T2", "T3"):
        sub = flagged.filter(pl.col("tier") == tier)
        if sub.height == 0:
            continue
        print(
            f"{tier:<5}{sub.height:>5}{sub['pnl'].sum():>12,.2f}"
            f"{(sub['resolved'] == 'no').mean() * 100:>7.1f}{sub['pnl'].mean():>11,.2f}"
        )

    print("\n=== by divergence bucket ===")
    print(f"{'bucket':<11}{'n':>5}{'pnl':>12}{'win%':>7}{'avg/trade':>11}")
    for b in ("0.10-0.20", "0.20-0.30", "0.30+"):
        sub = flagged.filter(pl.col("div_bucket") == b)
        if sub.height == 0:
            continue
        print(
            f"{b:<11}{sub.height:>5}{sub['pnl'].sum():>12,.2f}"
            f"{(sub['resolved'] == 'no').mean() * 100:>7.1f}{sub['pnl'].mean():>11,.2f}"
        )

    print("\n=== monthly PnL ===")
    monthly = (
        flagged.with_columns(month=pl.col("resolution_date").dt.strftime("%Y-%m"))
        .group_by("month")
        .agg(n=pl.len(), pnl=pl.col("pnl").sum())
        .sort("month")
    )
    for month, mn, mpnl in monthly.iter_rows():
        print(f"{month}  n={mn:>4}  ${mpnl:>10,.2f}")

    # equity curve
    dates = flagged["resolution_date"].to_list()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(dates, cum.to_list(), label="cumulative PnL")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("resolution date")
    ax.set_ylabel("PnL ($)")
    ax.set_title("Backtest equity curve (flat 100 NO contracts)")
    fig.tight_layout()
    fig.savefig(args.plot, dpi=150)
    plt.close(fig)
    print(f"\nequity curve saved to {args.plot}")


def _selftest() -> None:
    from datetime import date

    flagged = pl.DataFrame(
        {
            "contract_id": ["c1", "c2", "c3"],
            "match_id": [1, 2, 3],
            "team": ["A", "B", "C"],
            "open_price": [0.95, 0.90, 0.92],
            "resolved": ["no", "yes", "no"],
            "tier": ["T1", "T2", "T3"],
            "divergence": [0.25, 0.15, 0.35],
            "resolution_date": [date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 3)],
        }
    )
    out = apply_pnl(flagged)

    c1 = out.filter(pl.col("contract_id") == "c1").row(0, named=True)
    entry1 = 100 * (1 - 0.95)  # 5.0
    fee1 = 100 * 0.0175 * 0.95 * 0.05  # 0.083125
    assert abs(c1["entry"] - entry1) < 1e-9, c1
    assert abs(c1["fee"] - fee1) < 1e-9, c1
    assert abs(c1["pnl"] - (100 - entry1 - fee1)) < 1e-9, c1

    c2 = out.filter(pl.col("contract_id") == "c2").row(0, named=True)
    entry2 = 100 * (1 - 0.90)  # 10.0
    fee2 = 100 * 0.0175 * 0.90 * 0.10  # 0.1575
    assert abs(c2["pnl"] - (-entry2 - fee2)) < 1e-9, c2

    buckets = out.with_columns(b=divergence_bucket(pl.col("divergence")))
    assert buckets.filter(pl.col("contract_id") == "c1")["b"][0] == "0.20-0.30"
    assert buckets.filter(pl.col("contract_id") == "c2")["b"][0] == "0.10-0.20"
    assert buckets.filter(pl.col("contract_id") == "c3")["b"][0] == "0.30+"
    print("selftest OK")


if __name__ == "__main__":
    main()
