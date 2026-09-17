"""Backtester for the overpriced-favorite short strategy.

For each flagged contract: deploy a flat $5 position in NO contracts
(contracts = floor($5 / NO price)), pay the maker fee, hold to resolution.
NO wins when the team loses.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import polars as pl

from cs2_edge.analysis.divergence_scorer import FLAG_MIN_DIVERGENCE, flags, load_contracts, score
from cs2_edge.db.db_init import DEFAULT_DB_PATH, init_db

POSITION = 5.0
MAKER_FEE = 0.0175
TRADING_DAYS = 252


def apply_pnl(flagged: pl.DataFrame) -> pl.DataFrame:
    open_p = flagged["open_price"]
    no_price = (1.0 - open_p).round(4)
    contracts = (POSITION / no_price).floor().cast(pl.Int64)
    fee = contracts * MAKER_FEE * open_p * no_price
    won = flagged["resolved"] == "no"
    pnl = pl.when(won).then(contracts - POSITION - fee).otherwise(-POSITION - fee)
    return flagged.with_columns(contracts=contracts, fee=fee, pnl=pnl)


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
    contracts = load_contracts(con)
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
    capital_deployed = n * POSITION

    daily = flagged.group_by("resolution_date").agg(pl.col("pnl").sum()).sort("resolution_date")
    cum = flagged["pnl"].cum_sum()

    print("=== overall ===")
    print(f"trades: {n}")
    print(f"total PnL: ${total_pnl:,.2f}")
    print(f"win rate: {win_rate * 100:.1f}%")
    print(f"avg PnL/trade: ${avg_pnl:,.2f}")
    print(f"Sharpe (daily): {sharpe(daily['pnl']):.2f}")
    print(f"max drawdown: ${max_drawdown(cum):,.2f}")
    print(f"total capital deployed: ${capital_deployed:,.2f}")

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
    ax.set_title("Backtest equity curve (flat $5 NO position)")
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
    contracts1 = 100  # floor(5 / 0.05)
    fee1 = contracts1 * 0.0175 * 0.95 * 0.05
    assert c1["contracts"] == contracts1, c1
    assert abs(c1["fee"] - fee1) < 1e-9, c1
    assert abs(c1["pnl"] - (contracts1 - 5.0 - fee1)) < 1e-9, c1

    c2 = out.filter(pl.col("contract_id") == "c2").row(0, named=True)
    contracts2 = 50  # floor(5 / 0.10)
    fee2 = contracts2 * 0.0175 * 0.90 * 0.10
    assert c2["contracts"] == contracts2, c2
    assert abs(c2["pnl"] - (-5.0 - fee2)) < 1e-9, c2

    buckets = out.with_columns(b=divergence_bucket(pl.col("divergence")))
    assert buckets.filter(pl.col("contract_id") == "c1")["b"][0] == "0.20-0.30"
    assert buckets.filter(pl.col("contract_id") == "c2")["b"][0] == "0.10-0.20"
    assert buckets.filter(pl.col("contract_id") == "c3")["b"][0] == "0.30+"
    print("selftest OK")


if __name__ == "__main__":
    main()
