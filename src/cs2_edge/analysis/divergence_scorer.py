"""Divergence scorer: market open price vs independent HLTV model probability.

Joins model predictions (win_prob_predictions.csv) to kalshi_contracts on
match_id, maps each contract's team to the model's reference side, and flags
actionable OVER signals (market overprices a near-certain contract).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from cs2_edge.collectors.kalshi import normalize_team
from cs2_edge.db.db_init import DEFAULT_DB_PATH, init_db

FLAG_MIN_PRICE = 0.90  # 90-100% bin
FLAG_MIN_DIVERGENCE = 0.10


def score(contracts: pl.DataFrame, predictions: pl.DataFrame) -> pl.DataFrame:
    """Join on match_id, map team -> model side, compute divergence + direction."""
    df = contracts.join(predictions, on="match_id", how="inner")
    df = df.with_columns(
        [
            pl.col("team").map_elements(normalize_team, return_dtype=pl.Utf8).alias("norm_team"),
            pl.col("team_a").map_elements(normalize_team, return_dtype=pl.Utf8).alias("norm_a"),
            pl.col("team_b").map_elements(normalize_team, return_dtype=pl.Utf8).alias("norm_b"),
        ]
    )
    df = df.with_columns(
        model_prob=(
            pl.when(pl.col("norm_team") == pl.col("norm_a"))
            .then(pl.col("prob_team_a"))
            .when(pl.col("norm_team") == pl.col("norm_b"))
            .then(1.0 - pl.col("prob_team_a"))
            .otherwise(None)
        )
    ).filter(pl.col("model_prob").is_not_null())

    return df.with_columns(
        divergence=pl.col("open_price") - pl.col("model_prob"),
        direction=pl.when(pl.col("open_price") > pl.col("model_prob"))
        .then(pl.lit("OVER"))
        .otherwise(pl.lit("UNDER")),
    )


def flags(scored: pl.DataFrame) -> pl.DataFrame:
    return scored.filter(
        (pl.col("open_price") >= FLAG_MIN_PRICE)
        & (pl.col("divergence") > FLAG_MIN_DIVERGENCE)
        & (pl.col("direction") == "OVER")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Market-vs-model divergence scorer")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Path to DuckDB file")
    parser.add_argument(
        "--pred", default="data/win_prob_predictions.csv", help="Model predictions CSV"
    )
    parser.add_argument("--out", default=None, help="CSV path for flagged contracts")
    parser.add_argument("--selftest", action="store_true", help="Run self-check and exit")
    args = parser.parse_args()

    if args.selftest:
        _selftest()
        return

    con = init_db(args.db)
    contracts = con.execute(
        "SELECT contract_id, match_id, team, open_price, resolved FROM kalshi_contracts "
        "WHERE resolved IN ('yes', 'no') AND open_price IS NOT NULL AND match_id IS NOT NULL"
    ).pl()
    con.close()

    predictions = pl.read_csv(args.pred)
    scored = score(contracts, predictions)
    flagged = flags(scored)

    print(f"joined contracts: {scored.height}")
    print(f"OVER / UNDER: {(scored['direction'] == 'OVER').sum()} / {(scored['direction'] == 'UNDER').sum()}\n")

    cols = ["contract_id", "team", "open_price", "model_prob", "divergence", "tier", "resolved"]
    print(f"=== flagged OVER contracts (open >= {FLAG_MIN_PRICE}, divergence > {FLAG_MIN_DIVERGENCE}) ===")
    print(f"{'contract_id':<40}{'team':<22}{'open':>6}{'model':>7}{'div':>7}  {'tier':<4} {'resolved'}")
    for r in flagged.select(cols).sort("divergence", descending=True).iter_rows():
        cid, team, open_p, model_p, div, tier, res = r
        print(f"{cid:<40}{team:<22}{open_p:>6.2f}{model_p:>7.3f}{div:>+7.3f}  {tier:<4} {res}")

    n = flagged.height
    if n:
        hit = (flagged["resolved"] == "no").sum()
        print(f"\nhit rate: {hit}/{n} = {hit / n * 100:.1f}% (short was correct)")
    else:
        print("\nno flagged contracts")

    if args.out:
        flagged.select(cols).write_csv(args.out)
        print(f"flagged saved to {args.out}")


def _selftest() -> None:
    contracts = pl.DataFrame(
        {
            "contract_id": ["c1", "c2", "c3", "c4"],
            "match_id": [1, 2, 3, 4],
            "team": ["Team A", "Team B", "Team A", "Team X"],
            "open_price": [0.95, 0.60, 0.92, 0.93],
            "resolved": ["no", "yes", "yes", "no"],
        }
    )
    predictions = pl.DataFrame(
        {
            "match_id": [1, 2, 3, 4],
            "team_a": ["Team A", "Team A", "Team A", "Team A"],
            "team_b": ["Team B", "Team B", "Team B", "Team B"],
            "tier": ["T1", "T2", "T3", "T1"],
            "prob_team_a": [0.70, 0.70, 0.70, 0.70],
        }
    )
    scored = score(contracts, predictions)
    # c1: team A == team_a -> model 0.70, div 0.25 OVER
    # c2: team B == team_b -> model 0.30, div 0.30 OVER
    # c3: team A == team_a -> model 0.70, div 0.22 OVER
    # c4: Team X matches neither -> dropped
    assert scored.height == 3, scored
    assert scored.filter(pl.col("contract_id") == "c1")["divergence"][0] > 0, scored
    assert abs(scored.filter(pl.col("contract_id") == "c2")["model_prob"][0] - 0.30) < 1e-9

    fl = flags(scored)
    assert fl.height == 2, fl  # c1 (0.95) and c3 (0.92); c2 open 0.60 below threshold
    hit = (fl["resolved"] == "no").sum()
    assert hit == 1  # only c1 resolved 'no'
    print("selftest OK")


if __name__ == "__main__":
    main()
