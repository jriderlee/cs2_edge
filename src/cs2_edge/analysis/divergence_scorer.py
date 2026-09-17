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

FLAG_MIN_PRICE = 0.90  # entry floor
FLAG_MAX_PRICE = 0.97  # above this behaves like a forfeit / in-progress match
FLAG_MIN_DIVERGENCE = 0.30  # confirmed edge concentrates here (backtest)


def load_contracts(con) -> pl.DataFrame:
    """Resolved, matched contracts with open_price, match_start_ts, and first-candle ts."""
    return con.execute(
        "SELECT c.contract_id, c.match_id, c.team, c.open_price, c.resolved, "
        "c.resolution_date, c.match_start_ts, MIN(k.end_period_ts) AS open_ts "
        "FROM kalshi_contracts c "
        "LEFT JOIN kalshi_candles k ON k.contract_id = c.contract_id "
        "WHERE c.resolved IN ('yes', 'no') AND c.open_price IS NOT NULL AND c.match_id IS NOT NULL "
        "GROUP BY c.contract_id, c.match_id, c.team, c.open_price, c.resolved, "
        "c.resolution_date, c.match_start_ts"
    ).pl()


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


def flags(scored: pl.DataFrame, prematch: bool = True) -> pl.DataFrame:
    f = scored.filter(
        (pl.col("open_price") >= FLAG_MIN_PRICE)
        & (pl.col("open_price") <= FLAG_MAX_PRICE)
        & (pl.col("divergence") > FLAG_MIN_DIVERGENCE)
        & (pl.col("direction") == "OVER")
    )
    if prematch:
        # only keep contracts whose open price was set before the match started
        f = f.filter(
            pl.col("open_ts").is_not_null()
            & pl.col("match_start_ts").is_not_null()
            & (pl.col("open_ts") < pl.col("match_start_ts"))
        )
    return f


def in_play_report(flagged: pl.DataFrame) -> None:
    """Classify flagged contracts as pre-match vs in-play by first-candle vs match start."""
    total = flagged.height
    df = flagged.with_columns(delta_h=(pl.col("open_ts") - pl.col("match_start_ts")) / 3600.0)
    known = df.filter(pl.col("match_start_ts").is_not_null() & pl.col("open_ts").is_not_null())
    unknown = df.filter(pl.col("match_start_ts").is_null() | pl.col("open_ts").is_null())
    prematch = known.filter(pl.col("delta_h") < 0)
    inplay = known.filter((pl.col("delta_h") >= 0) & (pl.col("delta_h") <= 2))
    late = known.filter(pl.col("delta_h") > 2)

    def pct(n: int) -> str:
        return f"{n}/{total} ({n / total * 100:.1f}%)" if total else "0"

    print("=== in-play vs pre-match integrity check ===")
    print(f"flagged contracts: {total}")
    print(f"pre-match (open before match start): {pct(prematch.height)}")
    print(f"in-play (open within 0-2h after start): {pct(inplay.height)}")
    print(f"late (open >2h after start): {pct(late.height)}")
    print(f"unknown match start: {pct(unknown.height)}")
    print("\nunknown match start by open_price:")
    for lo, hi, label in [(0.90, 0.93, "0.90-0.93"), (0.93, 0.95, "0.93-0.95"), (0.95, 0.97, "0.95-0.97"), (0.97, 1.0, "0.97+")]:
        sub = unknown.filter((pl.col("open_price") >= lo) & (pl.col("open_price") < hi))
        print(f"  {label}: {sub.height}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Market-vs-model divergence scorer")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Path to DuckDB file")
    parser.add_argument(
        "--pred", default="data/win_prob_predictions_all.csv", help="Model predictions CSV"
    )
    parser.add_argument("--out", default=None, help="CSV path for flagged contracts")
    parser.add_argument("--integrity", action="store_true", help="In-play vs pre-match report and exit")
    parser.add_argument("--selftest", action="store_true", help="Run self-check and exit")
    args = parser.parse_args()

    if args.selftest:
        _selftest()
        return

    con = init_db(args.db)
    contracts = load_contracts(con)
    con.close()

    predictions = pl.read_csv(args.pred)
    scored = score(contracts, predictions)

    if args.integrity:
        in_play_report(flags(scored, prematch=False))
        return

    flagged = flags(scored)

    print(f"joined contracts: {scored.height}")
    print(f"OVER / UNDER: {(scored['direction'] == 'OVER').sum()} / {(scored['direction'] == 'UNDER').sum()}\n")

    cols = ["contract_id", "team", "open_price", "model_prob", "divergence", "tier", "resolved"]
    print(f"=== flagged OVER contracts ({FLAG_MIN_PRICE} <= open <= {FLAG_MAX_PRICE}, divergence > {FLAG_MIN_DIVERGENCE}, pre-match) ===")
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
            "contract_id": ["c1", "c2", "c3", "c4", "c5"],
            "match_id": [1, 2, 3, 4, 5],
            "team": ["Team A", "Team B", "Team A", "Team X", "Team A"],
            "open_price": [0.95, 0.60, 0.92, 0.93, 0.98],
            "resolved": ["no", "yes", "yes", "no", "no"],
            "open_ts": [100, 100, 150, 100, 100],
            "match_start_ts": [200, 200, 100, 200, 200],
        }
    )
    predictions = pl.DataFrame(
        {
            "match_id": [1, 2, 3, 4, 5],
            "team_a": ["Team A", "Team A", "Team A", "Team A", "Team A"],
            "team_b": ["Team B", "Team B", "Team B", "Team B", "Team B"],
            "tier": ["T1", "T2", "T3", "T1", "T1"],
            "prob_team_a": [0.50, 0.50, 0.50, 0.50, 0.50],
        }
    )
    scored = score(contracts, predictions)
    # c1: team A == team_a -> model 0.50, div 0.45 OVER (pre-match)
    # c2: team B == team_b -> model 0.50, div 0.10 OVER (open 0.60 below floor)
    # c3: team A == team_a -> model 0.50, div 0.42 OVER (in-play: open_ts >= start)
    # c4: Team X matches neither -> dropped
    # c5: team A, open 0.98 above max -> excluded
    assert scored.height == 4, scored
    assert scored.filter(pl.col("contract_id") == "c1")["divergence"][0] > 0, scored
    assert abs(scored.filter(pl.col("contract_id") == "c2")["model_prob"][0] - 0.50) < 1e-9

    fl_all = flags(scored, prematch=False)
    assert fl_all.height == 2, fl_all  # c1 and c3

    fl = flags(scored)  # default: pre-match only
    assert fl.height == 1, fl  # only c1 (pre-match); c3 dropped (in-play)
    assert fl["resolved"][0] == "no"
    print("selftest OK")


if __name__ == "__main__":
    main()
