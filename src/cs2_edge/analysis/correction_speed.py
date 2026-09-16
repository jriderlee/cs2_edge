"""Price correction speed + early liquidity by open-price bin.

For each contract: how many candles until price is within 5% of close, and how
much tradeable volume sits in the first 3 candles (i.e. is the opening
mispricing actually accessible before it corrects).
"""

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

# "within 5%" = 5 percentage points (not relative), for probabilities in [0, 1]
THRESHOLD = 0.05


def analyze(contracts: pl.DataFrame, candles: pl.DataFrame) -> pl.DataFrame:
    frame = (
        candles.join(contracts.select(["contract_id", "open_price", "close_price"]), on="contract_id")
        .sort("end_period_ts")
        .with_columns(candle_idx=pl.col("end_period_ts").cum_count().over("contract_id") - 1)
    )

    # first candle (0-based) within threshold of close
    hit = frame.filter(
        pl.col("price").is_not_null()
        & ((pl.col("price") - pl.col("close_price")).abs() <= THRESHOLD)
    )
    corr = hit.group_by("contract_id").agg(pl.col("candle_idx").min().alias("correction_time"))

    first3 = frame.filter(pl.col("candle_idx") < 3)
    first3_vol = first3.group_by("contract_id").agg(
        first3_volume=pl.col("volume").sum()
    )
    vwap = first3.filter(pl.col("volume") > 0).group_by("contract_id").agg(
        vwap_first3=(pl.col("price") * pl.col("volume")).sum() / pl.col("volume").sum()
    )
    avg_vol = frame.group_by("contract_id").agg(avg_volume=pl.col("volume").mean())

    per = (
        first3_vol.join(vwap, on="contract_id", how="left")
        .join(avg_vol, on="contract_id", how="left")
        .join(corr, on="contract_id", how="left")
    )

    df = contracts.join(per, on="contract_id", how="inner").with_columns(
        bin=(pl.col("open_price") * 10).floor().clip(0, 9).cast(pl.Int8),
        first3_zero=(pl.col("first3_volume") == 0),
        vwap_vs_open=pl.col("vwap_first3") - pl.col("open_price"),
    )

    return (
        df.group_by("bin")
        .agg(
            n=pl.len(),
            avg_open_price=pl.col("open_price").mean(),
            avg_first3_volume=pl.col("first3_volume").mean(),
            avg_volume=pl.col("avg_volume").mean(),
            avg_vwap_first3=pl.col("vwap_first3").mean(),
            avg_vwap_vs_open=pl.col("vwap_vs_open").mean(),
            zero_first3_pct=pl.col("first3_zero").mean() * 100,
            avg_correction_time=pl.col("correction_time").mean(),
        )
        .sort("bin")
    )


def _fmt(v: float | None, width: int, dec: int, sign: bool = False) -> str:
    if v is None:
        return "n/a".rjust(width)
    return format(v, f"{'+' if sign else ''}{width}.{dec}f")


def print_table(agg: pl.DataFrame) -> None:
    hdr = (
        f"{'bin':<10}{'n':>6}{'price':>8}{'first3vol':>11}{'avgvol':>10}"
        f"{'vwap3':>8}{'vwap-open':>10}{'%zero3':>8}{'candles':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for bin_, n, price, f3v, avgv, vwap3, vwap_open, zero_pct, candles in agg.iter_rows():
        print(
            f"{BIN_LABELS[bin_]:<10}{n:>6}{_fmt(price, 8, 3)}{_fmt(f3v, 11, 1)}"
            f"{_fmt(avgv, 10, 1)}{_fmt(vwap3, 8, 3)}{_fmt(vwap_open, 10, 3, sign=True)}"
            f"{_fmt(zero_pct, 7, 1)}%{_fmt(candles, 8, 2)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Price correction speed + early liquidity")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Path to DuckDB file")
    parser.add_argument(
        "--plot", default=None, help="Output path for price-correction path plot (0-10% vs 90-100%)"
    )
    parser.add_argument("--selftest", action="store_true", help="Run analysis self-check and exit")
    args = parser.parse_args()

    if args.selftest:
        _selftest()
        return

    con = init_db(args.db)
    contracts = con.execute(
        "SELECT contract_id, open_price, close_price FROM kalshi_contracts "
        "WHERE resolved IN ('yes', 'no') AND open_price IS NOT NULL AND close_price IS NOT NULL"
    ).pl()
    candles = con.execute(
        "SELECT contract_id, end_period_ts, price, volume FROM kalshi_candles"
    ).pl()
    con.close()

    print_table(analyze(contracts, candles))
    if args.plot:
        plot_price_path(contracts, candles, [0, 9], Path(args.plot))


def _bin_expr() -> pl.Expr:
    return (pl.col("open_price") * 10).floor().clip(0, 9).cast(pl.Int8)


def price_path(contracts: pl.DataFrame, candles: pl.DataFrame, bins: list[int]) -> pl.DataFrame:
    """Mean forward-filled price per candle index, for contracts in the given bins."""
    with_bin = contracts.with_columns(bin=_bin_expr()).select(
        ["contract_id", "bin", "close_price"]
    )
    df = (
        candles.join(with_bin, on="contract_id")
        .filter(pl.col("bin").is_in(bins))
        .sort("end_period_ts")
    )
    first = (
        df.filter(pl.col("price").is_not_null())
        .group_by("contract_id")
        .agg(pl.col("end_period_ts").min().alias("first_ts"))
    )
    df = (
        df.join(first, on="contract_id")
        .filter(pl.col("end_period_ts") >= pl.col("first_ts"))
        .with_columns(
            candle_idx=pl.col("end_period_ts").cum_count().over("contract_id") - 1,
            price=pl.col("price").forward_fill().over("contract_id"),
        )
    )
    return (
        df.filter(pl.col("price").is_not_null())
        .group_by(["bin", "candle_idx"])
        .agg(mean_price=pl.col("price").mean(), n=pl.len())
        .sort(["bin", "candle_idx"])
    )


def plot_price_path(
    contracts: pl.DataFrame, candles: pl.DataFrame, bins: list[int], out_path: Path
) -> None:
    path = price_path(contracts, candles, bins)
    close_by_bin = (
        contracts.with_columns(bin=_bin_expr())
        .filter(pl.col("bin").is_in(bins))
        .group_by("bin")
        .agg(avg_close=pl.col("close_price").mean())
    )
    colors = {0: "tab:blue", 9: "tab:red"}

    fig, ax = plt.subplots(figsize=(8, 5))
    for b in bins:
        sub = path.filter(pl.col("bin") == b)
        close = close_by_bin.filter(pl.col("bin") == b)
        if sub.height == 0:
            continue
        ax.plot(
            sub["candle_idx"], sub["mean_price"], "o-",
            color=colors.get(b), label=BIN_LABELS[b],
        )
        if close.height:
            ax.axhline(
                close["avg_close"][0], color=colors.get(b), linestyle="--", alpha=0.5
            )
    ax.set_xlabel("candles since first trade (1h each)")
    ax.set_ylabel("mean price")
    ax.set_ylim(0, 1)
    ax.set_title("Price correction path by opening-price bin")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"price path saved to {out_path}")


def _selftest() -> None:
    contracts = pl.DataFrame(
        {
            "contract_id": ["a", "b"],
            "open_price": [0.1, 0.5],
            "close_price": [0.7, 0.5],
        }
    )
    candles = pl.DataFrame(
        {
            "contract_id": ["a", "a", "a", "a", "b", "b"],
            "end_period_ts": [0, 1, 2, 3, 0, 1],
            "price": [0.1, 0.3, 0.6, 0.7, 0.5, 0.5],
            "volume": [10.0, 20.0, 0.0, 5.0, 30.0, 0.0],
        }
    )
    agg = analyze(contracts, candles)
    by_bin = {r[0]: r for r in agg.iter_rows()}

    # contract a: open 0.1 -> bin 1; first3 vol=30, vwap=(0.1*10+0.3*20)/30=0.2333, corr time=3
    a = by_bin[1]
    assert a[1] == 1, a
    assert abs(a[3] - 30.0) < 1e-9, a
    assert abs(a[5] - 0.233333) < 1e-4, a
    assert abs(a[6] - (0.233333 - 0.1)) < 1e-4, a
    assert a[7] == 0.0, a
    assert abs(a[8] - 3.0) < 1e-9, a

    # contract b: open 0.5 -> bin 5; first3 vol=30, vwap=0.5, corr time=0
    b = by_bin[5]
    assert b[1] == 1, b
    assert abs(b[5] - 0.5) < 1e-9, b
    assert abs(b[8] - 0.0) < 1e-9, b
    print("selftest OK")


if __name__ == "__main__":
    main()
