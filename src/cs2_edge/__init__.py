"""CS2 Edge: CS2 win-probability model vs Kalshi prediction markets."""

COMMANDS = [
    ("cs2_edge.db.db_init", "initialize the DuckDB database"),
    ("cs2_edge.db.migrate_backfill_tier", "one-shot: populate match_results.tier"),
    ("cs2_edge.collectors.hltv", "collect HLTV results/ratings (--roster, --maps)"),
    ("cs2_edge.collectors.kalshi", "collect Kalshi contracts + candles"),
    ("cs2_edge.models.win_probability", "train + evaluate the win-probability model"),
    ("cs2_edge.analysis.backtester", "simulate the overpriced-favorite short"),
    ("cs2_edge.analysis.bias_test", "open price vs realized win rate"),
    ("cs2_edge.analysis.divergence_scorer", "model probability vs market price"),
    ("cs2_edge.analysis.correction_speed", "mispricing correction speed"),
    ("cs2_edge.analysis.trade_frequency", "90-100% contract frequency"),
    ("cs2_edge.analysis.charts", "build interactive HTML charts"),
    ("cs2_edge.monitor.live_scanner", "hourly scanner + Discord alerts"),
]


def main() -> None:
    print("cs2-edge — CS2 win-probability model vs Kalshi markets\n")
    print("usage: python -m <module> [options]\n")
    print("modules:")
    for mod, desc in COMMANDS:
        print(f"  {mod:<38} {desc}")
