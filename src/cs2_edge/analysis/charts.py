"""Generate web-ready interactive charts from backtest and live alert data.

Single CLI entrypoint:
    uv run python -m cs2_edge.analysis.charts
    uv run python -m cs2_edge.analysis.charts --chart equity_curve

Each chart is written as a standalone HTML file (plotly.js inlined, dark
theme, mobile responsive) under data/charts/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import plotly.graph_objects as go
import polars as pl

from cs2_edge.analysis.backtester import apply_pnl
from cs2_edge.analysis.bias_test import BIN_LABELS, analyze
from cs2_edge.analysis.divergence_scorer import flags, load_contracts, score
from cs2_edge.db.db_init import DEFAULT_DB_PATH, init_db

GREEN = "#00cc96"
RED = "#ef553b"
GREY = "#7f7f7f"

CHART_NAMES = (
    "equity_curve",
    "bias_curve",
    "monthly_pnl",
    "tier_performance",
    "live_alerts",
    "signal_distribution",
)

SOURCES = {
    "equity_curve": "Source: backtester trade log (flat $5 NO position)",
    "bias_curve": "Source: kalshi_contracts resolved data",
    "monthly_pnl": "Source: backtester trade log",
    "tier_performance": "Source: backtester trade log",
    "live_alerts": "Source: live_alerts (Kalshi live scanner)",
    "signal_distribution": "Source: backtester trade log + live_alerts",
}


def _style(fig: go.Figure, source: str) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        autosize=True,
        margin=dict(l=50, r=20, t=70, b=60),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.add_annotation(
        text=source,
        xref="paper",
        yref="paper",
        x=0,
        y=-0.16,
        showarrow=False,
        font=dict(size=11, color="#8b8b8b"),
    )
    return fig


def _write(fig: go.Figure, path: Path, source: str) -> None:
    _style(fig, source)
    html = fig.to_html(
        full_html=True,
        include_plotlyjs=True,
        config={"responsive": True, "displayModeBar": False},
    )
    html = html.replace(
        "<head>",
        '<head>\n<meta name="viewport" content="width=device-width, initial-scale=1">',
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html)
    print(f"saved {path}")


def build_equity_curve(flagged: pl.DataFrame) -> go.Figure:
    f = flagged.sort("resolution_date")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=f["resolution_date"].to_list(),
            y=f["pnl"].cum_sum().to_list(),
            mode="lines",
            name="cumulative PnL",
            line=dict(color=GREEN, width=2),
            fill="tozeroy",
            fillcolor="rgba(0,204,150,0.12)",
        )
    )
    fig.add_hline(y=0, line=dict(color="#555555", width=1))
    fig.update_layout(
        title="Equity Curve",
        xaxis_title="date",
        yaxis_title="cumulative PnL ($)",
    )
    return fig


def build_bias_curve(bias: pl.DataFrame) -> go.Figure:
    labels = [BIN_LABELS[int(b)] for b in bias["bin"]]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=bias["avg_open_price"].to_list(),
            mode="lines+markers",
            name="market price",
            line=dict(color="#636efa", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=bias["win_rate"].to_list(),
            mode="lines+markers",
            name="actual win rate",
            line=dict(color=RED, width=2),
        )
    )
    fig.update_layout(
        title="Bias Curve: Market Price vs Actual Win Rate",
        xaxis_title="market price bin",
        yaxis_title="rate",
        yaxis=dict(range=[0, 1], tickformat=".0%"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def build_monthly_pnl(flagged: pl.DataFrame) -> go.Figure:
    monthly = (
        flagged.with_columns(month=pl.col("resolution_date").dt.strftime("%Y-%m"))
        .group_by("month")
        .agg(pnl=pl.col("pnl").sum())
        .sort("month")
    )
    months = monthly["month"].to_list()
    pnl = monthly["pnl"].to_list()
    colors = [GREEN if v >= 0 else RED for v in pnl]
    fig = go.Figure(go.Bar(x=months, y=pnl, marker_color=colors))
    fig.update_layout(
        title="Monthly PnL",
        xaxis_title="month",
        yaxis_title="PnL ($)",
    )
    return fig


def build_tier_performance(flagged: pl.DataFrame) -> go.Figure:
    tiers: list[str] = []
    win_rates: list[float] = []
    counts: list[int] = []
    for tier in ("T1", "T2", "T3"):
        sub = flagged.filter(pl.col("tier") == tier)
        if sub.height == 0:
            continue
        tiers.append(tier)
        win_rates.append(float((sub["resolved"] == "no").mean()))
        counts.append(sub.height)
    fig = go.Figure(
        go.Bar(
            x=tiers,
            y=win_rates,
            marker_color="#636efa",
            text=[f"{w:.0%} (n={n})" for w, n in zip(win_rates, counts)],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Win Rate by Tier",
        xaxis_title="tier",
        yaxis_title="win rate",
        yaxis=dict(range=[0, 1], tickformat=".0%"),
    )
    return fig


def _row_color(resolved: str | None) -> str:
    if resolved == "no":
        return "rgba(0,204,150,0.20)"
    if resolved == "yes":
        return "rgba(239,85,59,0.20)"
    return "rgba(127,127,127,0.12)"


def build_live_alerts(alerts: pl.DataFrame) -> go.Figure:
    rows = alerts.head(20)
    n = rows.height
    resolved = rows["resolved"].to_list()
    dates = [
        (a.strftime("%Y-%m-%d %H:%M") if a is not None else "")
        for a in rows["alerted_at"].to_list()
    ]
    resolved_labels = [("no" if r == "no" else "yes" if r == "yes" else "—") for r in resolved]
    row_fill = [_row_color(r) for r in resolved]

    fig = go.Figure(
        go.Table(
            header=dict(
                values=["date", "team", "open_price", "model_prob", "divergence", "tier", "resolved"],
                fill_color="#2b2b2b",
                font=dict(color="#ffffff", size=13),
                align="left",
                line_color="#444444",
                height=32,
            ),
            cells=dict(
                values=[
                    dates,
                    rows["team"].to_list(),
                    [f"{v:.3f}" for v in rows["open_price"].to_list()],
                    [f"{v:.3f}" for v in rows["model_prob"].to_list()],
                    [f"{v:+.3f}" for v in rows["divergence"].to_list()],
                    rows["tier"].to_list(),
                    resolved_labels,
                ],
                fill_color=[row_fill] * 7,
                font=dict(color="#e0e0e0", size=12),
                align="left",
                line_color="#444444",
                height=30,
            ),
        )
    )
    fig.update_layout(
        title=f"Live Alerts Feed (last {n})",
        margin=dict(l=10, r=10, t=70, b=10),
    )
    return fig


def build_signal_distribution(flagged: pl.DataFrame, alerts: pl.DataFrame) -> go.Figure:
    bt = flagged.select(["open_price", "model_prob", "resolved"]).with_columns(
        outcome=pl.when(pl.col("resolved") == "no")
        .then(pl.lit("win"))
        .otherwise(pl.lit("loss"))
    ).select(["open_price", "model_prob", "outcome"])
    lv = alerts.select(["open_price", "model_prob"]).drop_nulls().with_columns(
        outcome=pl.lit("unresolved")
    )

    fig = go.Figure()
    for outcome, color in (("win", GREEN), ("loss", RED), ("unresolved", GREY)):
        sub = pl.concat(
            [bt.filter(pl.col("outcome") == outcome), lv.filter(pl.col("outcome") == outcome)]
        )
        if sub.height == 0:
            continue
        fig.add_trace(
            go.Scatter(
                x=sub["open_price"].to_list(),
                y=sub["model_prob"].to_list(),
                mode="markers",
                name=outcome,
                marker=dict(color=color, size=8, opacity=0.7),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="no edge (y=x)",
            line=dict(color="#555555", width=1, dash="dash"),
        )
    )
    fig.update_layout(
        title="Signal Distribution: Open Price vs Model Probability",
        xaxis_title="open_price",
        yaxis_title="model_prob",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def load_data(args: argparse.Namespace) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    con = init_db(args.db)
    contracts = load_contracts(con)
    bias = con.execute(
        "SELECT open_price, resolved FROM kalshi_contracts "
        "WHERE resolved IN ('yes', 'no') AND open_price IS NOT NULL"
    ).pl()
    alerts = con.execute(
        "SELECT a.team, a.open_price, a.model_prob, a.divergence, a.tier, "
        "a.alerted_at, c.resolved "
        "FROM live_alerts a LEFT JOIN kalshi_contracts c ON c.contract_id = a.contract_id "
        "ORDER BY a.alerted_at DESC"
    ).pl()
    con.close()

    predictions = pl.read_csv(args.pred)
    flagged = apply_pnl(flags(score(contracts, predictions))).sort("resolution_date")
    return flagged, analyze(bias), alerts


def generate(name: str, data: tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame], path: Path) -> None:
    flagged, bias, alerts = data
    if name == "equity_curve":
        fig = build_equity_curve(flagged)
    elif name == "bias_curve":
        fig = build_bias_curve(bias)
    elif name == "monthly_pnl":
        fig = build_monthly_pnl(flagged)
    elif name == "tier_performance":
        fig = build_tier_performance(flagged)
    elif name == "live_alerts":
        fig = build_live_alerts(alerts)
    elif name == "signal_distribution":
        fig = build_signal_distribution(flagged, alerts)
    else:
        raise ValueError(f"unknown chart: {name}")
    _write(fig, path, SOURCES[name])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate interactive charts from backtest and live data")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Path to DuckDB file")
    parser.add_argument("--pred", default="data/win_prob_predictions_all.csv", help="Model predictions CSV")
    parser.add_argument("--out-dir", default="data/charts", help="Output directory for HTML files")
    parser.add_argument("--chart", choices=CHART_NAMES, help="Generate a single chart")
    parser.add_argument("--selftest", action="store_true", help="Run self-check and exit")
    args = parser.parse_args()

    if args.selftest:
        _selftest()
        return

    data = load_data(args)
    names = [args.chart] if args.chart else list(CHART_NAMES)
    for name in names:
        generate(name, data, Path(args.out_dir) / f"{name}.html")


def _selftest() -> None:
    from datetime import date, datetime

    flagged = pl.DataFrame(
        {
            "open_price": [0.95, 0.91, 0.93, 0.94],
            "model_prob": [0.50, 0.40, 0.45, 0.50],
            "resolved": ["no", "yes", "no", "no"],
            "tier": ["T1", "T2", "T1", "T3"],
            "pnl": [4.0, -5.0, 3.5, 2.0],
            "resolution_date": [
                date(2026, 1, 1),
                date(2026, 1, 2),
                date(2026, 2, 1),
                date(2026, 2, 5),
            ],
        }
    )
    bias = analyze(
        pl.DataFrame(
            {"open_price": [0.05, 0.05, 0.15, 0.95], "resolved": ["yes", "no", "yes", "yes"]}
        )
    )
    alerts = pl.DataFrame(
        {
            "team": ["Vitality", "FaZe"],
            "open_price": [0.905, 0.920],
            "model_prob": [0.460, 0.500],
            "divergence": [0.445, 0.420],
            "tier": ["T3", "T1"],
            "alerted_at": [datetime(2026, 1, 1, 12, 0), datetime(2026, 1, 2, 12, 0)],
            "resolved": ["no", None],
        }
    )

    import tempfile

    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        data = (flagged, bias, alerts)
        for name in CHART_NAMES:
            generate(name, data, out / f"{name}.html")
        assert len(list(out.glob("*.html"))) == len(CHART_NAMES)
    print("selftest OK")


if __name__ == "__main__":
    main()
