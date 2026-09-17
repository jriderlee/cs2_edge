"""Independent CS2 win-probability model built from HLTV data only.

No Kalshi / market-price features. Labels come from HLTV match results; all
predictor features are computed from past HLTV data only (no lookahead).
Validated with log loss and Brier score, overall and per tournament tier.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from datetime import date
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from cs2_edge.db.db_init import DEFAULT_DB_PATH, init_db

T1_KEYWORDS = ("iem", "blast", "pgl", "major", "pro league")
T2_KEYWORDS = ("challenger", "cct", "thunderpick")

WINDOWS = (30, 60, 90)

ACTIVE_MAPS = ("Mirage", "Inferno", "Nuke", "Ancient", "Anubis", "Dust2", "Vertigo")
MAP_WINDOW_DAYS = 730


def infer_tier(event_name: str | None) -> str:
    """Name-based tournament tier heuristic."""
    if not event_name:
        return "T3"
    n = event_name.lower()
    if "qualifier" in n:
        return "T3"
    if any(k in n for k in T1_KEYWORDS):
        return "T1"
    if any(k in n for k in T2_KEYWORDS):
        return "T2"
    return "T3"


def infer_bo(best_of: int) -> str:
    if best_of == 5:
        return "BO5"
    if best_of == 3:
        return "BO3"
    return "BO1"  # best_of 1 (walkover) or >5 (BO1 round scores)


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


def load_roster(con) -> dict[str, list[dict]]:
    """team -> sorted roster-change events (change_date, from_team, to_team, player_id)."""
    roster: dict[str, list[dict]] = defaultdict(list)
    for pid, ft, tt, cd in con.execute(
        "SELECT player_id, from_team, to_team, change_date FROM roster_changes"
    ).fetchall():
        e = {"player_id": pid, "from_team": ft, "to_team": tt, "change_date": cd}
        if ft:
            roster[ft].append(e)
        if tt:
            roster[tt].append(e)
    for k in roster:
        roster[k].sort(key=lambda e: e["change_date"])
    return roster


def load_player_rating(con) -> dict[int, float]:
    return {
        pid: r
        for pid, r in con.execute(
            "SELECT player_id, AVG(rating) FROM player_ratings GROUP BY player_id"
        ).fetchall()
    }


def load_match_maps(con) -> dict[int, list[tuple[str, str]]]:
    """match_id -> list[(map_name, winner)]."""
    out: dict[int, list[tuple[str, str]]] = defaultdict(list)
    for mid, mn, w in con.execute(
        "SELECT match_id, map_name, winner FROM match_maps"
    ).fetchall():
        out[mid].append((mn, w))
    return out


def _team_results(matches: pl.DataFrame) -> dict[str, list[tuple[date, bool]]]:
    tr: dict[str, list[tuple[date, bool]]] = defaultdict(list)
    for m in matches.iter_rows(named=True):
        a, b, w = m["team_a"], m["team_b"], m["winner"]
        tr[a].append((m["match_date"], w == a))
        tr[b].append((m["match_date"], w == b))
    return tr


def _mean(xs: list) -> float | None:
    return (sum(xs) / len(xs)) if xs else None


def _map_winrate(team: str, d: date, map_hist: dict) -> float | None:
    rates = []
    for m in ACTIVE_MAPS:
        dq = map_hist.get(team, {}).get(m)
        if not dq:
            continue
        while dq and (d - dq[0][0]).days > MAP_WINDOW_DAYS:
            dq.popleft()
        if dq:
            rates.append(sum(1 for _, w in dq if w) / len(dq))
    return _mean(rates) if rates else None


def _roster_features(
    team: str,
    d: date,
    roster: dict,
    player_rating: dict,
    team_results: dict,
) -> dict:
    """Per-team roster-change signals, all past-only (no lookahead)."""
    past = [e for e in roster.get(team, []) if e["change_date"] <= d]
    had_30d = any(0 <= (d - e["change_date"]).days <= 30 for e in past)

    added = [
        player_rating[e["player_id"]]
        for e in past
        if e["to_team"] == team and e["player_id"] in player_rating
    ]
    removed = [
        player_rating[e["player_id"]]
        for e in past
        if e["from_team"] == team and e["player_id"] in player_rating
    ]
    ma = _mean(added)
    mr = _mean(removed)
    if ma is not None and mr is not None:
        rating_delta = ma - mr
    elif ma is not None:
        rating_delta = ma
    elif mr is not None:
        rating_delta = -mr
    else:
        rating_delta = None

    games_since = None
    wr_before = wr_after = None
    if past:
        cd = past[-1]["change_date"]
        tr = team_results.get(team, [])
        games_since = sum(1 for md, _ in tr if cd < md <= d)
        before = [won for md, won in tr if 0 <= (cd - md).days <= 30]
        after = [won for md, won in tr if 0 < (md - cd).days <= 30 and md <= d]
        wr_before = _mean(before) if before else None
        wr_after = _mean(after) if after else None

    return {
        "rating_delta": rating_delta,
        "wr_before": wr_before,
        "wr_after": wr_after,
        "games_since": games_since,
        "had_change_30d": int(had_30d),
    }


def _team_features(
    win_dq: deque, rating_dq: deque, d: date
) -> tuple[dict[int, float], float | None]:
    while win_dq and (d - win_dq[0][0]).days > WINDOWS[-1]:
        win_dq.popleft()
    while rating_dq and (d - rating_dq[0][0]).days > 30:
        rating_dq.popleft()

    wr: dict[int, float] = {}
    for n in WINDOWS:
        wins = tot = 0
        for md, won in win_dq:
            if (d - md).days <= n:
                tot += 1
                wins += int(won)
        wr[n] = (wins / tot) if tot else 0.5

    ratings = [r for _, r in rating_dq]
    return wr, (sum(ratings) / len(ratings) if ratings else None)


def _h2h(h2h_dq: deque, d: date, a: str, b: str) -> tuple[int, int]:
    while h2h_dq and (d - h2h_dq[0][0]).days > 730:
        h2h_dq.popleft()
    a_wins = sum(1 for _, w in h2h_dq if w == a)
    b_wins = sum(1 for _, w in h2h_dq if w == b)
    return a_wins, b_wins


def compute_features(
    matches: pl.DataFrame,
    team_rating: dict,
    roster: dict,
    player_rating: dict,
    match_maps: dict,
) -> pl.DataFrame:
    """Rolling past-only features; team_a is the reference team (label = team_a wins)."""
    win_hist: dict[str, deque] = defaultdict(deque)
    rating_hist: dict[str, deque] = defaultdict(deque)
    h2h_hist: dict[frozenset, deque] = defaultdict(deque)
    map_hist: dict[str, dict[str, deque]] = defaultdict(lambda: defaultdict(deque))
    team_results = _team_results(matches)

    rows: list[dict] = []
    for m in matches.iter_rows(named=True):
        d = m["match_date"]
        a, b, winner = m["team_a"], m["team_b"], m["winner"]

        fa, ra = _team_features(win_hist[a], rating_hist[a], d)
        fb, rb = _team_features(win_hist[b], rating_hist[b], d)
        a_wins, b_wins = _h2h(h2h_hist[frozenset((a, b))], d, a, b)
        ma = _map_winrate(a, d, map_hist)
        mb = _map_winrate(b, d, map_hist)
        rf_a = _roster_features(a, d, roster, player_rating, team_results)
        rf_b = _roster_features(b, d, roster, player_rating, team_results)

        rows.append(
            {
                "match_id": m["match_id"],
                "match_date": d,
                "team_a": a,
                "team_b": b,
                "tier": m["tier"],
                "bo": m["bo"],
                "y": int(winner == a),
                "wr30_diff": fa[30] - fb[30],
                "wr60_diff": fa[60] - fb[60],
                "wr90_diff": fa[90] - fb[90],
                "rating30_diff": (ra - rb) if (ra is not None and rb is not None) else None,
                "h2h_diff": a_wins - b_wins,
                "h2h_count": a_wins + b_wins,
                "map_winrate_a": ma,
                "map_winrate_diff": (ma - mb) if (ma is not None and mb is not None) else None,
                "rating_delta_diff": (rf_a["rating_delta"] - rf_b["rating_delta"])
                if (rf_a["rating_delta"] is not None and rf_b["rating_delta"] is not None)
                else None,
                "roster_wr_before_diff": (rf_a["wr_before"] - rf_b["wr_before"])
                if (rf_a["wr_before"] is not None and rf_b["wr_before"] is not None)
                else None,
                "roster_wr_after_diff": (rf_a["wr_after"] - rf_b["wr_after"])
                if (rf_a["wr_after"] is not None and rf_b["wr_after"] is not None)
                else None,
                "games_since_change_diff": (rf_a["games_since"] - rf_b["games_since"])
                if (rf_a["games_since"] is not None and rf_b["games_since"] is not None)
                else None,
                "had_change_30d_diff": rf_a["had_change_30d"] - rf_b["had_change_30d"],
            }
        )

        a_won = winner == a
        win_hist[a].append((d, a_won))
        win_hist[b].append((d, not a_won))
        ra_cur = team_rating.get((m["match_id"], a))
        rb_cur = team_rating.get((m["match_id"], b))
        if ra_cur is not None:
            rating_hist[a].append((d, ra_cur))
        if rb_cur is not None:
            rating_hist[b].append((d, rb_cur))
        h2h_hist[frozenset((a, b))].append((d, winner))
        for mn, mw in match_maps.get(m["match_id"], []):
            if mw == a:
                map_hist[a][mn].append((d, True))
                map_hist[b][mn].append((d, False))
            elif mw == b:
                map_hist[b][mn].append((d, True))
                map_hist[a][mn].append((d, False))

    return pl.DataFrame(
        rows,
        schema={
            "match_id": pl.Int64,
            "match_date": pl.Date,
            "team_a": pl.Utf8,
            "team_b": pl.Utf8,
            "tier": pl.Utf8,
            "bo": pl.Utf8,
            "y": pl.Int64,
            "wr30_diff": pl.Float64,
            "wr60_diff": pl.Float64,
            "wr90_diff": pl.Float64,
            "rating30_diff": pl.Float64,
            "h2h_diff": pl.Float64,
            "h2h_count": pl.Int64,
            "map_winrate_a": pl.Float64,
            "map_winrate_diff": pl.Float64,
            "rating_delta_diff": pl.Float64,
            "roster_wr_before_diff": pl.Float64,
            "roster_wr_after_diff": pl.Float64,
            "games_since_change_diff": pl.Int64,
            "had_change_30d_diff": pl.Int64,
        },
    )


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def _feature_frame(df: pl.DataFrame) -> tuple[list[str], pl.DataFrame]:
    df = df.with_columns(
        [
            (pl.col("bo") == "BO3").cast(pl.Int8).alias("bo_BO3"),
            (pl.col("bo") == "BO5").cast(pl.Int8).alias("bo_BO5"),
            (pl.col("tier") == "T2").cast(pl.Int8).alias("tier_T2"),
            (pl.col("tier") == "T3").cast(pl.Int8).alias("tier_T3"),
        ]
    )
    feats = [
        "wr30_diff",
        "wr60_diff",
        "wr90_diff",
        "rating30_diff",
        "h2h_diff",
        "h2h_count",
        "map_winrate_a",
        "map_winrate_diff",
        "rating_delta_diff",
        "roster_wr_before_diff",
        "roster_wr_after_diff",
        "games_since_change_diff",
        "had_change_30d_diff",
        "bo_BO3",
        "bo_BO5",
        "tier_T2",
        "tier_T3",
    ]
    feats = [f for f in feats if f in df.columns]
    feats = [f for f in feats if df[f].null_count() < df.height]
    return feats, df


def _fit_predict(df: pl.DataFrame, feats: list[str], rounds: int = 150) -> tuple:
    df = df.sort("match_date")
    n = df.height
    train = df[: int(n * 0.70)]
    val = df[int(n * 0.70) : int(n * 0.85)]
    test = df[int(n * 0.85) :]

    def xy(sub: pl.DataFrame):
        return sub.select(feats).to_numpy().astype(np.float32), sub["y"].to_numpy()

    X_tr, y_tr = xy(train)
    X_va, y_va = xy(val)
    X_te, y_te = xy(test)

    model = lgb.train(
        {"objective": "binary", "verbosity": -1, "seed": 0},
        lgb.Dataset(X_tr, label=y_tr),
        num_boost_round=rounds,
        valid_sets=[lgb.Dataset(X_va, label=y_va)],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
    )
    return model, X_te, y_te, test


def _next_month(d: date) -> date:
    return d.replace(year=d.year + 1, month=1, day=1) if d.month == 12 else d.replace(month=d.month + 1, day=1)


def walk_forward_predict(
    df: pl.DataFrame, feats: list[str], start: date, rounds: int = 150
) -> pl.DataFrame:
    """Expanding-window walk-forward: train on all prior data, predict each month."""
    df = df.sort("match_date")
    frames: list[pl.DataFrame] = []
    cur = start
    last = df["match_date"].max()
    while cur <= last:
        nxt = _next_month(cur)
        train = df.filter(pl.col("match_date") < cur)
        window = df.filter((pl.col("match_date") >= cur) & (pl.col("match_date") < nxt))
        if window.height:
            n = train.height
            tr, va = train[: int(n * 0.85)], train[int(n * 0.85) :]
            model = lgb.train(
                {"objective": "binary", "verbosity": -1, "seed": 0},
                lgb.Dataset(tr.select(feats).to_numpy().astype(np.float32), label=tr["y"].to_numpy()),
                num_boost_round=rounds,
                valid_sets=[
                    lgb.Dataset(va.select(feats).to_numpy().astype(np.float32), label=va["y"].to_numpy())
                ],
                callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
            )
            p = model.predict(
                window.select(feats).to_numpy().astype(np.float32),
                num_iteration=model.best_iteration,
            )
            frames.append(window.with_columns(prob_team_a=pl.Series("prob_team_a", p)))
        cur = nxt
    return pl.concat(frames) if frames else df[:0]


def build_state(
    matches: pl.DataFrame, team_rating: dict, roster: dict, match_maps: dict
) -> tuple[dict, dict, dict, dict]:
    """Roll past matches into per-team / per-pair histories for live feature lookup."""
    win_hist: dict[str, deque] = defaultdict(deque)
    rating_hist: dict[str, deque] = defaultdict(deque)
    h2h_hist: dict[frozenset, deque] = defaultdict(deque)
    map_hist: dict[str, dict[str, deque]] = defaultdict(lambda: defaultdict(deque))
    for m in matches.iter_rows(named=True):
        d, a, b, winner = m["match_date"], m["team_a"], m["team_b"], m["winner"]
        a_won = winner == a
        win_hist[a].append((d, a_won))
        win_hist[b].append((d, not a_won))
        ra = team_rating.get((m["match_id"], a))
        rb = team_rating.get((m["match_id"], b))
        if ra is not None:
            rating_hist[a].append((d, ra))
        if rb is not None:
            rating_hist[b].append((d, rb))
        h2h_hist[frozenset((a, b))].append((d, winner))
        for mn, mw in match_maps.get(m["match_id"], []):
            if mw == a:
                map_hist[a][mn].append((d, True))
                map_hist[b][mn].append((d, False))
            elif mw == b:
                map_hist[b][mn].append((d, True))
                map_hist[a][mn].append((d, False))
    return win_hist, rating_hist, h2h_hist, map_hist


def feature_vector(
    a: str,
    b: str,
    d: date,
    bo: str,
    tier: str,
    state: tuple[dict, dict, dict, dict],
    roster: dict,
    player_rating: dict,
    team_results: dict,
    feats: list[str],
) -> list[float | None]:
    """Compute the model's feature vector for a (team_a, team_b, date) from past data."""
    win_hist, rating_hist, h2h_hist, map_hist = state
    fa, ra = _team_features(win_hist[a], rating_hist[a], d)
    fb, rb = _team_features(win_hist[b], rating_hist[b], d)
    aw, bw = _h2h(h2h_hist[frozenset((a, b))], d, a, b)
    ma = _map_winrate(a, d, map_hist)
    mb = _map_winrate(b, d, map_hist)
    rf_a = _roster_features(a, d, roster, player_rating, team_results)
    rf_b = _roster_features(b, d, roster, player_rating, team_results)
    vals = {
        "wr30_diff": fa[30] - fb[30],
        "wr60_diff": fa[60] - fb[60],
        "wr90_diff": fa[90] - fb[90],
        "rating30_diff": (ra - rb) if (ra is not None and rb is not None) else None,
        "h2h_diff": aw - bw,
        "h2h_count": aw + bw,
        "map_winrate_a": ma,
        "map_winrate_diff": (ma - mb) if (ma is not None and mb is not None) else None,
        "rating_delta_diff": (rf_a["rating_delta"] - rf_b["rating_delta"])
        if (rf_a["rating_delta"] is not None and rf_b["rating_delta"] is not None)
        else None,
        "roster_wr_before_diff": (rf_a["wr_before"] - rf_b["wr_before"])
        if (rf_a["wr_before"] is not None and rf_b["wr_before"] is not None)
        else None,
        "roster_wr_after_diff": (rf_a["wr_after"] - rf_b["wr_after"])
        if (rf_a["wr_after"] is not None and rf_b["wr_after"] is not None)
        else None,
        "games_since_change_diff": (rf_a["games_since"] - rf_b["games_since"])
        if (rf_a["games_since"] is not None and rf_b["games_since"] is not None)
        else None,
        "had_change_30d_diff": rf_a["had_change_30d"] - rf_b["had_change_30d"],
        "bo_BO3": 1 if bo == "BO3" else 0,
        "bo_BO5": 1 if bo == "BO5" else 0,
        "tier_T2": 1 if tier == "T2" else 0,
        "tier_T3": 1 if tier == "T3" else 0,
    }
    return [vals.get(f) for f in feats]


def train_model(df: pl.DataFrame, feats: list[str], rounds: int = 150):
    """Train a LightGBM model on all available data (early stop on a trailing holdout)."""
    df = df.sort("match_date")
    n = df.height
    tr, va = df[: int(n * 0.85)], df[int(n * 0.85) :]
    model = lgb.train(
        {"objective": "binary", "verbosity": -1, "seed": 0},
        lgb.Dataset(tr.select(feats).to_numpy().astype(np.float32), label=tr["y"].to_numpy()),
        num_boost_round=rounds,
        valid_sets=[
            lgb.Dataset(va.select(feats).to_numpy().astype(np.float32), label=va["y"].to_numpy())
        ],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="HLTV-only win probability model")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Path to DuckDB file")
    parser.add_argument("--rounds", type=int, default=150, help="LightGBM boosting rounds")
    parser.add_argument("--out", default=None, help="CSV path for predictions")
    parser.add_argument(
        "--predict-all",
        action="store_true",
        help="Walk-forward predictions for all matches from --start (defaults to all-pred CSV)",
    )
    parser.add_argument(
        "--start", default="2025-11-01", help="Walk-forward start month (YYYY-MM-DD)"
    )
    parser.add_argument("--selftest", action="store_true", help="Run self-check and exit")
    args = parser.parse_args()

    if args.selftest:
        _selftest()
        return

    con = init_db(args.db)
    backfill_tier(con)
    matches = con.execute(
        "SELECT match_id, match_date, team_a, team_b, winner, best_of, tier "
        "FROM match_results WHERE winner IS NOT NULL ORDER BY match_date"
    ).pl()
    matches = matches.with_columns(bo=pl.col("best_of").map_elements(infer_bo, return_dtype=pl.Utf8))

    team_rating = {
        (mid, team): r
        for mid, team, r in con.execute(
            "SELECT match_id, team, AVG(rating) FROM player_ratings GROUP BY match_id, team"
        ).fetchall()
    }
    roster = load_roster(con)
    player_rating = load_player_rating(con)
    match_maps = load_match_maps(con)
    con.close()

    df = compute_features(matches, team_rating, roster, player_rating, match_maps)
    feats, df = _feature_frame(df)

    # Ablation: confirm no market/price features are present.
    assert not any("price" in f or "kalshi" in f or "open" in f or "close" in f for f in feats), feats
    print(f"features ({len(feats)}): {feats}")
    print("ablation: no Kalshi / market-price features — confirmed\n")

    if args.predict_all:
        start = date.fromisoformat(args.start)
        preds = walk_forward_predict(df, feats, start, args.rounds)
        out = args.out or "data/win_prob_predictions_all.csv"
        preds.select(
            ["match_id", "match_date", "team_a", "team_b", "tier", "bo", "prob_team_a", "y"]
        ).write_csv(out)
        print(f"walk-forward predictions: {preds.height} matches from {start} -> {out}")
        return

    model, X_te, y_te, test = _fit_predict(df, feats, args.rounds)
    p = model.predict(X_te, num_iteration=model.best_iteration)

    print(f"train/val/test: {int(df.height*0.70)}/{int(df.height*0.15)}/{test.height} matches")
    print(f"overall  log loss = {_log_loss(y_te, p):.4f}  Brier = {_brier(y_te, p):.4f}")
    print(f"baseline log loss = {_log_loss(y_te, np.full(y_te.shape, 0.5)):.4f}  Brier = 0.2500\n")

    print("per tier:")
    for tier in ("T1", "T2", "T3"):
        mask = (test["tier"].to_numpy() == tier)
        if mask.sum() == 0:
            continue
        print(
            f"  {tier}: n={mask.sum():>5}  log loss = {_log_loss(y_te[mask], p[mask]):.4f}  "
            f"Brier = {_brier(y_te[mask], p[mask]):.4f}"
        )

    if args.out:
        out = test.with_columns(prob_team_a=pl.Series(p)).select(
            ["match_id", "match_date", "team_a", "team_b", "tier", "bo", "prob_team_a", "y"]
        )
        out.write_csv(args.out)
        print(f"\npredictions saved to {args.out}")


def _selftest() -> None:
    assert infer_tier("IEM Cologne Major 2026") == "T1"
    assert infer_tier("BLAST Bounty 2026 Season 1") == "T1"
    assert infer_tier("ESL Pro League Season 22") == "T1"
    assert infer_tier("ESL Challenger League Season 51") == "T2"
    assert infer_tier("CCT 2026 Europe Series 1") == "T2"
    assert infer_tier("PGL Masters Bucharest 2026 Europe Open Qualifier 2") == "T3"
    assert infer_tier("WINLINE MPKBK CIS LAN Season 7") == "T3"

    assert infer_bo(5) == "BO5"
    assert infer_bo(3) == "BO3"
    assert infer_bo(25) == "BO1"
    assert infer_bo(1) == "BO1"

    matches = pl.DataFrame(
        {
            "match_id": [1, 2, 3],
            "match_date": [date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 3)],
            "team_a": ["A", "A", "B"],
            "team_b": ["B", "C", "A"],
            "winner": ["A", "C", "A"],
            "tier": ["T1", "T2", "T3"],
            "bo": ["BO1", "BO3", "BO5"],
        }
    )
    team_rating = {
        (1, "A"): 1.1, (1, "B"): 0.9,
        (2, "A"): 1.0, (2, "C"): 0.8,
        (3, "B"): 1.2, (3, "A"): 0.7,
    }
    df = compute_features(matches, team_rating, {}, {}, {})

    m2 = df.filter(pl.col("match_id") == 2).row(0, named=True)
    assert m2["wr30_diff"] == 0.5, m2  # A=1.0 (won m1), C=0.5 (no history)
    assert m2["y"] == 0

    m3 = df.filter(pl.col("match_id") == 3).row(0, named=True)
    assert abs(m3["wr30_diff"] - (-0.5)) < 1e-9, m3  # B=0.0 (lost m1), A=0.5
    assert m3["h2h_diff"] == -1 and m3["h2h_count"] == 1, m3
    print("selftest OK")


if __name__ == "__main__":
    main()
