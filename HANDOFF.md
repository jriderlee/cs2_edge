# CS2 Edge — Handoff / Architecture Notes

> Written for the next person (or chatbot) to pick this up. Covers what the
> program does, how it is structured, and the issues still open as of this
> writing. Reflects the code in its current (committed) state.

---

## 1. What this project does

CS2 Edge predicts **Counter-Strike 2 match winners**, then compares those
predictions against **Kalshi prediction-market prices** to find bets where the
market is likely wrong. The specific edge it targets: when the market prices a
*favorite* far above what the model thinks is fair, that favorite is overpriced,
so the trade is to bet *against* them ("BUY NO").

The pipeline, end to end:

1. **Collect** — scrape HLTV (match results, player ratings, roster moves, maps)
   and pull Kalshi contract prices/outcomes.
2. **Store** — everything lands in a local DuckDB file (`data/cs2_edge.duckdb`).
3. **Predict** — a machine-learning model trained on HLTV data only outputs each
   team's win probability. It deliberately never sees market prices, so its
   opinion is independent of Kalshi.
4. **Find the edge** — compare the model's probability to the market price. A
   gap of `price >= 0.90` and `divergence >= 0.30` is a signal.
5. **Verify / act** — a backtester replays history to check profitability; a
   live scanner runs hourly, posts Discord alerts, and logs signals.

---

## 2. Repository layout

```
cs2_edge/
├── pyproject.toml            # root project (name "cs2-edge", entry "cs2-edge")
├── uv.lock                   # locked deps (duckdb, polars, lightgbm, httpx, ...)
├── README.md                 # EMPTY (see open issues)
├── .env                      # KALSHI_API_KEY, DISCORD_WEBHOOK_URL (gitignored)
├── data/                     # gitignored: cs2_edge.duckdb, CSVs, charts/
├── cs2_edge/                 # ⚠️ accidental nested git repo (see open issues)
└── src/cs2_edge/             # ← the actual application code
    ├── __init__.py           # main() → hello-world print (entry point NOT wired)
    ├── db/
    │   ├── db_init.py        # init_db(), DEFAULT_DB_PATH, schema application
    │   └── schema.sql        # all CREATE TABLE / INDEX statements
    ├── collectors/
    │   ├── hltv.py           # HLTV scraper (results, ratings, roster, maps, map_stats)
    │   └── kalshi.py         # Kalshi contract/candle collector + HLTV matching
    ├── models/
    │   └── win_probability.py# feature engineering + LightGBM model
    ├── analysis/
    │   ├── backtester.py         # $5/flat NO-bet simulation → PnL/equity
    │   ├── bias_test.py          # open price vs realized win rate by decile
    │   ├── divergence_scorer.py  # model prob vs market price → signal flags
    │   ├── correction_speed.py   # how fast mispricing corrects
    │   ├── trade_frequency.py    # how often 90-100% contracts open
    │   └── charts.py             # plotly HTML charts (NEW, untracked)
    └── monitor/
        └── live_scanner.py   # hourly scanner + Discord alerts
```

---

## 3. Data flow

```
HLTV.org ──hltv.py──► match_results, player_ratings,
                       roster_changes, match_maps, map_stats
Kalshi  ──kalshi.py►  kalshi_contracts, kalshi_candles
                             │
                             ▼
                     data/cs2_edge.duckdb
                             │
              win_probability.py  (reads HLTV tables ONLY)
                             │
                             ▼
                  win probability per match
                             │
              divergence_scorer / live_scanner
                  model_prob  vs  market price
                             │
                             ▼
                  signal (price ≥ 0.90, gap ≥ 0.30)
                             │
               backtester (historical PnL)  /  Discord alert + live_alerts
```

Key design point: the **model is trained on HLTV data only** — it never reads
`kalshi_contracts` or any price column. This keeps the prediction independent of
the market it is compared against (enforced by an `assert` in
`win_probability.main()`).

---

## 4. Database (DuckDB)

Single file, `data/cs2_edge.duckdb`. Schema lives in `src/cs2_edge/db/schema.sql`
and is applied by `init_db()` on every connection (all statements are
`IF NOT EXISTS`, so reapplying is safe). **There is currently no read-only mode**
— `init_db(db_path=DEFAULT_DB_PATH)` always opens read-write.

Tables:

| table | purpose |
|---|---|
| `match_results` | HLTV match results (winner, teams, date, `tier`, best-of) |
| `player_ratings` | per-player HLTV rating per match |
| `match_maps` | which team won each map of a match |
| `map_stats` | aggregated per-team/per-map win rates |
| `roster_changes` | player transfers (team A → team B on a date) |
| `kalshi_contracts` | Kalshi contract open/close price + resolution, matched to `match_id` |
| `kalshi_candles` | minute-level price/volume candles |
| `model_training_log` | log of model (re)train events (n_matches, n_ratings) |
| `live_alerts` | signals fired by the live scanner |

---

## 5. The model (win_probability.py)

For every past match it turns each team's history into ~16 numeric "features":

- recent win rate (last 30 / 60 / 90 days),
- player-rating strength,
- head-to-head record vs this opponent,
- map win rates,
- roster-change signals (did a star leave recently?),
- tournament tier (T1/T2/T3, inferred from the event name),
- match format (BO1/BO3/BO5).

A LightGBM classifier learns how these features predicted real winners in the
past. Output: probability that team A wins. Validation uses log-loss and Brier
score, reported overall and per tier. There is also a `--predict-all` walk-forward
mode that simulates "train on the past, predict the future" month by month.

---

## 6. How to run each piece

All modules run as `python -m cs2_edge.<module>` (via `uv run` if using uv):

```bash
# initialize / inspect the DB
python -m cs2_edge.db.db_init [db_path]

# collect data (writers)
python -m cs2_edge.collectors.hltv              # match results + ratings
python -m cs2_edge.collectors.hltv --roster     # transfers
python -m cs2_edge.collectors.hltv --maps       # per-map results → map_stats
python -m cs2_edge.collectors.kalshi            # Kalshi contracts + candles

# train / predict
python -m cs2_edge.models.win_probability                 # train + eval
python -m cs2_edge.models.win_probability --predict-all   # walk-forward CSV

# analysis (read-only)
python -m cs2_edge.analysis.backtester
python -m cs2_edge.analysis.bias_test
python -m cs2_edge.analysis.divergence_scorer
python -m cs2_edge.analysis.correction_speed
python -m cs2_edge.analysis.trade_frequency
python -m cs2_edge.analysis.charts

# live monitoring
python -m cs2_edge.monitor.live_scanner --once    # single scan
python -m cs2_edge.monitor.live_scanner           # hourly + daily scheduler
```

The `cs2-edge` console script (`[project.scripts]`) is **not wired up** — it
still points at the hello-world `main()`.

---

## 7. Current state & open issues

### 7.1 DuckDB concurrency (UNSOLVED — the main problem)
The live scanner and the background collectors all open the same DuckDB file
**read-write** (`init_db()` has no read-only option), so they can collide on the
file lock when running simultaneously. In the current code the scanner also
performs writes:

- `_load_history()` calls `backfill_tier(con)` (`ALTER TABLE` + `UPDATE`),
  then `INSERT INTO model_training_log`.
- `_run_once()` does `INSERT INTO live_alerts`.

So the scanner is not read-only today. A proposed fix (not yet applied): give
`init_db()` a `read_only` flag, open the scanner's connections read-only, and
route its writes through a separate short-lived writable connection with retry
on `duckdb.IOException`. This needs a decision on where the writes
(`model_training_log`, `live_alerts`) ultimately live, since a read-only scanner
can no longer write them directly.

### 7.2 `backfill_tier` is a write tied to the scan path
`backfill_tier()` populates `match_results.tier` from the event name. It is a
write (`ALTER TABLE` + `UPDATE`) and is currently called from
`win_probability.main()` and from `live_scanner._load_history()`. A candidate
cleanup is to extract it into a standalone one-shot migration script
(e.g. `migrate_backfill_tier.py`) so it isn't run by the read-only scanner.

### 7.3 Accidental nested git repo
`cs2_edge/` at the project root is a **separate git repository** (has its own
`.git`), containing a duplicate hello-world project. It shows as untracked
(`? cs2_edge`) in the outer repo. Needs to be removed or converted to a proper
submodule/ignored.

### 7.4 Uncommitted work
The following are uncommitted in the outer repo:
- `pyproject.toml` + `uv.lock`: `plotly` was added (for `charts.py`).
- `src/cs2_edge/analysis/charts.py`: new, untracked.

### 7.5 Empty README & unwired entry point
- `README.md` is empty (no user docs).
- `cs2-edge` console entry point still prints "Hello from cs2-edge!".
