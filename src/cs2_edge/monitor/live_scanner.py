"""Live scanner: hourly scan of active Kalshi CS2 markets against the model.

Trains the HLTV-only model at startup, then on each scan pulls active markets,
scores them, and alerts (Discord webhook + DuckDB) on contracts meeting
open_price >= 0.90 and divergence >= 0.30.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

import httpx
import numpy as np
import polars as pl
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from cs2_edge.collectors.kalshi import (
    BASE_URL,
    DEFAULT_SERIES,
    parse_rules_primary,
    normalize_team,
)
from cs2_edge.db.db_init import DEFAULT_DB_PATH, init_db
from cs2_edge.models.win_probability import (
    _feature_frame,
    build_state,
    compute_features,
    feature_vector,
    infer_bo,
    infer_tier,
    train_model,
)

MIN_PRICE = 0.90
MIN_DIVERGENCE = 0.30
USER_AGENT = "cs2-edge/0.1"


def load_env(path: str = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def _to_float(v: str | float | None) -> float | None:
    if v in (None, ""):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _event_name(rules: str) -> str:
    m = re.search(r"wins the (.+?):", rules)
    return m.group(1).strip() if m else ""


def _infer_bo(rules: str) -> str:
    r = rules.lower()
    if "best of five" in r or "bo5" in r:
        return "BO5"
    if "best of one" in r or "bo1" in r:
        return "BO1"
    return "BO3"  # Kalshi CS2 is predominantly BO3


class LiveScanner:
    def __init__(
        self,
        db_path: str | Path = DEFAULT_DB_PATH,
        webhook_url: str | None = None,
        series: list[str] | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.webhook_url = webhook_url
        self.series = series or DEFAULT_SERIES
        self.model = None
        self.feats: list[str] = []
        self.state = None
        self.roster: dict[str, list[date]] = {}
        self.hltv_norm: dict[str, str] = {}

    def _load_history(self) -> None:
        con = init_db(self.db_path)
        matches = con.execute(
            "SELECT match_id, match_date, team_a, team_b, winner, best_of, tier "
            "FROM match_results WHERE winner IS NOT NULL ORDER BY match_date"
        ).pl()
        matches = matches.with_columns(
            bo=pl.col("best_of").map_elements(infer_bo, return_dtype=pl.Utf8)
        )
        team_rating = {
            (mid, team): r
            for mid, team, r in con.execute(
                "SELECT match_id, team, AVG(rating) FROM player_ratings GROUP BY match_id, team"
            ).fetchall()
        }
        roster: dict[str, list[date]] = defaultdict(list)
        for ft, tt, cd in con.execute(
            "SELECT from_team, to_team, change_date FROM roster_changes"
        ).fetchall():
            if ft:
                roster[ft].append(cd)
            if tt:
                roster[tt].append(cd)
        for k in roster:
            roster[k].sort()
        names = con.execute(
            "SELECT DISTINCT team_a FROM match_results UNION SELECT DISTINCT team_b FROM match_results"
        ).fetchall()
        con.close()

        self.hltv_norm = {normalize_team(t[0]): t[0] for t in names}
        df = compute_features(matches, team_rating, roster)
        self.feats, df = _feature_frame(df)
        self.model = train_model(df, self.feats)
        self.state = build_state(matches, team_rating, roster)
        self.roster = roster
        print(f"model trained ({len(self.feats)} features)")

    @staticmethod
    def _mid(m: dict) -> float | None:
        bid = _to_float(m.get("yes_bid_dollars"))
        ask = _to_float(m.get("yes_ask_dollars"))
        if bid is not None and ask is not None and bid > 0 and ask > 0:
            return (bid + ask) / 2
        return _to_float(m.get("last_price_dollars"))

    def _score(self, m: dict) -> dict | None:
        rules = parse_rules_primary(m.get("rules_primary") or "")
        ka, kb, kyes = rules.get("team_a"), rules.get("team_b"), rules.get("yes_team")
        md = rules.get("match_date")
        if not ka or not kb or not kyes or md is None:
            return None
        a = self.hltv_norm.get(normalize_team(ka))
        b = self.hltv_norm.get(normalize_team(kb))
        y = self.hltv_norm.get(normalize_team(kyes))
        if a is None or b is None or y is None:
            return None

        price = self._mid(m)
        if price is None or price < MIN_PRICE:
            return None

        tier = infer_tier(_event_name(m.get("rules_primary") or ""))
        bo = _infer_bo(m.get("rules_primary") or "")
        vec = feature_vector(a, b, md, bo, tier, self.state, self.roster, self.feats)
        p_a = float(self.model.predict(np.array([vec], dtype=np.float32), num_iteration=self.model.best_iteration)[0])
        model_prob = p_a if y == a else (1 - p_a)
        divergence = price - model_prob
        if divergence < MIN_DIVERGENCE:
            return None

        return {
            "contract_id": m["ticker"],
            "team": y,
            "opponent": b if y == a else a,
            "open_price": price,
            "model_prob": model_prob,
            "divergence": divergence,
            "tier": tier,
            "event_ticker": m.get("event_ticker"),
            "match_date": md,
        }

    async def _fetch_active_markets(self, client: httpx.AsyncClient) -> list[dict]:
        markets: list[dict] = []
        for st in self.series:
            cursor = ""
            while True:
                params: dict = {"series_ticker": st, "limit": 1000}
                if cursor:
                    params["cursor"] = cursor
                resp = await client.get("/markets", params=params)
                resp.raise_for_status()
                data = resp.json()
                markets.extend(m for m in data.get("markets", []) if m.get("status") == "active")
                cursor = data.get("cursor") or ""
                if not cursor:
                    break
        return markets

    async def _post_discord(self, s: dict) -> None:
        if not self.webhook_url:
            return
        msg = (
            f"🚨 OVER signal\n"
            f"Match: {s['team']} vs {s['opponent']}\n"
            f"BUY NO on: {s['team']}\n"
            f"open_price: {s['open_price']:.2f} | model: {s['model_prob']:.2f} | "
            f"divergence: {s['divergence']:+.3f}\n"
            f"tier: {s['tier']} | match date: {s['match_date']}\n"
            f"contract: `{s['contract_id']}`"
        )
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                await c.post(self.webhook_url, json={"content": msg})
        except httpx.HTTPError as e:
            print(f"discord post failed: {e}")

    async def _run_once(self) -> None:
        async with httpx.AsyncClient(
            base_url=BASE_URL, timeout=30, headers={"User-Agent": USER_AGENT}
        ) as client:
            markets = await self._fetch_active_markets(client)

        signals = [s for m in markets if (s := self._score(m))]
        con = init_db(self.db_path)
        existing = {r[0] for r in con.execute("SELECT contract_id FROM live_alerts").fetchall()}
        new = [s for s in signals if s["contract_id"] not in existing]
        for s in new:
            await self._post_discord(s)
            con.execute(
                "INSERT INTO live_alerts (contract_id, team, open_price, model_prob, divergence, tier, event_ticker, match_date) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [s["contract_id"], s["team"], s["open_price"], s["model_prob"], s["divergence"], s["tier"], s["event_ticker"], s["match_date"]],
            )
        con.close()
        print(
            f"{datetime.now():%Y-%m-%d %H:%M:%S} scan: {len(markets)} active markets, "
            f"{len(signals)} signals, {len(new)} new alerts"
        )

    async def run(self, once: bool = False) -> None:
        self._load_history()
        await self._run_once()
        if once:
            return
        scheduler = AsyncIOScheduler()
        scheduler.add_job(self._run_once, "interval", hours=1, id="scan")
        scheduler.start()
        print("live scanner running (hourly scan)")
        await asyncio.Event().wait()


def main() -> None:
    parser = argparse.ArgumentParser(description="Live divergence scanner")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Path to DuckDB file")
    parser.add_argument("--once", action="store_true", help="Run a single scan and exit")
    parser.add_argument("--series", default=",".join(DEFAULT_SERIES), help="Comma-separated series")
    args = parser.parse_args()

    load_env()
    webhook = os.getenv("DISCORD_WEBHOOK_URL")
    scanner = LiveScanner(
        db_path=args.db,
        webhook_url=webhook,
        series=args.series.split(",") if args.series else None,
    )
    asyncio.run(scanner.run(once=args.once))


if __name__ == "__main__":
    main()
