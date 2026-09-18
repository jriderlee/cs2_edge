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
import time
from datetime import datetime
from pathlib import Path

import duckdb
import httpx
import numpy as np
import polars as pl
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from cs2_edge.collectors.hltv import HLTVCollector
from cs2_edge.collectors.kalshi import (
    BASE_URL,
    DEFAULT_SERIES,
    _RateLimiter,
    parse_match_start_ts,
    parse_rules_primary,
    normalize_team,
)
from cs2_edge.db.db_init import DEFAULT_DB_PATH, init_db
from cs2_edge.models.win_probability import (
    _feature_frame,
    _team_results,
    build_state,
    compute_features,
    feature_vector,
    infer_bo,
    infer_tier,
    load_match_maps,
    load_player_rating,
    load_roster,
    train_model,
)

MIN_PRICE = 0.90
MIN_DIVERGENCE = 0.30
MAX_PRICE_MOVE = 0.15        # drop if current price moved >0.15 from open
MIN_RECENT_VOLUME = 1.0      # drop if last-2-candle volume is below this ("near zero")
USER_AGENT = "cs2-edge/0.1"


def write_with_retry(
    db_path: Path, sql: str, params: list, retries: int = 5, delay: float = 0.5
) -> None:
    """Run a single INSERT on a short-lived writable connection, retrying on lock
    collisions with the background collector. Closes immediately after the write."""
    for attempt in range(1, retries + 1):
        try:
            con = duckdb.connect(str(db_path))
            try:
                con.execute(sql, params)
                return
            finally:
                con.close()
        except duckdb.IOException:
            if attempt == retries:
                raise
            time.sleep(delay)


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


def _to_unix(ts: str | None) -> int | None:
    if not ts:
        return None
    try:
        return int(datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp())
    except ValueError:
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


def _candle_volume(c: dict) -> float | None:
    v = c.get("volume_fp")
    if v is None:
        v = c.get("volume")
    return _to_float(v)


def _first_candle_mid(c: dict) -> float | None:
    """Representative YES price of a candle: mid of bid/ask open (live) or flat open (hist)."""
    ask = _to_float((c.get("yes_ask") or {}).get("open_dollars")) if isinstance(c.get("yes_ask"), dict) else None
    bid = _to_float((c.get("yes_bid") or {}).get("open_dollars")) if isinstance(c.get("yes_bid"), dict) else None
    if ask is not None and bid is not None and bid > 0:
        return (ask + bid) / 2
    if ask is not None:
        return ask
    if bid is not None:
        return bid
    p = c.get("price") or {}
    if isinstance(p, dict):
        v = p.get("open_dollars") if "open_dollars" in p else p.get("open")
        return _to_float(v)
    return None


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
        self.roster: dict = {}
        self.player_rating: dict = {}
        self.team_results: dict = {}
        self.hltv_norm: dict[str, str] = {}

    def _load_history(self) -> None:
        con = init_db(self.db_path, read_only=True)
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
        roster = load_roster(con)
        player_rating = load_player_rating(con)
        match_maps = load_match_maps(con)
        names = con.execute(
            "SELECT DISTINCT team_a FROM match_results UNION SELECT DISTINCT team_b FROM match_results"
        ).fetchall()

        self.hltv_norm = {normalize_team(t[0]): t[0] for t in names}
        df = compute_features(matches, team_rating, roster, player_rating, match_maps)
        self.feats, df = _feature_frame(df)
        self.model = train_model(df, self.feats)
        self.state = build_state(matches, team_rating, roster, match_maps)
        self.roster = roster
        self.player_rating = player_rating
        self.team_results = _team_results(matches)
        n_ratings = con.execute("SELECT count(*) FROM player_ratings").fetchone()[0]
        con.close()
        write_with_retry(
            self.db_path,
            "INSERT INTO model_training_log (n_matches, n_ratings) VALUES (?, ?)",
            [len(matches), n_ratings],
        )
        print(f"model trained ({len(self.feats)} features, {len(matches)} matches); retrain logged")

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
        vec = feature_vector(
            a, b, md, bo, tier, self.state, self.roster, self.player_rating, self.team_results, self.feats
        )
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
                for m in data.get("markets", []):
                    if m.get("status") == "active":
                        m["_series"] = st
                        markets.append(m)
                cursor = data.get("cursor") or ""
                if not cursor:
                    break
        return markets

    async def _fetch_candles(self, client: httpx.AsyncClient, markets: list[dict]) -> dict[str, list[dict]]:
        """Fetch candle history for each market (concurrent, rate-limited)."""
        sem = asyncio.Semaphore(8)
        limiter = _RateLimiter(5.0)
        now = int(time.time())

        async def one(m: dict) -> tuple[str, list[dict]]:
            async with sem:
                await limiter.wait()
                st = m.get("_series") or self.series[0]
                start = _to_unix(m.get("open_time")) or (now - 7 * 86400)
                try:
                    resp = await client.get(
                        f"/series/{st}/markets/{m['ticker']}/candlesticks",
                        params={"start_ts": start - 3600, "end_ts": now + 3600, "period_interval": 60},
                    )
                    resp.raise_for_status()
                    return m["ticker"], resp.json().get("candlesticks", [])
                except httpx.HTTPError:
                    return m["ticker"], []

        results = await asyncio.gather(*(one(m) for m in markets))
        return dict(results)

    @staticmethod
    def _mature_reason(m: dict, candles: list[dict]) -> str | None:
        """Return the reason a market should be dropped, or None if it is fresh."""
        cs = sorted(candles, key=lambda c: c.get("end_period_ts") or 0)
        if not cs:
            return "no-candles"
        recent_vol = sum(_candle_volume(c) or 0.0 for c in cs[-2:])
        if recent_vol < MIN_RECENT_VOLUME:
            return "volume"
        open_price = _first_candle_mid(cs[0])
        current = LiveScanner._mid(m)
        if open_price is not None and current is not None and abs(current - open_price) > MAX_PRICE_MOVE:
            return "moved"
        return None

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
        now = int(time.time())
        async with httpx.AsyncClient(
            base_url=BASE_URL, timeout=30, headers={"User-Agent": USER_AGENT}
        ) as client:
            markets = await self._fetch_active_markets(client)
            total = len(markets)

            # CHANGE 1: pre-match filter — keep only markets that haven't started
            prematch: list[dict] = []
            for m in markets:
                mts = parse_match_start_ts(m.get("ticker") or "", m.get("rules_primary") or "")
                if mts is not None and now < mts:
                    m["_match_start_ts"] = mts
                    prematch.append(m)

            # CHANGE 2: mature market filter — needs candle data
            candles = await self._fetch_candles(client, prematch)
            fresh: list[dict] = []
            drop_reasons: dict[str, int] = {}
            for m in prematch:
                reason = self._mature_reason(m, candles.get(m["ticker"], []))
                if reason is None:
                    fresh.append(m)
                else:
                    drop_reasons[reason] = drop_reasons.get(reason, 0) + 1

        signals = [s for m in fresh if (s := self._score(m))]
        con = init_db(self.db_path, read_only=True)
        existing = {r[0] for r in con.execute("SELECT contract_id FROM live_alerts").fetchall()}
        new = [s for s in signals if s["contract_id"] not in existing]
        for s in new:
            await self._post_discord(s)
            write_with_retry(
                self.db_path,
                "INSERT INTO live_alerts (contract_id, team, open_price, model_prob, divergence, tier, event_ticker, match_date) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [s["contract_id"], s["team"], s["open_price"], s["model_prob"], s["divergence"], s["tier"], s["event_ticker"], s["match_date"]],
            )
        con.close()
        reasons = " ".join(f"{k}={v}" for k, v in sorted(drop_reasons.items())) or "-"
        print(
            f"{datetime.now():%Y-%m-%d %H:%M:%S} scan: {total} markets -> "
            f"{len(prematch)} pre-match -> {len(fresh)} not-mature -> {len(signals)} signals "
            f"({len(new)} new alerts; mature drops: {reasons})"
        )

    async def _daily_update(self) -> None:
        collector = HLTVCollector(db_path=self.db_path)
        result = await collector.collect(days=7)
        self._load_history()
        print(
            f"{datetime.now():%Y-%m-%d %H:%M:%S} daily HLTV update: "
            f"{result['matches']} new matches, {result['ratings']} new ratings; model retrained"
        )

    async def run(self, once: bool = False) -> None:
        self._load_history()
        await self._run_once()
        if once:
            return
        scheduler = AsyncIOScheduler()
        scheduler.add_job(self._run_once, "interval", hours=1, id="scan")
        scheduler.add_job(self._daily_update, "interval", hours=24, id="hltv_update")
        scheduler.start()
        print("live scanner running (hourly scan + daily HLTV update)")
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
