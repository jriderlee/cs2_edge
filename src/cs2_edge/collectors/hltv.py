"""HLTV.org collector using the hltv-async-api package as transport."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import re
from datetime import date, datetime, timedelta
from pathlib import Path

import duckdb
import polars as pl
from bs4 import BeautifulSoup
from hltv_async_api import Hltv

from cs2_edge.db.db_init import DEFAULT_DB_PATH, init_db

BASE_URL = "https://www.hltv.org"
RESULTS_URL = f"{BASE_URL}/results"
TRANSFERS_URL = f"{BASE_URL}/transfers"

MATCH_SCHEMA = {
    "match_id": pl.Int64,
    "event_name": pl.Utf8,
    "match_date": pl.Date,
    "team_a": pl.Utf8,
    "team_b": pl.Utf8,
    "team_a_score": pl.Int32,
    "team_b_score": pl.Int32,
    "winner": pl.Utf8,
    "best_of": pl.Int32,
    "map_count": pl.Int32,
    "hltv_url": pl.Utf8,
}
RATING_SCHEMA = {
    "player_id": pl.Int64,
    "match_id": pl.Int64,
    "player_name": pl.Utf8,
    "team": pl.Utf8,
    "rating_date": pl.Date,
    "rating": pl.Float64,
    "maps_played": pl.Int32,
}
ROSTER_SCHEMA = {
    "change_id": pl.Int64,
    "player_id": pl.Int64,
    "player_name": pl.Utf8,
    "from_team": pl.Utf8,
    "to_team": pl.Utf8,
    "change_date": pl.Date,
    "change_type": pl.Utf8,
    "source_url": pl.Utf8,
}
MATCH_KEY_COLS = ["match_id", "match_date", "team_a", "team_b", "winner", "best_of", "map_count"]
RATING_KEY_COLS = ["player_id", "match_id", "player_name", "team", "rating_date", "rating", "maps_played"]
ROSTER_KEY_COLS = ["change_id"]


def parse_hltv_date(text: str | None) -> date | None:
    if not text:
        return None
    text = re.sub(r"\b(\d+)(st|nd|rd|th)\b", r"\1", text.strip())
    for fmt in ("%B %d %Y", "%B %d, %Y", "%b %d %Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _unix_to_date(ms: str) -> date | None:
    try:
        return datetime.fromtimestamp(int(ms) / 1000).date()
    except (ValueError, OSError):
        return None


def infer_best_of(team_a_score: int, team_b_score: int) -> int:
    return 2 * max(team_a_score, team_b_score) - 1


def parse_results(page: BeautifulSoup) -> list[dict]:
    matches: list[dict] = []
    current_date: date | None = None

    for el in page.select(".standard-headline, .result-con"):
        classes = el.get("class") or []
        if "standard-headline" in classes:
            text = re.sub(r"^(Results|Matches) for ", "", el.get_text())
            current_date = parse_hltv_date(text)
            continue

        a = el.select_one("a[href*='/matches/']")
        if not a:
            continue
        m = re.search(r"/matches/(\d+)/", a["href"])
        if not m:
            continue

        teams = [d.get_text(strip=True) for d in el.select(".team-name")]
        if len(teams) < 2:
            teams = [d.get_text(strip=True) for d in el.select(".team")]
        if len(teams) < 2:
            continue

        scores = [s.get_text(strip=True) for s in el.select(".result-score span")]
        if len(scores) < 2:
            continue
        try:
            a_score, b_score = int(scores[0]), int(scores[1])
        except ValueError:
            continue
        if a_score == b_score:
            continue

        ts = el.get("data-zonedgrouping-entry-unix")
        match_date = _unix_to_date(ts) if ts else current_date
        if not match_date:
            continue

        event = el.select_one(".event-name")
        team_a, team_b = teams[0], teams[1]
        matches.append(
            {
                "match_id": int(m.group(1)),
                "event_name": event.get_text(strip=True) if event else None,
                "match_date": match_date,
                "team_a": team_a,
                "team_b": team_b,
                "team_a_score": a_score,
                "team_b_score": b_score,
                "winner": team_a if a_score > b_score else team_b,
                "best_of": infer_best_of(a_score, b_score),
                "map_count": a_score + b_score,
                "hltv_url": BASE_URL + a["href"],
            }
        )
    return matches


def parse_match_stats(page: BeautifulSoup) -> list[dict]:
    rows: list[dict] = []
    for table in page.find_all("table", class_="totalstats")[:2]:
        team_el = table.select_one("a.teamName")
        team = team_el.get_text(strip=True) if team_el else None
        if not team:
            continue
        for tr in table.select("tr:not(.header-row)"):
            a = tr.select_one("a[href^='/player/']")
            nick = tr.select_one(".player-nick")
            rating_el = tr.select_one("td.rating")
            if not a or not nick or rating_el is None:
                continue
            pid = re.search(r"/player/(\d+)/", a["href"])
            if not pid:
                continue
            try:
                rating = float(rating_el.get_text(strip=True))
            except ValueError:
                continue
            rows.append(
                {
                    "player_id": int(pid.group(1)),
                    "player_name": nick.get_text(strip=True),
                    "team": team,
                    "rating": rating,
                }
            )
    return rows


def _change_id(player_id: int, change_date: date, ctype: str, from_team: str | None, to_team: str | None) -> int:
    key = f"{player_id}|{change_date}|{ctype}|{from_team or ''}|{to_team or ''}"
    return int(hashlib.md5(key.encode()).hexdigest()[:15], 16)


def parse_transfers(page: BeautifulSoup) -> list[dict]:
    rows: list[dict] = []
    for r in page.select(".transfer-row"):
        pimg = r.select_one("a.transfer-player-image-container")
        player_id = player_name = None
        if pimg:
            m = re.search(r"/player/(\d+)/([^/]+)", pimg.get("href") or "")
            if m:
                player_id, player_name = int(m.group(1)), m.group(2)

        def team_name(c) -> str | None:
            if c.name != "a" or not (c.get("href") or "").startswith("/team/"):
                return None
            img = c.select_one("img")
            if img:
                return img.get("alt") or img.get("title")
            m = re.search(r"/team/\d+/(.+)", c["href"])
            return m.group(1) if m else None

        teams = r.select(".transfer-team-container")
        from_team = team_name(teams[0]) if teams else None
        to_team = team_name(teams[1]) if len(teams) > 1 else None

        mov = r.select_one(".transfer-movement")
        mov_text = mov.get_text(" ", strip=True).lower() if mov else ""
        if "is benched" in mov_text:
            ctype = "bench"
        elif "joins" in mov_text:
            ctype = "join"
        elif "parts ways" in mov_text:
            ctype = "leave"
        elif "transfers" in mov_text:
            ctype = "transfer"
        else:
            ctype = "other"

        d = r.select_one(".transfer-date")
        change_date = parse_hltv_date(d.get_text(strip=True)) if d else None
        if player_id is None or change_date is None:
            continue
        rows.append(
            {
                "change_id": _change_id(player_id, change_date, ctype, from_team, to_team),
                "player_id": player_id,
                "player_name": player_name,
                "from_team": from_team,
                "to_team": to_team,
                "change_date": change_date,
                "change_type": ctype,
                "source_url": TRANSFERS_URL,
            }
        )
    return rows


class HLTVCollector:
    def __init__(
        self,
        db_path: str | Path = DEFAULT_DB_PATH,
        delay: float = 1.5,
        timeout: float = 15.0,
        max_retries: int = 10,
    ) -> None:
        self.db_path = Path(db_path)
        self.delay = delay
        self.timeout = timeout
        self.max_retries = max_retries
        self.hltv: Hltv | None = None

    def _insert(self, con: duckdb.DuckDBPyConnection, table: str, rows: list[dict], schema: dict, key_cols: list[str]) -> int:
        if not rows:
            return 0
        df = pl.DataFrame(rows, schema=schema).drop_nulls(subset=key_cols)
        dropped = len(rows) - df.height
        if dropped:
            print(f"  validation dropped {dropped} row(s) from {table}")
        if df.height == 0:
            return 0
        con.register("_df", df)
        con.execute(f"INSERT INTO {table} BY NAME SELECT * FROM _df")
        return df.height

    async def collect(self, years: int = 2, max_matches: int | None = None) -> dict:
        start_date = date.today() - timedelta(days=365 * years)
        self.hltv = Hltv(
            min_delay=1.0,
            max_delay=self.delay + 1.0,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        con = init_db(self.db_path)
        existing = {r[0] for r in con.execute("SELECT match_id FROM match_results").fetchall()}
        total_matches = 0
        total_ratings = 0
        offset = 0

        try:
            while True:
                page = await self.hltv._fetch(f"{RESULTS_URL}?offset={offset}")
                if page is None:
                    print(f"Failed to fetch results page offset={offset}; stopping.")
                    break
                matches = parse_results(page)
                if not matches:
                    break

                new = [
                    m
                    for m in matches
                    if m["match_date"] >= start_date and m["match_id"] not in existing
                ]

                match_rows: list[dict] = []
                rating_rows: list[dict] = []
                for m in new:
                    if max_matches and len(match_rows) >= max_matches:
                        break
                    stats_page = await self.hltv._fetch(m["hltv_url"])
                    await asyncio.sleep(self.delay)
                    if stats_page is not None:
                        for r in parse_match_stats(stats_page):
                            # ponytail: maps_played == map_count (stand-ins mid-series are rare);
                            # pull per-player "Maps" from /stats/matches/ if it stops being Cloudflare-blocked
                            r["match_id"] = m["match_id"]
                            r["rating_date"] = m["match_date"]
                            r["maps_played"] = m["map_count"]
                            rating_rows.append(r)
                    match_rows.append(m)
                    existing.add(m["match_id"])

                n_matches = self._insert(con, "match_results", match_rows, MATCH_SCHEMA, MATCH_KEY_COLS)
                n_ratings = self._insert(con, "player_ratings", rating_rows, RATING_SCHEMA, RATING_KEY_COLS)
                total_matches += n_matches
                total_ratings += n_ratings
                counts = con.execute(
                    "SELECT (SELECT count(*) FROM match_results), (SELECT count(*) FROM player_ratings)"
                ).fetchone()
                print(
                    f"offset={offset}: inserted {n_matches} matches, {n_ratings} ratings "
                    f"(totals: {counts[0]} matches, {counts[1]} ratings)"
                )

                if matches[-1]["match_date"] < start_date:
                    break
                if max_matches and total_matches >= max_matches:
                    break
                offset += 100
                await asyncio.sleep(self.delay)
        finally:
            con.close()
            await self.hltv.close()

        return {"matches": total_matches, "ratings": total_ratings}

    async def collect_roster(self, pages: int = 90) -> int:
        self.hltv = Hltv(min_delay=0.5, max_delay=1.0, timeout=self.timeout, max_retries=self.max_retries)
        con = init_db(self.db_path)
        existing = {r[0] for r in con.execute("SELECT change_id FROM roster_changes").fetchall()}
        rows: list[dict] = []
        try:
            for pg in range(1, pages + 1):
                page = await self.hltv._fetch(f"{TRANSFERS_URL}?page={pg}")
                if page is None:
                    print(f"transfers page {pg} fetch failed; stopping")
                    break
                parsed = parse_transfers(page)
                if not parsed:
                    break
                rows.extend(r for r in parsed if r["change_id"] not in existing)
                if pg % 10 == 0:
                    print(f"transfers page {pg}: {len(rows)} new so far")
                await asyncio.sleep(0.5)
        finally:
            await self.hltv.close()

        n = self._insert(con, "roster_changes", rows, ROSTER_SCHEMA, ROSTER_KEY_COLS)
        con.close()
        print(f"roster changes inserted: {n}")
        return n


async def _run(args: argparse.Namespace) -> None:
    collector = HLTVCollector(
        db_path=args.db, delay=args.delay, timeout=args.timeout, max_retries=args.retries
    )
    result = await collector.collect(years=args.years, max_matches=args.limit)
    print(f"Done: {result['matches']} matches, {result['ratings']} player ratings")


async def _run_roster(args: argparse.Namespace) -> None:
    collector = HLTVCollector(db_path=args.db, timeout=args.timeout, max_retries=args.retries)
    n = await collector.collect_roster(pages=args.pages)
    print(f"Done: {n} roster changes")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect CS2 match results from HLTV.org")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Path to DuckDB file")
    parser.add_argument("--years", type=int, default=2, help="Years of history to collect")
    parser.add_argument("--delay", type=float, default=1.5, help="Seconds between requests")
    parser.add_argument("--timeout", type=float, default=15.0, help="Request timeout")
    parser.add_argument("--retries", type=int, default=10, help="Max retries per request")
    parser.add_argument("--limit", type=int, default=None, help="Max matches to collect (testing)")
    parser.add_argument("--roster", action="store_true", help="Scrape roster changes (transfers)")
    parser.add_argument("--pages", type=int, default=90, help="Transfers pages to scrape (~1 year)")
    parser.add_argument("--selftest", action="store_true", help="Run parser self-check and exit")
    args = parser.parse_args()

    if args.selftest:
        _selftest()
        return

    if args.roster:
        asyncio.run(_run_roster(args))
        return

    asyncio.run(_run(args))


def _selftest() -> None:
    results_html = """
    <div class="results-sublist">
      <div class="standard-headline">Results for September 15th 2026</div>
      <div class="result-con" data-zonedgrouping-entry-unix="1757952000000">
        <a href="/matches/2375678/nip-vs-faze" class="a-reset">
          <div class="result">
            <table><tbody><tr>
              <td class="team-cell"><div class="team team1"><div class="team-name">NiP</div></div></td>
              <td class="result-score"><span class="score-won">2</span><span class="score-lost">1</span></td>
              <td class="team-cell"><div class="team team2"><div class="team-name">FaZe</div></div></td>
            </tr></tbody></table>
            <div class="event-name">IEM Katowice 2026</div>
          </div>
        </a>
      </div>
    </div>
    """
    stats_html = """
    <table class="table totalstats">
      <tr class="header-row"><td class="players"><div class="align-logo"><a class="teamName team" href="/team/7969/nemiga">Nemiga</a></div></td><td class="rating text-center">Rating<span class="ratingDesc">3.0</span></td></tr>
      <tr><td class="players"><div class="flagAlign"><a class="text-ellipsis" href="/player/23317/khan"><div class="statsPlayerName">Beksultan '<span class="player-nick">khaN</span>' Ospan</div></a></div></td><td class="rating text-center ratingPositive">1.26</td></tr>
    </table>
    <table class="table totalstats">
      <tr class="header-row"><td class="players"><div class="align-logo"><a class="teamName team" href="/team/1234/betm">BET-M</a></div></td><td class="rating text-center">Rating</td></tr>
      <tr><td class="players"><div class="flagAlign"><a class="text-ellipsis" href="/player/9999/x"><div class="statsPlayerName">X '<span class="player-nick">x</span>' Y</div></a></div></td><td class="rating text-center">0.95</td></tr>
    </table>
    """

    matches = parse_results(BeautifulSoup(results_html, "html.parser"))
    assert len(matches) == 1, matches
    m = matches[0]
    assert m["match_id"] == 2375678
    assert m["team_a"] == "NiP" and m["team_b"] == "FaZe"
    assert m["winner"] == "NiP"
    assert m["best_of"] == 3 and m["map_count"] == 3
    assert m["match_date"] is not None

    ratings = parse_match_stats(BeautifulSoup(stats_html, "html.parser"))
    assert len(ratings) == 2, ratings
    assert ratings[0]["player_id"] == 23317
    assert ratings[0]["player_name"] == "khaN"
    assert ratings[0]["team"] == "Nemiga"
    assert ratings[0]["rating"] == 1.26
    assert ratings[1]["team"] == "BET-M" and ratings[1]["rating"] == 0.95

    transfers_html = """
    <div class="transfer-row">
      <a class="transfer-player-image-container" href="/player/25111/redzed"><img title="x"/></a>
      <div class="transfer-teams-container">
        <a class="transfer-team-container a-reset" href="/team/12366/aurora-young-blud">
          <div class="transfer-team-logo-container"><img alt="Aurora Young Blud"/></div>
        </a>
        <div class="transfer-arrow"></div>
        <a class="transfer-team-container a-reset" href="/team/13613/bebop">
          <div class="transfer-team-logo-container"><img alt="Bebop"/></div>
        </a>
      </div>
      <div class="transfer-movement">redzed transfers from Aurora Young Blud to Bebop</div>
      <div class="transfer-date">Sep 17th 2026</div>
    </div>
    """
    transfers = parse_transfers(BeautifulSoup(transfers_html, "html.parser"))
    assert len(transfers) == 1, transfers
    t = transfers[0]
    assert t["player_id"] == 25111 and t["player_name"] == "redzed", t
    assert t["from_team"] == "Aurora Young Blud" and t["to_team"] == "Bebop", t
    assert t["change_type"] == "transfer" and t["change_date"] == date(2026, 9, 17), t

    print("selftest OK")


if __name__ == "__main__":
    main()
