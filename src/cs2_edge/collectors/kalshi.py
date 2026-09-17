"""Kalshi CS2 match contract collector.

Pulls historical CS2 match markets, extracts open/close price and resolution,
and matches each Kalshi event to an HLTV match_result by team names + date.
Unmatched events are stored with a NULL match_id.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import unicodedata
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import httpx
import polars as pl

from cs2_edge.db.db_init import DEFAULT_DB_PATH, SCHEMA_PATH, init_db

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
DEFAULT_SERIES = ["KXCS2GAME", "KXCSGOGAME"]
USER_AGENT = "cs2-edge/0.1"

ET = ZoneInfo("America/New_York")
_MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

CONTRACT_SCHEMA = {
    "contract_id": pl.Utf8,
    "match_id": pl.Int64,
    "team": pl.Utf8,
    "open_price": pl.Float64,
    "close_price": pl.Float64,
    "resolved": pl.Utf8,
    "resolution_date": pl.Date,
    "match_start_ts": pl.Int64,
}

CANDLE_SCHEMA = {
    "contract_id": pl.Utf8,
    "end_period_ts": pl.Int64,
    "price": pl.Float64,
    "volume": pl.Float64,
}

# normalized Kalshi name -> normalized HLTV name (Kalshi tends to shorten / rebrand)
ALIASES = {
    "nip": "ninjasinpyjamas",
    "navi": "natusvincere",
    "sementedomal": "sementesdomal",
}

# tokens Kalshi appends that HLTV drops; stripped before aliasing
GENERIC_TOKENS = {"esports", "esport", "team", "gaming"}

_DATE_RE = re.compile(r"scheduled for\s+([A-Z][a-z]{2}\s+\d{1,2},\s+\d{4})")
_TIME_RE = re.compile(r"at\s+(\d{1,2}):(\d{2})\s+([AP]M)")
_VS_RE = re.compile(r"\s+vs\.?\s+(.+?)\s+(?:Counter[\s-]?Strike|CS2)?\s*match")
_YES_TEAM_RE = re.compile(r"If\s+(.+?)\s+wins?\s+the")


def _et_to_unix(dt: datetime) -> int:
    return int(dt.replace(tzinfo=ET).timestamp())


def _ticker_match_start_ts(ticker: str) -> int | None:
    """Match start time from ticker 'KXCS2GAME-26JUL181700IMPBHE-IMP' (date+HHMM in ET)."""
    body = ticker.split("-", 1)[1] if "-" in ticker else ticker
    m = re.match(r"^(\d{2})([A-Z]{3})(\d{2})(\d{4})?", body)
    if not m or not m.group(4):
        return None
    mon = _MONTHS.get(m.group(2))
    if mon is None:
        return None
    try:
        dt = datetime(2000 + int(m.group(1)), mon, int(m.group(3)), int(m.group(4)[:2]), int(m.group(4)[2:]))
    except ValueError:
        return None
    return _et_to_unix(dt)


def _rules_match_start_ts(rules_primary: str) -> int | None:
    d = _DATE_RE.search(rules_primary)
    t = _TIME_RE.search(rules_primary)
    if not d or not t:
        return None
    try:
        day = datetime.strptime(d.group(1), "%b %d, %Y")
    except ValueError:
        return None
    hh = int(t.group(1))
    mm = int(t.group(2))
    if t.group(3) == "PM" and hh != 12:
        hh += 12
    elif t.group(3) == "AM" and hh == 12:
        hh = 0
    try:
        return _et_to_unix(datetime(day.year, day.month, day.day, hh, mm))
    except ValueError:
        return None


def parse_match_start_ts(ticker: str, rules_primary: str) -> int | None:
    """Match start time (unix, UTC). Ticker wins; rules text is the fallback."""
    return _ticker_match_start_ts(ticker) or _rules_match_start_ts(rules_primary or "")


def _fold(name: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", name) if not unicodedata.combining(c)
    )


def normalize_team(name: str | None) -> str:
    """Lowercase, fold accents, drop generic tokens, strip non-alphanumerics, then aliases."""
    if not name:
        return ""
    tokens = [
        t
        for t in re.split(r"[^a-z0-9]+", _fold(name).lower())
        if t and t not in GENERIC_TOKENS
    ]
    norm = "".join(tokens)
    return ALIASES.get(norm, norm)


def parse_rules_primary(text: str) -> dict:
    """Pull the yes-team, the two matchup teams, and the match date from a market's rules."""
    out: dict = {}
    m = _YES_TEAM_RE.search(text)
    if m:
        out["yes_team"] = m.group(1).strip()
    m = _VS_RE.search(text)
    if m:
        out["team_b"] = m.group(1).strip()
        # team_a is the trailing segment after the last ":" before the " vs."
        out["team_a"] = re.split(r":\s*", text[: m.start()])[-1].strip()
    m = _DATE_RE.search(text)
    if m:
        try:
            out["match_date"] = datetime.strptime(m.group(1), "%b %d, %Y").date()
        except ValueError:
            pass
    return out


def _first_open(candles: list[dict]) -> float | None:
    """First candle with a real YES-trade price open, in either API field format."""
    for c in candles:
        p = c.get("price") or {}
        v = p.get("open_dollars") if "open_dollars" in p else p.get("open")
        if v is not None:
            return float(v)
    return None


def _candle_close(c: dict) -> float | None:
    p = c.get("price") or {}
    v = p.get("close_dollars") if "close_dollars" in p else p.get("close")
    return float(v) if v is not None else None


def _candle_volume(c: dict) -> float | None:
    v = c.get("volume_fp")
    if v is None:
        v = c.get("volume")
    return float(v) if v not in (None, "") else None


def _to_unix(ts: str | None) -> int | None:
    if not ts:
        return None
    try:
        return int(datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp())
    except ValueError:
        return None


class _RateLimiter:
    """Serializes request starts to a max sustained rate (Kalshi public reads 429 past ~5/s)."""

    def __init__(self, rate: float) -> None:
        self.interval = 1.0 / rate
        self._lock = asyncio.Lock()
        self._next = 0.0

    async def wait(self) -> None:
        async with self._lock:
            loop = asyncio.get_event_loop()
            now = loop.time()
            if now < self._next:
                await asyncio.sleep(self._next - now)
                now = loop.time()
            self._next = now + self.interval


class KalshiCollector:
    def __init__(
        self,
        db_path: str | Path = DEFAULT_DB_PATH,
        series: list[str] | None = None,
        timeout: float = 30.0,
        concurrency: int = 20,
        rate: float = 5.0,
    ) -> None:
        self.db_path = Path(db_path)
        self.series = series or DEFAULT_SERIES
        self.timeout = timeout
        self.concurrency = concurrency
        self._limiter = _RateLimiter(rate)

    async def _get(self, client: httpx.AsyncClient, path: str, params: dict) -> dict:
        for attempt in range(6):
            await self._limiter.wait()
            resp = await client.get(path, params=params)
            if resp.status_code == 429:
                await asyncio.sleep(2.0 * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp.json()
        raise httpx.HTTPStatusError("rate limited", request=resp.request, response=resp)

    async def _paginate(self, client: httpx.AsyncClient, path: str, series_ticker: str) -> list[dict]:
        items: list[dict] = []
        cursor = ""
        while True:
            params: dict = {"series_ticker": series_ticker, "limit": 1000}
            if cursor:
                params["cursor"] = cursor
            data = await self._get(client, path, params)
            batch = data.get("markets", [])
            items.extend(batch)
            cursor = data.get("cursor") or ""
            if not cursor or not batch:
                break
        return items

    async def _fetch_markets(self, client: httpx.AsyncClient) -> list[dict]:
        by_ticker: dict[str, dict] = {}
        for st in self.series:
            live = await self._paginate(client, "/markets", st)
            hist = await self._paginate(client, "/historical/markets", st)
            for m in live:
                m["_source"] = "live"
                by_ticker.setdefault(m["ticker"], m)
            for m in hist:
                m["_source"] = "historical"
                by_ticker.setdefault(m["ticker"], m)
        return list(by_ticker.values())

    async def _candles(self, client: httpx.AsyncClient, m: dict) -> list[dict]:
        start = (_to_unix(m.get("open_time")) or 0) - 3600
        end = (_to_unix(m.get("close_time")) or int(datetime.now().timestamp())) + 3600
        params = {"start_ts": start, "end_ts": end, "period_interval": 60}
        if m["_source"] == "historical":
            path = f"/historical/markets/{m['ticker']}/candlesticks"
        else:
            st = m.get("series_ticker") or self.series[0]
            path = f"/series/{st}/markets/{m['ticker']}/candlesticks"
        try:
            data = await self._get(client, path, params)
        except httpx.HTTPStatusError:
            return []
        return data.get("candlesticks", [])

    async def _fetch_open_prices(
        self, client: httpx.AsyncClient, markets: list[dict]
    ) -> tuple[dict[str, float | None], list[dict]]:
        sem = asyncio.Semaphore(self.concurrency)

        async def one(m: dict) -> tuple[str, float | None, list[dict]]:
            async with sem:
                candles = await self._candles(client, m)
                return m["ticker"], _first_open(candles), candles

        results = await asyncio.gather(*(one(m) for m in markets))
        opens = {t: o for t, o, _ in results}
        candle_rows = [
            {
                "contract_id": t,
                "end_period_ts": c.get("end_period_ts"),
                "price": _candle_close(c),
                "volume": _candle_volume(c),
            }
            for t, _, candles in results
            for c in candles
            if c.get("end_period_ts") is not None
        ]
        return opens, candle_rows

    def _migrate(self, con: duckdb.DuckDBPyConnection) -> None:
        contract_cols = {r[0] for r in con.execute("DESCRIBE kalshi_contracts").fetchall()}
        if "open_price" not in contract_cols:
            con.execute("DROP TABLE IF EXISTS kalshi_contracts")
            con.execute(SCHEMA_PATH.read_text())
        elif "match_start_ts" not in contract_cols:
            con.execute("ALTER TABLE kalshi_contracts ADD COLUMN match_start_ts BIGINT")
        candle_cols = {r[0] for r in con.execute("DESCRIBE kalshi_candles").fetchall()}
        if "volume" not in candle_cols:
            con.execute("DROP TABLE IF EXISTS kalshi_candles")
            con.execute(SCHEMA_PATH.read_text())

    def _insert(self, con: duckdb.DuckDBPyConnection, rows: list[dict]) -> int:
        if not rows:
            return 0
        df = pl.DataFrame(rows, schema=CONTRACT_SCHEMA)
        con.register("_kdf", df)
        con.execute(
            "INSERT INTO kalshi_contracts BY NAME SELECT * FROM _kdf "
            "ON CONFLICT (contract_id) DO UPDATE SET "
            "match_id = EXCLUDED.match_id, "
            "team = EXCLUDED.team, "
            "open_price = COALESCE(EXCLUDED.open_price, kalshi_contracts.open_price), "
            "close_price = EXCLUDED.close_price, "
            "resolved = EXCLUDED.resolved, "
            "resolution_date = EXCLUDED.resolution_date, "
            "match_start_ts = COALESCE(EXCLUDED.match_start_ts, kalshi_contracts.match_start_ts)"
        )
        return df.height

    def _insert_candles(self, con: duckdb.DuckDBPyConnection, rows: list[dict]) -> int:
        if not rows:
            return 0
        df = pl.DataFrame(rows, schema=CANDLE_SCHEMA)
        con.register("_cdf", df)
        con.execute("INSERT OR REPLACE INTO kalshi_candles BY NAME SELECT * FROM _cdf")
        return df.height

    async def collect(
        self,
        limit: int | None = None,
        fetch_open: bool = True,
        tolerance_days: int = 1,
    ) -> dict:
        async with httpx.AsyncClient(
            base_url=BASE_URL, timeout=self.timeout, headers={"User-Agent": USER_AGENT}
        ) as client:
            print(f"Fetching Kalshi markets for series: {', '.join(self.series)}")
            markets = await self._fetch_markets(client)
            markets = [m for m in markets if m.get("result") in ("yes", "no")]
            print(f"Resolved markets: {len(markets)}")
            if limit:
                markets = markets[:limit]

            opens: dict[str, float | None] = {}
            candle_rows: list[dict] = []
            if fetch_open:
                print("Fetching opening prices...")
                opens, candle_rows = await self._fetch_open_prices(client, markets)

        con = init_db(self.db_path)
        self._migrate(con)

        hltv_index: dict[frozenset, list[tuple[date, int]]] = {}
        for mid, a, b, d in con.execute(
            "SELECT match_id, team_a, team_b, match_date FROM match_results"
        ).fetchall():
            hltv_index.setdefault(frozenset((normalize_team(a), normalize_team(b))), []).append((d, mid))

        events: dict[str, list[dict]] = {}
        for m in markets:
            events.setdefault(m["event_ticker"], []).append(m)

        rows: list[dict] = []
        matched = unmatched = 0
        for et, emarkets in events.items():
            meta = parse_rules_primary(emarkets[0].get("rules_primary") or "")
            match_id = self._match(hltv_index, meta, tolerance_days)
            if match_id is None:
                unmatched += 1
            else:
                matched += 1

            for m in emarkets:
                rules = parse_rules_primary(m.get("rules_primary") or "")
                team = rules.get("yes_team") or m.get("yes_sub_title") or ""
                resolved = m.get("result")
                close_price = self._to_float(m.get("last_price_dollars"))
                res_ts = m.get("settlement_ts") or m.get("close_time")
                rows.append(
                    {
                        "contract_id": m["ticker"],
                        "match_id": match_id,
                        "team": team,
                        "open_price": opens.get(m["ticker"]),
                        "close_price": close_price,
                        "resolved": resolved,
                        "resolution_date": self._to_date(res_ts),
                        "match_start_ts": parse_match_start_ts(m["ticker"], m.get("rules_primary") or ""),
                    }
                )

        n = self._insert(con, rows)
        n_candles = self._insert_candles(con, candle_rows)
        total = len(events)
        rate = (matched / total * 100) if total else 0.0
        print(
            f"Matched {matched}/{total} Kalshi matches to HLTV ({rate:.1f}%); "
            f"unmatched: {unmatched}; contracts stored: {n}; candles stored: {n_candles}"
        )
        con.close()
        return {"contracts": n, "matches": total, "matched": matched, "unmatched": unmatched}

    @staticmethod
    def _match(
        index: dict[frozenset, list[tuple[date, int]]], meta: dict, tolerance_days: int
    ) -> int | None:
        a = normalize_team(meta.get("team_a"))
        b = normalize_team(meta.get("team_b"))
        d = meta.get("match_date")
        if not a or not b or d is None:
            return None
        cands = index.get(frozenset((a, b)))
        if not cands:
            return None
        best, best_diff, tie = None, tolerance_days + 1, 0
        for cd, mid in cands:
            diff = abs((cd - d).days)
            if diff < best_diff:
                best, best_diff, tie = mid, diff, 1
            elif diff == best_diff:
                tie += 1
        return best if tie == 1 else None

    @staticmethod
    def _to_float(v: str | float | None) -> float | None:
        if v is None or v == "":
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _to_date(ts: str | None) -> date | None:
        if not ts:
            return None
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00")).date()
        except ValueError:
            return None


async def _run(args: argparse.Namespace) -> None:
    series = args.series.split(",") if args.series else DEFAULT_SERIES
    collector = KalshiCollector(
        db_path=args.db, series=series, timeout=args.timeout, rate=args.rate
    )
    result = await collector.collect(
        limit=args.limit, fetch_open=not args.no_open, tolerance_days=args.tolerance
    )
    print(
        f"Done: {result['contracts']} contracts, {result['matched']}/{result['matches']} matched "
        f"({result['unmatched']} unmatched)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect CS2 match contracts from Kalshi")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Path to DuckDB file")
    parser.add_argument(
        "--series", default=",".join(DEFAULT_SERIES), help="Comma-separated Kalshi series tickers"
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="Request timeout")
    parser.add_argument("--limit", type=int, default=None, help="Max markets to collect (testing)")
    parser.add_argument("--tolerance", type=int, default=1, help="Date tolerance in days for matching")
    parser.add_argument("--rate", type=float, default=5.0, help="Max requests/sec to Kalshi (429 past ~5/s)")
    parser.add_argument("--no-open", action="store_true", help="Skip opening-price fetch")
    parser.add_argument("--selftest", action="store_true", help="Run parser self-check and exit")
    args = parser.parse_args()

    if args.selftest:
        _selftest()
        return

    asyncio.run(_run(args))


def _selftest() -> None:
    rules = (
        "If Vitality wins the StarLadder Major Budapest 2025: FaZe vs. Vitality "
        "Counter Strike match originally scheduled for Dec 14, 2025, then the market resolves to Yes."
    )
    meta = parse_rules_primary(rules)
    assert meta["yes_team"] == "Vitality", meta
    assert meta["team_a"] == "FaZe" and meta["team_b"] == "Vitality", meta
    assert meta["match_date"] == date(2025, 12, 14), meta

    rules2 = (
        "If BORRACHEIROS wins the Gamers Club Liga Série A 2026: Semente do Mal vs. BORRACHEIROS "
        "CS2 match originally scheduled for Sep 18, 2026 at 5:00 PM EDT, then the market resolves to Yes."
    )
    meta2 = parse_rules_primary(rules2)
    assert meta2["yes_team"] == "BORRACHEIROS", meta2
    assert meta2["team_a"] == "Semente do Mal" and meta2["team_b"] == "BORRACHEIROS", meta2
    assert meta2["match_date"] == date(2026, 9, 18), meta2

    rules3 = (
        "If Rune Eaters wins the Stake Ranked Episode 5: Open Qualifier 2026: "
        "Rune Eaters vs. ASTRAL CS2 match originally scheduled for Sep 15, 2026, "
        "then the market resolves to Yes."
    )
    meta3 = parse_rules_primary(rules3)
    assert meta3["team_a"] == "Rune Eaters" and meta3["team_b"] == "ASTRAL", meta3
    assert meta3["match_date"] == date(2026, 9, 15), meta3

    assert normalize_team("Ninjas in Pyjamas") == "ninjasinpyjamas"
    assert normalize_team("NiP") == "ninjasinpyjamas"
    assert normalize_team("Natus Vincere") == "natusvincere"
    assert normalize_team("NAVI") == "natusvincere"
    assert normalize_team("Virtus.pro") == "virtuspro"
    assert normalize_team("Virtus Pro") == "virtuspro"
    assert normalize_team("BetBoom") == normalize_team("BETBOOM") == "betboom"
    assert normalize_team("Grêmio") == normalize_team("Gremio Esports") == "gremio"

    live_candles = [
        {"price": {}},
        {"price": {"open_dollars": "0.6600", "close_dollars": "0.6800"}},
    ]
    assert _first_open(live_candles) == 0.66
    hist_candles = [{"price": {"open": "0.6800", "close": "0.7100"}}]
    assert _first_open(hist_candles) == 0.68

    # match start time: ticker wins, rules text fallback
    assert parse_match_start_ts("KXCS2GAME-26JUL181700IMPBHE-IMP", "") == _et_to_unix(datetime(2026, 7, 18, 17, 0))
    assert parse_match_start_ts(
        "KXCS2GAME-25NOV29IMPNAVI-NAVI",
        "If Natus Vincere wins the ... scheduled for Nov 29, 2025 at 5:00 PM EDT, then ...",
    ) == _et_to_unix(datetime(2025, 11, 29, 17, 0))
    assert parse_match_start_ts("KXCS2GAME-25NOV29IMPNAVI-NAVI", "scheduled for Nov 29, 2025, then") is None

    print("selftest OK")


if __name__ == "__main__":
    main()
