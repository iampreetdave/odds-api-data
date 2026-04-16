"""
The Odds API v4 — ASYNC Bulk Data Fetcher (Maximum Speed)
https://the-odds-api.com/liveapi/guides/v4/

Uses asyncio + aiohttp + token bucket rate limiter.

Two knobs:
  CONCURRENCY       — max simultaneous open TCP connections
  REQUESTS_PER_SECOND — token bucket throughput cap

On 429:  exponential backoff + auto-reduce rate temporarily.
On quota low: hard-stop.

Replace ODDS_API_KEY with your actual key.
"""

import asyncio
import aiohttp
import aiofiles
import pandas as pd
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
import os

ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "2654835dd9c6779ec43efbe938a8ebe3")

BASE_URL = "https://api.the-odds-api.com"

REGIONS     = "us,uk,eu"
MARKETS     = "h2h,spreads,totals"
ODDS_FORMAT = "decimal"

# Max simultaneous open connections
CONCURRENCY = 10

# Token bucket: requests allowed per second
# Start at 5, bump to 10 if zero 429s after first 100 requests
REQUESTS_PER_SECOND = 5

# Hard-stop if remaining credits fall below this
QUOTA_STOP_THRESHOLD = 50

HISTORICAL_START     = "2015-01-01"
HISTORICAL_END       = "2025-12-31"
HISTORICAL_STEP_DAYS = 1            # daily snapshots — maximises data extracted

PRIORITY_SPORTS = [
    "americanfootball_nfl",
    "americanfootball_ncaaf",
    "baseball_mlb",
    "basketball_nba",
    "basketball_ncaab",
    "icehockey_nhl",
    "soccer_epl",
    "soccer_uefa_champs_league",
    "soccer_spain_la_liga",
    "soccer_germany_bundesliga",
    "soccer_italy_serie_a",
    "soccer_france_ligue_one",
    "soccer_usa_mls",
    "mma_mixed_martial_arts",
]

SCORES_DAYS_FROM = 3

OUTPUT_DIR      = Path("odds_data")
CHECKPOINT_FILE = OUTPUT_DIR / "checkpoint.json"
LOG_FILE        = OUTPUT_DIR / "fetch.log"

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
OUTPUT_DIR.mkdir(exist_ok=True)
for d in ["current", "historical", "scores", "events"]:
    (OUTPUT_DIR / d).mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(stream=open(
            __import__("sys").stdout.fileno(),
            mode="w", encoding="utf-8", buffering=1, closefd=False
        )),
    ]
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# TOKEN BUCKET RATE LIMITER
# ─────────────────────────────────────────────
class TokenBucket:
    """
    Async token bucket. Allows burst up to `capacity` then refills at `rate/sec`.
    On 429s the caller reduces the rate dynamically.
    """
    def __init__(self, rate: float, capacity: float = None):
        self.rate     = rate                       # tokens added per second
        self.capacity = capacity or rate * 2       # burst headroom
        self._tokens  = self.capacity
        self._last    = time.monotonic()
        self._lock    = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last
            self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
            self._last = now
            if self._tokens < 1:
                wait = (1 - self._tokens) / self.rate
                await asyncio.sleep(wait)
                self._tokens = 0
            else:
                self._tokens -= 1

    def throttle(self, factor: float = 0.5):
        """Reduce rate by factor on 429. Never goes below 0.5 req/s."""
        self.rate = max(0.5, self.rate * factor)
        log.warning(f"Rate throttled to {self.rate:.1f} req/s")

    def recover(self, target: float):
        """Slowly recover rate back toward original."""
        self.rate = min(target, self.rate * 1.2)


# ─────────────────────────────────────────────
# SHARED STATE (async-safe)
# ─────────────────────────────────────────────
class State:
    def __init__(self):
        self.quota_remaining: int | None = None
        self.quota_used:      int | None = None
        self.request_count:   int        = 0
        self.success_streak:  int        = 0       # consecutive successes since last 429
        self.lock        = asyncio.Lock()
        self.checkpoint  : dict          = {"completed": []}
        self.stop_event  = asyncio.Event()
        self.bucket      : TokenBucket   = None    # set in main_async

state = State()


# ─────────────────────────────────────────────
# CHECKPOINT (async file writes)
# ─────────────────────────────────────────────
async def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        async with aiofiles.open(CHECKPOINT_FILE) as f:
            state.checkpoint = json.loads(await f.read())
    else:
        state.checkpoint = {"completed": []}


async def save_checkpoint():
    async with aiofiles.open(CHECKPOINT_FILE, "w") as f:
        await f.write(json.dumps(state.checkpoint, indent=2))


async def mark_done(key: str):
    async with state.lock:
        if key not in state.checkpoint["completed"]:
            state.checkpoint["completed"].append(key)
        await save_checkpoint()


def is_done(key: str) -> bool:
    return key in state.checkpoint["completed"]


# ─────────────────────────────────────────────
# ASYNC HTTP CLIENT
# ─────────────────────────────────────────────
async def fetch(session: aiohttp.ClientSession, semaphore: asyncio.Semaphore,
                endpoint: str, params: dict = None) -> dict | list | None:
    """
    Token-bucket + semaphore gated async GET.
    - Token bucket controls req/sec throughput
    - Semaphore caps max open connections
    - 429 -> throttle bucket + exponential sleep
    - Quota low -> set stop flag
    """
    if state.stop_event.is_set():
        return None

    params = dict(params or {})
    params["apiKey"] = ODDS_API_KEY
    url = f"{BASE_URL}{endpoint}"

    for attempt in range(4):
        # Wait for token bucket before acquiring connection slot
        await state.bucket.acquire()

        async with semaphore:
            try:
                async with session.get(url, params=params,
                                       timeout=aiohttp.ClientTimeout(total=30)) as resp:

                    # Update quota + counters
                    async with state.lock:
                        if "x-requests-remaining" in resp.headers:
                            state.quota_remaining = int(resp.headers["x-requests-remaining"])
                            state.quota_used      = int(resp.headers.get("x-requests-used", 0))
                        state.request_count += 1
                        if state.request_count % 25 == 0:
                            log.info(
                                f"Requests: {state.request_count} | "
                                f"Rate: {state.bucket.rate:.1f}/s | "
                                f"Credits remaining: {state.quota_remaining} | "
                                f"Used: {state.quota_used}"
                            )
                        if (state.quota_remaining is not None
                                and state.quota_remaining < QUOTA_STOP_THRESHOLD):
                            log.warning(f"QUOTA LOW ({state.quota_remaining}). Stopping.")
                            state.stop_event.set()
                            return None

                    if resp.status == 401:
                        log.error("Invalid API key.")
                        state.stop_event.set()
                        return None

                    if resp.status == 422:
                        return None

                    if resp.status == 429:
                        # Read Retry-After if provided, else exponential backoff
                        retry_after = int(resp.headers.get("Retry-After", 0))
                        wait = retry_after if retry_after > 0 else (2 ** (attempt + 1))
                        log.warning(f"429 — backing off {wait}s (attempt {attempt+1}/4) | throttling rate")
                        state.bucket.throttle(0.5)
                        await asyncio.sleep(wait)
                        continue

                    if resp.status != 200:
                        log.warning(f"HTTP {resp.status}: {url}")
                        return None

                    # Success — slowly recover rate toward original target
                    async with state.lock:
                        state.success_streak += 1
                        if state.success_streak % 20 == 0:
                            state.bucket.recover(REQUESTS_PER_SECOND)

                    return await resp.json()

            except asyncio.TimeoutError:
                log.warning(f"Timeout (attempt {attempt+1}/4): {endpoint}")
                await asyncio.sleep(2 ** attempt)
            except Exception as e:
                log.error(f"Error (attempt {attempt+1}/4): {e}")
                await asyncio.sleep(2 ** attempt)

    return None


# ─────────────────────────────────────────────
# FLATTENERS
# ─────────────────────────────────────────────
def flatten_odds(events: list, snapshot_time: str = None) -> list:
    rows = []
    snap = snapshot_time or datetime.now(timezone.utc).isoformat()
    for event in events:
        base = {
            "event_id":      event.get("id"),
            "sport_key":     event.get("sport_key"),
            "sport_title":   event.get("sport_title"),
            "commence_time": event.get("commence_time"),
            "home_team":     event.get("home_team"),
            "away_team":     event.get("away_team"),
            "snapshot_time": snap,
        }
        bookmakers = event.get("bookmakers", [])
        if not bookmakers:
            rows.append({**base, "bookmaker_key": None, "market_key": None,
                         "outcome_name": None, "outcome_price": None, "outcome_point": None})
            continue
        for bm in bookmakers:
            bm_base = {**base,
                       "bookmaker_key":         bm.get("key"),
                       "bookmaker_title":        bm.get("title"),
                       "bookmaker_last_update":  bm.get("last_update")}
            for market in bm.get("markets", []):
                for outcome in market.get("outcomes", []):
                    rows.append({**bm_base,
                                 "market_key":    market.get("key"),
                                 "outcome_name":  outcome.get("name"),
                                 "outcome_price": outcome.get("price"),
                                 "outcome_point": outcome.get("point")})
    return rows


def flatten_scores(events: list) -> list:
    rows = []
    for event in events:
        base = {
            "event_id":      event.get("id"),
            "sport_key":     event.get("sport_key"),
            "sport_title":   event.get("sport_title"),
            "commence_time": event.get("commence_time"),
            "home_team":     event.get("home_team"),
            "away_team":     event.get("away_team"),
            "completed":     event.get("completed"),
            "last_update":   event.get("last_update"),
        }
        scores = event.get("scores") or []
        if not scores:
            rows.append({**base, "score_name": None, "score_value": None})
        else:
            for sc in scores:
                rows.append({**base, "score_name": sc.get("name"), "score_value": sc.get("score")})
    return rows


# ─────────────────────────────────────────────
# CSV WRITER (append-safe, thread-safe via lock)
# ─────────────────────────────────────────────
_csv_lock = asyncio.Lock()

async def write_csv(rows: list, path: Path):
    if not rows:
        return
    async with _csv_lock:
        df = pd.DataFrame(rows)
        header = not path.exists()
        df.to_csv(path, mode="a", header=header, index=False)


# ─────────────────────────────────────────────
# TASK FUNCTIONS (each is one API call)
# ─────────────────────────────────────────────
async def task_current_odds(session, sem, sport_key: str):
    key = f"current_odds_{sport_key}"
    if is_done(key) or state.stop_event.is_set():
        return
    data = await fetch(session, sem, f"/v4/sports/{sport_key}/odds", {
        "regions": REGIONS, "markets": MARKETS,
        "oddsFormat": ODDS_FORMAT, "dateFormat": "iso",
    })
    if data is not None:
        rows = flatten_odds(data, snapshot_time=datetime.now(timezone.utc).isoformat())
        await write_csv(rows, OUTPUT_DIR / "current" / f"{sport_key}.csv")
        log.info(f"[current] {sport_key}: {len(data)} events, {len(rows)} rows")
    await mark_done(key)


async def task_scores(session, sem, sport_key: str):
    key = f"scores_{sport_key}"
    if is_done(key) or state.stop_event.is_set():
        return
    data = await fetch(session, sem, f"/v4/sports/{sport_key}/scores", {
        "daysFrom": SCORES_DAYS_FROM, "dateFormat": "iso",
    })
    if data is not None:
        rows = flatten_scores(data)
        await write_csv(rows, OUTPUT_DIR / "scores" / f"{sport_key}.csv")
    await mark_done(key)


async def task_events(session, sem, sport_key: str) -> list:
    key = f"events_{sport_key}"
    if is_done(key):
        fpath = OUTPUT_DIR / "events" / f"{sport_key}.csv"
        if fpath.exists():
            try:
                df = pd.read_csv(fpath)
                return df["id"].dropna().tolist() if "id" in df.columns else []
            except Exception:
                return []
        return []
    if state.stop_event.is_set():
        return []
    data = await fetch(session, sem, f"/v4/sports/{sport_key}/events", {"dateFormat": "iso"})
    if data:
        await write_csv(data, OUTPUT_DIR / "events" / f"{sport_key}.csv")
        await mark_done(key)
        return [e["id"] for e in data if "id" in e]
    await mark_done(key)
    return []


async def task_historical_odds(session, sem, sport_key: str, date_iso: str):
    date_slug = date_iso[:10]
    key = f"hist_odds_{sport_key}_{date_slug}"
    if is_done(key) or state.stop_event.is_set():
        return
    data = await fetch(session, sem, f"/v4/historical/sports/{sport_key}/odds", {
        "date": date_iso, "regions": REGIONS,
        "markets": MARKETS, "oddsFormat": ODDS_FORMAT, "dateFormat": "iso",
    })
    if data is not None:
        events = data.get("data", []) if isinstance(data, dict) else data
        rows = flatten_odds(events, snapshot_time=date_iso)
        await write_csv(rows, OUTPUT_DIR / "historical" / f"odds_{sport_key}.csv")
        if rows:
            log.info(f"[hist] {sport_key} @ {date_slug}: {len(events)} events, {len(rows)} rows | remaining: {state.quota_remaining}")
    await mark_done(key)


# ─────────────────────────────────────────────
# ORCHESTRATOR
# ─────────────────────────────────────────────
async def main_async():
    semaphore = asyncio.Semaphore(CONCURRENCY)

    # Init token bucket with configured rate
    state.bucket = TokenBucket(rate=REQUESTS_PER_SECOND)

    connector = aiohttp.TCPConnector(
        limit=CONCURRENCY + 5,   # connection pool size
        ttl_dns_cache=300,
        enable_cleanup_closed=True,
    )

    async with aiohttp.ClientSession(connector=connector) as session:

        # ── 1. Sports list (free, sequential) ──────────────────
        log.info("Fetching sports list...")
        sports_data = await fetch(session, semaphore, "/v4/sports", {"all": "true"})
        if not sports_data:
            log.error("Could not fetch sports list. Check API key.")
            return
        await write_csv(sports_data, OUTPUT_DIR / "sports.csv")
        all_sports = [s["key"] for s in sports_data]
        log.info(f"{len(all_sports)} sports found")

        # ── 2. Current odds + scores — ALL sports, fully concurrent ──
        log.info(f"Firing current odds + scores for {len(all_sports)} sports at CONCURRENCY={CONCURRENCY}...")
        t0 = time.time()

        current_tasks = []
        for s in all_sports:
            current_tasks.append(task_current_odds(session, semaphore, s))
            current_tasks.append(task_scores(session, semaphore, s))

        await asyncio.gather(*current_tasks)
        log.info(f"Current odds + scores done in {time.time()-t0:.1f}s")

        if state.stop_event.is_set():
            log.warning("Quota exhausted after current odds. Stopping.")
            return

        # ── 3. Events + per-event odds — concurrent per sport ──
        log.info("Fetching events for all sports...")
        t0 = time.time()
        event_id_map: dict[str, list] = {}

        event_tasks = [task_events(session, semaphore, s) for s in all_sports]
        results = await asyncio.gather(*event_tasks)
        for sport_key, eids in zip(all_sports, results):
            event_id_map[sport_key] = eids

        log.info(f"Events fetched in {time.time()-t0:.1f}s")

        if state.stop_event.is_set():
            return

        # ── 4. Historical odds — concurrent across all sport+date combos ──
        # Use all_sports if --all-sports flag set, otherwise PRIORITY_SPORTS
        hist_sports = all_sports if USE_ALL_SPORTS else [s for s in PRIORITY_SPORTS if s in all_sports]
        skipped = [s for s in PRIORITY_SPORTS if s not in all_sports] if not USE_ALL_SPORTS else []
        for s in skipped:
            log.warning(f"  {s} not in active sports list, skipping")

        dates = []
        current_dt = datetime.strptime(HISTORICAL_START, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_dt     = datetime.strptime(HISTORICAL_END,   "%Y-%m-%d").replace(tzinfo=timezone.utc)
        while current_dt <= end_dt:
            dates.append(current_dt.strftime("%Y-%m-%dT%H:%M:%SZ"))
            current_dt += timedelta(days=HISTORICAL_STEP_DAYS)

        total_tasks = len(hist_sports) * len(dates)
        log.info(
            f"Historical: {len(hist_sports)} sports x {len(dates)} days "
            f"({HISTORICAL_START} -> {HISTORICAL_END}, step={HISTORICAL_STEP_DAYS}d) "
            f"= {total_tasks:,} tasks"
        )
        log.info(f"At current rate (~{REQUESTS_PER_SECOND}/s) this will take ~{total_tasks/REQUESTS_PER_SECOND/3600:.1f}h "
                 f"or until quota runs out at {QUOTA_STOP_THRESHOLD} credits remaining")
        t0 = time.time()

        hist_tasks = [
            task_historical_odds(session, semaphore, sport_key, d)
            for sport_key in hist_sports
            for d in dates
        ]

        CHUNK = 500
        for i in range(0, len(hist_tasks), CHUNK):
            if state.stop_event.is_set():
                log.warning("Quota exhausted during historical fetch. Stopping.")
                break
            chunk = hist_tasks[i:i+CHUNK]
            await asyncio.gather(*chunk)
            pct = min(100, (i + CHUNK) / len(hist_tasks) * 100)
            log.info(
                f"Historical chunk {i//CHUNK + 1}/{(len(hist_tasks)+CHUNK-1)//CHUNK} "
                f"({pct:.0f}%) | credits remaining: {state.quota_remaining}"
            )

        log.info(f"Historical fetch done in {time.time()-t0:.1f}s")


# ─────────────────────────────────────────────
# CONSOLIDATOR
# ─────────────────────────────────────────────
def consolidate():
    log.info("Consolidating CSVs...")

    for folder, out_name in [
        ("historical", "ALL_historical_odds.csv"),
        ("current",    "ALL_current_odds.csv"),
        ("scores",     "ALL_scores.csv"),
    ]:
        files = list((OUTPUT_DIR / folder).glob("*.csv"))
        if not files:
            continue
        dfs = []
        for f in files:
            try:
                dfs.append(pd.read_csv(f, low_memory=False, dtype={
                    "outcome_price": "float64",
                    "outcome_point": "float64",
                }))
            except Exception:
                pass
        if dfs:
            merged = pd.concat(dfs, ignore_index=True)
            merged.to_csv(OUTPUT_DIR / out_name, index=False)
            log.info(f"  {out_name}: {len(merged):,} rows from {len(dfs)} files")


def print_summary():
    log.info("\n" + "=" * 60)
    log.info("SUMMARY")
    log.info(f"  Total requests : {state.request_count}")
    log.info(f"  Credits used   : {state.quota_used}")
    log.info(f"  Credits left   : {state.quota_remaining}")
    log.info(f"  Output dir     : {OUTPUT_DIR.resolve()}")
    log.info("=" * 60)
    for fpath in sorted(OUTPUT_DIR.glob("*.csv")):
        try:
            rows = sum(1 for _ in open(fpath)) - 1
            kb   = fpath.stat().st_size // 1024
            log.info(f"  {fpath.name}  ({rows} rows, {kb} KB)")
        except Exception:
            pass


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="The Odds API v4 Async Bulk Fetcher")
    parser.add_argument("--reset-checkpoint", action="store_true")
    parser.add_argument("--consolidate-only", action="store_true")
    parser.add_argument("--current-only",     action="store_true", help="Only current odds + scores")
    parser.add_argument("--historical-only",  action="store_true", help="Skip current, only historical")
    parser.add_argument("--concurrency",      type=int,   default=CONCURRENCY,
                        help=f"Max open connections (default: {CONCURRENCY})")
    parser.add_argument("--rate",             type=float, default=REQUESTS_PER_SECOND,
                        help=f"Requests per second token bucket (default: {REQUESTS_PER_SECOND})")
    parser.add_argument("--all-sports",       action="store_true",
                        help="Use ALL 162 sports for historical fetch instead of priority list only")
    parser.add_argument("--sports",           nargs="+", default=None,
                        help="Override sports list e.g. --sports americanfootball_nfl baseball_mlb")
    parser.add_argument("--date-start",       default=HISTORICAL_START)
    parser.add_argument("--date-end",         default=HISTORICAL_END)
    parser.add_argument("--step-days",        type=int, default=HISTORICAL_STEP_DAYS)
    args = parser.parse_args()

    if not ODDS_API_KEY:
        print("ERROR: ODDS_API_KEY is empty.")
        raise SystemExit(1)

    CONCURRENCY         = args.concurrency
    REQUESTS_PER_SECOND = args.rate
    USE_ALL_SPORTS      = args.all_sports

    if args.sports:
        PRIORITY_SPORTS.clear()
        PRIORITY_SPORTS.extend(args.sports)

    HISTORICAL_START     = args.date_start
    HISTORICAL_END       = args.date_end
    HISTORICAL_STEP_DAYS = args.step_days

    if args.reset_checkpoint and CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        log.info("Checkpoint cleared.")

    if args.consolidate_only:
        consolidate()
        print_summary()
        raise SystemExit(0)

    # Load checkpoint (sync before event loop)
    if CHECKPOINT_FILE.exists():
        state.checkpoint = json.loads(CHECKPOINT_FILE.read_text())

    start = datetime.now()
    log.info(f"Starting async fetch | CONCURRENCY={CONCURRENCY} | RATE={REQUESTS_PER_SECOND}/s | {start}")
    log.info(f"Regions: {REGIONS} | Markets: {MARKETS}")

    asyncio.run(main_async())

    consolidate()
    print_summary()
    log.info(f"Total time: {datetime.now() - start}")
