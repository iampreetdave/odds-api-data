"""
Sports Odds Pipeline — single file
Fetches The Odds API v4, stores raw per sport, builds ML tables.

Usage:
  python fetch.py                    # full run
  python fetch.py --current-only     # today's lines only
  python fetch.py --historical-only  # historical only
  python fetch.py --reset            # clear checkpoint, start fresh

Env:
  ODDS_API_KEY  — API key (falls back to hardcoded below)

Output:
  odds_data/
    sports.csv
    checkpoint.json
    raw/{sport}/
      odds.csv      <- every bookmaker row ever fetched
      scores.csv    <- completed game scores
    ml/{sport}/
      consensus.csv     <- per snapshot: avg odds + implied prob + overround
      line_movement.csv <- opening vs closing line + sharp signals
      features.csv      <- one row per event, all features as columns
      ml_ready.csv      <- features + labels merged, feed into model directly
"""

import os, sys, json, asyncio, logging, time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import aiohttp, aiofiles
import pandas as pd
import numpy as np

# ──────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────
API_KEY   = os.environ.get("ODDS_API_KEY", "2654835dd9c6779ec43efbe938a8ebe3")
BASE_URL  = "https://api.the-odds-api.com"

REGIONS      = "us,uk,eu"
MARKETS      = "h2h,spreads,totals"
ODDS_FORMAT  = "decimal"
CONCURRENCY  = 10
RATE_PER_SEC = 5.0
QUOTA_STOP   = 50

HIST_START    = "2015-01-01"
HIST_END      = datetime.now(timezone.utc).strftime("%Y-%m-%d")
HIST_STEP     = 1          # days between snapshots

MIN_BOOKS     = 3          # min bookmakers for consensus to be trusted

OUT   = Path("odds_data")
CKPT  = OUT / "checkpoint.json"

# ──────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────
OUT.mkdir(exist_ok=True)
fmt = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(
    level=logging.INFO, format=fmt,
    handlers=[
        logging.FileHandler(OUT / "run.log", encoding="utf-8"),
        logging.StreamHandler(stream=open(
            sys.stdout.fileno(), mode="w",
            encoding="utf-8", buffering=1, closefd=False
        )),
    ]
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────
# STATE
# ──────────────────────────────────────────────────────────
class State:
    def __init__(self):
        self.remaining = None
        self.used      = None
        self.count     = 0
        self.lock      = asyncio.Lock()
        self.stop      = asyncio.Event()
        self.ckpt      = {"done": []}
        self.bucket    = None   # set in run()

state = State()


# ──────────────────────────────────────────────────────────
# CHECKPOINT
# ──────────────────────────────────────────────────────────
def load_ckpt():
    if CKPT.exists():
        state.ckpt = json.loads(CKPT.read_text())

def save_ckpt():
    CKPT.write_text(json.dumps(state.ckpt, indent=2))

async def done(key):
    async with state.lock:
        if key not in state.ckpt["done"]:
            state.ckpt["done"].append(key)
        save_ckpt()

def is_done(key):
    return key in state.ckpt["done"]


# ──────────────────────────────────────────────────────────
# TOKEN BUCKET
# ──────────────────────────────────────────────────────────
class Bucket:
    def __init__(self, rate):
        self.rate     = rate
        self.tokens   = rate * 2
        self.last     = time.monotonic()
        self._lock    = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            self.tokens = min(self.rate * 2, self.tokens + (now - self.last) * self.rate)
            self.last = now
            if self.tokens < 1:
                await asyncio.sleep((1 - self.tokens) / self.rate)
                self.tokens = 0
            else:
                self.tokens -= 1

    def throttle(self):
        self.rate = max(0.5, self.rate * 0.5)
        log.warning(f"Rate throttled -> {self.rate:.1f}/s")

    def recover(self, target):
        self.rate = min(target, self.rate * 1.2)


# ──────────────────────────────────────────────────────────
# HTTP
# ──────────────────────────────────────────────────────────
async def get(session, sem, endpoint, params=None):
    if state.stop.is_set():
        return None

    p = dict(params or {})
    p["apiKey"] = API_KEY

    for attempt in range(4):
        await state.bucket.acquire()
        async with sem:
            try:
                async with session.get(
                    f"{BASE_URL}{endpoint}", params=p,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as r:
                    async with state.lock:
                        if "x-requests-remaining" in r.headers:
                            state.remaining = int(r.headers["x-requests-remaining"])
                            state.used      = int(r.headers.get("x-requests-used", 0))
                        state.count += 1
                        if state.count % 50 == 0:
                            log.info(f"Requests: {state.count} | Rate: {state.bucket.rate:.1f}/s | Credits left: {state.remaining}")
                        if state.remaining is not None and state.remaining < QUOTA_STOP:
                            log.warning(f"Quota low ({state.remaining}). Stopping.")
                            state.stop.set()
                            return None

                    if r.status == 401:
                        log.error("Bad API key.")
                        state.stop.set()
                        return None
                    if r.status == 422:
                        return None
                    if r.status == 429:
                        wait = int(r.headers.get("Retry-After", 2 ** (attempt + 1)))
                        log.warning(f"429 - waiting {wait}s")
                        state.bucket.throttle()
                        await asyncio.sleep(wait)
                        continue
                    if r.status != 200:
                        return None

                    async with state.lock:
                        state.bucket.recover(RATE_PER_SEC)
                    return await r.json()

            except asyncio.TimeoutError:
                await asyncio.sleep(2 ** attempt)
            except Exception as e:
                log.error(f"Request error: {e}")
                await asyncio.sleep(2 ** attempt)
    return None


# ──────────────────────────────────────────────────────────
# CSV HELPERS
# ──────────────────────────────────────────────────────────
_wlock = asyncio.Lock()

async def append(rows, path):
    if not rows:
        return
    async with _wlock:
        df  = pd.DataFrame(rows)
        hdr = not path.exists()
        df.to_csv(path, mode="a", header=hdr, index=False)

def write(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info(f"  Saved {len(df):,} rows -> {path.relative_to(OUT)}")


# ──────────────────────────────────────────────────────────
# FLATTEN
# ──────────────────────────────────────────────────────────
def flatten_odds(events, snap=None):
    snap = snap or datetime.now(timezone.utc).isoformat()
    rows = []
    for ev in events:
        base = {
            "event_id":      ev.get("id"),
            "sport_key":     ev.get("sport_key"),
            "sport_title":   ev.get("sport_title"),
            "commence_time": ev.get("commence_time"),
            "home_team":     ev.get("home_team"),
            "away_team":     ev.get("away_team"),
            "snapshot_time": snap,
        }
        for bm in ev.get("bookmakers", []):
            b = {**base,
                 "bookmaker_key":   bm.get("key"),
                 "bookmaker_title": bm.get("title"),
                 "last_update":     bm.get("last_update")}
            for mkt in bm.get("markets", []):
                for out in mkt.get("outcomes", []):
                    rows.append({**b,
                                 "market":        mkt.get("key"),
                                 "outcome_name":  out.get("name"),
                                 "price":         out.get("price"),
                                 "point":         out.get("point"),
                                 "implied_prob":  round(1 / out["price"], 6) if out.get("price") else None})
    return rows

def flatten_scores(events):
    rows = []
    for ev in events:
        base = {
            "event_id":      ev.get("id"),
            "sport_key":     ev.get("sport_key"),
            "home_team":     ev.get("home_team"),
            "away_team":     ev.get("away_team"),
            "commence_time": ev.get("commence_time"),
            "completed":     ev.get("completed"),
            "last_update":   ev.get("last_update"),
        }
        for sc in (ev.get("scores") or []):
            rows.append({**base, "score_name": sc.get("name"), "score_value": sc.get("score")})
        if not ev.get("scores"):
            rows.append({**base, "score_name": None, "score_value": None})
    return rows


# ──────────────────────────────────────────────────────────
# FETCHERS
# ──────────────────────────────────────────────────────────
async def fetch_current(session, sem, sport, raw_dir):
    key = f"current_{sport}"
    if is_done(key) or state.stop.is_set():
        return
    data = await get(session, sem, f"/v4/sports/{sport}/odds", {
        "regions": REGIONS, "markets": MARKETS,
        "oddsFormat": ODDS_FORMAT, "dateFormat": "iso"
    })
    if data:
        rows = flatten_odds(data, datetime.now(timezone.utc).isoformat())
        await append(rows, raw_dir / "odds.csv")
    await done(key)


async def fetch_scores(session, sem, sport, raw_dir):
    key = f"scores_{sport}"
    if is_done(key) or state.stop.is_set():
        return
    data = await get(session, sem, f"/v4/sports/{sport}/scores", {"daysFrom": 3, "dateFormat": "iso"})
    if data:
        rows = flatten_scores(data)
        await append(rows, raw_dir / "scores.csv")
    await done(key)


async def fetch_historical(session, sem, sport, raw_dir):
    start = datetime.strptime(HIST_START, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end   = datetime.strptime(HIST_END,   "%Y-%m-%d").replace(tzinfo=timezone.utc)
    dates = []
    cur = start
    while cur <= end:
        dates.append(cur.strftime("%Y-%m-%dT%H:%M:%SZ"))
        cur += timedelta(days=HIST_STEP)

    tasks = []
    for d in dates:
        key = f"hist_{sport}_{d[:10]}"
        if is_done(key):
            continue
        tasks.append(_fetch_hist_date(session, sem, sport, d, raw_dir))

    if not tasks:
        log.info(f"  {sport}: all historical dates already fetched")
        return

    log.info(f"  {sport}: {len(tasks)} historical dates to fetch")
    CHUNK = 200
    for i in range(0, len(tasks), CHUNK):
        if state.stop.is_set():
            break
        await asyncio.gather(*tasks[i:i+CHUNK])


async def _fetch_hist_date(session, sem, sport, date_iso, raw_dir):
    key = f"hist_{sport}_{date_iso[:10]}"
    if is_done(key) or state.stop.is_set():
        return
    data = await get(session, sem, f"/v4/historical/sports/{sport}/odds", {
        "date": date_iso, "regions": REGIONS,
        "markets": MARKETS, "oddsFormat": ODDS_FORMAT, "dateFormat": "iso"
    })
    if data is not None:
        events = data.get("data", []) if isinstance(data, dict) else data
        rows = flatten_odds(events, snap=date_iso)
        if rows:
            await append(rows, raw_dir / "odds.csv")
    await done(key)


# ──────────────────────────────────────────────────────────
# ML PROCESSING
# ──────────────────────────────────────────────────────────
def build_ml(sport, raw_dir, ml_dir):
    log.info(f"Building ML tables for {sport}...")
    ml_dir.mkdir(parents=True, exist_ok=True)

    odds_path   = raw_dir / "odds.csv"
    scores_path = raw_dir / "scores.csv"

    if not odds_path.exists():
        log.warning(f"  No odds.csv for {sport}, skipping ML")
        return

    # ── Load ──────────────────────────────────────────────
    odds = pd.read_csv(odds_path, low_memory=False,
                       dtype={"price": "float64", "point": "float64", "implied_prob": "float64"})
    odds["snapshot_time"] = pd.to_datetime(odds["snapshot_time"], utc=True, errors="coerce")
    odds["commence_time"] = pd.to_datetime(odds["commence_time"], utc=True, errors="coerce")
    odds = odds.dropna(subset=["event_id", "price", "market"])
    odds = odds[odds["market"].isin(["h2h","spreads","totals"])]

    if odds.empty:
        log.warning(f"  {sport}: no usable odds rows")
        return

    # ── Consensus per event x snapshot ────────────────────
    h2h      = odds[odds["market"] == "h2h"]
    spreads  = odds[odds["market"] == "spreads"]
    totals   = odds[odds["market"] == "totals"]

    rows = []
    for (eid, snap), grp in h2h.groupby(["event_id","snapshot_time"]):
        n_books = grp["bookmaker_key"].nunique()
        if n_books < MIN_BOOKS:
            continue

        meta = grp.iloc[0]
        home_prices = grp[grp["outcome_name"] == meta["home_team"]]["price"]
        away_prices = grp[grp["outcome_name"] == meta["away_team"]]["price"]
        draw_prices = grp[~grp["outcome_name"].isin([meta["home_team"], meta["away_team"]])]["price"]

        def stats(s):
            if s.empty:
                return dict(mean=np.nan, std=np.nan, min=np.nan, max=np.nan)
            return dict(mean=s.mean(), std=s.std() if len(s)>1 else 0.0, min=s.min(), max=s.max())

        hs = stats(home_prices)
        as_ = stats(away_prices)
        ds  = stats(draw_prices)

        ip_home = 1/hs["mean"]  if pd.notna(hs["mean"])  else np.nan
        ip_away = 1/as_["mean"] if pd.notna(as_["mean"]) else np.nan
        ip_draw = 1/ds["mean"]  if pd.notna(ds["mean"])  else 0.0

        total_ip = (ip_home or 0) + (ip_away or 0) + (ip_draw or 0)
        overround = total_ip - 1

        sp = spreads[(spreads["event_id"]==eid) & (spreads["snapshot_time"]==snap)]
        sp_home = sp[sp["outcome_name"]==meta["home_team"]]
        tot = totals[(totals["event_id"]==eid) & (totals["snapshot_time"]==snap)]
        tot_over = tot[tot["outcome_name"].str.lower().str.contains("over", na=False)]

        rows.append({
            "event_id":              eid,
            "sport_key":             meta["sport_key"],
            "home_team":             meta["home_team"],
            "away_team":             meta["away_team"],
            "commence_time":         meta["commence_time"],
            "snapshot_time":         snap,
            "hours_before_game":     round((meta["commence_time"] - snap).total_seconds()/3600, 2),
            "num_bookmakers":        n_books,
            # H2H consensus
            "h2h_home_mean":         round(hs["mean"],4)  if pd.notna(hs["mean"])  else np.nan,
            "h2h_home_std":          round(hs["std"],4),
            "h2h_home_min":          hs["min"],
            "h2h_home_max":          hs["max"],
            "h2h_away_mean":         round(as_["mean"],4) if pd.notna(as_["mean"]) else np.nan,
            "h2h_away_std":          round(as_["std"],4),
            "h2h_draw_mean":         round(ds["mean"],4)  if pd.notna(ds["mean"])  else np.nan,
            # Implied probabilities
            "implied_prob_home":     round(ip_home,4) if pd.notna(ip_home) else np.nan,
            "implied_prob_away":     round(ip_away,4) if pd.notna(ip_away) else np.nan,
            "implied_prob_draw":     round(ip_draw,4) if ip_draw else np.nan,
            # No-vig fair probabilities (overround removed = best ML feature)
            "prob_home_fair":        round(ip_home/total_ip,4) if total_ip else np.nan,
            "prob_away_fair":        round(ip_away/total_ip,4) if total_ip else np.nan,
            "prob_draw_fair":        round(ip_draw/total_ip,4) if total_ip and ip_draw else np.nan,
            "overround":             round(overround,4),
            "books_disagreement":    round(hs["std"],4),
            # Spreads
            "spread_point":          sp_home["point"].mean()  if not sp_home.empty else np.nan,
            "spread_price":          sp_home["price"].mean()  if not sp_home.empty else np.nan,
            # Totals
            "total_line":            tot_over["point"].mean() if not tot_over.empty else np.nan,
            "total_price":           tot_over["price"].mean() if not tot_over.empty else np.nan,
        })

    if not rows:
        log.warning(f"  {sport}: consensus built 0 rows (not enough bookmakers?)")
        return

    consensus = pd.DataFrame(rows)
    write(consensus, ml_dir / "consensus.csv")

    # ── Line movement ──────────────────────────────────────
    mv_rows = []
    for eid, grp in consensus.groupby("event_id"):
        grp = grp.sort_values("snapshot_time")
        if len(grp) < 2:
            continue
        first, last = grp.iloc[0], grp.iloc[-1]

        def mv(col):
            o, c = first.get(col), last.get(col)
            if pd.isna(o) or pd.isna(c):
                return np.nan, np.nan
            return round(c-o,4), round((c-o)/o*100,2) if o else np.nan

        hm, hm_pct = mv("h2h_home_mean")
        am, am_pct = mv("h2h_away_mean")
        sm, sm_pct = mv("spread_point")
        tm, tm_pct = mv("total_line")

        mv_rows.append({
            "event_id":             eid,
            "sport_key":            first["sport_key"],
            "home_team":            first["home_team"],
            "away_team":            first["away_team"],
            "commence_time":        first["commence_time"],
            "num_snapshots":        len(grp),
            "first_snapshot":       grp["snapshot_time"].min(),
            "last_snapshot":        grp["snapshot_time"].max(),
            # Opening
            "h2h_home_open":        first["h2h_home_mean"],
            "h2h_away_open":        first["h2h_away_mean"],
            "spread_open":          first["spread_point"],
            "total_open":           first["total_line"],
            "prob_home_open":       first["prob_home_fair"],
            "prob_away_open":       first["prob_away_fair"],
            # Closing
            "h2h_home_close":       last["h2h_home_mean"],
            "h2h_away_close":       last["h2h_away_mean"],
            "spread_close":         last["spread_point"],
            "total_close":          last["total_line"],
            "prob_home_close":      last["prob_home_fair"],
            "prob_away_close":      last["prob_away_fair"],
            # Movement
            "h2h_home_movement":    hm,
            "h2h_home_move_pct":    hm_pct,
            "h2h_away_movement":    am,
            "h2h_away_move_pct":    am_pct,
            "spread_movement":      sm,
            "spread_move_pct":      sm_pct,
            "total_movement":       tm,
            "total_move_pct":       tm_pct,
            # Sharp signal: line moved in same direction on both sides
            "sharp_on_home":        int(not pd.isna(hm) and hm < -0.05 and not pd.isna(am) and am > 0.05),
            "sharp_on_away":        int(not pd.isna(hm) and hm > 0.05  and not pd.isna(am) and am < -0.05),
            "overround_close":      last["overround"],
        })

    if not mv_rows:
        log.info(f"  {sport}: no multi-snapshot events for line movement")
    else:
        movement = pd.DataFrame(mv_rows)
        write(movement, ml_dir / "line_movement.csv")

    # ── Features wide (one row per event, closing line) ───
    closing = consensus.sort_values("snapshot_time").groupby("event_id").last().reset_index()
    closing["day_of_week"]    = closing["commence_time"].dt.dayofweek
    closing["month"]          = closing["commence_time"].dt.month
    closing["year"]           = closing["commence_time"].dt.year
    closing["hour_utc"]       = closing["commence_time"].dt.hour
    closing["is_weekend"]     = (closing["day_of_week"] >= 5).astype(int)
    closing["home_is_fav"]    = (closing["h2h_home_mean"] < closing["h2h_away_mean"]).astype(int)
    closing["market_edge"]    = (closing["prob_home_fair"] - closing["prob_away_fair"]).abs().round(4)

    if mv_rows and len(pd.DataFrame(mv_rows)) > 0:
        movement_df = pd.DataFrame(mv_rows)
        mv_cols = ["event_id","h2h_home_open","h2h_away_open","spread_open","total_open",
                   "prob_home_open","prob_away_open","h2h_home_movement","h2h_home_move_pct",
                   "h2h_away_movement","h2h_away_move_pct","spread_movement","total_movement",
                   "num_snapshots","sharp_on_home","sharp_on_away"]
        mv_cols = [c for c in mv_cols if c in movement_df.columns]
        features = closing.merge(movement_df[mv_cols], on="event_id", how="left")
    else:
        features = closing.copy()

    write(features, ml_dir / "features.csv")

    # ── Labels from scores ────────────────────────────────
    if not scores_path.exists():
        log.info(f"  {sport}: no scores.csv, skipping labels")
        return

    scores = pd.read_csv(scores_path, low_memory=False)
    completed = scores[scores["completed"] == True].copy()
    if completed.empty:
        return

    labels_list = []
    for eid, grp in completed.groupby("event_id"):
        meta = grp.iloc[0]
        score_map = dict(zip(grp["score_name"].astype(str), pd.to_numeric(grp["score_value"], errors="coerce")))
        home_score = score_map.get(str(meta["home_team"]))
        away_score = score_map.get(str(meta["away_team"]))
        if pd.isna(home_score) or pd.isna(away_score):
            # try first two keys
            vals = [v for v in score_map.values() if pd.notna(v)]
            if len(vals) >= 2:
                home_score, away_score = vals[0], vals[1]
            else:
                continue
        labels_list.append({
            "event_id":     eid,
            "home_team":    meta["home_team"],
            "away_team":    meta["away_team"],
            "home_score":   home_score,
            "away_score":   away_score,
            "total_points": home_score + away_score,
            "home_win":     int(home_score > away_score),
            "away_win":     int(away_score > home_score),
            "draw":         int(home_score == away_score),
            "score_diff":   round(home_score - away_score, 1),
        })

    if not labels_list:
        return

    labels = pd.DataFrame(labels_list)
    write(labels, ml_dir / "labels.csv")

    # ── ML ready (features + labels) ─────────────────────
    label_cols = ["event_id","home_win","away_win","draw","total_points","score_diff"]
    ml = features.merge(labels[[c for c in label_cols if c in labels.columns]], on="event_id", how="inner")
    ml["label_source"] = "scores_api"
    write(ml, ml_dir / "ml_ready.csv")

    log.info(f"  {sport} ML done: {len(features):,} feature rows, {len(ml):,} labeled rows")


# ──────────────────────────────────────────────────────────
# ORCHESTRATOR
# ──────────────────────────────────────────────────────────
async def run(mode):
    state.bucket = Bucket(RATE_PER_SEC)
    sem = asyncio.Semaphore(CONCURRENCY)

    conn = aiohttp.TCPConnector(limit=CONCURRENCY+5, ttl_dns_cache=300)
    async with aiohttp.ClientSession(connector=conn) as session:

        # Sports list
        log.info("Fetching sports list...")
        sports_data = await get(session, sem, "/v4/sports", {"all": "true"})
        if not sports_data:
            log.error("Could not fetch sports. Check API key.")
            return
        pd.DataFrame(sports_data).to_csv(OUT / "sports.csv", index=False)
        all_sports = [s["key"] for s in sports_data]
        log.info(f"{len(all_sports)} sports found")

        # Current odds + scores
        if mode in ("both", "current"):
            log.info("Fetching current odds + scores...")
            tasks = []
            for s in all_sports:
                raw_dir = OUT / "raw" / s
                raw_dir.mkdir(parents=True, exist_ok=True)
                tasks.append(fetch_current(session, sem, s, raw_dir))
                tasks.append(fetch_scores(session, sem, s, raw_dir))
            await asyncio.gather(*tasks)
            log.info("Current odds done.")

        if state.stop.is_set():
            return

        # Historical — sport by sport (each sport's dates in parallel)
        if mode in ("both", "historical"):
            log.info(f"Historical fetch: {HIST_START} -> {HIST_END}, step={HIST_STEP}d")
            for sport in all_sports:
                if state.stop.is_set():
                    break
                raw_dir = OUT / "raw" / sport
                raw_dir.mkdir(parents=True, exist_ok=True)
                await fetch_historical(session, sem, sport, raw_dir)

    log.info("Fetch complete.")


def build_all_ml():
    log.info("Building ML tables for all sports...")
    raw_base = OUT / "raw"
    if not raw_base.exists():
        log.warning("No raw/ directory found.")
        return
    sports = [d for d in raw_base.iterdir() if d.is_dir()]
    for sport_dir in sorted(sports):
        sport = sport_dir.name
        ml_dir = OUT / "ml" / sport
        try:
            build_ml(sport, sport_dir, ml_dir)
        except Exception as e:
            log.error(f"  {sport} ML failed: {e}")


def summary():
    log.info("\n" + "="*60)
    log.info("SUMMARY")
    log.info(f"  Requests  : {state.count}")
    log.info(f"  Credits used     : {state.used}")
    log.info(f"  Credits remaining: {state.remaining}")
    raw_sports = list((OUT/"raw").iterdir()) if (OUT/"raw").exists() else []
    ml_sports  = list((OUT/"ml").iterdir())  if (OUT/"ml").exists()  else []
    log.info(f"  Sports with raw data : {len(raw_sports)}")
    log.info(f"  Sports with ML data  : {len(ml_sports)}")
    log.info(f"  Output: {OUT.resolve()}")
    log.info("="*60)


# ──────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--current-only",    action="store_true")
    p.add_argument("--historical-only", action="store_true")
    p.add_argument("--reset",           action="store_true", help="Clear checkpoint")
    p.add_argument("--ml-only",         action="store_true", help="Skip fetch, rebuild ML tables only")
    p.add_argument("--concurrency",     type=int,   default=CONCURRENCY)
    p.add_argument("--rate",            type=float, default=RATE_PER_SEC)
    args = p.parse_args()

    if not API_KEY:
        print("Set ODDS_API_KEY env var.")
        sys.exit(1)

    CONCURRENCY  = args.concurrency
    RATE_PER_SEC = args.rate

    if args.reset and CKPT.exists():
        CKPT.unlink()
        log.info("Checkpoint cleared.")

    load_ckpt()

    mode = "both"
    if args.current_only:    mode = "current"
    if args.historical_only: mode = "historical"

    start = datetime.now()
    log.info(f"Starting | mode={mode} | concurrency={CONCURRENCY} | rate={RATE_PER_SEC}/s")

    if not args.ml_only:
        asyncio.run(run(mode))

    build_all_ml()
    summary()
    log.info(f"Total time: {datetime.now() - start}")
