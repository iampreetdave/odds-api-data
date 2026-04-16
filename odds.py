"""
Sports Odds Pipeline — Maximum Data Extraction
Hits every endpoint The Odds API offers, all 5 regions, all markets.
Fetches most recent data first so if credits run out you still have
complete coverage of recent years.

Credits: ~4.9M needed for full historical (5 regions x 162 sports x 2024 days)
         Script auto-stops at QUOTA_STOP credits remaining.

Usage:
  python fetch.py                    # full run
  python fetch.py --current-only
  python fetch.py --historical-only
  python fetch.py --reset            # clear checkpoint, start fresh
  python fetch.py --ml-only          # skip fetch, rebuild ML tables

Env: ODDS_API_KEY
"""

import os, sys, json, asyncio, logging, time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import aiohttp, aiofiles
import pandas as pd
import numpy as np

API_KEY   = os.environ.get("ODDS_API_KEY", "2654835dd9c6779ec43efbe938a8ebe3")
BASE_URL  = "https://api.the-odds-api.com"

REGIONS_MAIN     = "us,us2,uk,eu,au"
REGIONS_OUTRIGHT = "us,uk,eu"
MARKETS_MAIN     = "h2h,spreads,totals"
MARKETS_OUTRIGHT = "outrights"
ODDS_FORMAT      = "decimal"
CONCURRENCY      = 10
RATE_PER_SEC     = 5.0
QUOTA_STOP       = 50
HIST_END         = datetime.now(timezone.utc).strftime("%Y-%m-%d")
HIST_START       = "2020-10-01"
HIST_STEP        = 1
MIN_BOOKS        = 3

OUT  = Path("odds_data")
CKPT = OUT / "checkpoint.json"

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

class State:
    def __init__(self):
        self.remaining = None
        self.used      = None
        self.count     = 0
        self.lock      = asyncio.Lock()
        self.stop      = asyncio.Event()
        self.ckpt      = {"done": []}
        self.bucket    = None

state = State()

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

class Bucket:
    def __init__(self, rate):
        self.rate   = rate
        self.tokens = rate * 2
        self.last   = time.monotonic()
        self._lock  = asyncio.Lock()

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
                            log.info(f"Requests: {state.count:,} | Rate: {state.bucket.rate:.1f}/s | Credits left: {state.remaining:,}")
                        if state.remaining is not None and state.remaining < QUOTA_STOP:
                            log.warning(f"Quota exhausted ({state.remaining} left). Stopping.")
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

_wlock = asyncio.Lock()

async def append(rows, path):
    if not rows:
        return
    async with _wlock:
        path.parent.mkdir(parents=True, exist_ok=True)
        df  = pd.DataFrame(rows)
        hdr = not path.exists()
        df.to_csv(path, mode="a", header=hdr, index=False)

def write(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info(f"  Saved {len(df):,} rows -> {path.relative_to(OUT)}")

def flatten_odds(events, snap=None, market_type="main"):
    snap = snap or datetime.now(timezone.utc).isoformat()
    rows = []
    for ev in events:
        base = {
            "event_id": ev.get("id"), "sport_key": ev.get("sport_key"),
            "sport_title": ev.get("sport_title"), "commence_time": ev.get("commence_time"),
            "home_team": ev.get("home_team"), "away_team": ev.get("away_team"),
            "snapshot_time": snap, "market_type": market_type,
        }
        for bm in ev.get("bookmakers", []):
            b = {**base, "bookmaker_key": bm.get("key"), "bookmaker_title": bm.get("title"),
                 "last_update": bm.get("last_update")}
            for mkt in bm.get("markets", []):
                for out in mkt.get("outcomes", []):
                    price = out.get("price")
                    rows.append({**b, "market": mkt.get("key"), "outcome_name": out.get("name"),
                                 "price": price, "point": out.get("point"),
                                 "implied_prob": round(1/price, 6) if price and price > 0 else None})
    return rows

def flatten_scores(events):
    rows = []
    for ev in events:
        base = {"event_id": ev.get("id"), "sport_key": ev.get("sport_key"),
                "home_team": ev.get("home_team"), "away_team": ev.get("away_team"),
                "commence_time": ev.get("commence_time"), "completed": ev.get("completed"),
                "last_update": ev.get("last_update")}
        for sc in (ev.get("scores") or []):
            rows.append({**base, "score_name": sc.get("name"), "score_value": sc.get("score")})
        if not ev.get("scores"):
            rows.append({**base, "score_name": None, "score_value": None})
    return rows

def flatten_events(events, sport, snap=None):
    snap = snap or datetime.now(timezone.utc).isoformat()
    return [{"event_id": ev.get("id"), "sport_key": sport, "sport_title": ev.get("sport_title"),
             "home_team": ev.get("home_team"), "away_team": ev.get("away_team"),
             "commence_time": ev.get("commence_time"), "snapshot_time": snap}
            for ev in events if ev.get("id")]

async def fetch_current_main(session, sem, sport, raw_dir):
    key = f"cur_main_{sport}"
    if is_done(key) or state.stop.is_set(): return
    data = await get(session, sem, f"/v4/sports/{sport}/odds", {
        "regions": REGIONS_MAIN, "markets": MARKETS_MAIN, "oddsFormat": ODDS_FORMAT, "dateFormat": "iso"})
    if data:
        await append(flatten_odds(data, market_type="main"), raw_dir / "odds.csv")
    await done(key)

async def fetch_current_outrights(session, sem, sport, raw_dir, has_outrights):
    if not has_outrights: return
    key = f"cur_out_{sport}"
    if is_done(key) or state.stop.is_set(): return
    data = await get(session, sem, f"/v4/sports/{sport}/odds", {
        "regions": REGIONS_OUTRIGHT, "markets": MARKETS_OUTRIGHT, "oddsFormat": ODDS_FORMAT, "dateFormat": "iso"})
    if data:
        await append(flatten_odds(data, market_type="outrights"), raw_dir / "odds.csv")
    await done(key)

async def fetch_scores(session, sem, sport, raw_dir):
    key = f"scores_{sport}"
    if is_done(key) or state.stop.is_set(): return
    data = await get(session, sem, f"/v4/sports/{sport}/scores", {"daysFrom": 3, "dateFormat": "iso"})
    if data:
        await append(flatten_scores(data), raw_dir / "scores.csv")
    await done(key)

async def fetch_events_and_per_event_odds(session, sem, sport, raw_dir):
    key = f"events_{sport}"
    if is_done(key) or state.stop.is_set(): return
    data = await get(session, sem, f"/v4/sports/{sport}/events", {"dateFormat": "iso"})
    if not data:
        await done(key); return
    rows = flatten_events(data, sport)
    await append(rows, raw_dir / "events.csv")
    tasks = [_fetch_event_odds(session, sem, sport, ev["event_id"], raw_dir) for ev in rows]
    await asyncio.gather(*tasks)
    await done(key)

async def _fetch_event_odds(session, sem, sport, event_id, raw_dir):
    key = f"ev_odds_{event_id}"
    if is_done(key) or state.stop.is_set(): return
    data = await get(session, sem, f"/v4/sports/{sport}/events/{event_id}/odds", {
        "regions": REGIONS_MAIN, "markets": f"{MARKETS_MAIN},{MARKETS_OUTRIGHT}",
        "oddsFormat": ODDS_FORMAT, "dateFormat": "iso"})
    if data:
        events = [data] if isinstance(data, dict) else data
        await append(flatten_odds(events, market_type="event_detail"), raw_dir / "odds.csv")
    await done(key)

async def _fetch_hist_date(session, sem, sport, date_iso, raw_dir):
    key = f"hist_{sport}_{date_iso[:10]}"
    if is_done(key) or state.stop.is_set(): return
    data = await get(session, sem, f"/v4/historical/sports/{sport}/odds", {
        "date": date_iso, "regions": REGIONS_MAIN, "markets": MARKETS_MAIN,
        "oddsFormat": ODDS_FORMAT, "dateFormat": "iso"})
    if data is not None:
        events = data.get("data", []) if isinstance(data, dict) else data
        rows = flatten_odds(events, snap=date_iso, market_type="historical_main")
        if rows:
            await append(rows, raw_dir / "odds.csv")
    await done(key)

async def _fetch_hist_outrights(session, sem, sport, date_iso, raw_dir):
    key = f"hist_out_{sport}_{date_iso[:10]}"
    if is_done(key) or state.stop.is_set(): return
    data = await get(session, sem, f"/v4/historical/sports/{sport}/odds", {
        "date": date_iso, "regions": REGIONS_OUTRIGHT, "markets": MARKETS_OUTRIGHT,
        "oddsFormat": ODDS_FORMAT, "dateFormat": "iso"})
    if data is not None:
        events = data.get("data", []) if isinstance(data, dict) else data
        rows = flatten_odds(events, snap=date_iso, market_type="historical_outrights")
        if rows:
            await append(rows, raw_dir / "odds.csv")
    await done(key)

async def _fetch_hist_events(session, sem, sport, date_iso, raw_dir):
    key = f"hist_ev_{sport}_{date_iso[:10]}"
    if is_done(key) or state.stop.is_set(): return
    data = await get(session, sem, f"/v4/historical/sports/{sport}/events", {
        "date": date_iso, "dateFormat": "iso"})
    if data is not None:
        events = data.get("data", []) if isinstance(data, dict) else data
        rows = flatten_events(events, sport, snap=date_iso)
        if rows:
            await append(rows, raw_dir / "events.csv")
    await done(key)

async def run(mode):
    state.bucket = Bucket(RATE_PER_SEC)
    sem  = asyncio.Semaphore(CONCURRENCY)
    conn = aiohttp.TCPConnector(limit=CONCURRENCY+5, ttl_dns_cache=300)

    async with aiohttp.ClientSession(connector=conn) as session:

        log.info("Fetching sports list...")
        sports_data = await get(session, sem, "/v4/sports", {"all": "true"})
        if not sports_data:
            log.error("Could not fetch sports. Check API key."); return
        pd.DataFrame(sports_data).to_csv(OUT / "sports.csv", index=False)
        sports_meta = {s["key"]: s for s in sports_data}
        all_sports  = list(sports_meta.keys())
        log.info(f"{len(all_sports)} sports found")

        if mode in ("both", "current"):
            log.info("Fetching current data: odds, outrights, scores, events, per-event odds...")
            tasks = []
            for sport in all_sports:
                raw_dir       = OUT / "raw" / sport
                has_outrights = sports_meta[sport].get("has_outrights", False)
                tasks += [
                    fetch_current_main(session, sem, sport, raw_dir),
                    fetch_current_outrights(session, sem, sport, raw_dir, has_outrights),
                    fetch_scores(session, sem, sport, raw_dir),
                    fetch_events_and_per_event_odds(session, sem, sport, raw_dir),
                ]
            await asyncio.gather(*tasks)
            log.info("Current data done.")

        if state.stop.is_set(): return

        if mode in ("both", "historical"):
            end   = datetime.strptime(HIST_END,   "%Y-%m-%d").replace(tzinfo=timezone.utc)
            start = datetime.strptime(HIST_START, "%Y-%m-%d").replace(tzinfo=timezone.utc)

            # Newest first
            all_dates = []
            cur = end
            while cur >= start:
                all_dates.append(cur.strftime("%Y-%m-%dT12:00:00Z"))
                cur -= timedelta(days=HIST_STEP)

            total_tasks = len(all_dates) * len(all_sports)
            log.info(
                f"Historical: {len(all_dates)} dates x {len(all_sports)} sports = {total_tasks:,} tasks\n"
                f"  Strategy: date-first (all sports per date) so credits give even coverage\n"
                f"  Regions: {REGIONS_MAIN} | Markets: {MARKETS_MAIN} + outrights for applicable sports"
            )

            # Process one date at a time, all sports concurrently per date
            # This gives even temporal coverage: if credits run out on date X,
            # you have complete data for ALL sports from today back to X
            CHUNK = 30  # dates per batch
            for i in range(0, len(all_dates), CHUNK):
                if state.stop.is_set(): break
                date_batch = all_dates[i:i+CHUNK]
                tasks = []
                for date_iso in date_batch:
                    for sport in all_sports:
                        raw_dir       = OUT / "raw" / sport
                        raw_dir.mkdir(parents=True, exist_ok=True)
                        has_outrights = sports_meta[sport].get("has_outrights", False)
                        tasks.append(_fetch_hist_date(session, sem, sport, date_iso, raw_dir))
                        if has_outrights:
                            tasks.append(_fetch_hist_outrights(session, sem, sport, date_iso, raw_dir))
                        # Historical event metadata (weekly only, cheap)
                        if datetime.strptime(date_iso[:10], "%Y-%m-%d").weekday() == 0:  # Mondays only
                            tasks.append(_fetch_hist_events(session, sem, sport, date_iso, raw_dir))
                await asyncio.gather(*tasks)
                pct = min(100, (i+CHUNK)/len(all_dates)*100)
                log.info(
                    f"  Batch done: {date_batch[0][:10]} | "
                    f"{pct:.1f}% of dates | "
                    f"Credits left: {state.remaining:,}"
                )

    log.info("Fetch complete.")

def build_ml(sport, raw_dir, ml_dir):
    odds_path   = raw_dir / "odds.csv"
    scores_path = raw_dir / "scores.csv"
    if not odds_path.exists(): return
    try:
        odds = pd.read_csv(odds_path, low_memory=False,
                           dtype={"price": "float64", "point": "float64", "implied_prob": "float64"})
    except Exception as e:
        log.warning(f"  {sport}: {e}"); return

    odds["snapshot_time"] = pd.to_datetime(odds["snapshot_time"], utc=True, errors="coerce")
    odds["commence_time"] = pd.to_datetime(odds["commence_time"], utc=True, errors="coerce")
    odds = odds.dropna(subset=["event_id","price","market"])
    core = odds[odds["market"].isin(["h2h","spreads","totals"])].copy()
    if core.empty: return

    h2h     = core[core["market"] == "h2h"]
    spreads = core[core["market"] == "spreads"]
    totals  = core[core["market"] == "totals"]

    rows = []
    for (eid, snap), grp in h2h.groupby(["event_id","snapshot_time"]):
        n = grp["bookmaker_key"].nunique()
        if n < MIN_BOOKS: continue
        meta = grp.iloc[0]
        hp = grp[grp["outcome_name"] == meta["home_team"]]["price"]
        ap = grp[grp["outcome_name"] == meta["away_team"]]["price"]
        dp = grp[~grp["outcome_name"].isin([meta["home_team"], meta["away_team"]])]["price"]

        def st(s):
            if s.empty: return dict(mean=np.nan, std=np.nan, min=np.nan, max=np.nan)
            return dict(mean=s.mean(), std=s.std() if len(s)>1 else 0.0, min=s.min(), max=s.max())

        hs, as_, ds = st(hp), st(ap), st(dp)
        ih = 1/hs["mean"] if pd.notna(hs["mean"]) else np.nan
        ia = 1/as_["mean"] if pd.notna(as_["mean"]) else np.nan
        id_ = 1/ds["mean"] if pd.notna(ds["mean"]) else 0.0
        tip = (ih or 0) + (ia or 0) + (id_ or 0)

        sp  = spreads[(spreads["event_id"]==eid) & (spreads["snapshot_time"]==snap)]
        sph = sp[sp["outcome_name"]==meta["home_team"]]
        tot = totals[(totals["event_id"]==eid) & (totals["snapshot_time"]==snap)]
        tov = tot[tot["outcome_name"].str.lower().str.contains("over", na=False)]

        rows.append({
            "event_id": eid, "sport_key": meta["sport_key"],
            "home_team": meta["home_team"], "away_team": meta["away_team"],
            "commence_time": meta["commence_time"], "snapshot_time": snap,
            "hours_before_game": round((meta["commence_time"]-snap).total_seconds()/3600, 2),
            "num_bookmakers": n,
            "h2h_home_mean": round(hs["mean"],4) if pd.notna(hs["mean"]) else np.nan,
            "h2h_home_std": round(hs["std"],4), "h2h_home_min": hs["min"], "h2h_home_max": hs["max"],
            "h2h_away_mean": round(as_["mean"],4) if pd.notna(as_["mean"]) else np.nan,
            "h2h_away_std": round(as_["std"],4),
            "h2h_draw_mean": round(ds["mean"],4) if pd.notna(ds["mean"]) else np.nan,
            "implied_prob_home": round(ih,4) if pd.notna(ih) else np.nan,
            "implied_prob_away": round(ia,4) if pd.notna(ia) else np.nan,
            "implied_prob_draw": round(id_,4) if id_ else np.nan,
            "prob_home_fair": round(ih/tip,4) if tip else np.nan,
            "prob_away_fair": round(ia/tip,4) if tip else np.nan,
            "prob_draw_fair": round(id_/tip,4) if tip and id_ else np.nan,
            "overround": round(tip-1,4), "books_disagreement": round(hs["std"],4),
            "spread_point": sph["point"].mean() if not sph.empty else np.nan,
            "spread_price": sph["price"].mean() if not sph.empty else np.nan,
            "total_line": tov["point"].mean() if not tov.empty else np.nan,
            "total_price": tov["price"].mean() if not tov.empty else np.nan,
        })

    if not rows: return
    consensus = pd.DataFrame(rows)
    write(consensus, ml_dir / "consensus.csv")

    mv_rows = []
    for eid, grp in consensus.groupby("event_id"):
        grp = grp.sort_values("snapshot_time")
        if len(grp) < 2: continue
        f, l = grp.iloc[0], grp.iloc[-1]
        def mv(col):
            o, c = f.get(col), l.get(col)
            if pd.isna(o) or pd.isna(c): return np.nan, np.nan
            return round(c-o,4), round((c-o)/o*100,2) if o else np.nan
        hm, hmp = mv("h2h_home_mean"); am, amp = mv("h2h_away_mean")
        sm, _   = mv("spread_point");  tm, _   = mv("total_line")
        mv_rows.append({
            "event_id": eid, "sport_key": f["sport_key"],
            "home_team": f["home_team"], "away_team": f["away_team"],
            "commence_time": f["commence_time"], "num_snapshots": len(grp),
            "first_snapshot": grp["snapshot_time"].min(), "last_snapshot": grp["snapshot_time"].max(),
            "h2h_home_open": f["h2h_home_mean"], "h2h_away_open": f["h2h_away_mean"],
            "spread_open": f["spread_point"], "total_open": f["total_line"],
            "prob_home_open": f["prob_home_fair"], "prob_away_open": f["prob_away_fair"],
            "h2h_home_close": l["h2h_home_mean"], "h2h_away_close": l["h2h_away_mean"],
            "spread_close": l["spread_point"], "total_close": l["total_line"],
            "prob_home_close": l["prob_home_fair"], "prob_away_close": l["prob_away_fair"],
            "h2h_home_movement": hm, "h2h_home_move_pct": hmp,
            "h2h_away_movement": am, "h2h_away_move_pct": amp,
            "spread_movement": sm, "total_movement": tm,
            "sharp_on_home": int(pd.notna(hm) and hm < -0.05 and pd.notna(am) and am > 0.05),
            "sharp_on_away": int(pd.notna(hm) and hm > 0.05  and pd.notna(am) and am < -0.05),
            "overround_close": l["overround"],
        })

    if mv_rows:
        write(pd.DataFrame(mv_rows), ml_dir / "line_movement.csv")

    closing = consensus.sort_values("snapshot_time").groupby("event_id").last().reset_index()
    closing["day_of_week"] = closing["commence_time"].dt.dayofweek
    closing["month"]       = closing["commence_time"].dt.month
    closing["year"]        = closing["commence_time"].dt.year
    closing["hour_utc"]    = closing["commence_time"].dt.hour
    closing["is_weekend"]  = (closing["day_of_week"] >= 5).astype(int)
    closing["home_is_fav"] = (closing["h2h_home_mean"] < closing["h2h_away_mean"]).astype(int)
    closing["market_edge"] = (closing["prob_home_fair"] - closing["prob_away_fair"]).abs().round(4)

    if mv_rows:
        mv_df = pd.DataFrame(mv_rows)
        mc = [c for c in ["event_id","h2h_home_open","h2h_away_open","spread_open","total_open",
                           "prob_home_open","prob_away_open","h2h_home_movement","h2h_home_move_pct",
                           "h2h_away_movement","h2h_away_move_pct","spread_movement","total_movement",
                           "num_snapshots","sharp_on_home","sharp_on_away"] if c in mv_df.columns]
        features = closing.merge(mv_df[mc], on="event_id", how="left")
    else:
        features = closing.copy()
    write(features, ml_dir / "features.csv")

    if not scores_path.exists(): return
    try:
        scores = pd.read_csv(scores_path, low_memory=False)
    except Exception: return
    completed = scores[scores["completed"] == True].copy()
    if completed.empty: return

    labels_list = []
    for eid, grp in completed.groupby("event_id"):
        meta = grp.iloc[0]
        sm = dict(zip(grp["score_name"].astype(str), pd.to_numeric(grp["score_value"], errors="coerce")))
        hs = sm.get(str(meta["home_team"])); as_ = sm.get(str(meta["away_team"]))
        if pd.isna(hs) or pd.isna(as_):
            vals = [v for v in sm.values() if pd.notna(v)]
            if len(vals) >= 2: hs, as_ = vals[0], vals[1]
            else: continue
        labels_list.append({
            "event_id": eid, "home_score": hs, "away_score": as_,
            "total_points": hs+as_, "home_win": int(hs>as_), "away_win": int(as_>hs),
            "draw": int(hs==as_), "score_diff": round(hs-as_, 1),
        })

    if not labels_list: return
    labels = pd.DataFrame(labels_list)
    write(labels, ml_dir / "labels.csv")
    ml = features.merge(
        labels[["event_id","home_win","away_win","draw","total_points","score_diff"]],
        on="event_id", how="inner"
    )
    write(ml, ml_dir / "ml_ready.csv")
    log.info(f"  {sport}: {len(features):,} feature rows | {len(ml):,} labeled")

def build_all_ml():
    log.info("Building ML tables...")
    raw_base = OUT / "raw"
    if not raw_base.exists(): return
    for sport_dir in sorted(raw_base.iterdir()):
        if not sport_dir.is_dir(): continue
        try:
            build_ml(sport_dir.name, sport_dir, OUT / "ml" / sport_dir.name)
        except Exception as e:
            log.error(f"  {sport_dir.name} ML failed: {e}")

def summary():
    raw_sports = list((OUT/"raw").iterdir()) if (OUT/"raw").exists() else []
    ml_sports  = list((OUT/"ml").iterdir())  if (OUT/"ml").exists()  else []
    log.info("\n" + "="*60)
    log.info("SUMMARY")
    log.info(f"  Requests          : {state.count:,}")
    log.info(f"  Credits used      : {state.used}")
    log.info(f"  Credits remaining : {state.remaining}")
    log.info(f"  Checkpoint tasks  : {len(state.ckpt.get('done',[])  ):,}")
    log.info(f"  Sports with raw   : {len(raw_sports)}")
    log.info(f"  Sports with ML    : {len(ml_sports)}")
    log.info(f"  Output            : {OUT.resolve()}")
    log.info("="*60)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--current-only",    action="store_true")
    p.add_argument("--historical-only", action="store_true")
    p.add_argument("--reset",           action="store_true")
    p.add_argument("--ml-only",         action="store_true")
    p.add_argument("--concurrency",     type=int,   default=CONCURRENCY)
    p.add_argument("--rate",            type=float, default=RATE_PER_SEC)
    args = p.parse_args()

    if not API_KEY:
        print("Set ODDS_API_KEY env var."); sys.exit(1)

    CONCURRENCY  = args.concurrency
    RATE_PER_SEC = args.rate

    if args.reset and CKPT.exists():
        CKPT.unlink(); log.info("Checkpoint cleared.")

    load_ckpt()
    mode = "both"
    if args.current_only:    mode = "current"
    if args.historical_only: mode = "historical"

    start = datetime.now()
    log.info(f"Starting | mode={mode} | concurrency={CONCURRENCY} | rate={RATE_PER_SEC}/s")
    log.info(f"Regions: {REGIONS_MAIN} | Markets: {MARKETS_MAIN} + {MARKETS_OUTRIGHT}")

    if not args.ml_only:
        asyncio.run(run(mode))

    build_all_ml()
    summary()
    log.info(f"Total time: {datetime.now() - start}")
