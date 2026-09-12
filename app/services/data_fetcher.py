import asyncio
import httpx
from datetime import datetime
from pymongo import UpdateOne
from app.core.config import get_settings
from app.core.database import candles_collection
from app.models.schemas import Candle

settings = get_settings()
BASE_URL = "https://api.twelvedata.com/time_series"

# Twelve Data's free tier caps requests per minute (separately from the 800/day credit
# cap) -- ingesting several intervals back-to-back (each looping all 4 pairs with zero
# delay between them, see /ingest/{interval} in main.py) can burst past that per-minute
# limit even while staying well under the daily one. Confirmed live: a 4-interval sweep
# (5min/15min/1h/4h) 429'd on every pair for 4h specifically, then 1day succeeded right
# after -- proof the limit is a transient, self-resolving per-minute window, not a hard
# stop. Retry with backoff here instead of failing the whole ingest outright.
MAX_RETRIES_ON_RATE_LIMIT = 3
RETRY_BACKOFF_SECONDS = 20


async def fetch_candles(
    pair: str, interval: str, output_size: int = 100, end_date: str | None = None
) -> list[Candle]:
    """
    Fetch OHLCV candles for a forex pair from Twelve Data.
    interval examples: '1min', '5min', '15min', '1h', '4h', '1day'
    end_date: optional 'YYYY-MM-DD HH:MM:SS' (or 'YYYY-MM-DD') to page backward through
    history instead of returning the most recent output_size candles -- see backfill_batch.
    """
    params = {
        "symbol": pair,
        "interval": interval,
        "outputsize": output_size,
        "apikey": settings.twelve_data_api_key,
        "timezone": "UTC",
    }
    if end_date is not None:
        params["end_date"] = end_date

    for attempt in range(MAX_RETRIES_ON_RATE_LIMIT + 1):
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(BASE_URL, params=params)

        if resp.status_code == 429 and attempt < MAX_RETRIES_ON_RATE_LIMIT:
            await asyncio.sleep(RETRY_BACKOFF_SECONDS)
            continue

        resp.raise_for_status()
        data = resp.json()
        break

    if data.get("status") == "error":
        raise ValueError(f"Twelve Data error for {pair}/{interval}: {data.get('message')}")

    values = data.get("values", [])
    candles = []
    for v in values:
        candles.append(
            Candle(
                pair=pair,
                interval=interval,
                timestamp=datetime.strptime(v["datetime"], "%Y-%m-%d %H:%M:%S")
                if len(v["datetime"]) > 10
                else datetime.strptime(v["datetime"], "%Y-%m-%d"),
                open=float(v["open"]),
                high=float(v["high"]),
                low=float(v["low"]),
                close=float(v["close"]),
                volume=float(v.get("volume", 0)) if v.get("volume") else None,
            )
        )
    return candles


async def store_candles(candles: list[Candle]) -> int:
    """
    Upsert candles into MongoDB, avoiding duplicates on (pair, interval, timestamp).
    Uses one bulk_write instead of one round-trip per candle — with a few hundred
    candles per ingest, sequential update_one calls turn minutes of network latency
    into what should be a single request.
    """
    if not candles:
        return 0

    operations = [
        UpdateOne(
            {"pair": c.pair, "interval": c.interval, "timestamp": c.timestamp},
            {"$set": c.model_dump()},
            upsert=True,
        )
        for c in candles
    ]
    result = await candles_collection.bulk_write(operations, ordered=False)
    return result.upserted_count + result.modified_count


INTERVAL_MINUTES = {
    "1min": 1, "5min": 5, "15min": 15, "30min": 30,
    "1h": 60, "2h": 120, "4h": 240, "8h": 480, "1day": 1440,
}

# Caps how far a single auto-backfill (see fetch_and_store) will reach back, even if the
# true gap is larger -- a single Twelve Data call costs the same 1 credit regardless of
# outputsize, so there's no quota reason to cap this low, but an unbounded cap risks a
# single routine ingest call unexpectedly pulling years of history if `latest` is ever
# wildly wrong (e.g. a bad timestamp). 1000 candles covers ~3.5 days at 5min -- comfortably
# more than any gap this project has actually seen (the GitHub Actions schedule-trigger gap
# that motivated this was ~2 hours). A genuinely bigger deliberate backfill still works the
# same way it always has: call /ingest with an explicit larger output_size.
MAX_AUTO_BACKFILL_CANDLES = 1000


async def fetch_and_store(pair: str, interval: str, output_size: int = 100) -> int:
    """
    Fetches and stores candles, automatically widening output_size to cover any gap since
    the last candle already stored for this pair/interval. Without this, a delayed or
    dropped scheduled trigger (GitHub Actions `schedule` runs are documented as best-effort
    and can be silently skipped under load -- confirmed in practice: every run in this
    project's Actions history shows "success", but one scheduled trigger simply never fired
    for ~2 hours) would leave a permanent hole in candles_collection, since the routine cron
    call only ever requests the last few candles (output_size=5), not enough to close a gap
    once one has opened. The next successful ingest call for that pair/interval now closes
    it on its own instead of requiring a manual deep backfill.
    """
    if interval in INTERVAL_MINUTES:
        latest = await candles_collection.find_one(
            {"pair": pair, "interval": interval}, sort=[("timestamp", -1)]
        )
        if latest is not None:
            gap_minutes = (datetime.utcnow() - latest["timestamp"]).total_seconds() / 60
            candles_needed = int(gap_minutes / INTERVAL_MINUTES[interval]) + 2  # +2: rounding/in-progress bar buffer
            output_size = min(max(output_size, candles_needed), MAX_AUTO_BACKFILL_CANDLES)

    candles = await fetch_candles(pair, interval, output_size)
    return await store_candles(candles)


# Twelve Data's binding constraint for a deep backfill is the per-minute rate limit, not
# the 800/day credit cap -- outputsize doesn't change a call's credit cost (see
# MAX_AUTO_BACKFILL_CANDLES's comment above), so this just paces calls comfortably under
# that per-minute ceiling rather than trying to find its exact edge.
BACKFILL_CALL_DELAY_SECONDS = 8

# Twelve Data's documented max outputsize per call, even on paid plans.
BACKFILL_PAGE_SIZE = 5000


async def backfill_batch(pair: str, interval: str, start_date: str, max_calls: int = 5) -> dict:
    """
    Walks candle history backward from whatever's already stored for this pair/interval (or
    from "now" if nothing is stored yet) toward start_date, one Twelve Data call per page,
    stopping after max_calls so a single call stays short enough for one HTTP request/gateway
    timeout -- a full 2010-to-now backfill at 5min is 300+ calls per pair, far more than should
    ever run inside one synchronous request. Progress is just "the earliest timestamp already
    in candles_collection for this pair/interval", so calling this repeatedly (see
    POST /ingest/backfill/{interval} and .github/workflows/backfill-history.yml) resumes
    correctly with no separate state to track, and is safe to re-run after a failed/timed-out
    call since every page is stored before the next one is requested.
    """
    earliest = await candles_collection.find_one(
        {"pair": pair, "interval": interval}, sort=[("timestamp", 1)]
    )
    cursor = earliest["timestamp"] if earliest else None
    start = datetime.strptime(start_date, "%Y-%m-%d")

    if cursor is not None and cursor <= start:
        return {"done": True, "calls_made": 0, "candles_stored": 0, "oldest": cursor.isoformat()}

    calls_made = 0
    candles_stored = 0
    done = False
    while calls_made < max_calls:
        end_date_param = cursor.strftime("%Y-%m-%d %H:%M:%S") if cursor else None
        candles = await fetch_candles(pair, interval, output_size=BACKFILL_PAGE_SIZE, end_date=end_date_param)
        calls_made += 1
        if not candles:
            done = True
            break
        candles_stored += await store_candles(candles)
        oldest_in_page = min(c.timestamp for c in candles)
        if cursor is not None and oldest_in_page >= cursor:
            # Same page as last time -- Twelve Data has nothing older to give us.
            done = True
            break
        cursor = oldest_in_page
        if cursor <= start:
            done = True
            break
        if calls_made < max_calls:
            await asyncio.sleep(BACKFILL_CALL_DELAY_SECONDS)

    return {
        "done": done,
        "calls_made": calls_made,
        "candles_stored": candles_stored,
        "oldest": cursor.isoformat() if cursor else None,
    }
