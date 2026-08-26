import httpx
from datetime import datetime
from pymongo import UpdateOne
from app.core.config import get_settings
from app.core.database import candles_collection
from app.models.schemas import Candle

settings = get_settings()
BASE_URL = "https://api.twelvedata.com/time_series"


async def fetch_candles(pair: str, interval: str, output_size: int = 100) -> list[Candle]:
    """
    Fetch OHLCV candles for a forex pair from Twelve Data.
    interval examples: '1min', '5min', '15min', '1h', '4h', '1day'
    """
    params = {
        "symbol": pair,
        "interval": interval,
        "outputsize": output_size,
        "apikey": settings.twelve_data_api_key,
        "timezone": "UTC",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(BASE_URL, params=params)
        resp.raise_for_status()
        data = resp.json()

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
