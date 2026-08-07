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


async def fetch_and_store(pair: str, interval: str, output_size: int = 100) -> int:
    candles = await fetch_candles(pair, interval, output_size)
    return await store_candles(candles)
