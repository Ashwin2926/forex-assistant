from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import get_settings

settings = get_settings()

client = AsyncIOMotorClient(settings.mongodb_uri)
db = client[settings.mongodb_db_name]

# Collections
candles_collection = db["candles"]                    # raw OHLCV data
signals_collection = db["signals"]                    # live buy/sell/hold signals + outcomes
backtest_signals_collection = db["backtest_signals"]  # per-signal backtest replay results
backtest_runs_collection = db["backtest_runs"]        # one summary doc per backtest run
paper_trades_collection = db["paper_trades"]          # Deriv Multipliers demo-account executions


async def init_indexes():
    """Call once at startup to ensure efficient queries."""
    await candles_collection.create_index(
        [("pair", 1), ("interval", 1), ("timestamp", -1)], unique=True
    )
    await signals_collection.create_index([("pair", 1), ("timestamp", -1)])
    await signals_collection.create_index([("status", 1)])
    await backtest_signals_collection.create_index([("run_id", 1), ("timestamp", 1)])
    await backtest_runs_collection.create_index([("run_id", 1)], unique=True)
    await backtest_runs_collection.create_index([("pair", 1), ("interval", 1), ("profile", 1), ("created_at", -1)])
    await paper_trades_collection.create_index([("status", 1)])
    await paper_trades_collection.create_index([("opened_at", -1)])
