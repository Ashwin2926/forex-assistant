from pathlib import Path

import pandas as pd

# scripts/export_candles.py writes here; committed into the repo so it survives
# independently of Mongo Atlas's storage cap and is available to anything that checks out
# this repo (the FastAPI Cloud backend, since it also deploys from here, and GitHub Actions
# training runs).
ARCHIVE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "candles"


def _slug(pair: str) -> str:
    return pair.replace("/", "_")


def load_archived_candles(pair: str, interval: str) -> pd.DataFrame:
    """
    Reads the committed CSV export for this pair/interval, or an empty frame (same columns)
    if none exists yet -- callers should treat a missing archive as "no older history",
    not an error, since not every pair/interval has been backfilled/exported.
    """
    path = ARCHIVE_DIR / f"{_slug(pair)}_{interval}.csv.gz"
    columns = ["timestamp", "open", "high", "low", "close", "volume"]
    if not path.exists():
        return pd.DataFrame(columns=columns)
    return pd.read_csv(path, compression="gzip", parse_dates=["timestamp"])


async def load_full_candle_history(pair: str, interval: str) -> list[dict]:
    """
    Merges the committed archive (deep history, exported periodically) with whatever's
    currently live in candles_collection (ingestion continues forward from wherever it last
    left off -- see data_fetcher.fetch_and_store's gap-fill logic -- so Mongo always covers
    at least "recent", regardless of how far back the archive file goes). Overlap is
    resolved in Mongo's favor (keep="last" after concatenating archive-then-Mongo) since
    Mongo is always at least as fresh, and is exactly what a "Sync now" catch-up run updates.

    Used by the deep-history backtest endpoints (/backtest, /backtest/sweep,
    /backtest/optimize, /consensus/backtest) -- NOT by live signal generation, which
    deliberately stays Mongo-only (a bounded recent window is all it needs, and archive
    files only get refreshed periodically, not on every candle).

    Imports candles_collection lazily (rather than at module load) so this module stays
    importable from scripts/train_rl.py, which deliberately avoids app.core.database/config
    (see that script's own docstring) -- only this function, which the FastAPI backend
    calls with full settings already loaded, needs it.
    """
    from app.core.database import candles_collection

    archive_df = load_archived_candles(pair, interval)

    cursor = candles_collection.find(
        {"pair": pair, "interval": interval}, {"_id": 0, "pair": 0, "interval": 0}
    ).sort("timestamp", 1)
    mongo_docs = await cursor.to_list(length=None)
    mongo_df = pd.DataFrame(mongo_docs, columns=["timestamp", "open", "high", "low", "close", "volume"])

    combined = pd.concat([archive_df, mongo_df], ignore_index=True)
    if combined.empty:
        return []
    combined = combined.drop_duplicates(subset="timestamp", keep="last").sort_values("timestamp")

    combined["pair"] = pair
    combined["interval"] = interval
    return combined.to_dict("records")
