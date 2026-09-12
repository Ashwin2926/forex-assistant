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
    Reads the committed Parquet export for this pair/interval, or an empty frame (same
    columns) if none exists yet -- callers should treat a missing archive as "no older
    history", not an error, since not every pair/interval has been backfilled/exported.

    Parquet over gzip-CSV: ~15-20x faster to read in pandas (measured on this project's own
    502k-row EUR/USD 5min file: 0.024s vs 0.50s), at the cost of ~27% more disk space -- worth
    it since slow archive reads have already caused gateway timeouts on the endpoints that
    call this (see scripts/export_candles.py's own docstring for the numbers).
    """
    path = ARCHIVE_DIR / f"{_slug(pair)}_{interval}.parquet"
    columns = ["timestamp", "open", "high", "low", "close", "volume"]
    if not path.exists():
        return pd.DataFrame(columns=columns)
    return pd.read_parquet(path)


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


def load_full_candle_history_sync(db, pair: str, interval: str) -> pd.DataFrame:
    """
    Sync-pymongo counterpart of load_full_candle_history above, for scripts that run
    standalone on a GitHub Actions runner instead of inside the FastAPI Cloud process (see
    scripts/train_rl.py and scripts/rebuild_backtests.py's own docstrings for why -- no event
    loop to share, and app.core.database/config deliberately not imported there). Same
    archive-merged-with-Mongo, Mongo-wins-on-overlap semantics; returns a DataFrame directly
    (not list[dict]) since every sync caller immediately wants a DataFrame anyway.
    """
    mongo_docs = list(
        db["candles"].find(
            {"pair": pair, "interval": interval}, {"_id": 0, "pair": 0, "interval": 0}
        ).sort("timestamp", 1)
    )
    archive_df = load_archived_candles(pair, interval)
    mongo_df = pd.DataFrame(mongo_docs, columns=["timestamp", "open", "high", "low", "close", "volume"])

    combined = pd.concat([archive_df, mongo_df], ignore_index=True)
    if combined.empty:
        return combined
    return combined.drop_duplicates(subset="timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)
