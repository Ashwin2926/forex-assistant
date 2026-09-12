"""
Exports candles_collection to Parquet files committed into the repo, so the deep history
built up by .github/workflows/backfill-history.yml survives independently of Mongo Atlas
storage limits (Cluster2's M0 tier is a hard 512MB cap -- see PROGRESS.md/chat around
2026-09-12) and is readable by anything that checks out this repo: ML/RL training
(scripts/train_rl.py, already runs on a GitHub Actions runner with a full checkout) and the
live backend's own deep-history backtest endpoints (app/services/candle_archive.py), since
FastAPI Cloud also deploys from this same repo.

Parquet (brotli-compressed) over gzip-CSV: measured on this project's own EUR/USD 5min file
(502k rows), brotli-parquet is ~27% LARGER on disk (5.2MB vs 4.1MB) but reads ~15-20x FASTER
in pandas (0.024s vs 0.50s) -- worth it here since the deep-history endpoints that read this
archive have already hit FastAPI Cloud's gateway timeout more than once this same session;
faster parsing directly reduces that risk. Repo size at these totals (tens of MB) isn't a
real constraint either way.

MERGES with whatever's already in each existing .parquet file instead of overwriting it
outright -- candles_collection is deliberately kept trimmed to a rolling recent window (see
/candles/prune-history), so a naive overwrite-from-Mongo-only export would silently replace
years of archived history with just that trimmed window the moment it runs after a trim.
(This bit exactly once: 2026-09-12's second export call did precisely that to 16 of the 20
files before being caught and recovered from git history -- see PROGRESS.md.) Merging keeps
this monotonic: coverage can only grow, never shrink, regardless of Mongo's current state
or how many times this runs.

Run via .github/workflows/export-candles.yml, which commits data/candles/*.parquet back to
the branch afterward -- same "do the DB work on a GitHub Actions runner, not the FastAPI
Cloud process" pattern train_rl.py already established, and avoids ever needing Mongo
credentials outside of Actions secrets.

Deliberately mirrors train_rl.py's approach: plain sync pymongo (no event loop here to
share, so motor would be pure overhead), Mongo env vars read directly rather than through
app.core.config (this script has no use for twelve_data_api_key etc.).
"""
import os
from pathlib import Path

import pandas as pd
from pymongo import MongoClient

DEFAULT_PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD"]
INTERVALS = ["5min", "15min", "1h", "4h", "1day"]

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "candles"

CANDLE_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def pairs_list() -> list[str]:
    raw = os.environ.get("FOREX_PAIRS")
    if not raw:
        return DEFAULT_PAIRS
    return [p.strip() for p in raw.split(",") if p.strip()]


def slug(pair: str) -> str:
    return pair.replace("/", "_")


def main() -> None:
    mongodb_uri = os.environ["MONGODB_URI"]
    mongodb_db_name = os.environ.get("MONGODB_DB_NAME", "forex_assistant")
    client = MongoClient(mongodb_uri)
    db = client[mongodb_db_name]
    candles_collection = db["candles"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for pair in pairs_list():
        for interval in INTERVALS:
            docs = list(
                candles_collection.find(
                    {"pair": pair, "interval": interval},
                    {"_id": 0, "pair": 0, "interval": 0},
                ).sort("timestamp", 1)
            )
            mongo_df = pd.DataFrame(docs, columns=CANDLE_COLUMNS)

            out_path = OUTPUT_DIR / f"{slug(pair)}_{interval}.parquet"
            existing_df = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame(columns=CANDLE_COLUMNS)

            combined = pd.concat([existing_df, mongo_df], ignore_index=True)
            if combined.empty:
                print(f"{pair} {interval}: no candles (existing or Mongo), skipping")
                continue
            combined = combined.drop_duplicates(subset="timestamp", keep="last").sort_values("timestamp")

            combined.to_parquet(out_path, index=False, compression="brotli")
            print(
                f"{pair} {interval}: {len(existing_df)} existing + {len(mongo_df)} from Mongo "
                f"-> {len(combined)} merged candles -> {out_path.relative_to(OUTPUT_DIR.parent.parent)}"
            )


if __name__ == "__main__":
    main()
