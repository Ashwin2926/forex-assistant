"""
Exports candles_collection to compressed CSV files committed into the repo, so the deep
history built up by .github/workflows/backfill-history.yml survives independently of
Mongo Atlas storage limits (Cluster2's M0 tier is a hard 512MB cap -- see PROGRESS.md/chat
around 2026-09-12) and is readable by anything that checks out this repo: ML/RL training
(scripts/train_rl.py, already runs on a GitHub Actions runner with a full checkout) and the
live backend's own deep-history backtest endpoints (app/services/candle_archive.py), since
FastAPI Cloud also deploys from this same repo.

Run via .github/workflows/export-candles.yml, which commits data/candles/*.csv.gz back to
the branch afterward -- same "do the DB work on a GitHub Actions runner, not the FastAPI
Cloud process" pattern train_rl.py already established, and avoids ever needing Mongo
credentials outside of Actions secrets.

Deliberately mirrors train_rl.py's approach: plain sync pymongo (no event loop here to
share, so motor would be pure overhead), Mongo env vars read directly rather than through
app.core.config (this script has no use for twelve_data_api_key etc.).
"""
import gzip
import os
from pathlib import Path

import pandas as pd
from pymongo import MongoClient

DEFAULT_PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD"]
INTERVALS = ["5min", "15min", "1h", "4h", "1day"]

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "candles"


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
            if not docs:
                print(f"{pair} {interval}: no candles, skipping")
                continue

            df = pd.DataFrame(docs)
            out_path = OUTPUT_DIR / f"{slug(pair)}_{interval}.csv.gz"
            with gzip.open(out_path, "wt", newline="") as f:
                df.to_csv(f, index=False)
            print(f"{pair} {interval}: {len(df)} candles -> {out_path.relative_to(OUTPUT_DIR.parent.parent)}")


if __name__ == "__main__":
    main()
