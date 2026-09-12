"""
One-off archive of backtest_signals_collection to Parquet, ahead of an emergency collection
drop -- see PROGRESS.md's 2026-09-12 outage entry: rebuild_backtests.py wrote ~90,000+ signals
in one run, pushing Atlas over its 512MB M0 quota and blocking the app from starting (writes
are blocked cluster-wide once over quota, including the index-creation call in app startup).

Deleting documents doesn't reliably reclaim space on M0: WiredTiger doesn't shrink a
collection's on-disk file when documents are deleted from it (freed space is reused
internally for future writes, not returned to the OS, until a `compact` runs), and Atlas's
free/shared tiers don't allow running `compact` manually -- confirmed live: a full
scripts/trim_backtest_signals.py --confirm run left Atlas reporting the exact same "518 MB of
512 MB" afterward. Dropping the collection entirely is the one operation that frees the
underlying file immediately (scripts/drop_backtest_signals.py does that, as a separate script
run only after THIS script's output is committed to git -- see
archive-and-reset-backtest-signals.yml's own step ordering for why the two are split).

Deliberately does NOT import app.core.config or app.core.database, same as the other
standalone scripts here -- talks to Mongo via a plain sync pymongo client.
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from pymongo import MongoClient

ARCHIVE_DIR = Path(__file__).resolve().parent.parent / "data" / "backtest_signals_archive"


def main():
    mongodb_uri = os.environ["MONGODB_URI"]
    mongodb_db_name = os.environ.get("MONGODB_DB_NAME", "forex_assistant")
    client = MongoClient(mongodb_uri)
    db = client[mongodb_db_name]

    docs = list(db["backtest_signals"].find({}))
    print(f"{len(docs)} backtest_signals documents to archive.")
    if not docs:
        print("Nothing to archive -- collection is already empty.")
        return

    for d in docs:
        d["_id"] = str(d["_id"])
        # reasons is a list[dict] (SignalReason) -- JSON-stringified rather than stored as a
        # nested Parquet struct column, the simplest representation that round-trips cleanly
        # regardless of which fields any individual SignalReason happens to carry.
        if d.get("reasons") is not None:
            d["reasons"] = json.dumps(d["reasons"], default=str)

    df = pd.DataFrame(docs)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_path = ARCHIVE_DIR / f"backtest_signals_{today}.parquet"
    df.to_parquet(out_path, compression="brotli", index=False)
    print(f"Wrote {len(df)} rows to {out_path} ({out_path.stat().st_size / 1e6:.2f} MB).")


if __name__ == "__main__":
    main()
