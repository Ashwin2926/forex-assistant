"""
Drops every Mongo collection downstream of raw candle history, keeping candles_collection
(and data/candles/*.parquet, untouched by this script entirely) intact -- for resetting the
app's derived state (signals, backtests, trained models, job logs) without re-fetching price
history from Twelve Data, which is slow and rate-limited.

Built for the SMC-only migration reset (see PROGRESS.md's rule-engine-removal entry) but
generic enough to reuse for any future "wipe derived state, keep candles" reset -- pass
--dry-run to see counts without dropping anything.

Deliberately does NOT import app.core.config/app.core.database, same reasoning as
scripts/train_rl.py and scripts/rebuild_backtests.py -- reads MONGODB_URI directly from the
environment and talks to Mongo via a plain sync pymongo client.
"""
import argparse
import os
import sys

from pymongo import MongoClient

# Every collection except "candles" itself (see app/core/database.py for the full list) --
# anything derived from a signal-generation or training run, regardless of which engine
# produced it. Kept as an explicit list (not "every collection except candles" computed from
# the live db) so a newly-added collection doesn't get silently swept into a reset it was
# never intended for -- a future reset script must deliberately add its name here.
DROP_COLLECTIONS = [
    "signals", "backtest_signals", "backtest_runs", "paper_trades", "consensus_signals",
    "ml_runs", "rl_policies", "rl_signals", "rl_train_jobs", "ppo_policies",
    "run_all_flows_jobs", "backtest_rebuild_jobs",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Report counts without dropping anything.")
    args = parser.parse_args()

    mongodb_uri = os.environ["MONGODB_URI"]
    mongodb_db_name = os.environ.get("MONGODB_DB_NAME", "forex_assistant")
    client = MongoClient(mongodb_uri)
    db = client[mongodb_db_name]

    for name in DROP_COLLECTIONS:
        count = db[name].count_documents({})
        if args.dry_run:
            print(f"[dry-run] would drop {name} ({count} docs)")
        else:
            db.drop_collection(name)
            print(f"dropped {name} ({count} docs)")

    print(f"candles kept: {db['candles'].count_documents({})}")
    sys.exit(0)


if __name__ == "__main__":
    main()
