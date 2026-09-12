"""
Emergency storage-reclaim tool, run directly on a GitHub Actions runner because the FastAPI
Cloud app itself cannot start while Atlas is over its space quota -- writes get blocked
cluster-wide, including the index-creation call app/main.py's startup handler runs
(app.core.database.init_indexes), which crashes the whole app before it can serve a single
request. See PROGRESS.md's 2026-09-12 entry: rebuild_backtests.py wrote ~90,000+ backtest
signals in one run (a full-archive backtest on 5min/15min naturally produces tens of
thousands of directional signals per pair), pushing Cluster2 from ~308MB to over its 512MB
M0 cap.

Why this can't just be POST /backtest/prune-history: that endpoint keeps the 50 MOST RECENT
non-RL runs and deletes the rest -- but today's oversized rebuild runs ARE the most recent,
so that policy would keep exactly the bloat and delete only the old, already-tiny pre-rebuild
history. This script instead caps how many signals are RETAINED PER RUN, regardless of
recency -- every run over --keep-per-run gets trimmed down to a random sample of that many,
freeing the bulk of the space while leaving each run with enough of a sample for
app/services/ml_training_data.py's own $sample-based fetch (capped at MAX_BACKTEST_SIGNALS,
currently 5,000 across ALL qualifying runs combined) to keep working.

Deliberately does NOT import app.core.config or app.core.database, same reasoning as
scripts/train_rl.py -- talks to Mongo via a plain sync pymongo client.

Defaults to a dry run (--confirm to actually delete) -- same convention as the existing
POST /backtest/prune-history endpoint.
"""
import argparse
import os
import random

from pymongo import MongoClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep-per-run", type=int, default=1000, help="Max backtest_signals docs to retain for any single run_id -- the rest are deleted (a random sample is kept, not the first N).")
    parser.add_argument("--confirm", action="store_true", help="Actually delete. Omit for a dry run that only reports what would be deleted.")
    args = parser.parse_args()

    mongodb_uri = os.environ["MONGODB_URI"]
    mongodb_db_name = os.environ.get("MONGODB_DB_NAME", "forex_assistant")
    client = MongoClient(mongodb_uri)
    db = client[mongodb_db_name]

    run_ids = db["backtest_signals"].distinct("run_id")
    print(f"{len(run_ids)} distinct runs in backtest_signals.")

    total_deleted = 0
    for run_id in run_ids:
        # Only need _id here -- fetching full documents (each carrying a reasons list) for
        # every oversized run is exactly the cost this script exists to avoid.
        ids = [d["_id"] for d in db["backtest_signals"].find({"run_id": run_id}, {"_id": 1})]
        if len(ids) <= args.keep_per_run:
            continue

        run_doc = db["backtest_runs"].find_one({"run_id": run_id}, {"pair": 1, "interval": 1})
        label = f"{run_doc['pair']}/{run_doc['interval']}" if run_doc else "(no matching backtest_runs doc)"

        random.shuffle(ids)
        to_delete = ids[args.keep_per_run:]
        print(f"{run_id} {label}: {len(ids)} docs -> keeping {args.keep_per_run}, deleting {len(to_delete)}")

        if args.confirm:
            # Batched -- a single delete_many with tens of thousands of _ids in one $in can
            # itself be a slow/heavy query; chunking keeps each command small and cheap.
            for i in range(0, len(to_delete), 5000):
                batch = to_delete[i:i + 5000]
                db["backtest_signals"].delete_many({"_id": {"$in": batch}})
        total_deleted += len(to_delete)

    print(f"{'Deleted' if args.confirm else 'Would delete'} {total_deleted} backtest_signals docs total.")
    if not args.confirm:
        print("Dry run -- re-run with --confirm to actually delete.")


if __name__ == "__main__":
    main()
