"""
Drops backtest_signals_collection entirely -- the actual space-reclaiming step of the
emergency archive-and-reset (see scripts/export_backtest_signals.py's docstring and
PROGRESS.md's 2026-09-12 outage entry). Dropping a collection removes its underlying storage
file immediately, unlike delete_many, which doesn't reclaim disk space on Atlas's M0 tier.

Only ever run as the LAST step of archive-and-reset-backtest-signals.yml, after the Parquet
archive has been committed and pushed -- the workflow's own step ordering (a failed export or
git push stops the job before this step runs) is what guarantees this never executes without
a durable copy of the data existing first.

Deliberately does NOT import app.core.config or app.core.database, same as the other
standalone scripts here.
"""
import os

from pymongo import MongoClient


def main():
    mongodb_uri = os.environ["MONGODB_URI"]
    mongodb_db_name = os.environ.get("MONGODB_DB_NAME", "forex_assistant")
    client = MongoClient(mongodb_uri)
    db = client[mongodb_db_name]

    count_before = db["backtest_signals"].estimated_document_count()
    db["backtest_signals"].drop()
    print(f"Dropped backtest_signals ({count_before} documents) -- recreated empty on next write.")


if __name__ == "__main__":
    main()
