"""
Standalone rule-engine backtest runner, run directly on a GitHub Actions runner instead of
inside the FastAPI Cloud process -- same reasoning as scripts/train_rl.py (see that file's
own docstring and PROGRESS.md's 2026-09-10 entry): a deep-archive-backed backtest on a
5min/15min combo replays hundreds of thousands of candles, and looping that per-request
against FastAPI Cloud's ~125s gateway timeout has already caused documented failures on
similarly slow endpoints (see PROGRESS.md's 2026-09-12 storage-crisis/candle-archive entries).
Doing it here removes that timeout entirely.

Why this script exists (separate from just calling POST /backtest per combo): the ML
classifier's training data (see app/services/ml_training_data.py) needs a large pool of
resolved signals generated under today's live default config -- deliberately, so the
classifier never learns from a superseded ruleset. Checked live on 2026-09-12: every backtest
run that existed before this script did predates the 2026-09-07 EMA 9/21 default (they used
the older EMA 12/26), so none of them were usable. This script re-runs the CURRENT default
config across the full archive for every pair/interval, producing fresh training data. Re-run
it any time PROFILE_DEFAULTS["intraday"] changes, for the same reason.

Per-signal detail is written straight to a git-committed Parquet archive
(data/backtest_signals_archive/*.parquet, one file per pair/interval, overwritten each run --
NOT merged/appended, since a rebuild always represents "the complete current-config replay,"
not incremental new history the way candle data is), NOT into
backtest_signals_collection -- see PROGRESS.md's 2026-09-12 outage entry: writing ~90,000+
signals into Mongo pushed Atlas over its 512MB M0 quota and crashed the entire app (writes
are blocked cluster-wide once over quota, including the index-creation call in app startup).
The Parquet archive has no comparable size cap (same reasoning as data/candles/*.parquet
already uses), so this can't repeat that outage regardless of how much history it replays.
Each run's lightweight summary (hit-rate, expectancy, etc. -- no per-signal detail) still goes
into backtest_runs_collection, which stays small (under 1MB for 800+ docs) and is what powers
the existing GET /backtest/runs listing.

Deliberately does NOT import app.core.config or app.core.database, same as train_rl.py --
reads only the two Mongo env vars it needs directly from the environment and talks to Mongo
via a plain sync pymongo client for backtest_runs/backtest_rebuild_jobs only.
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from pymongo import MongoClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.backtester import run_backtest
from app.services.candle_archive import load_full_candle_history_sync
from app.services.signal_engine import default_config_for

DEFAULT_PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD"]
INTERVALS = ["5min", "15min", "1h", "4h", "1day"]

PROFILE = "intraday"

ARCHIVE_DIR = Path(__file__).resolve().parent.parent / "data" / "backtest_signals_archive"


def pairs_list() -> list[str]:
    raw = os.environ.get("FOREX_PAIRS")
    if not raw:
        return DEFAULT_PAIRS
    return [p.strip() for p in raw.split(",") if p.strip()]


def _write_signals_parquet(pair: str, interval: str, signals: list) -> None:
    """Overwrites data/backtest_signals_archive/{pair}_{interval}.parquet with this run's
    signals -- see module docstring for why overwrite, not merge. reasons is JSON-stringified
    (a list[dict] doesn't round-trip through Parquet as cleanly as a plain column);
    ml_training_data.load_backtest_signals_archive() restores it on read."""
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    rows = [s.model_dump() for s in signals]
    for r in rows:
        r["reasons"] = json.dumps(r["reasons"], default=str)
    df = pd.DataFrame(rows)
    slug = pair.replace("/", "_")
    df.to_parquet(ARCHIVE_DIR / f"{slug}_{interval}.parquet", compression="brotli", index=False)


def run_one(db, pair: str, interval: str, max_lookforward: int) -> tuple[int, int]:
    """Returns (directional_signals, hits) for progress printing. Raises on failure (not
    enough candle history, etc.) -- same "let the caller decide skip vs. abort" contract
    train_rl.py's run_one leaves to main()."""
    config = default_config_for(PROFILE)
    df = load_full_candle_history_sync(db, pair, interval)
    if df.empty:
        raise ValueError(f"No candle history for {pair}/{interval}.")

    run, signals = run_backtest(df, pair, interval, PROFILE, config=config, max_lookforward=max_lookforward)

    if signals:
        _write_signals_parquet(pair, interval, signals)
    db["backtest_runs"].insert_one(run.model_dump())
    return run.directional_signals, run.hits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", default=None, help="backtest_rebuild_jobs_collection doc to report progress into; created if it doesn't already exist (e.g. a schedule-triggered run with no pre-created job doc).")
    parser.add_argument("--pair", default=None, help="Single pair (e.g. 'EUR/USD'). Omit together with --interval to sweep every pair x interval.")
    parser.add_argument("--interval", default=None, help="Single interval (e.g. '15min'). Omit together with --pair to sweep every pair x interval.")
    parser.add_argument("--max-lookforward", type=int, default=20)
    args = parser.parse_args()

    if bool(args.pair) != bool(args.interval):
        parser.error("--pair and --interval must be given together, or both omitted for a full sweep.")
    combos = [(args.pair, args.interval)] if args.pair else [
        (pair, interval) for pair in pairs_list() for interval in INTERVALS
    ]

    mongodb_uri = os.environ["MONGODB_URI"]
    mongodb_db_name = os.environ.get("MONGODB_DB_NAME", "forex_assistant")
    client = MongoClient(mongodb_uri)
    db = client[mongodb_db_name]

    job_id = args.job_id
    if job_id:
        job_doc = db["backtest_rebuild_jobs"].find_one({"job_id": job_id})
        if not job_doc:
            db["backtest_rebuild_jobs"].insert_one({
                "job_id": job_id, "status": "running", "created_at": datetime.utcnow(),
                "max_lookforward": args.max_lookforward, "total": len(combos), "completed": 0,
                "results": [],
            })

    exit_code = 0
    for pair, interval in combos:
        if job_id:
            job_doc = db["backtest_rebuild_jobs"].find_one({"job_id": job_id}, {"status": 1})
            if not job_doc or job_doc.get("status") != "running":
                print(f"Job {job_id} is no longer running (cancelled) -- stopping before {pair}/{interval}.")
                break
        print(f"Backtesting {pair}/{interval} (today's live default config)...")
        try:
            directional, hits = run_one(db, pair, interval, args.max_lookforward)
            cell = {"pair": pair, "interval": interval, "ok": True, "directional_signals": directional, "hits": hits}
            print(f"  ok: {directional} directional signals, {hits} hits")
        except Exception as e:
            cell = {"pair": pair, "interval": interval, "ok": False, "error": str(e)}
            exit_code = 1
            print(f"  FAILED: {e}")
        if job_id:
            db["backtest_rebuild_jobs"].update_one(
                {"job_id": job_id, "status": "running"},
                {"$push": {"results": cell}, "$inc": {"completed": 1}},
            )

    if job_id:
        db["backtest_rebuild_jobs"].update_one(
            {"job_id": job_id, "status": "running"},
            {"$set": {"status": "done", "finished_at": datetime.utcnow()}},
        )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
