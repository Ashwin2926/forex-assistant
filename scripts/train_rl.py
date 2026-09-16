"""
Standalone PPO training entry point, run directly on a GitHub Actions runner instead of
inside the FastAPI Cloud process (see .github/workflows/train-rl.yml and PROGRESS.md's
2026-09-10 entry). Training is CPU-heavy -- a single pair/interval combo already runs
60-90s -- and running 20 of them sequentially inside an HTTP request/BackgroundTasks
handler risks both gateway timeouts (already seen: a 524) and silent death if the FastAPI
Cloud instance recycles mid-run (the same failure class that killed the old in-process
APScheduler ingestion cron). The GitHub Actions runner has none of those constraints, so
this script does the actual training there and writes results straight to Mongo.

Deliberately does NOT import app.core.config or app.core.database: those pull in required
settings (twelve_data_api_key, etc.) this script has no use for and the Actions runner has
no reason to be given. Reads only the two Mongo env vars it actually needs directly from
the environment, and talks to Mongo via a plain sync pymongo client -- there's no event
loop here to share and nothing else running concurrently that async would avoid blocking,
so motor (the async driver app/core/database.py uses) would be pure overhead.

Persists into the exact same collections/shapes app/main.py's _run_rl_training and
run_train_all_job already use (ppo_policies_collection, backtest_runs_collection,
backtest_signals_collection, rl_train_jobs_collection) so GET /rl/train-all/{job_id},
/rl/train-all-latest, /rl/policies, and every backtest-run endpoint keep working unchanged
regardless of where training actually ran.
"""
import argparse
import os
import sys
from datetime import datetime

# Imported before pandas (below) or anything that transitively imports pandas/pyarrow gets a
# chance to -- Windows-only DLL-init-order conflict, same one app/main.py's own top-of-file
# comment documents (WinError 1114 loading torch/lib/c10.dll if pandas claims the relevant DLL
# slot first). Only matters when this script runs on a local Windows dev machine; the GitHub
# Actions runner it's normally meant for is Linux and has never hit this.
import torch  # noqa: F401

import pandas as pd
from pymongo import MongoClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.candle_archive import load_full_candle_history_sync
from app.services.case_memory import RESOLVED_STATUSES
from app.services.ml_training_data import load_backtest_signals_archive
from app.services.ml_features import is_current_smc_signal
from app.services.ppo_engine import train_ppo_policy
from app.services.rl_engine import (
    rl_config_profile, RL_FEATURE_NAMES, is_usable_warm_start, should_keep_new_policy,
)
from app.services.signal_engine import default_config_for

# Same default set app/core/config.py's Settings.forex_pairs and app/main.py's
# RL_INTERVALS use -- kept as a literal default here (overridable via env) rather than
# importing Settings, see module docstring for why.
DEFAULT_PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD"]
RL_INTERVALS = ["5min", "15min", "1h", "4h", "1day"]


DEFAULT_STARTING_BALANCE = 50.0


def pairs_list() -> list[str]:
    raw = os.environ.get("FOREX_PAIRS")
    if not raw:
        return DEFAULT_PAIRS
    return [p.strip() for p in raw.split(",") if p.strip()]


def find_warm_start_policy(db, pair: str, interval: str) -> tuple[str | None, bytes | None, float]:
    """Mirrors app/main.py's _find_warm_start_policy exactly (same is_usable_warm_start check,
    same baseline_return_pct contract) but against a sync pymongo db. See that function's own
    docstring for the full reasoning -- both delegate to rl_engine.is_usable_warm_start so the
    two paths can't drift on what counts as a usable prior policy."""
    prev = db["ppo_policies"].find_one({"pair": pair, "interval": interval}, sort=[("created_at", -1)])
    if not prev or prev.get("feature_names") != RL_FEATURE_NAMES:
        return None, None, 0.0
    eval_run = db["backtest_runs"].find_one({"run_id": prev.get("eval_run_id")})
    if not is_usable_warm_start(eval_run):
        return None, None, 0.0
    return prev["policy_id"], prev["model_bytes"], eval_run.get("total_return_pct") or 0.0


def get_ml_reference_signals_sync(db) -> list[dict]:
    """Mirrors app/services/ml_training_data.py's async get_ml_reference_signals exactly (live
    consensus signals from Mongo, unioned with backtest signals from the Parquet archive) but
    against a sync pymongo db for the live-signal half -- load_backtest_signals_archive itself
    needs no sync/async split at all, since it's pure local file I/O. consensus_signals, not
    the retired rule engine's signals collection -- see PROGRESS.md's rule-engine-removal entry."""
    live_signals = list(db["consensus_signals"].find({"source": "live", "status": {"$in": list(RESOLVED_STATUSES)}}))
    live_signals = [s for s in live_signals if is_current_smc_signal(s)]
    return live_signals + load_backtest_signals_archive()


def run_one(db, pair: str, interval: str, total_timesteps: int, train_frac: float,
            max_lookforward: int, starting_balance: float, random_seed: int | None,
            target_atr_mult: float | None = None, stop_atr_mult: float | None = None):
    """Mirrors app/main.py's _run_rl_training exactly (same query, same training call, same
    warm-start selection, same persistence) but against a sync pymongo db instead of the
    app's async motor collections.

    target_atr_mult/stop_atr_mult: same override _run_rl_training/POST /rl/train/{interval}
    accept -- omit (both None) to use rl_engine.rl_atr_mults(interval, pair)'s configured
    default. Lets a candidate value be swept via this (reliable, GitHub-Actions-backed) path
    instead of the direct endpoint, which has been seen to 524 on a slow combo.

    Only persists the newly trained policy (letting it become "latest" for the next
    warm-start / live serving) if rl_engine.should_keep_new_policy says it's at least as good
    as whatever it warm-started from -- mirrors app/main.py's _run_rl_training exactly, see
    that function's own docstring for why. Returns (policy, eval_run, kept)."""
    config = default_config_for(rl_config_profile(interval), pair)
    df = load_full_candle_history_sync(db, pair, interval)
    if df.empty:
        raise ValueError(f"No candle history for {pair}/{interval}. Run /ingest/{interval} first.")

    ml_reference_signals = get_ml_reference_signals_sync(db)

    warm_start_policy_id, warm_start_model_bytes, baseline_return_pct = find_warm_start_policy(db, pair, interval)

    policy, eval_run, trade_signals, _poc_diagnostics = train_ppo_policy(
        df, pair, interval, config, total_timesteps=total_timesteps, train_frac=train_frac,
        max_lookforward=max_lookforward, starting_balance=starting_balance,
        ml_reference_signals=ml_reference_signals, random_seed=random_seed,
        target_atr_mult=target_atr_mult, stop_atr_mult=stop_atr_mult,
        warm_start_policy_id=warm_start_policy_id, warm_start_model_bytes=warm_start_model_bytes,
    )

    kept = should_keep_new_policy(eval_run.total_return_pct, baseline_return_pct)
    if kept:
        if trade_signals:
            db["backtest_signals"].insert_many([s.model_dump() for s in trade_signals])
        db["ppo_policies"].insert_one(policy.model_dump())
    db["backtest_runs"].insert_one(eval_run.model_dump())
    return policy, eval_run, kept


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", default=None, help="rl_train_jobs_collection doc to write progress into; created if it doesn't already exist (e.g. a schedule-triggered run with no pre-created job doc).")
    parser.add_argument("--pair", default=None, help="Single pair (e.g. 'EUR/USD'). Omit together with --interval to sweep every pair x interval.")
    parser.add_argument("--interval", default=None, help="Single interval (e.g. '5min'). Omit together with --pair to sweep every pair x interval.")
    parser.add_argument("--total-timesteps", type=int, default=50_000)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--max-lookforward", type=int, default=20)
    parser.add_argument("--starting-balance", type=float, default=DEFAULT_STARTING_BALANCE)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--target-atr-mult", type=float, default=None, help="Override rl_atr_mults(interval, pair)'s target ATR multiple -- must be given with --stop-atr-mult and a single --pair/--interval combo, see main.py's POST /rl/train/{interval} for the same override.")
    parser.add_argument("--stop-atr-mult", type=float, default=None, help="Override rl_atr_mults(interval, pair)'s stop ATR multiple -- see --target-atr-mult.")
    args = parser.parse_args()

    mongodb_uri = os.environ["MONGODB_URI"]
    mongodb_db_name = os.environ.get("MONGODB_DB_NAME", "forex_assistant")
    client = MongoClient(mongodb_uri)
    db = client[mongodb_db_name]

    if bool(args.pair) != bool(args.interval):
        parser.error("--pair and --interval must be given together, or both omitted for a full sweep.")
    if (args.target_atr_mult is not None) != (args.stop_atr_mult is not None):
        parser.error("--target-atr-mult and --stop-atr-mult must be given together.")
    if args.target_atr_mult is not None and not args.pair:
        parser.error("--target-atr-mult/--stop-atr-mult require a single --pair/--interval combo, not a full sweep.")
    combos = [(args.pair, args.interval)] if args.pair else [
        (pair, interval) for pair in pairs_list() for interval in RL_INTERVALS
    ]

    job_id = args.job_id
    if job_id:
        job_doc = db["rl_train_jobs"].find_one({"job_id": job_id})
        if not job_doc:
            db["rl_train_jobs"].insert_one({
                "job_id": job_id, "status": "running", "created_at": datetime.utcnow(),
                "total_timesteps": args.total_timesteps, "train_frac": args.train_frac,
                "starting_balance": args.starting_balance, "total": len(combos), "completed": 0,
                "results": [],
            })

    exit_code = 0
    for pair, interval in combos:
        if job_id:
            job_doc = db["rl_train_jobs"].find_one({"job_id": job_id}, {"status": 1})
            if not job_doc or job_doc.get("status") != "running":
                print(f"Job {job_id} is no longer running (cancelled) -- stopping before {pair}/{interval}.")
                break
        print(f"Training {pair}/{interval}...")
        try:
            policy, eval_run, kept = run_one(
                db, pair, interval, args.total_timesteps, args.train_frac, args.max_lookforward,
                args.starting_balance, args.random_seed,
                target_atr_mult=args.target_atr_mult, stop_atr_mult=args.stop_atr_mult,
            )
            cell = {
                "pair": pair, "interval": interval, "ok": True, "policy_id": policy.policy_id,
                "hit_rate_pct": eval_run.hit_rate_pct, "expectancy_pct": eval_run.expectancy_pct,
                "directional_signals": eval_run.directional_signals, "hold_signals": eval_run.hold_signals,
                "starting_balance": eval_run.starting_balance, "ending_balance": eval_run.ending_balance,
                "total_return_pct": eval_run.total_return_pct, "kept": kept,
            }
            print(
                f"  {'kept' if kept else 'REJECTED (worse than prior policy, not persisted)'}: "
                f"hit_rate={eval_run.hit_rate_pct}% expectancy={eval_run.expectancy_pct}% "
                f"return={eval_run.total_return_pct}%"
            )
        except Exception as e:
            cell = {"pair": pair, "interval": interval, "ok": False, "error": str(e)}
            exit_code = 1
            print(f"  FAILED: {e}")
        if job_id:
            db["rl_train_jobs"].update_one(
                {"job_id": job_id, "status": "running"},
                {"$push": {"results": cell}, "$inc": {"completed": 1}},
            )

    if job_id:
        db["rl_train_jobs"].update_one(
            {"job_id": job_id, "status": "running"},
            {"$set": {"status": "done", "finished_at": datetime.utcnow()}},
        )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
