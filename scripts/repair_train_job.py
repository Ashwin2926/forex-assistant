"""
Rebuilds a Train-all job's per-combo results (rl_train_jobs) from what training actually
wrote, for jobs recorded wrong by the pre-2026-10-14 scripts/train_rl.py: with one combo per
train-rl.yml matrix job, the first matrix job to finish marked the whole job "done", and
every later combo's result was dropped (the RL page showed 1 of 20).

Every trained combo inserts its evaluation into backtest_runs (profile "rl_ppo") whether or
not the policy was kept, and a kept policy's ppo_policies doc points back at it through
eval_run_id -- so each combo's latest rl_ppo eval inside the job's time window, plus whether
a policy references it, reproduces exactly the cell train_rl.py would have recorded.
(policy_id is only recoverable for kept policies; a rejected policy was never stored.)

Dry run by default; --confirm writes. --job-id defaults to the most recent job.
"""
import argparse
import os
import sys
from datetime import timedelta

from pymongo import MongoClient

WINDOW = timedelta(hours=7)  # train-rl.yml's matrix jobs time out at 6h each


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()

    db = MongoClient(os.environ["MONGODB_URI"], serverSelectionTimeoutMS=15000)[
        os.environ.get("MONGODB_DB_NAME", "forex_assistant")
    ]
    jobs = db["rl_train_jobs"]
    job = (jobs.find_one({"job_id": args.job_id}) if args.job_id
           else jobs.find_one(sort=[("created_at", -1)]))
    if not job:
        print("No such job.")
        return 1

    start = job["created_at"]
    print(f"Job {job['job_id']}: created {start:%Y-%m-%d %H:%M:%S}, status={job.get('status')}, "
          f"recorded {len(job.get('results', []))}/{job.get('total')} results")

    latest = {}
    for run in db["backtest_runs"].find(
        {"profile": "rl_ppo", "created_at": {"$gte": start, "$lt": start + WINDOW}},
        sort=[("created_at", 1)],
    ):
        latest[(run["pair"], run["interval"])] = run  # sorted ascending, so last one wins

    cells, last_at = [], start
    for (pair, interval), run in sorted(latest.items()):
        policy = db["ppo_policies"].find_one({"eval_run_id": run["run_id"]}, {"policy_id": 1})
        cells.append({
            "pair": pair, "interval": interval, "ok": True,
            "policy_id": policy["policy_id"] if policy else None,
            "hit_rate_pct": run.get("hit_rate_pct"), "expectancy_pct": run.get("expectancy_pct"),
            "directional_signals": run.get("directional_signals"), "hold_signals": run.get("hold_signals"),
            "starting_balance": run.get("starting_balance"), "ending_balance": run.get("ending_balance"),
            "total_return_pct": run.get("total_return_pct"), "kept": policy is not None,
        })
        last_at = max(last_at, run["created_at"])
        print(f"  {pair:8} {interval:6} {'kept    ' if policy else 'rejected'} "
              f"return={run.get('total_return_pct')}% hit={run.get('hit_rate_pct')}%")

    print(f"Rebuilt {len(cells)} of {job.get('total')} combos.")
    if not args.confirm:
        print("Dry run -- pass --confirm to write.")
        return 0
    jobs.update_one({"job_id": job["job_id"]}, {"$set": {
        "results": cells, "completed": len(cells), "status": "done", "finished_at": last_at,
    }})
    print("Written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
