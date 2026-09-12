"""
Widens the supervised hit/miss classifier's (ml_model.py) training data beyond live-only
signals to also include qualifying backtest signals -- see PROGRESS.md for the gap this
closes: live resolved signals total in the hundreds, while backtest_signals_collection
already holds tens of thousands of resolved examples from replaying the full historical
Parquet archive, previously untouched by the classifier.

Deliberately has NO app.core.* imports (not even app.core.database) -- this needs to be
importable from scripts/train_rl.py, which runs standalone on a GitHub Actions runner and
avoids app.core.* on purpose (see that script's own module docstring). Both the async
(app/main.py) and sync (scripts/train_rl.py) fetch paths delegate to is_qualifying_backtest_run
below so a future RuleConfig/PROFILE_DEFAULTS change can't silently diverge between them.
"""
from app.services.case_memory import RESOLVED_STATUSES
from app.services.signal_engine import default_config_for

QUALIFYING_BACKTEST_PROFILE = "intraday"

# Hard cap on how many qualifying backtest signals a single fetch returns, via Mongo's
# $sample (a random subset, not "first N" -- avoids skewing toward whichever combo happens to
# sort first). Added after this went uncapped and immediately broke: rebuilding all 20
# pair/interval combos on 2026-09-12 (scripts/rebuild_backtests.py) produced ~90,000+
# qualifying rows overnight, and this fetch runs SYNCHRONOUSLY on every live signal-generation
# call (create_signal's ML gate, POST /rl/signal, not just the POST /ml/train diagnostic) --
# fetching + per-row feature extraction + refitting XGBoost on that many rows blew well past
# FastAPI Cloud's ~125s gateway timeout (confirmed live: a direct POST /ml/train call hit a
# 524 at exactly 126s) and left the instance visibly degraded for a while afterward (even the
# public, DB-free GET / stopped responding). 5,000 is generous relative to the ~662 live
# signals this whole project started with, while keeping fit time in the tenths-of-a-second
# range this classifier was designed around (see ml_model.py's own N_ESTIMATORS/MAX_DEPTH
# comment) -- this is a hard operational ceiling, not a data-quality judgment.
MAX_BACKTEST_SIGNALS = 5000


def is_qualifying_backtest_run(run: dict) -> bool:
    """
    True iff this backtest_runs document was produced with exactly today's live default
    RuleConfig -- not a /backtest/sweep or /backtest/optimize candidate that lost (or won a
    since-superseded) search, and not an RL/PPO run (profile="rl_ppo"/"rl"). Signals from a
    non-default config were labeled hit/miss/expired against a different rule/target/stop
    setup than what's live today, so training the classifier on them would teach it
    feature/label relationships that don't describe current behavior.

    Checks BOTH the embedded rule_config dict AND the run's own top-level
    target_atr_mult/stop_atr_mult -- run_backtest accepts those two as separate query-param
    overrides independent of the RuleConfig body, so a run can have a byte-identical
    rule_config to today's default while still being labeled against a different target/stop
    distance. Comparing only rule_config would miss that.

    This is strict-equality matching, not a fuzzy/subset comparison: if RuleConfig ever gains
    a new field, older qualifying runs recorded before that change will have a rule_config
    dict missing that key and will stop qualifying, even though they used "the default" at the
    time. Accepted tradeoff -- a fuzzy match would risk silently including stale configs,
    which is the worse failure direction here.
    """
    if run.get("profile") != QUALIFYING_BACKTEST_PROFILE:
        return False
    default = default_config_for(QUALIFYING_BACKTEST_PROFILE)
    if run.get("rule_config") != default.model_dump():
        return False
    return (
        run.get("target_atr_mult") == default.target_atr_mult
        and run.get("stop_atr_mult") == default.stop_atr_mult
    )


async def get_ml_reference_signals(signals_collection, backtest_signals_collection, backtest_runs_collection) -> list[dict]:
    """
    Live resolved signals UNION qualifying backtest signals -- the shared data source for
    every ML-classifier call site (the create_signal quality gate, POST /ml/train, POST
    /ml/predict, and both RL training's ml_reference_signals/frozen_ml_snapshot). Async/motor
    version for app/main.py; scripts/train_rl.py has its own sync/pymongo mirror that
    delegates to is_qualifying_backtest_run above for the actual qualifying logic.

    backtest_runs_collection is small enough (low thousands at most) to fetch in full and
    filter in Python rather than build a fragile exact-match Mongo query against a nested
    rule_config subdocument.
    """
    live_signals = await signals_collection.find(
        {"source": "live", "status": {"$in": list(RESOLVED_STATUSES)}}
    ).to_list(length=None)

    candidate_runs = await backtest_runs_collection.find(
        {"profile": QUALIFYING_BACKTEST_PROFILE}
    ).to_list(length=None)
    qualifying_run_ids = [r["run_id"] for r in candidate_runs if is_qualifying_backtest_run(r)]

    backtest_signals = await backtest_signals_collection.aggregate([
        {"$match": {
            "run_id": {"$in": qualifying_run_ids},
            "status": {"$in": list(RESOLVED_STATUSES)},
            "size_tier": None,
        }},
        {"$sample": {"size": MAX_BACKTEST_SIGNALS}},
    ]).to_list(length=None) if qualifying_run_ids else []

    return live_signals + backtest_signals
