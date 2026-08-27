from typing import Optional
import pandas as pd
from app.core.database import candles_collection, signals_collection, consensus_signals_collection, rl_signals_collection
from app.services.signal_engine import label_outcome, spread_cost_pct

DEFAULT_MAX_LOOKFORWARD = 20

# Live resolution gets meaningfully more real time to actually hit target or stop than
# backtest/training's tight max_lookforward=20 -- that value exists for tractability inside a
# bounded training loop (rl_engine.train_rl_policy walks forward through a fixed historical
# slice; letting one simulated trade run indefinitely would stall the whole training walk).
# None of that applies to a live signal: it costs nothing to just check it again next cron
# cycle, for as long as it takes, the same way a manually-traded position would actually be
# held. Per-interval, not one flat candle count, since the same count means a wildly
# different real-world duration at 5min vs 1day (20 candles is ~100min at 5min but ~20 days
# at 1day). Starting guesses, not independently validated against live outcomes yet -- same
# caveat as every other unvalidated constant in this project (PROXIMITY_ATR_MULT,
# RL_ATR_MULTS_BY_PROFILE, etc.); revisit once enough live resolutions at these windows exist
# to check hit/miss/expired mix against.
LIVE_MAX_LOOKFORWARD_BY_INTERVAL: dict[str, int] = {
    "5min": 200,   # ~16.7 hours
    "15min": 150,  # ~37.5 hours
    "1h": 72,      # ~3 days
    "4h": 42,      # ~7 days
    "1day": 30,    # ~30 days
}


def _live_max_lookforward(interval: str, override: Optional[int]) -> int:
    """
    override (a caller-supplied max_lookforward) always wins -- e.g. a manual call that wants
    the old tight window for a quick comparison. A bare call (no override, what the cron
    always does) falls through to this interval's entry in LIVE_MAX_LOOKFORWARD_BY_INTERVAL,
    or DEFAULT_MAX_LOOKFORWARD for any interval not in that table.
    """
    if override is not None:
        return override
    return LIVE_MAX_LOOKFORWARD_BY_INTERVAL.get(interval, DEFAULT_MAX_LOOKFORWARD)


async def score_pending_signals(max_lookforward: Optional[int] = None) -> dict:
    """
    Closes the loop the whole project is built around: finds live directional signals
    still marked "pending", checks the candles that have actually arrived since each was
    generated, and updates status to hit/miss/expired using the exact same label_outcome
    logic the backtester validates against history.

    A signal only has something to check once it has target_price/stop_price attached
    (see /signals — HOLD signals and any generated before that was added have neither and
    are skipped, not stuck as false negatives). A signal with fewer than its window's real
    candles since it fired stays "pending" — there's nothing wrong with it, it just hasn't
    had time to play out yet, and this job will look at it again next run.

    max_lookforward: explicit override applied to every doc regardless of interval, if given.
    Left as None (the default -- what the cron always uses), each doc gets ITS OWN interval's
    entry from LIVE_MAX_LOOKFORWARD_BY_INTERVAL instead of one flat window for every interval
    -- see that table's own comment for why.
    """
    query = {
        "status": "pending",
        "direction": {"$in": ["BUY", "SELL"]},
        "target_price": {"$ne": None},
        "stop_price": {"$ne": None},
    }
    cursor = signals_collection.find(query)
    pending = await cursor.to_list(length=None)

    tally = {"hit": 0, "miss": 0, "expired": 0, "still_pending": 0, "skipped_no_data": 0}

    for doc in pending:
        window = _live_max_lookforward(doc["interval"], max_lookforward)
        future_cursor = (
            candles_collection.find({
                "pair": doc["pair"],
                "interval": doc["interval"],
                "timestamp": {"$gt": doc["timestamp"]},
            })
            .sort("timestamp", 1)
            .limit(window)
        )
        future_docs = await future_cursor.to_list(length=window)
        if not future_docs:
            tally["skipped_no_data"] += 1
            continue

        future_df = pd.DataFrame(future_docs)
        status, outcome_price, outcome_timestamp, candles_to_outcome = label_outcome(
            future_df, doc["direction"], doc["target_price"], doc["stop_price"], window,
        )

        if status == "pending":
            tally["still_pending"] += 1
            continue

        pct_move = ((outcome_price - doc["price_at_signal"]) / doc["price_at_signal"]) * 100
        if doc["direction"] == "SELL":
            pct_move = -pct_move
        pct_move -= spread_cost_pct(doc["pair"], doc["price_at_signal"])  # every real trade pays this, win or lose

        await signals_collection.update_one(
            {"_id": doc["_id"]},
            {"$set": {
                "status": status,
                "outcome_price": round(float(outcome_price), 5),
                "outcome_timestamp": outcome_timestamp,
                "outcome_pct_move": round(pct_move, 5),
                "candles_to_outcome": candles_to_outcome,
            }},
        )
        tally[status] += 1

    return tally


async def score_pending_consensus_signals(max_lookforward: Optional[int] = None) -> dict:
    """
    Same closing-the-loop job as score_pending_signals, for consensus_signals_collection
    instead. A separate function rather than a generalized one because the two collections'
    documents aren't quite the same shape (ConsensusSignal has no HOLD direction to filter
    out and no target_price/stop_price existence check needed -- both are always set by
    construction, unlike a regular Signal which can be a directionless HOLD) and reusing
    price_at_signal's name would be misleading here (ConsensusSignal calls it entry_price).

    max_lookforward: see score_pending_signals' docstring -- None (the default) gives each
    doc its own interval-appropriate window instead of one flat value for every interval.
    """
    cursor = consensus_signals_collection.find({"status": "pending"})
    pending = await cursor.to_list(length=None)

    tally = {"hit": 0, "miss": 0, "expired": 0, "still_pending": 0, "skipped_no_data": 0}

    for doc in pending:
        window = _live_max_lookforward(doc["interval"], max_lookforward)
        future_cursor = (
            candles_collection.find({
                "pair": doc["pair"],
                "interval": doc["interval"],
                "timestamp": {"$gt": doc["timestamp"]},
            })
            .sort("timestamp", 1)
            .limit(window)
        )
        future_docs = await future_cursor.to_list(length=window)
        if not future_docs:
            tally["skipped_no_data"] += 1
            continue

        future_df = pd.DataFrame(future_docs)
        status, outcome_price, outcome_timestamp, candles_to_outcome = label_outcome(
            future_df, doc["direction"], doc["target_price"], doc["stop_price"], window,
        )

        if status == "pending":
            tally["still_pending"] += 1
            continue

        pct_move = ((outcome_price - doc["entry_price"]) / doc["entry_price"]) * 100
        if doc["direction"] == "SELL":
            pct_move = -pct_move
        pct_move -= spread_cost_pct(doc["pair"], doc["entry_price"])  # every real trade pays this, win or lose

        await consensus_signals_collection.update_one(
            {"_id": doc["_id"]},
            {"$set": {
                "status": status,
                "outcome_price": round(float(outcome_price), 5),
                "outcome_timestamp": outcome_timestamp,
                "outcome_pct_move": round(pct_move, 5),
                "candles_to_outcome": candles_to_outcome,
            }},
        )
        tally[status] += 1

    return tally


async def score_pending_rl_signals(max_lookforward: Optional[int] = None) -> dict:
    """
    Same closing-the-loop job as score_pending_consensus_signals, for rl_signals_collection.
    RLSignal is shaped identically to ConsensusSignal for exactly this reason (always
    directional, entry_price/target_price/stop_price always set) -- this is the live,
    forward-going half of "backtest its signals and learn" (the historical-replay half is
    train_rl_policy itself).

    max_lookforward: see score_pending_signals' docstring -- None (the default) gives each
    doc its own interval-appropriate window instead of one flat value for every interval.
    """
    cursor = rl_signals_collection.find({"status": "pending"})
    pending = await cursor.to_list(length=None)

    tally = {"hit": 0, "miss": 0, "expired": 0, "still_pending": 0, "skipped_no_data": 0}

    for doc in pending:
        window = _live_max_lookforward(doc["interval"], max_lookforward)
        future_cursor = (
            candles_collection.find({
                "pair": doc["pair"],
                "interval": doc["interval"],
                "timestamp": {"$gt": doc["timestamp"]},
            })
            .sort("timestamp", 1)
            .limit(window)
        )
        future_docs = await future_cursor.to_list(length=window)
        if not future_docs:
            tally["skipped_no_data"] += 1
            continue

        future_df = pd.DataFrame(future_docs)
        status, outcome_price, outcome_timestamp, candles_to_outcome = label_outcome(
            future_df, doc["direction"], doc["target_price"], doc["stop_price"], window,
        )

        if status == "pending":
            tally["still_pending"] += 1
            continue

        pct_move = ((outcome_price - doc["entry_price"]) / doc["entry_price"]) * 100
        if doc["direction"] == "SELL":
            pct_move = -pct_move
        pct_move -= spread_cost_pct(doc["pair"], doc["entry_price"])  # every real trade pays this, win or lose

        await rl_signals_collection.update_one(
            {"_id": doc["_id"]},
            {"$set": {
                "status": status,
                "outcome_price": round(float(outcome_price), 5),
                "outcome_timestamp": outcome_timestamp,
                "outcome_pct_move": round(pct_move, 5),
                "candles_to_outcome": candles_to_outcome,
            }},
        )
        tally[status] += 1

    return tally
