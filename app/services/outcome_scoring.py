import pandas as pd
from app.core.database import candles_collection, signals_collection, consensus_signals_collection, rl_signals_collection
from app.services.signal_engine import label_outcome, spread_cost_pct

DEFAULT_MAX_LOOKFORWARD = 20


async def score_pending_signals(max_lookforward: int = DEFAULT_MAX_LOOKFORWARD) -> dict:
    """
    Closes the loop the whole project is built around: finds live directional signals
    still marked "pending", checks the candles that have actually arrived since each was
    generated, and updates status to hit/miss/expired using the exact same label_outcome
    logic the backtester validates against history.

    A signal only has something to check once it has target_price/stop_price attached
    (see /signals — HOLD signals and any generated before that was added have neither and
    are skipped, not stuck as false negatives). A signal with fewer than max_lookforward
    real candles since it fired stays "pending" — there's nothing wrong with it, it just
    hasn't had time to play out yet, and this job will look at it again next run.
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
        future_cursor = (
            candles_collection.find({
                "pair": doc["pair"],
                "interval": doc["interval"],
                "timestamp": {"$gt": doc["timestamp"]},
            })
            .sort("timestamp", 1)
            .limit(max_lookforward)
        )
        future_docs = await future_cursor.to_list(length=max_lookforward)
        if not future_docs:
            tally["skipped_no_data"] += 1
            continue

        future_df = pd.DataFrame(future_docs)
        status, outcome_price, outcome_timestamp, candles_to_outcome = label_outcome(
            future_df, doc["direction"], doc["target_price"], doc["stop_price"], max_lookforward,
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


async def score_pending_consensus_signals(max_lookforward: int = DEFAULT_MAX_LOOKFORWARD) -> dict:
    """
    Same closing-the-loop job as score_pending_signals, for consensus_signals_collection
    instead. A separate function rather than a generalized one because the two collections'
    documents aren't quite the same shape (ConsensusSignal has no HOLD direction to filter
    out and no target_price/stop_price existence check needed -- both are always set by
    construction, unlike a regular Signal which can be a directionless HOLD) and reusing
    price_at_signal's name would be misleading here (ConsensusSignal calls it entry_price).
    """
    cursor = consensus_signals_collection.find({"status": "pending"})
    pending = await cursor.to_list(length=None)

    tally = {"hit": 0, "miss": 0, "expired": 0, "still_pending": 0, "skipped_no_data": 0}

    for doc in pending:
        future_cursor = (
            candles_collection.find({
                "pair": doc["pair"],
                "interval": doc["interval"],
                "timestamp": {"$gt": doc["timestamp"]},
            })
            .sort("timestamp", 1)
            .limit(max_lookforward)
        )
        future_docs = await future_cursor.to_list(length=max_lookforward)
        if not future_docs:
            tally["skipped_no_data"] += 1
            continue

        future_df = pd.DataFrame(future_docs)
        status, outcome_price, outcome_timestamp, candles_to_outcome = label_outcome(
            future_df, doc["direction"], doc["target_price"], doc["stop_price"], max_lookforward,
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


async def score_pending_rl_signals(max_lookforward: int = DEFAULT_MAX_LOOKFORWARD) -> dict:
    """
    Same closing-the-loop job as score_pending_consensus_signals, for rl_signals_collection.
    RLSignal is shaped identically to ConsensusSignal for exactly this reason (always
    directional, entry_price/target_price/stop_price always set) -- this is the live,
    forward-going half of "backtest its signals and learn" (the historical-replay half is
    train_rl_policy itself).
    """
    cursor = rl_signals_collection.find({"status": "pending"})
    pending = await cursor.to_list(length=None)

    tally = {"hit": 0, "miss": 0, "expired": 0, "still_pending": 0, "skipped_no_data": 0}

    for doc in pending:
        future_cursor = (
            candles_collection.find({
                "pair": doc["pair"],
                "interval": doc["interval"],
                "timestamp": {"$gt": doc["timestamp"]},
            })
            .sort("timestamp", 1)
            .limit(max_lookforward)
        )
        future_docs = await future_cursor.to_list(length=max_lookforward)
        if not future_docs:
            tally["skipped_no_data"] += 1
            continue

        future_df = pd.DataFrame(future_docs)
        status, outcome_price, outcome_timestamp, candles_to_outcome = label_outcome(
            future_df, doc["direction"], doc["target_price"], doc["stop_price"], max_lookforward,
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
