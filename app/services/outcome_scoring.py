import pandas as pd
from app.core.database import candles_collection, signals_collection
from app.services.signal_engine import label_outcome

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
