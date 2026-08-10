"""
One-off cleanup for the live-signal duplication bug (fixed in app/main.py's create_signal:
it was reading candles oldest-first, so every dashboard load before a real candle arrived
regenerated the identical signal and inserted it again — one lucky/unlucky signal ended up
counted 20-30x in /signals/accuracy).

Groups resolved live signals by (pair, interval, profile, direction, price_at_signal, status)
and deletes all but the earliest doc in each group, since same-price duplicates are the
same underlying signal inserted repeatedly, not independent occurrences.

Run from the project root with the real .env in place:
    python -m scripts.dedupe_signals            # dry run, prints what would be deleted
    python -m scripts.dedupe_signals --apply     # actually deletes
"""
import asyncio
import sys
from collections import defaultdict

from app.core.database import signals_collection


async def main(apply: bool):
    cursor = signals_collection.find(
        {"source": "live", "status": {"$in": ["hit", "miss", "expired"]}}
    ).sort("timestamp", 1)
    docs = await cursor.to_list(length=None)

    groups = defaultdict(list)
    for d in docs:
        key = (d["pair"], d["interval"], d["profile"], d["direction"], d["price_at_signal"], d["status"])
        groups[key].append(d)

    to_delete = []
    for key, group in groups.items():
        if len(group) > 1:
            keep, drop = group[0], group[1:]
            to_delete.extend(d["_id"] for d in drop)
            print(f"{key}: keeping 1 (timestamp={keep['timestamp']}), dropping {len(drop)}")

    print(f"\n{len(to_delete)} duplicate signal(s) to delete out of {len(docs)} resolved live signals.")

    if not apply:
        print("Dry run only — re-run with --apply to actually delete.")
        return

    if to_delete:
        result = await signals_collection.delete_many({"_id": {"$in": to_delete}})
        print(f"Deleted {result.deleted_count} documents.")


if __name__ == "__main__":
    asyncio.run(main(apply="--apply" in sys.argv))
