"""
One-off cleanup for pending live signals whose interval isn't part of the actively-ingested
set (15min/1h, per .github/workflows/keep-fresh.yml) -- they can never resolve, because no
new candles for that interval ever arrive for /signals/score to check them against, so
they'd stay "pending" forever (e.g. a stray EUR/USD 1day signal from manual curl testing).

Run from the project root with the real .env in place:
    python -m scripts.cleanup_orphaned_signals            # dry run, prints what would be deleted
    python -m scripts.cleanup_orphaned_signals --apply     # actually deletes
"""
import asyncio
import sys

from app.core.database import signals_collection

ACTIVELY_INGESTED_INTERVALS = {"15min", "1h"}


async def main(apply: bool):
    cursor = signals_collection.find({
        "source": "live",
        "status": "pending",
        "interval": {"$nin": list(ACTIVELY_INGESTED_INTERVALS)},
    })
    docs = await cursor.to_list(length=None)

    for d in docs:
        print(
            f"{d['pair']} {d['interval']} {d['profile']} {d['direction']} @ {d['price_at_signal']} "
            f"(created {d['timestamp']}) -- orphaned, can never resolve"
        )

    print(f"\n{len(docs)} orphaned pending signal(s) found.")

    if not apply:
        print("Dry run only -- re-run with --apply to actually delete.")
        return

    if docs:
        result = await signals_collection.delete_many({"_id": {"$in": [d["_id"] for d in docs]}})
        print(f"Deleted {result.deleted_count} documents.")


if __name__ == "__main__":
    asyncio.run(main(apply="--apply" in sys.argv))
