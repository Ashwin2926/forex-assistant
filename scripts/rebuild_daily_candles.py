"""
Rebuilds every stored 1day candle from 1h history and deletes the ones that don't belong --
the same thing POST /ingest/1day?full_rebuild=true does, but run on a GitHub Actions runner
(.github/workflows/rebuild-daily-candles.yml) instead of inside FastAPI Cloud, where it read
every pair's whole 1h history, took ~2 min, timed out and blocked the queue behind it
(2026-10-12; see app/main.py's _rebuild_daily_candles).

Why it's needed (found 2026-10-14): daily bars are built from 1h and open at 17:00 New York
(see app/services/daily_candles.py), but the manual candle catch-up still fetched the
provider's own daily feed, stamped 00:00 UTC -- so recent days were stored twice (00:00 and
21:00 bars covering overlapping hours, plus provider weekend bars). Live 1day signals read
these straight from Mongo, so the duplicates roughly doubled the bar count -- making 30-bar
expiries fire about twice as early -- and let one price move be scored against two bars.

Default is a dry run that only reports, per pair: daily bars built from 1h, stored bars,
what would be inserted / changed / deleted (deletions broken down by year, newest listed).
--confirm applies it: upserts every built bar, then deletes stored 1day docs that aren't
one of them. A pair whose 1h history builds fewer than MIN_SAFE_BARS daily bars is skipped
(deleting its stored bars could starve live 1day signals of history).

Deliberately avoids app.core.config / app.core.database (same as scripts/train_rl.py) --
only needs MONGODB_URI / MONGODB_DB_NAME.
"""
import argparse
import os
import sys
from collections import Counter

from pymongo import MongoClient, UpdateOne

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.services.candle_archive import load_full_candle_history_sync  # noqa: E402

DEFAULT_PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD"]
MIN_SAFE_BARS = 250  # well above rl_engine.MIN_WARMUP_BARS (30) and any EMA/ATR warmup
PRICE_TOLERANCE = 1e-9


def pairs_list() -> list[str]:
    raw = os.environ.get("FOREX_PAIRS")
    return [p.strip() for p in raw.split(",") if p.strip()] if raw else DEFAULT_PAIRS


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm", action="store_true", help="apply changes (default: dry run)")
    args = parser.parse_args()

    db = MongoClient(os.environ["MONGODB_URI"], serverSelectionTimeoutMS=15000)[
        os.environ.get("MONGODB_DB_NAME", "forex_assistant")
    ]
    candles = db["candles"]
    mode = "APPLYING" if args.confirm else "DRY RUN (pass --confirm to apply)"
    print(f"{mode}\n")

    for pair in pairs_list():
        daily = load_full_candle_history_sync(db, pair, "1day")
        stored = {
            d["timestamp"]: d
            for d in candles.find(
                {"pair": pair, "interval": "1day"},
                {"_id": 0, "timestamp": 1, "open": 1, "high": 1, "low": 1, "close": 1},
            )
        }
        built = {row.timestamp.to_pydatetime(): row for row in daily.itertuples(index=False)}

        to_delete = sorted(set(stored) - set(built))
        to_insert = sorted(set(built) - set(stored))
        changed = [
            ts for ts in set(built) & set(stored)
            if any(abs(float(getattr(built[ts], f)) - float(stored[ts].get(f, 0))) > PRICE_TOLERANCE
                   for f in ("open", "high", "low", "close"))
        ]

        print(f"== {pair}")
        if built:
            first, last = min(built), max(built)
            print(f"  built from 1h : {len(built)} daily bars, {first:%Y-%m-%d %H:%M} .. {last:%Y-%m-%d %H:%M} UTC")
        else:
            print("  built from 1h : 0 daily bars")
        print(f"  stored now    : {len(stored)}")
        print(f"  would insert  : {len(to_insert)}   would change: {len(changed)}   would delete: {len(to_delete)}")
        if to_delete:
            by_year = Counter(ts.year for ts in to_delete)
            print("  deletions by year: " + ", ".join(f"{y}: {n}" for y, n in sorted(by_year.items())))
            hours = Counter(f"{ts:%H:%M}" for ts in to_delete)
            print("  deletions by UTC open time: " + ", ".join(f"{h}: {n}" for h, n in hours.most_common()))
            print("  newest deletions: " + ", ".join(f"{ts:%Y-%m-%d %H:%M} ({ts:%a})" for ts in to_delete[-8:]))

        if len(built) < MIN_SAFE_BARS:
            print(f"  SKIPPED: only {len(built)} bars built from 1h (< {MIN_SAFE_BARS}) -- not touching this pair")
            continue
        if not args.confirm:
            continue

        ops = [
            UpdateOne(
                {"pair": pair, "interval": "1day", "timestamp": ts},
                {"$set": {"open": float(r.open), "high": float(r.high), "low": float(r.low),
                          "close": float(r.close), "volume": float(r.volume)}},
                upsert=True,
            )
            for ts, r in built.items()
        ]
        for i in range(0, len(ops), 1000):
            candles.bulk_write(ops[i:i + 1000], ordered=False)
        removed = candles.delete_many(
            {"pair": pair, "interval": "1day", "timestamp": {"$in": to_delete}}
        ).deleted_count if to_delete else 0
        remaining = candles.count_documents({"pair": pair, "interval": "1day"})
        print(f"  APPLIED: upserted {len(ops)}, deleted {removed}, now stored {remaining}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
