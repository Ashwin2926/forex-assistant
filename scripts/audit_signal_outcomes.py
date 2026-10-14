"""
Read-only audit of closed live RL signals: how each one actually closed, and whether the
recorded outcome matches what price did. Run via .github/workflows/audit-signal-outcomes.yml.

A "hit" can mean two different things in rl_signals:
  - the target was touched (signal_engine.label_outcome -- outcome_price == target_price), or
  - the agent closed the trade early (POST /rl/signal's learned EXIT, closed_early=True) and
    the realized move was >= 0 after spread -- often a fraction of the target distance.
This splits them apart, then replays the stored candles of the signal's own interval from
entry to the recorded close and checks:
  - stop touched BEFORE the recorded close  -> should have been a miss
  - target touched before an early exit      -> should have been a full target hit
  - a candle touching both levels            -> resolved as a miss by label_outcome's
    conservative rule; the 1h candles inside it say which really came first
Candles may have changed since scoring: 1day signals opened before the 2026-10-03 switch to
1h-built daily bars were scored on the old 00:00 UTC provider bars, which no longer exist
(2026-10-14 de-dup), so their replays use differently-aligned bars and are approximate.
"""
import os
from collections import Counter
from datetime import datetime, timedelta

from pymongo import MongoClient

INTERVAL_STEP = {"5min": timedelta(minutes=5), "15min": timedelta(minutes=15), "1h": timedelta(hours=1),
                 "4h": timedelta(hours=4), "1day": timedelta(days=1)}


def touched(c, direction, level, kind):
    if kind == "stop":
        return c["low"] <= level if direction == "BUY" else c["high"] >= level
    return c["high"] >= level if direction == "BUY" else c["low"] <= level


def first_touch_hourly(db, s, start, end):
    """Which level the 1h candles inside [start, end) reach first, or None."""
    for c in db["candles"].find({"pair": s["pair"], "interval": "1h", "timestamp": {"$gte": start, "$lt": end}},
                                sort=[("timestamp", 1)]):
        t, st = touched(c, s["direction"], s["target_price"], "target"), touched(c, s["direction"], s["stop_price"], "stop")
        if t and st:
            return "both within one hour"
        if st:
            return "stop first"
        if t:
            return "target first"
    return None


def main():
    db = MongoClient(os.environ["MONGODB_URI"], serverSelectionTimeoutMS=15000)[os.environ["MONGODB_DB_NAME"]]
    since = datetime.utcnow() - timedelta(days=int(os.environ.get("DAYS", "60")))
    signals = list(db["rl_signals"].find(
        {"source": {"$ne": "backtest"}, "status": {"$in": ["hit", "miss", "expired"]}, "timestamp": {"$gte": since}},
        {"_id": 0, "state": 0, "q_values": 0},
    ).sort("timestamp", 1))

    kinds = Counter()
    print("| Opened (UTC) | Pair | TF | Dir | Status | How it closed | Move | Target dist | Replay check |")
    print("|---|---|---|---|---|---|---|---|---|")
    for s in signals:
        entry, tgt, stp = s["entry_price"], s["target_price"], s["stop_price"]
        tgt_dist = abs(tgt - entry) / entry * 100
        if s.get("closed_early"):
            how = f"early exit ({'profit' if s['status'] == 'hit' else 'loss'})"
        elif s["status"] == "hit":
            how = "target reached"
        elif s["status"] == "miss":
            how = "stop reached"
        else:
            how = "expired"
        kinds[how] += 1

        # Replay own-interval candles strictly after entry up to and including the close.
        end = s.get("outcome_timestamp")
        check = "-"
        if end is not None:
            # label_outcome stamps a target/stop close with the bar that touched it, so that bar
            # counts. An early exit or expiry is stamped with the bar that was still forming at
            # decision time -- its final high/low partly happened AFTER the exit, so stop there.
            upper = {"$lte": end} if how in ("target reached", "stop reached") else {"$lt": end}
            candles = list(db["candles"].find(
                {"pair": s["pair"], "interval": s["interval"], "timestamp": {"$gt": s["timestamp"], **upper}},
                sort=[("timestamp", 1)],
            ))
            problems = []
            for c in candles:
                t, st = touched(c, s["direction"], tgt, "target"), touched(c, s["direction"], stp, "stop")
                if t and st:
                    order = first_touch_hourly(db, s, c["timestamp"], c["timestamp"] + INTERVAL_STEP[s["interval"]])
                    problems.append(f"{c['timestamp']:%m-%d %H:%M} touched both ({order or 'no 1h data'})")
                    break
                if st and how != "stop reached":
                    problems.append(f"stop touched {c['timestamp']:%m-%d %H:%M} before close")
                    break
                if t and how not in ("target reached",):
                    problems.append(f"target touched {c['timestamp']:%m-%d %H:%M} before close")
                    break
            if how == "target reached" and not problems and not any(touched(c, s["direction"], tgt, "target") for c in candles):
                problems.append("target NOT touched in stored candles")
            if how == "stop reached" and not problems and not any(touched(c, s["direction"], stp, "stop") for c in candles):
                problems.append("stop NOT touched in stored candles")
            check = "; ".join(problems) if problems else f"OK ({len(candles)} bars)"

        move = s.get("outcome_pct_move")
        print(f"| {s['timestamp']:%Y-%m-%d %H:%M} | {s['pair']} | {s['interval']} | {s['direction']} | {s['status']} "
              f"| {how} | {'' if move is None else f'{move:+.3f}%'} | {tgt_dist:.3f}% | {check} |")

    print(f"\n{len(signals)} closed live RL signals since {since:%Y-%m-%d}: " + ", ".join(f"{k}: {v}" for k, v in kinds.most_common()))


if __name__ == "__main__":
    main()
