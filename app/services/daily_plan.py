"""
"Plan for today" numbers for GET /dashboard/daily-plan -- only what tested as reliable in the
2026-09-30 daily-bias backtest (2020-2026, daily bars rebuilt from 1h):
  - expected range: 14-day ATR of daily candles; 82-84% of days landed within 0.5-1.5x it.
  - yesterday's high/low: one or the other was reached on ~85% of days (each ~50%).
  - last week's high/low.
No direction: no rule tested (the 5 strategies, trend, momentum) called the day's direction
better than a coin flip -- see PROGRESS.md's 2026-09-30/10-03 entries.
"""
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

ATR_DAYS = 14


def pip_size(pair: str) -> float:
    return 0.01 if pair.endswith("JPY") else 0.0001


def build_daily_plan(
    pair: str, daily: list[dict], hourly_today: list[dict], price: Optional[float],
    price_at: Optional[datetime] = None,
) -> dict:
    """
    daily: finished 1day candles (timestamp = open, 17:00 New York the evening before), any
    order. hourly_today: 1h candles since today's open. price: latest close (5min), falls back
    to yesterday's close.
    """
    if len(daily) < ATR_DAYS + 1:
        return {"pair": pair, "error": "not enough daily candles yet"}
    pip = pip_size(pair)
    d = pd.DataFrame(daily).sort_values("timestamp").reset_index(drop=True)
    prev_close = d["close"].shift()
    tr = pd.concat(
        [d["high"] - d["low"], (d["high"] - prev_close).abs(), (d["low"] - prev_close).abs()], axis=1,
    ).max(axis=1)
    atr = float(tr.iloc[-ATR_DAYS:].mean())
    yday = d.iloc[-1]
    today_open = pd.Timestamp(yday["timestamp"]) + pd.Timedelta(days=1)

    # Weeks by close date (a bar's close date = open + 1 day); last week = latest finished
    # Mon-Fri week before the one today belongs to.
    close_date = (d["timestamp"] + pd.Timedelta(days=1)).dt.normalize()
    d["week"] = close_date - pd.to_timedelta(close_date.dt.dayofweek, unit="D")
    today_close_date = (today_open + pd.Timedelta(days=1)).normalize()
    if today_close_date.dayofweek >= 5:  # after Friday's close the next trading day is Monday
        today_close_date += pd.Timedelta(days=7 - today_close_date.dayofweek)
    this_week = today_close_date - pd.Timedelta(days=today_close_date.dayofweek)
    past = d[d["week"] < this_week]
    last_week = past[past["week"] == past["week"].max()] if len(past) else past

    price = float(price) if price is not None else float(yday["close"])
    highs = [h["high"] for h in hourly_today]
    lows = [h["low"] for h in hourly_today]
    today_high = max(highs + [price]) if highs else None
    today_low = min(lows + [price]) if lows else None

    def level(name: str, value: Optional[float]) -> Optional[dict]:
        if value is None or pd.isna(value):
            return None
        value = float(value)
        return {
            "name": name, "price": round(value, 5),
            "distance_pips": round((value - price) / pip, 1),
            "touched_today": today_high is not None and today_low <= value <= today_high,
        }

    return {
        "pair": pair,
        "price": round(price, 5),
        "price_at": price_at,
        "day_opened_at": today_open.to_pydatetime(),
        "atr_pips": round(atr / pip, 1),
        "expected_range_pips": [round(0.5 * atr / pip, 1), round(1.5 * atr / pip, 1)],
        "today_range_pips": round((today_high - today_low) / pip, 1) if today_high is not None else None,
        "levels": [lv for lv in (
            level("Yesterday high", yday["high"]),
            level("Yesterday low", yday["low"]),
            level("Last week high", last_week["high"].max() if len(last_week) else None),
            level("Last week low", last_week["low"].min() if len(last_week) else None),
        ) if lv],
    }
