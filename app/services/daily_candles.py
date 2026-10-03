"""
Daily candles built from 1h candles instead of the data provider's own daily feed.

Found 2026-10-03: the stored 1day candles for EUR/USD and GBP/USD were broken before 2024
(each day's open typically 30-40 pips from the previous close -- about half the day's range;
real FX trades continuously so the two should be ~equal) and still had some bad days after.
The 1h candles are clean for every pair, so daily bars are rebuilt from them everywhere
1day data is used: training (candle_archive.load_full_candle_history*), live signals and
scoring (POST /ingest/1day writes these into candles_collection).

A trading day closes at 17:00 New York -- the standard FX daily close, and what MT5/most
brokers' daily candles use -- so the bar for day D covers [D-1 17:00 NY, D 17:00 NY).
Timestamps are the bar's OPEN time in naive UTC, the same convention as every other interval
(data_fetcher.interval_due relies on it). Only finished days are returned; weekend-labelled
fragments and days with too few hourly bars (holidays, data gaps) are dropped.
"""
from datetime import datetime
from typing import Optional

import pandas as pd

NY = "America/New_York"
MIN_HOURLY_BARS = 18  # a normal trading day has 24; fewer means a holiday or a data gap
CANDLE_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def daily_from_hourly(hourly: pd.DataFrame, now: Optional[datetime] = None) -> pd.DataFrame:
    """hourly: 1h candles with naive-UTC open timestamps. Returns finished daily candles."""
    if hourly.empty:
        return pd.DataFrame(columns=CANDLE_COLUMNS)
    h = hourly.sort_values("timestamp").drop_duplicates(subset="timestamp", keep="last")
    ny = pd.to_datetime(h["timestamp"]).dt.tz_localize("UTC").dt.tz_convert(NY)
    # 17:00 NY onwards belongs to the next calendar day's bar.
    day = (ny + pd.Timedelta(hours=7)).dt.normalize().dt.tz_localize(None)
    h = h.assign(day=day.values)
    d = h.groupby("day", sort=True).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum"), bars=("open", "size"),
    ).reset_index()
    d = d[(d["day"].dt.dayofweek < 5) & (d["bars"] >= MIN_HOURLY_BARS)]

    # Bar for day D opens at D-1 17:00 NY and closes at D 17:00 NY.
    close_ny = (d["day"] + pd.Timedelta(hours=17)).dt.tz_localize(NY, ambiguous="NaT", nonexistent="shift_forward")
    open_utc = (close_ny - pd.Timedelta(days=1)).dt.tz_convert("UTC").dt.tz_localize(None)
    close_utc = close_ny.dt.tz_convert("UTC").dt.tz_localize(None)
    now = now or datetime.utcnow()
    d = d.assign(timestamp=open_utc.values, _close=close_utc.values)
    d = d[d["_close"] <= pd.Timestamp(now)]
    return d[CANDLE_COLUMNS].reset_index(drop=True)
