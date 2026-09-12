"""
Backfills historical candles directly into the Parquet archive (data/candles/), bypassing
Mongo entirely. See PROGRESS.md's 2026-09-12 entries for why: routing deep backfills through
Mongo first (POST /ingest/backfill/{interval}) repeatedly grew candles_collection past
Atlas's free-tier storage cap, needed multiple manual trim cycles afterward, and every
individual page fetch was a separate HTTP request to FastAPI Cloud subject to its gateway's
~125s timeout -- runs kept silently continuing server-side past what the client saw,
needing several retries just to find out whether they'd actually finished.

This script has none of those constraints: it's one continuous process in a single GitHub
Actions job (backfill-archive.yml, up to 6h), talks to Twelve Data directly, and writes
straight into data/candles/*.parquet the same way scripts/export_candles.py merges (read
existing archive, concat, dedupe, write back) -- coverage only ever grows. Commits and
pushes periodically (not just at the very end) so progress survives a job timeout or
cancellation partway through a pair.

Deliberately does NOT touch candles_collection -- keeping Mongo's rolling recent window
fresh for live signal generation is data_fetcher.fetch_and_store's job (driven by
keep-fresh.yml), a completely separate concern from this deep archival backfill.

Mirrors export_candles.py/train_rl.py's own conventions: plain env vars read directly
(no app.core.config, which would require settings this script has no use for), pairs list
resolved the same way.
"""
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import httpx
import pandas as pd

DEFAULT_PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD"]
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "candles"
CANDLE_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

BASE_URL = "https://api.twelvedata.com/time_series"
PAGE_SIZE = 5000  # Twelve Data's documented max outputsize per call, even on paid plans.

# Paces calls comfortably under Twelve Data's per-minute rate limit -- the binding
# constraint for a deep backfill, not the 800/day credit cap (outputsize doesn't change a
# call's credit cost). Same value data_fetcher.BACKFILL_CALL_DELAY_SECONDS uses.
CALL_DELAY_SECONDS = 8
MAX_RETRIES_ON_RATE_LIMIT = 3
RETRY_BACKOFF_SECONDS = 20

# Commits/pushes after this many Twelve Data calls within a single pair/interval's loop (in
# addition to always doing so at the end of that loop) -- so a job that times out or gets
# cancelled mid-pair still keeps most of its progress instead of losing an entire pair's
# work because nothing was pushed yet.
COMMIT_EVERY_N_CALLS = 10


def pairs_list() -> list[str]:
    raw = os.environ.get("FOREX_PAIRS")
    if not raw:
        return DEFAULT_PAIRS
    return [p.strip() for p in raw.split(",") if p.strip()]


def slug(pair: str) -> str:
    return pair.replace("/", "_")


def fetch_page(pair: str, interval: str, api_key: str, end_date: str | None) -> list[dict]:
    params = {
        "symbol": pair, "interval": interval, "outputsize": PAGE_SIZE,
        "apikey": api_key, "timezone": "UTC",
    }
    if end_date is not None:
        params["end_date"] = end_date

    resp = None
    for attempt in range(MAX_RETRIES_ON_RATE_LIMIT + 1):
        resp = httpx.get(BASE_URL, params=params, timeout=30.0)
        if resp.status_code == 429 and attempt < MAX_RETRIES_ON_RATE_LIMIT:
            time.sleep(RETRY_BACKOFF_SECONDS)
            continue
        resp.raise_for_status()
        break

    data = resp.json()
    if data.get("status") == "error":
        raise ValueError(f"Twelve Data error for {pair}/{interval}: {data.get('message')}")

    rows = []
    for v in data.get("values", []):
        ts = (
            datetime.strptime(v["datetime"], "%Y-%m-%d %H:%M:%S")
            if len(v["datetime"]) > 10
            else datetime.strptime(v["datetime"], "%Y-%m-%d")
        )
        rows.append({
            "timestamp": ts,
            "open": float(v["open"]), "high": float(v["high"]),
            "low": float(v["low"]), "close": float(v["close"]),
            "volume": float(v["volume"]) if v.get("volume") else None,
        })
    return rows


def commit_and_push(message: str) -> None:
    subprocess.run(["git", "config", "user.name", "github-actions[bot]"], check=True)
    subprocess.run(
        ["git", "config", "user.email", "github-actions[bot]@users.noreply.github.com"], check=True,
    )
    subprocess.run(["git", "add", "data/candles/"], check=True)
    diff = subprocess.run(["git", "diff", "--cached", "--quiet"])
    if diff.returncode == 0:
        print("No archive changes to commit.")
        return
    subprocess.run(["git", "commit", "-m", message], check=True)
    subprocess.run(["git", "push"], check=True)


def backfill_pair_interval(pair: str, interval: str, start_date: datetime, api_key: str, max_calls: int) -> None:
    out_path = OUTPUT_DIR / f"{slug(pair)}_{interval}.parquet"
    archive_df = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame(columns=CANDLE_COLUMNS)

    cursor = archive_df["timestamp"].min() if len(archive_df) else None
    if cursor is not None and cursor <= start_date:
        print(f"{pair} {interval}: already covers back to {cursor}, nothing to do")
        return

    calls_made = 0
    while calls_made < max_calls:
        end_date_param = cursor.strftime("%Y-%m-%d %H:%M:%S") if cursor is not None else None
        rows = fetch_page(pair, interval, api_key, end_date_param)
        calls_made += 1

        if not rows:
            print(f"{pair} {interval}: Twelve Data has no older history to give (stopping before {start_date})")
            break

        page_df = pd.DataFrame(rows, columns=CANDLE_COLUMNS)
        archive_df = pd.concat([archive_df, page_df], ignore_index=True)
        archive_df = archive_df.drop_duplicates(subset="timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)
        archive_df.to_parquet(out_path, index=False, compression="brotli")

        oldest = page_df["timestamp"].min()
        print(f"{pair} {interval}: call {calls_made}/{max_calls} -> {len(rows)} candles, oldest now {oldest}")

        if cursor is not None and oldest >= cursor:
            print(f"{pair} {interval}: same page again, Twelve Data has nothing older")
            break
        cursor = oldest

        if calls_made % COMMIT_EVERY_N_CALLS == 0:
            commit_and_push(f"Backfill {pair} {interval} to archive (in progress, oldest={cursor.date()})")

        if cursor <= start_date:
            print(f"{pair} {interval}: reached {start_date}")
            break
        time.sleep(CALL_DELAY_SECONDS)

    commit_and_push(f"Backfill {pair} {interval} to archive (oldest={cursor.date() if cursor is not None else 'n/a'})")


def main() -> None:
    api_key = os.environ["TWELVE_DATA_API_KEY"]
    interval = os.environ["INTERVAL"]
    start_date = datetime.strptime(os.environ.get("START_DATE", "2020-01-01"), "%Y-%m-%d")
    max_calls_per_pair = int(os.environ.get("MAX_CALLS_PER_PAIR", "400"))

    for pair in pairs_list():
        backfill_pair_interval(pair, interval, start_date, api_key, max_calls_per_pair)


if __name__ == "__main__":
    main()
