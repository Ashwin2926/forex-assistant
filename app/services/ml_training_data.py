"""
Widens the supervised hit/miss classifier's (ml_model.py) training data beyond live-only
signals to also include backtest signals -- see PROGRESS.md for the gap this closes: live
resolved signals total in the hundreds, while a full-archive backtest produces tens of
thousands of resolved examples per pair/interval, previously untouched by the classifier.

Backtest signals are read from a git-committed Parquet archive (data/backtest_signals_archive/
*.parquet, one file per pair/interval, written by scripts/rebuild_backtests.py), NOT from
Mongo -- see PROGRESS.md's 2026-09-12 outage entry for why: writing ~90,000+ signals into
backtest_signals_collection pushed Atlas over its 512MB M0 quota and crashed the whole app
(writes are blocked cluster-wide once over quota, including the index-creation call in app
startup). The Parquet archive has no comparable size cap (same reasoning as
data/candles/*.parquet already uses for deep candle history), so a rebuild can never repeat
that outage regardless of how much history it replays.

Every file in the archive is, by construction, always "today's live default config" --
scripts/rebuild_backtests.py only ever writes it with default_config_for("intraday") and no
target/stop override, overwriting the previous file each run -- so unlike the old
Mongo-run-matching approach, there's no separate "is this run still current" check needed:
the archive being present at all IS the qualification.

Deliberately has NO app.core.* imports -- this needs to be importable from
scripts/rebuild_backtests.py and scripts/train_rl.py, which run standalone on a GitHub
Actions runner and avoid app.core.* on purpose (see those scripts' own module docstrings).
Reading local Parquet files is fast enough (~0.024s for a 502k-row file, see
candle_archive.py's own measurement) to call synchronously from the async app/main.py path
too, same precedent as candle_archive.load_archived_candles.
"""
import json
from pathlib import Path

import pandas as pd

from app.services.case_memory import RESOLVED_STATUSES
from app.services.ml_features import KNOWN_PAIRS, KNOWN_INTERVALS

ARCHIVE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "backtest_signals_archive"

# Only files matching {pair_slug}_{interval}.parquet for a known pair/interval are trusted --
# NOT a bare "*.parquet" glob. Learned the hard way: a one-off incident-recovery export
# (backtest_signals_2026-09-12.parquet, written by scripts/export_backtest_signals.py during
# the 2026-09-12 outage) sat in this same directory and would otherwise have been silently
# read back in as if it were current data -- it was actually a raw, unfiltered dump of the
# whole old Mongo collection: stale pre-2026-09-07 EMA 12/26 signals, RL trade-log noise, and
# the handful of genuinely-current signals all mixed together, exactly what the qualifying-
# config design was built to keep out. A stray or manually-dropped file can't repeat that
# once only exact expected filenames are accepted.
_VALID_ARCHIVE_FILENAMES = {
    f"{pair.replace('/', '_')}_{interval}.parquet" for pair in KNOWN_PAIRS for interval in KNOWN_INTERVALS
}

# Hard cap on how many archived backtest signals a single fetch returns, via a random sample
# (not "first N" -- avoids skewing toward whichever combo happens to have the most rows, e.g.
# 5min naturally producing far more signals than 1day). This fetch runs SYNCHRONOUSLY on every
# live signal-generation call (create_signal's ML gate, POST /rl/signal, not just the
# POST /ml/train diagnostic) -- even with storage no longer a concern, feature-extracting and
# refitting XGBoost on an unbounded, ever-growing archive would still cost real CPU time on
# every request. 5,000 is generous relative to the ~662 live signals this project started
# with, while keeping fit time in the range ml_model.py's own N_ESTIMATORS/MAX_DEPTH comment
# was designed around -- an operational ceiling, not a data-quality judgment.
MAX_BACKTEST_SIGNALS = 5000


def load_backtest_signals_archive() -> list[dict]:
    """
    Reads every data/backtest_signals_archive/*.parquet file (one per pair/interval, see
    scripts/rebuild_backtests.py) and returns a randomly capped sample across all of them
    combined. Missing archive dir (nothing rebuilt yet) returns an empty list, not an error --
    same "absent archive means no data yet" convention as candle_archive.load_archived_candles.
    """
    if not ARCHIVE_DIR.exists():
        return []

    frames = [
        pd.read_parquet(p) for p in ARCHIVE_DIR.glob("*.parquet") if p.name in _VALID_ARCHIVE_FILENAMES
    ]
    if not frames:
        return []

    df = pd.concat(frames, ignore_index=True)
    if len(df) > MAX_BACKTEST_SIGNALS:
        df = df.sample(n=MAX_BACKTEST_SIGNALS)

    signals = df.to_dict("records")
    for s in signals:
        # reasons is stored JSON-stringified (a list[dict] doesn't round-trip through Parquet
        # as cleanly as a plain column) -- restored here so extract_features() sees the same
        # list-of-dicts shape it gets from a live Mongo document.
        if isinstance(s.get("reasons"), str):
            s["reasons"] = json.loads(s["reasons"])
    return signals


async def get_ml_reference_signals(signals_collection) -> list[dict]:
    """
    Live resolved signals (Mongo) UNION archived backtest signals (Parquet) -- the shared data
    source for every ML-classifier call site (the create_signal quality gate, POST /ml/train,
    POST /ml/predict, and both RL training's ml_reference_signals/frozen_ml_snapshot).
    """
    live_signals = await signals_collection.find(
        {"source": "live", "status": {"$in": list(RESOLVED_STATUSES)}}
    ).to_list(length=None)
    return live_signals + load_backtest_signals_archive()
