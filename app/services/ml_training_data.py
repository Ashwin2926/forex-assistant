"""
Widens the supervised hit/miss classifier's (ml_model.py) training data beyond live-only
signals to also include backtest signals -- see PROGRESS.md for the gap this closes: live
resolved signals total in the hundreds, while a full-archive backtest produces tens of
thousands of resolved examples per pair/interval, previously untouched by the classifier.

Signals here are ConsensusSignal-shaped (SMC), not the retired rule engine's Signal --
scripts/rebuild_backtests.py runs run_consensus_backtest, and every live caller passes
consensus_signals_collection, not signals_collection. See PROGRESS.md's rule-engine-removal
migration entry.

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
import random
from pathlib import Path

import pandas as pd

from app.services.case_memory import RESOLVED_STATUSES
from app.services.ml_features import KNOWN_PAIRS, KNOWN_INTERVALS, is_current_smc_signal

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


# Only these columns are ever read from an archive file -- extract_features()/
# train_hit_classifier() don't touch entry_price, target_price, outcome_*, or agreeing_count,
# so there's no reason to load them into memory at all. strategy_calls replaces the retired
# rule engine's reasons; ConsensusSignal has no profile field (unlike Signal) so it's dropped.
_ARCHIVE_COLUMNS = [
    "pair", "interval", "timestamp", "direction", "confidence", "atr_pct", "strategy_calls", "status", "source",
]


def load_backtest_signals_archive() -> list[dict]:
    """
    Reads every valid archive file (see _VALID_ARCHIVE_FILENAMES) and restores strategy_calls
    from its JSON-stringified column back to a real list[dict] -- see
    scripts/rebuild_backtests.py's _write_signals_parquet for why it's stringified on write.

    Re-enabled 2026-09-16 after being hard-disabled since 2026-09-13's OOM incident: that
    incident was against the OLD rule-engine-based archive (16 files, ~330,000 rows total --
    the rule engine's simple EMA/RSI/MACD votes fire on a large fraction of bars). SMC
    consensus fires far more rarely (needs a weighted majority of 5 strategies to actually
    agree, see consensus.check_consensus's own docstring) -- a full fresh rebuild across all
    20 pair/interval combos totaled ~535 rows / under 600KB combined (confirmed live
    2026-09-16). No per-file downsampling needed at this volume; MAX_BACKTEST_SIGNALS below
    stays as a cheap final safety cap in case a future rebuild ever grows the archive back
    toward the old scale, not because today's archive needs it.

    Also filters out any stale pre-SMC-rewrite rows the same way get_ml_reference_signals
    filters live ones (see is_current_smc_signal) -- defensive, since every file here should
    already be freshly rebuilt, but a future partial/interrupted rebuild could leave an old
    file sitting next to fresh ones.
    """
    if not ARCHIVE_DIR.exists():
        return []
    rows: list[dict] = []
    for path in sorted(ARCHIVE_DIR.iterdir()):
        if path.name not in _VALID_ARCHIVE_FILENAMES:
            continue
        df = pd.read_parquet(path, columns=_ARCHIVE_COLUMNS)
        for record in df.to_dict("records"):
            record["strategy_calls"] = json.loads(record["strategy_calls"])
            rows.append(record)
    rows = [r for r in rows if is_current_smc_signal(r)]
    if len(rows) > MAX_BACKTEST_SIGNALS:
        rows = random.sample(rows, MAX_BACKTEST_SIGNALS)
    return rows


async def get_ml_reference_signals(signals_collection) -> list[dict]:
    """
    Live resolved signals (Mongo) UNION archived backtest signals (Parquet) -- the shared data
    source for every ML-classifier call site (create_consensus_signal's quality gate,
    POST /ml/train, POST /ml/predict, and both RL training's
    ml_reference_signals/frozen_ml_snapshot).
    """
    live_signals = await signals_collection.find(
        {"source": "live", "status": {"$in": list(RESOLVED_STATUSES)}}
    ).to_list(length=None)
    # Excludes pre-SMC-rewrite consensus signals still sitting in the collection -- see
    # is_current_smc_signal's own docstring for why a stale-shaped record can't just be
    # included as extra data.
    live_signals = [s for s in live_signals if is_current_smc_signal(s)]
    return live_signals + load_backtest_signals_archive()
