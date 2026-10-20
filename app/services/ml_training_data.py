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
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from app.services.case_memory import RESOLVED_STATUSES
from app.services.ml_features import KNOWN_PAIRS, KNOWN_INTERVALS, is_current_smc_signal
from app.services.strategies import STRATEGY_NAMES

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

    Cached in memory, keyed by each file's name/size/mtime (2026-10-03): this used to re-read
    all 16 Parquet files on every call, and get_ml_reference_signals is awaited directly in
    POST /rl/signal's request handler, so the read ran on the event loop. keep-fresh.yml kept
    showing one /rl/signal call answer, then the next few hang to the gateway's 524 with
    app/main.py's 90s _run_native timeout never firing -- i.e. the event loop itself was stuck,
    which a stalled pyarrow read in a process that also runs torch/XGBoost would explain. The
    files only change on a redeploy, so after the first read this is a dict lookup.
    use_threads=False keeps pyarrow's own thread pool out of the web process entirely.
    """
    if not ARCHIVE_DIR.exists():
        return []
    paths = [p for p in sorted(ARCHIVE_DIR.iterdir()) if p.name in _VALID_ARCHIVE_FILENAMES]
    key = tuple((p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in paths)
    if _archive_cache["key"] == key:
        return list(_archive_cache["rows"])
    rows: list[dict] = []
    for path in paths:
        df = pd.read_parquet(path, columns=_ARCHIVE_COLUMNS, use_threads=False)
        for record in df.to_dict("records"):
            record["strategy_calls"] = json.loads(record["strategy_calls"])
            rows.append(record)
    rows = [r for r in rows if is_current_smc_signal(r)]
    if len(rows) > MAX_BACKTEST_SIGNALS:
        rows = random.sample(rows, MAX_BACKTEST_SIGNALS)
    _archive_cache["key"], _archive_cache["rows"] = key, rows
    return list(rows)


_archive_cache: dict = {"key": None, "rows": []}


# RL trades as ML training rows (added 2026-10-20). Until then the classifier only learned
# from resolved consensus signals, and none had resolved since 2026-09-26 -- every "retrain"
# refit the same 632 archived rows. RL trades resolve all the time: live RLSignals, plus the
# test-slice trades every nightly retrain logs to backtest_signals (each carrying its entry
# market state since this change, saved to ML_RL_TRADES_COLLECTION). Their state starts with the same 5 strategy votes (signed,
# BUY-positive) and atr_pct the ML features are built from (rl_engine.RL_MARKET_FEATURE_NAMES).
MAX_RL_TRADES = 5000  # same reasoning as MAX_BACKTEST_SIGNALS; sampled deterministically
RL_ROWS_TTL_SECONDS = 1800  # the set only grows nightly; don't reload it on every signal call
RL_TRADE_QUERY = {"status": {"$in": list(RESOLVED_STATUSES)}, f"state.{len(STRATEGY_NAMES)}": {"$exists": True}}
RL_TRADE_PROJECTION = {"_id": 0, "pair": 1, "interval": 1, "timestamp": 1, "direction": 1, "status": 1, "state": 1}


def rl_trade_to_ml_row(doc: dict) -> Optional[dict]:
    """A resolved RL trade (live RLSignal or logged backtest trade) in the signal-like shape
    extract_features() reads: one synthetic strategy_call per vote, atr_pct, and the same
    weighted agreement confidence rl_engine.compute_ml_scores scores hypotheticals with
    (consensus.direction_confidence; all strategy weights are 1.0)."""
    state = doc.get("state") or []
    n = len(STRATEGY_NAMES)
    direction = doc.get("direction")
    if len(state) <= n or direction not in ("BUY", "SELL"):
        return None
    calls, agreeing = [], 0.0
    for name, vote in zip(STRATEGY_NAMES, state[:n]):
        vote = float(vote)
        call_direction = "BUY" if vote > 0 else "SELL" if vote < 0 else "HOLD"
        calls.append({"strategy": name, "direction": call_direction, "strength": abs(vote)})
        if call_direction == direction:
            agreeing += abs(vote)
    return {
        "pair": doc.get("pair"), "interval": doc.get("interval"), "timestamp": doc.get("timestamp"),
        "direction": direction, "status": doc.get("status"), "strategy_calls": calls,
        "confidence": round(min(agreeing / n, 1.0) * 100, 1), "atr_pct": float(state[n]), "source": "rl",
    }


def select_rl_rows(docs: list[dict]) -> list[dict]:
    rows = [r for r in (rl_trade_to_ml_row(d) for d in docs) if r and r["timestamp"] is not None]
    rows.sort(key=lambda r: (r["timestamp"], r["pair"] or "", r["interval"] or "", r["direction"]))
    if len(rows) > MAX_RL_TRADES:
        # Fixed seed: the same stored trades always give the same sample, so the cached ML fit
        # (ml_model._cached_fit) isn't invalidated by a fresh random draw on every reload.
        rows = sorted(random.Random(0).sample(rows, MAX_RL_TRADES), key=lambda r: r["timestamp"])
    return rows


ML_RL_TRADES_COLLECTION = "ml_rl_trades"


def ml_rl_trade_upserts(trade_signals) -> list:
    """pymongo UpdateOne ops saving every RL eval trade (kept policy or not) to the ML's own
    collection -- backtest_signals only keeps the CURRENT policies' trade logs (prune_rl_history),
    so most trades would never reach the classifier. Keyed per bar+direction: nightly test
    slices overlap, and the same bar/direction resolves the same way under the same geometry."""
    from pymongo import UpdateOne
    ops = []
    for s in trade_signals:
        d = s.model_dump() if hasattr(s, "model_dump") else s
        if not d.get("state") or d.get("status") not in RESOLVED_STATUSES:
            continue
        key = {"pair": d["pair"], "interval": d["interval"], "timestamp": d["timestamp"], "direction": d["direction"]}
        ops.append(UpdateOne(key, {"$set": {"status": d["status"], "state": d["state"]}}, upsert=True))
    return ops


_rl_rows_cache: dict = {"at": None, "rows": []}


async def load_rl_trade_rows(fresh: bool = False) -> list[dict]:
    if not fresh and _rl_rows_cache["at"] is not None and time.monotonic() - _rl_rows_cache["at"] < RL_ROWS_TTL_SECONDS:
        return list(_rl_rows_cache["rows"])
    from app.core.database import db, rl_signals_collection
    docs = await rl_signals_collection.find(RL_TRADE_QUERY, RL_TRADE_PROJECTION).to_list(length=None)
    docs += await db[ML_RL_TRADES_COLLECTION].find(RL_TRADE_QUERY, RL_TRADE_PROJECTION).to_list(length=None)
    rows = select_rl_rows(docs)
    _rl_rows_cache["at"], _rl_rows_cache["rows"] = time.monotonic(), rows
    return list(rows)


async def get_ml_reference_signals(signals_collection, fresh: bool = False) -> list[dict]:
    """
    Live resolved signals (Mongo) UNION archived backtest signals (Parquet) UNION resolved RL
    trades (see load_rl_trade_rows) -- the shared data
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
    return live_signals + load_backtest_signals_archive() + await load_rl_trade_rows(fresh)
