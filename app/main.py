from fastapi import BackgroundTasks, Body, FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from datetime import datetime, timedelta
import asyncio
import time
import uuid
import httpx
import pandas as pd

from app.core.config import get_settings
from app.core.auth import AuthMiddleware, create_token, verify_credentials
from app.core.database import (
    db,
    init_indexes,
    candles_collection,
    signals_collection,
    backtest_signals_collection,
    backtest_runs_collection,
    paper_trades_collection,
    consensus_signals_collection,
    ml_runs_collection,
    rl_signals_collection,
    rl_train_jobs_collection,
    run_all_flows_jobs_collection,
    ppo_policies_collection,
)
from app.models.schemas import (
    LoginRequest, RuleConfig, RLSignal, RLTrainAllJob, RLTrainAllCell, RunAllFlowsJob, RLInsightFinding,
    PPOPolicy, MLTrainResult,
)
from app.services.data_fetcher import fetch_and_store, backfill_batch
from app.services.candle_archive import load_full_candle_history
from app.services.indicators import atr as compute_atr_series, add_all_indicators
from app.services.signal_engine import generate_signal, compute_atr_target_stop, default_config_for, apply_rules
from app.services.backtester import run_backtest, run_consensus_backtest
from app.services.strategies import STRATEGIES
from app.services.consensus import check_consensus
from app.services.ml_features import extract_features, signal_like_features
from app.services.ml_model import train_hit_classifier, predict_hit_probability
from app.services.rl_engine import (
    compute_strategy_vote_states, rl_config_profile, full_rl_state,
    position_size_units, rl_atr_mults, MIN_WARMUP_BARS, DEFAULT_STARTING_BALANCE,
    RISK_FRACTION_BY_TIER, RL_FEATURE_NAMES,
)
from app.services.case_memory import (
    memory_summary, memory_gate, nearest_cases, explain_divergence, find_diverging_neighbor, RESOLVED_STATUSES,
)
from app.services.ppo_engine import train_ppo_policy, choose_action_ppo
from app.services.deriv_client import deriv_session, DerivAuthError
from app.services.paper_trading import execute_paper_trade, sync_open_trade
from app.services.outcome_scoring import (
    score_pending_signals, score_pending_consensus_signals, score_pending_rl_signals,
    resolve_rl_signal_real_outcome, LIVE_MAX_LOOKFORWARD_BY_INTERVAL,
)

settings = get_settings()
app = FastAPI(title="Forex Trading Assistant")

# Default grid for /backtest/optimize when no configs are supplied — covers the knobs
# backtesting has actually shown to matter (EMA responsiveness, RSI sensitivity, and
# target/stop — RuleConfig's own 1.5x/1.0x default turned out to have negative expectancy
# on every EMA/RSI combination on every pair; a closer target with a more generous stop
# (0.5x/1.25x) is a validated, cross-pair-tested improvement, see signal_engine.PROFILE_DEFAULTS).
# Both target/stop ratios are included here so a plain "run optimize" has a chance to find
# the better one instead of only searching a doomed ratio.
DEFAULT_OPTIMIZE_GRID = [
    RuleConfig(),
    RuleConfig(ema_fast=20, ema_slow=100),
    RuleConfig(ema_fast=10, ema_slow=50),
    RuleConfig(rsi_oversold=25, rsi_overbought=75),
    RuleConfig(rsi_oversold=35, rsi_overbought=65),
    RuleConfig(ema_fast=20, ema_slow=100, rsi_oversold=25, rsi_overbought=75),
    RuleConfig(ema_fast=10, ema_slow=50, rsi_oversold=25, rsi_overbought=75),
    RuleConfig(target_atr_mult=0.5, stop_atr_mult=1.25),
    RuleConfig(ema_fast=20, ema_slow=100, target_atr_mult=0.5, stop_atr_mult=1.25),
    RuleConfig(rsi_oversold=25, rsi_overbought=75, target_atr_mult=0.5, stop_atr_mult=1.25),
]

# AuthMiddleware added first so CORSMiddleware ends up outermost (Starlette makes the
# *last*-added middleware the outermost one) — otherwise a 401 from AuthMiddleware would
# skip CORS entirely and the browser would report a CORS error instead of surfacing 401.
app.add_middleware(AuthMiddleware)

# Local Next.js dev server needs to call this API directly from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://forex-assistant-five.vercel.app"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    await init_indexes()


@app.get("/")
async def root():
    return {"status": "ok", "pairs": settings.pairs_list}


@app.post("/auth/login")
async def login(body: LoginRequest):
    if not verify_credentials(body.username, body.password):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    return {"access_token": create_token(body.username), "token_type": "bearer"}


@app.post("/ingest/{interval}")
async def ingest(interval: str, output_size: int = 300):
    """
    Pull latest candles for all configured pairs at the given interval.
    interval: '5min', '15min', '1h', '4h', '1day'
    output_size: candles to fetch per pair (Twelve Data allows up to 5000 on the free tier).
    Backtest optimization (train/test split) needs more history than a single live signal
    does — raise this when you want a meaningful split, e.g. ?output_size=2000.

    A short pause between pairs (not just between separate /ingest calls) -- confirmed live
    that firing all 4 pairs back-to-back, repeated across a few intervals in quick
    succession (e.g. the "Ingest all" dashboard button, or Sync now), can burst past Twelve
    Data's per-minute rate limit well before the daily credit cap is anywhere close.
    fetch_candles itself also retries with backoff on a 429 (see data_fetcher.py), so this
    is belt-and-suspenders: spacing to avoid triggering it, retry in case it happens anyway.
    """
    results = {}
    for i, pair in enumerate(settings.pairs_list):
        if i > 0:
            await asyncio.sleep(2)
        try:
            count = await fetch_and_store(pair, interval, output_size=output_size)
            results[pair] = f"{count} candles stored"
        except Exception as e:
            results[pair] = f"error: {e}"
    return results


@app.post("/ingest/backfill/{interval}")
async def ingest_backfill(interval: str, start_date: str = "2010-01-01", max_calls_per_pair: int = 5):
    """
    Deep historical backfill for ML/RL training data (candles_collection), separate from
    /ingest's live-tail pulls. One call only advances each pair by up to max_calls_per_pair
    pages toward start_date -- a full 2010-> now backfill at finer intervals is hundreds of
    calls per pair, too long for one synchronous HTTP request. Resumable with no extra state:
    progress is just "the earliest candle already stored" (see backfill_batch), so call this
    repeatedly (.github/workflows/backfill-history.yml does) until every pair's "done" is true.
    """
    results = {}
    for i, pair in enumerate(settings.pairs_list):
        if i > 0:
            await asyncio.sleep(2)
        try:
            results[pair] = await backfill_batch(pair, interval, start_date, max_calls=max_calls_per_pair)
        except Exception as e:
            results[pair] = {"error": str(e)}
    results["all_done"] = all(isinstance(r, dict) and r.get("done") for r in results.values())
    results["all_reached_start_date"] = all(
        isinstance(r, dict) and r.get("reached_start_date") for r in results.values()
    )
    return results


# "Good ones only" bar for GENERATING a live signal (create_signal, create_rl_signal) --
# NOT for training, which keeps using everything regardless of quality (that's how the
# classifier gets calibrated in the first place). Backed by GET /ml/runs' own calibration
# table: the 60-70% and 70-100% predicted buckets both actually hit ~70% of the time in
# reality, while everything below 50% hits LESS than half the time -- 60% is the first bucket
# boundary where "the model says this is good" and "this is actually good" agree. Starting
# guess at exactly that boundary, not independently swept -- same caveat as every other
# unvalidated threshold in this project; revisit once enough gated-vs-ungated live outcomes
# exist to check where the real cutoff should be.
GOOD_SIGNAL_ML_THRESHOLD = 0.6


@app.post("/signals/{interval}/{profile}")
async def create_signal(
    interval: str, profile: str, pair: str,
    target_atr_mult: Optional[float] = None, stop_atr_mult: Optional[float] = None,
):
    """
    Generate a signal for a pair using stored candle data.
    profile: always 'intraday' — the one validated ruleset, used across every interval.
    pair is a query param (e.g. ?pair=EUR/USD) —
    it contains a literal '/', which breaks Starlette path-parameter matching
    even when percent-encoded, so it can't live in the URL path.

    Directional signals get an ATR-based target_price/stop_price attached at creation
    time (same formula the backtester uses) — this is what /signals/score checks stored
    candles against later to resolve the signal's status from "pending" to hit/miss/expired.
    target_atr_mult/stop_atr_mult: explicit override; omit to use that profile's config
    values (RuleConfig.target_atr_mult/stop_atr_mult — see signal_engine.PROFILE_DEFAULTS).

    ML quality gate: a BUY/SELL the rule engine would otherwise fire only actually goes out
    as one if the ML classifier's calibrated hit-probability for it clears
    GOOD_SIGNAL_ML_THRESHOLD -- otherwise it's downgraded to HOLD (see Signal.ml_override).
    Fits fresh on every currently-resolved signal, same as GET /ml/predict -- genuinely live,
    no lookahead concern to guard against the way rl_engine.py's training-time frozen
    snapshot does. Fails OPEN (signal passes through ungated) when there isn't enough
    resolved history yet for predict_hit_probability to return a real number -- "not enough
    data" isn't evidence of a bad signal, so there's nothing to block on.
    """
    config = default_config_for(profile, pair)
    cursor = candles_collection.find(
        {"pair": pair, "interval": interval}
    ).sort("timestamp", -1).limit(500)
    docs = await cursor.to_list(length=500)
    docs.reverse()  # find() gave newest-first for the limit to bite correctly; generate_signal wants ascending

    min_needed = config.ema_slow
    if len(docs) < min_needed:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough candle history ({len(docs)} rows). Need {min_needed}+ for the slow EMA "
                    f"({profile} profile). Run /ingest/{interval} first."
        )

    df = pd.DataFrame(docs)
    signal = generate_signal(df, pair, interval, profile, config)

    if signal.direction in ("BUY", "SELL"):
        resolved_signals = await signals_collection.find(
            {"source": "live", "status": {"$in": ["hit", "miss", "expired"]}}
        ).to_list(length=None)
        features = extract_features(signal.model_dump())
        signal.ml_hit_probability = predict_hit_probability(resolved_signals, features)
        if signal.ml_hit_probability is not None and signal.ml_hit_probability < GOOD_SIGNAL_ML_THRESHOLD:
            signal.ml_override = (
                f"ML rated this {signal.direction} at only {signal.ml_hit_probability * 100:.0f}% hit "
                f"probability (below the {GOOD_SIGNAL_ML_THRESHOLD * 100:.0f}% bar for a live signal) "
                f"-- held instead."
            )
            signal.direction = "HOLD"

    if signal.direction in ("BUY", "SELL"):
        atr_val = float(compute_atr_series(df, config.atr_period).iloc[-1])
        signal.target_price, signal.stop_price = compute_atr_target_stop(
            signal.price_at_signal, atr_val, signal.direction,
            target_atr_mult if target_atr_mult is not None else config.target_atr_mult,
            stop_atr_mult if stop_atr_mult is not None else config.stop_atr_mult,
        )

    # The latest stored candle only advances when /ingest brings in a new one — calling
    # this endpoint again before that (e.g. every dashboard load) would otherwise insert
    # an identical duplicate for the same candle, inflating /signals/accuracy's counts.
    # target_price/stop_price are compared too (not just direction/price) so a repeat call
    # with different target_atr_mult/stop_atr_mult overrides is treated as a distinct
    # signal rather than silently returning the first call's target/stop.
    last = await signals_collection.find_one(
        {"pair": pair, "interval": interval, "profile": profile, "source": "live"},
        sort=[("timestamp", -1)],
    )
    if (
        last is not None
        and last["direction"] == signal.direction
        and last["price_at_signal"] == signal.price_at_signal
        and last["target_price"] == signal.target_price
        and last["stop_price"] == signal.stop_price
    ):
        last["_id"] = str(last["_id"])
        return last

    await signals_collection.insert_one(signal.model_dump())
    return signal


@app.get("/signals")
async def list_signals(pair: str | None = None, limit: int = 50):
    query = {"pair": pair} if pair else {}
    cursor = signals_collection.find(query).sort("timestamp", -1).limit(limit)
    docs = await cursor.to_list(length=limit)
    for d in docs:
        d["_id"] = str(d["_id"])
    return docs


@app.post("/signals/score")
async def score_signals(max_lookforward: int | None = None):
    """
    Checks every pending live signal against candles that have arrived since it fired,
    resolving status to hit/miss/expired wherever enough real data now exists — the live
    equivalent of what the backtester does against fixed history. Run this after each
    /ingest so newly-arrived candles get checked; the .github/workflows/keep-fresh.yml
    cron does both automatically every 15 minutes, this is for triggering it on demand.

    max_lookforward left unset (the default) uses each signal's own interval-appropriate
    window (outcome_scoring.LIVE_MAX_LOOKFORWARD_BY_INTERVAL) instead of one flat value for
    every interval -- pass an explicit value here only to force the same window everywhere
    (e.g. for a quick manual comparison against the old behavior).
    """
    return await score_pending_signals(max_lookforward=max_lookforward)


@app.get("/signals/accuracy")
async def signal_accuracy(pair: str | None = None, profile: str | None = None, limit: int = 100):
    """
    Rolling hit-rate over the most recent *resolved* live signals (hit/miss/expired) —
    excludes still-pending signals and anything from a backtest run. This is what tells
    you whether live performance is actually tracking what was backtested; it often won't
    match at first, and that gap is itself useful signal, not a bug to explain away.

    sample_size/hits/misses/expired are capped at `limit` (a rolling window, so hit-rate
    stays responsive to recent performance instead of getting diluted as history grows
    forever) and will plateau once enough signals have resolved — the total_* fields are
    the real, uncapped counts (count_documents per status on the same filter) so "is
    sample_size stuck" has an actual answer instead of looking like a stalled number.
    """
    query: dict = {"source": "live", "status": {"$in": ["hit", "miss", "expired"]}}
    if pair:
        query["pair"] = pair
    if profile:
        query["profile"] = profile
    cursor = signals_collection.find(query).sort("timestamp", -1).limit(limit)
    docs = await cursor.to_list(length=limit)

    hits = sum(1 for d in docs if d["status"] == "hit")
    misses = sum(1 for d in docs if d["status"] == "miss")
    expired = sum(1 for d in docs if d["status"] == "expired")
    total = len(docs)

    total_hits = await signals_collection.count_documents({**query, "status": "hit"})
    total_misses = await signals_collection.count_documents({**query, "status": "miss"})
    total_expired = await signals_collection.count_documents({**query, "status": "expired"})
    total_resolved = total_hits + total_misses + total_expired
    total_decided = total_hits + total_misses

    return {
        "pair": pair,
        "profile": profile,
        "sample_size": total,
        "hits": hits,
        "misses": misses,
        "expired": expired,
        "hit_rate_pct": round(hits / total * 100, 1) if total else None,
        "total_resolved": total_resolved,
        "total_hits": total_hits,
        "total_misses": total_misses,
        "total_expired": total_expired,
        # see RLSignal accuracy's directional_hit_rate_pct docstring -- same reasoning: this
        # divides by decided (hit+miss) trades only, so an interval with a lot of "expired"
        # timeouts doesn't drag the headline number toward 0 for reasons unrelated to whether
        # the model is directionally right when it actually resolves.
        "total_hit_rate_pct": round(total_hits / total_resolved * 100, 1) if total_resolved else None,
        "directional_hit_rate_pct": round(total_hits / total_decided * 100, 1) if total_decided else None,
    }


@app.get("/candles/{interval}")
async def get_candles(interval: str, pair: str, profile: str = "intraday", limit: int = 300):
    """
    Stored candles with EMA/RSI/MACD/ATR attached, for charting — computed with the
    intraday RuleConfig (so the EMA/RSI periods shown match whatever generate_signal
    actually used). Returns the most recent `limit` candles, ascending
    by timestamp. Indicators need config.ema_slow rows of preceding history to be
    meaningful; the earliest rows in a short result may show as null for that reason.
    """
    config = default_config_for(profile, pair)
    cursor = candles_collection.find({"pair": pair, "interval": interval}).sort("timestamp", -1).limit(limit)
    docs = await cursor.to_list(length=limit)
    if not docs:
        raise HTTPException(
            status_code=404,
            detail=f"No candle history for {pair}/{interval}. Run /ingest/{interval} first.",
        )
    docs.reverse()  # find() gave newest-first for the limit to bite correctly; chart wants ascending

    df = pd.DataFrame(docs)
    indicator_df = add_all_indicators(df, config)
    indicator_df = indicator_df.replace({float("nan"): None})

    return [
        {
            "timestamp": row["timestamp"],
            "open": row["open"], "high": row["high"], "low": row["low"], "close": row["close"],
            "ema_fast": row["ema_fast"], "ema_slow": row["ema_slow"],
            "rsi": row["rsi"], "macd": row["macd"], "macd_signal": row["macd_signal"], "macd_hist": row["macd_hist"],
            "atr": row["atr"],
        }
        for _, row in indicator_df.iterrows()
    ]


@app.post("/consensus/score")
async def score_consensus_signals(max_lookforward: int | None = None):
    """
    Closes the loop for live consensus signals the same way /signals/score does for regular
    ones -- resolves pending consensus signals to hit/miss/expired against candles that have
    arrived since they fired. Separate endpoint (not folded into /signals/score) because
    consensus signals live in their own collection with a different shape.

    Must stay declared before /consensus/{interval} below -- Starlette matches routes in
    declaration order, and {interval} is a single dynamic path segment that would otherwise
    swallow a request to the literal path "/consensus/score" (interval="score") before it
    ever reached this one. (No such collision for /consensus/backtest/{interval}, which has
    two segments after /consensus/ and so never matches the single-segment {interval} pattern.)
    """
    return await score_pending_consensus_signals(max_lookforward=max_lookforward)


@app.post("/consensus/{interval}")
async def create_consensus_signal(interval: str, pair: str):
    """
    Runs every independent strategy (app/services/strategies.py) against stored candles and
    checks for consensus (app/services/consensus.py) — a weighted majority (see
    consensus.STRATEGY_WEIGHTS/REQUIRED_WEIGHT_FRACTION) agreeing on direction, and within
    PROXIMITY_ATR_MULT of each other's entry/exit. Always returns every strategy_call
    alongside the verdict, so "no consensus right now" is visibly the calls disagreeing, not
    an opaque empty response.

    pair: query param (e.g. ?pair=EUR/USD) — a path param would break on the literal '/'.
    """
    config = default_config_for("intraday", pair)
    cursor = candles_collection.find(
        {"pair": pair, "interval": interval}
    ).sort("timestamp", -1).limit(500)
    docs = await cursor.to_list(length=500)
    docs.reverse()

    min_needed = 30  # covers every strategy's own warmup (see run_consensus_backtest)
    if len(docs) < min_needed:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough candle history ({len(docs)} rows). Need {min_needed}+ for consensus. "
                    f"Run /ingest/{interval} first."
        )

    df = pd.DataFrame(docs)
    indicator_df = add_all_indicators(df, config)
    calls = [fn(indicator_df, config) for fn in STRATEGIES]
    latest = indicator_df.iloc[-1]
    consensus = check_consensus(calls, pair, interval, latest["timestamp"], float(latest["atr"]))

    if consensus is None:
        return {"consensus": None, "strategy_calls": [c.model_dump() for c in calls]}

    # De-dupe against the last stored consensus signal for this pair/interval, same idea as
    # create_signal's dedup — avoid inserting an identical duplicate when the underlying
    # candle hasn't advanced since the last check.
    last = await consensus_signals_collection.find_one(
        {"pair": pair, "interval": interval, "source": "live"},
        sort=[("timestamp", -1)],
    )
    if (
        last is not None
        and last["direction"] == consensus.direction
        and last["entry_price"] == consensus.entry_price
        and last["target_price"] == consensus.target_price
    ):
        last["_id"] = str(last["_id"])
        return {"consensus": last, "strategy_calls": [c.model_dump() for c in calls]}

    await consensus_signals_collection.insert_one(consensus.model_dump())
    return {"consensus": consensus, "strategy_calls": [c.model_dump() for c in calls]}


@app.get("/consensus")
async def list_consensus_signals(pair: str | None = None, limit: int = 50):
    query = {"pair": pair} if pair else {}
    cursor = consensus_signals_collection.find(query).sort("timestamp", -1).limit(limit)
    docs = await cursor.to_list(length=limit)
    for d in docs:
        d["_id"] = str(d["_id"])
    return docs


@app.post("/consensus/backtest/{interval}")
async def backtest_consensus(interval: str, pair: str, train_frac: float = 0.7, max_lookforward: int = 20):
    """
    Train/test validated backtest of the consensus mechanism itself — same discipline as
    /backtest/optimize: run on the first train_frac of history, then re-score on the untouched
    remaining tail. There's no grid to search here, but the split still catches a mechanism
    that only looks good on one slice of history by chance rather than a real, repeatable edge.
    """
    if not 0 < train_frac < 1:
        raise HTTPException(status_code=400, detail="train_frac must be between 0 and 1 (exclusive).")

    config = default_config_for("intraday", pair)
    docs = await load_full_candle_history(pair, interval)
    if not docs:
        raise HTTPException(
            status_code=400,
            detail=f"No candle history for {pair}/{interval}. Run /ingest/{interval} first.",
        )

    df = pd.DataFrame(docs)
    split_idx = int(len(df) * train_frac)

    if split_idx < 31:
        raise HTTPException(
            status_code=400,
            detail=f"Train slice ({split_idx} candles) is too short for consensus's warmup needs (30+). "
                    f"Ingest more history or lower train_frac.",
        )
    if len(df) - split_idx - max_lookforward < 1:
        raise HTTPException(
            status_code=400,
            detail=f"Test slice is too short ({len(df) - split_idx} candles) for max_lookforward={max_lookforward}. "
                    f"Ingest more history or raise train_frac.",
        )

    train_df = df.iloc[:split_idx]
    try:
        train_run, _train_signals = run_consensus_backtest(
            train_df, pair, interval, config, max_lookforward=max_lookforward,
        )
        test_run, _test_signals = run_consensus_backtest(
            df, pair, interval, config, max_lookforward=max_lookforward, eval_start_index=split_idx,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Individual consensus signals aren't persisted to backtest_signals_collection -- that
    # collection stores Signal-shaped docs (reasons/profile/confidence) for the existing
    # /backtest/runs/{run_id}/signals endpoint, and ConsensusSignal's shape (strategy_calls,
    # no profile/confidence) doesn't match. The two BacktestRun summaries below (tagged
    # profile="consensus") are what the frontend train/test view actually needs.
    await backtest_runs_collection.insert_one(train_run.model_dump())
    await backtest_runs_collection.insert_one(test_run.model_dump())

    return {"train": train_run, "test": test_run}


@app.post("/backtest/{interval}/{profile}")
async def backtest(
    interval: str,
    profile: str,
    pair: str,
    target_atr_mult: Optional[float] = None,
    stop_atr_mult: Optional[float] = None,
    max_lookforward: int = 20,
    config: Optional[RuleConfig] = Body(default=None),
):
    """
    Replay the rule engine bar-by-bar over stored candle history and score hit-rate.

    Don't trust a live signal until its ruleset has a backtested track record here first.
    pair: query param (e.g. ?pair=EUR/USD) — a path param would break on the literal '/'.
    target_atr_mult / stop_atr_mult: explicit override for take-profit/stop-loss distance
    (multiple of ATR(14) at signal time) — omit to use config's own target_atr_mult/stop_atr_mult.
    max_lookforward: max candles to wait for target or stop before calling a signal "expired".
    config: optional JSON body overriding rule thresholds (EMA/RSI/MACD/ATR periods and
    cutoffs) — omit to use that profile's default RuleConfig (see signal_engine.PROFILE_DEFAULTS).
    For comparing several configs against the same data in one call, use /backtest/sweep instead.
    """
    config = config or default_config_for(profile, pair)
    docs = await load_full_candle_history(pair, interval)

    min_needed = config.ema_slow + max_lookforward + 1
    if len(docs) < min_needed:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough candle history ({len(docs)} rows). Need {min_needed}+ for a backtest "
                    f"with ema_slow={config.ema_slow} and max_lookforward={max_lookforward}. "
                    f"Run /ingest/{interval} first.",
        )

    df = pd.DataFrame(docs)
    try:
        run, signals = run_backtest(
            df, pair, interval, profile, config,
            target_atr_mult=target_atr_mult,
            stop_atr_mult=stop_atr_mult,
            max_lookforward=max_lookforward,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if signals:
        await backtest_signals_collection.insert_many([s.model_dump() for s in signals])
    await backtest_runs_collection.insert_one(run.model_dump())

    return run


@app.post("/backtest/sweep/{interval}/{profile}")
async def backtest_sweep(
    interval: str,
    profile: str,
    pair: str,
    target_atr_mult: Optional[float] = None,
    stop_atr_mult: Optional[float] = None,
    max_lookforward: int = 20,
    configs: list[RuleConfig] = Body(...),
):
    """
    Run multiple RuleConfig variations against the exact same historical candles in one
    call, so you can see what actually improves hit-rate instead of tweaking and hoping.
    Candles are fetched once and reused across every config for a fair comparison.
    pair: query param (e.g. ?pair=EUR/USD) — a path param would break on the literal '/'.
    target_atr_mult/stop_atr_mult: explicit override applied to every config in the sweep —
    omit to let each config use its own target_atr_mult/stop_atr_mult instead.

    Body: a JSON array of RuleConfig objects (partial overrides are fine — unset fields
    fall back to RuleConfig defaults), e.g.
    [{"rsi_oversold": 25, "rsi_overbought": 75}, {"rsi_oversold": 35, "rsi_overbought": 65}]

    Returns each run's summary sorted by hit_rate_pct descending (runs with no
    directional signals sort last).
    """
    if not configs:
        raise HTTPException(status_code=400, detail="Provide at least one config to sweep.")

    docs = await load_full_candle_history(pair, interval)
    if not docs:
        raise HTTPException(
            status_code=400,
            detail=f"No candle history for {pair}/{interval}. Run /ingest/{interval} first.",
        )
    df = pd.DataFrame(docs)

    runs = []
    for config in configs:
        min_needed = config.ema_slow + max_lookforward + 1
        if len(docs) < min_needed:
            raise HTTPException(
                status_code=400,
                detail=f"Not enough candle history ({len(docs)} rows) for config with "
                        f"ema_slow={config.ema_slow} (needs {min_needed}+). Run /ingest/{interval} first.",
            )
        try:
            run, signals = run_backtest(
                df, pair, interval, profile, config,
                target_atr_mult=target_atr_mult,
                stop_atr_mult=stop_atr_mult,
                max_lookforward=max_lookforward,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        if signals:
            await backtest_signals_collection.insert_many([s.model_dump() for s in signals])
        await backtest_runs_collection.insert_one(run.model_dump())
        runs.append(run)

    runs.sort(key=lambda r: (r.hit_rate_pct is None, -(r.hit_rate_pct or 0)))
    return runs


@app.post("/backtest/optimize/{interval}/{profile}")
async def backtest_optimize(
    interval: str,
    profile: str,
    pair: str,
    train_frac: float = 0.7,
    target_atr_mult: Optional[float] = None,
    stop_atr_mult: Optional[float] = None,
    max_lookforward: int = 20,
    min_directional_signals: int = 20,
    rank_by: str = "expectancy",
    configs: Optional[list[RuleConfig]] = Body(default=None),
):
    """
    Grid-search RuleConfig on the first train_frac of history, pick a winner there (among
    configs with at least min_directional_signals, so a "winner" isn't just 3 lucky
    signals), then validate that exact winner on the untouched remaining tail. The train
    and test numbers are reported side by side — a real edge should survive on data the
    config was never tuned against; a large drop from train to test means the "improvement"
    was curve-fit noise, not signal.

    rank_by: "expectancy" (default) or "hit_rate". Hit-rate alone can be misleading — a
    low-hit-rate config with much bigger wins than losses can have better expected value
    per signal than a high-hit-rate config with tiny wins and occasional large losses.
    expectancy_pct (mean pct_move across every directional signal, win or lose) answers
    "is this actually profitable on average"; hit_rate_pct only answers "how often is it
    right." Both are always returned regardless of which one you rank by.

    target_atr_mult/stop_atr_mult: explicit override applied uniformly to every candidate —
    omit (recommended) to let each candidate config use its own target_atr_mult/stop_atr_mult,
    which is what actually lets this search target/stop as part of the grid (they're
    RuleConfig fields now, so a configs body can vary them same as EMA/RSI/session/vol).

    configs: optional JSON array of RuleConfig overrides to search — omit to use the
    built-in default grid (varies EMA responsiveness and RSI sensitivity, the two
    parameters backtesting has actually shown to move hit-rate).
    """
    if rank_by not in ("expectancy", "hit_rate"):
        raise HTTPException(status_code=400, detail="rank_by must be 'expectancy' or 'hit_rate'.")
    grid = configs or DEFAULT_OPTIMIZE_GRID
    if not grid:
        raise HTTPException(status_code=400, detail="Provide at least one config, or omit configs to use the default grid.")
    if not 0 < train_frac < 1:
        raise HTTPException(status_code=400, detail="train_frac must be between 0 and 1 (exclusive).")

    docs = await load_full_candle_history(pair, interval)
    if not docs:
        raise HTTPException(
            status_code=400,
            detail=f"No candle history for {pair}/{interval}. Run /ingest/{interval} first.",
        )

    df = pd.DataFrame(docs)
    split_idx = int(len(df) * train_frac)

    max_ema_slow = max(c.ema_slow for c in grid)
    if split_idx < max_ema_slow + 1:
        raise HTTPException(
            status_code=400,
            detail=f"Train slice ({split_idx} candles) is too short for the largest config's warmup "
                    f"(ema_slow={max_ema_slow}). Ingest more history or lower train_frac.",
        )
    if len(df) - split_idx - max_lookforward < 1:
        raise HTTPException(
            status_code=400,
            detail=f"Test slice is too short ({len(df) - split_idx} candles) for max_lookforward={max_lookforward}. "
                    f"Ingest more history or raise train_frac.",
        )

    train_df = df.iloc[:split_idx]
    train_candidates = []
    for config in grid:
        try:
            run, signals = run_backtest(
                train_df, pair, interval, profile, config,
                target_atr_mult=target_atr_mult, stop_atr_mult=stop_atr_mult, max_lookforward=max_lookforward,
            )
        except ValueError:
            continue  # this config's warmup doesn't fit in the train slice — skip, don't fail the whole search
        train_candidates.append((config, run, signals))

    rank_field = "expectancy_pct" if rank_by == "expectancy" else "hit_rate_pct"
    eligible = [
        (c, r, s) for c, r, s in train_candidates
        if getattr(r, rank_field) is not None and r.directional_signals >= min_directional_signals
    ]
    if not eligible:
        raise HTTPException(
            status_code=400,
            detail=f"No config produced at least {min_directional_signals} directional signals on the "
                    f"train slice ({split_idx} candles) — try a longer train slice or lower min_directional_signals.",
        )
    eligible.sort(key=lambda x: -getattr(x[1], rank_field))
    best_config, best_train_run, best_train_signals = eligible[0]

    try:
        test_run, test_signals = run_backtest(
            df, pair, interval, profile, best_config,
            target_atr_mult=target_atr_mult, stop_atr_mult=stop_atr_mult, max_lookforward=max_lookforward,
            eval_start_index=split_idx,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if best_train_signals:
        await backtest_signals_collection.insert_many([s.model_dump() for s in best_train_signals])
    await backtest_runs_collection.insert_one(best_train_run.model_dump())
    if test_signals:
        await backtest_signals_collection.insert_many([s.model_dump() for s in test_signals])
    await backtest_runs_collection.insert_one(test_run.model_dump())

    return {
        "winning_config": best_config,
        "rank_by": rank_by,
        "train": best_train_run,
        "test": test_run,
        "candidates_evaluated": [
            {
                "config": c,
                "hit_rate_pct": r.hit_rate_pct,
                "expectancy_pct": r.expectancy_pct,
                "directional_signals": r.directional_signals,
            }
            for c, r, _ in train_candidates
        ],
    }


@app.post("/backtest/prune-history")
async def prune_backtest_history(confirm: bool = False, keep_latest_n: int = 50):
    """
    Reclaims Atlas storage from rule-based backtest/sweep/optimize runs (profile != "rl_ppo"
    -- see /rl/prune-history for that side) with the same zero-feature-loss reasoning: the
    frontend's only path to a run's per-signal detail is GET /backtest/runs?limit=50 (see
    frontend/src/app/backtest/page.tsx, no pair/profile filter -- a single global most-recent
    list) followed by clicking into one of those 50 (RunDetail.tsx calls
    GET /backtest/runs/{run_id}/signals). A run older than the 50 most recent overall is
    already unreachable from the app, so its backtest_signals_collection detail is dead
    weight -- this keeps backtest_runs_collection itself untouched (cheap, <1MB for 800+
    docs) and only deletes the per-trade signals for runs past keep_latest_n.

    confirm: defaults to False, a dry run that returns exactly what WOULD be deleted.
    """
    keep_run_ids = {
        d["run_id"] async for d in
        backtest_runs_collection.find({"profile": {"$ne": "rl_ppo"}}, {"run_id": 1})
            .sort("created_at", -1).limit(keep_latest_n)
    }
    prunable_run_ids = [
        d["run_id"] async for d in
        backtest_runs_collection.find(
            {"profile": {"$ne": "rl_ppo"}, "run_id": {"$nin": list(keep_run_ids)}}, {"run_id": 1},
        )
    ]

    signals_count = await backtest_signals_collection.count_documents(
        {"run_id": {"$in": prunable_run_ids}}
    ) if prunable_run_ids else 0

    result = {
        "confirmed": confirm,
        "runs_kept": len(keep_run_ids),
        "runs_pruned": len(prunable_run_ids),
        "backtest_signals_to_delete": signals_count,
    }
    if not confirm:
        return result

    if prunable_run_ids:
        await backtest_signals_collection.delete_many({"run_id": {"$in": prunable_run_ids}})
    return result


@app.get("/backtest/runs")
async def list_backtest_runs(pair: str | None = None, profile: str | None = None, limit: int = 20):
    query = {}
    if pair:
        query["pair"] = pair
    if profile:
        query["profile"] = profile
    cursor = backtest_runs_collection.find(query).sort("created_at", -1).limit(limit)
    docs = await cursor.to_list(length=limit)
    for d in docs:
        d["_id"] = str(d["_id"])
    return docs


@app.get("/backtest/runs/{run_id}")
async def get_backtest_run(run_id: str):
    doc = await backtest_runs_collection.find_one({"run_id": run_id})
    if not doc:
        raise HTTPException(status_code=404, detail=f"No backtest run found with run_id={run_id}")
    doc["_id"] = str(doc["_id"])
    return doc


@app.get("/backtest/runs/{run_id}/signals")
async def get_backtest_run_signals(run_id: str, status: str | None = None, limit: int = 500):
    query = {"run_id": run_id}
    if status:
        query["status"] = status
    cursor = backtest_signals_collection.find(query).sort("timestamp", 1).limit(limit)
    docs = await cursor.to_list(length=limit)
    for d in docs:
        d["_id"] = str(d["_id"])
    return docs


@app.post("/ml/train")
async def train_ml_model(train_frac: float = 0.7, force: bool = False):
    """
    Trains the supervised hit/miss classifier (app/services/ml_model.py, XGBoost --
    NOT reinforcement learning) on every resolved live signal across all pairs/profiles. One
    shared model, not per-pair -- splitting the current ~238 resolved signals further would
    leave too few examples per model to mean anything. Chronological train/test split, not
    random (see train_hit_classifier's own docstring) -- the same lookahead-bias discipline
    already applied to run_backtest's eval_start_index.

    keep-fresh.yml's cron calls this every 20 minutes unconditionally, but a signal takes
    100min-5hr to even resolve (max_lookforward candles) -- most firings have zero new
    resolved signals to learn from, and refitting the same rows just reproduces the same
    model byte-for-byte, making "is this improving" impossible to answer honestly. So: if the
    resolved count matches the most recent stored run's train+test sample count, skip the fit
    and return that prior result (with skipped=True) instead of pretending a no-op retrain
    is progress. Pass force=True to always refit regardless (e.g. after a feature-set change,
    when the row count is unchanged but what gets extracted from those rows isn't).
    """
    if not 0 < train_frac < 1:
        raise HTTPException(status_code=400, detail="train_frac must be between 0 and 1 (exclusive).")

    query = {"source": "live", "status": {"$in": ["hit", "miss", "expired"]}}
    resolved_count = await signals_collection.count_documents(query)

    if not force:
        last_run = await ml_runs_collection.find_one(sort=[("created_at", -1)])
        if last_run is not None and last_run["train_samples"] + last_run["test_samples"] == resolved_count:
            last_run.pop("_id", None)
            # A stored run from before feature_coefficients was renamed to feature_importances
            # (the LogisticRegression -> XGBoost switch) still has the old key under this exact
            # shortcut path -- everywhere else routes through MLTrainResult and would have
            # already 500'd loudly on such a document, but this branch used to return the raw
            # Mongo doc as-is, silently handing the frontend a payload with no
            # feature_importances key at all (Object.entries(undefined) then throws client-side).
            # Validating through the model instead both migrates that one renamed field and
            # supplies real defaults ({}, [], False) for any other field an old document predates.
            if "feature_importances" not in last_run and "feature_coefficients" in last_run:
                last_run["feature_importances"] = last_run.pop("feature_coefficients")
            result = MLTrainResult(**last_run)
            result.skipped = True
            return result

    signals = await signals_collection.find(query).to_list(length=None)

    try:
        result = train_hit_classifier(signals, train_frac=train_frac)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    await ml_runs_collection.insert_one(result.model_dump())
    return result


@app.get("/ml/runs")
async def list_ml_runs(limit: int = 20):
    cursor = ml_runs_collection.find().sort("created_at", -1).limit(limit)
    docs = await cursor.to_list(length=limit)
    for d in docs:
        d["_id"] = str(d["_id"])
    return docs


@app.post("/ml/predict/{interval}/{profile}")
async def predict_signal(interval: str, profile: str, pair: str):
    """
    Generates a fresh signal the same way create_signal does (reuses generate_signal), but
    does NOT insert it into signals_collection or touch that endpoint's dedup/history in any
    way -- purely advisory. Attaches ml_hit_probability from a model trained fresh on every
    currently resolved signal; null (not a fabricated number) when there isn't enough
    resolved data yet -- see ml_model.MIN_TRAIN_SIGNALS/MIN_TEST_SIGNALS.

    pair: query param (e.g. ?pair=EUR/USD) — a path param would break on the literal '/'.
    """
    config = default_config_for(profile, pair)
    cursor = candles_collection.find(
        {"pair": pair, "interval": interval}
    ).sort("timestamp", -1).limit(500)
    docs = await cursor.to_list(length=500)
    docs.reverse()

    min_needed = config.ema_slow
    if len(docs) < min_needed:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough candle history ({len(docs)} rows). Need {min_needed}+ for "
                    f"{profile}. Run /ingest/{interval} first."
        )

    df = pd.DataFrame(docs)
    signal = generate_signal(df, pair, interval, profile, config)

    if signal.direction in ("BUY", "SELL"):
        atr_val = float(compute_atr_series(df, config.atr_period).iloc[-1])
        signal.target_price, signal.stop_price = compute_atr_target_stop(
            signal.price_at_signal, atr_val, signal.direction,
            config.target_atr_mult, config.stop_atr_mult,
        )

    # "hit" only means anything for an actual trade -- HOLD signals never get target/stop,
    # never get scored (see score_pending_signals' direction filter), and so never appear in
    # the training set at all. Asking the model to score one isn't "not enough data," it's a
    # different question with no meaning: there's no trade to hit or miss. Skip it rather than
    # returning a number that looks like a real answer but isn't (the model's direction_buy
    # feature is 0 for both a real SELL and a HOLD -- without this guard it would silently
    # score a HOLD as if it were a SELL that never happened).
    ml_hit_probability = None
    if signal.direction in ("BUY", "SELL"):
        resolved_query = {"source": "live", "status": {"$in": ["hit", "miss", "expired"]}}
        resolved_signals = await signals_collection.find(resolved_query).to_list(length=None)
        features = extract_features(signal.model_dump())
        ml_hit_probability = predict_hit_probability(resolved_signals, features)

    response = signal.model_dump()
    response["ml_hit_probability"] = ml_hit_probability
    return response


def _rl_state_from_candles(
    docs: list[dict], config: RuleConfig, pair: str, interval: str, profile: str, resolved_signals: list[dict],
) -> tuple[list[float], pd.DataFrame, Optional[float], Optional[float]]:
    """
    Builds the indicator dataframe and the current (latest-bar) market state vector the same
    way compute_strategy_vote_states does internally, without recomputing every prior bar's
    state just to read the last one -- then appends the two live ml_hit_probability_buy/sell
    features (see rl_engine.RL_MARKET_FEATURE_NAMES) so the returned vector is ready to pass
    straight into full_rl_state, same shape train_rl_policy's market_states entries have.

    Unlike train_rl_policy's frozen-snapshot fit (see that function's ml_reference_signals
    docstring), this is genuinely live -- fits on every currently-resolved signal fresh, no
    lookahead concern, exactly what GET /ml/predict already does for the same reason.
    resolved_signals is fetched by the caller (async) rather than here, since this function is
    plain sync.

    Also returns the raw (buy_score, sell_score) -- None, not the state vector's 0.5
    placeholder, when there isn't enough resolved history yet -- so create_rl_signal's own ML
    quality gate can fail OPEN on "not enough data" the same way create_signal's does, rather
    than reading a placeholder 0.5 as if it were a real (bad) score.
    """
    df = pd.DataFrame(docs)
    indicator_df = add_all_indicators(df, config)
    states = compute_strategy_vote_states(indicator_df, config)

    latest = indicator_df.iloc[-1]
    prev = indicator_df.iloc[-2]
    reasons, _bullish_votes, _bearish_votes, total_rules, rule_votes, rule_strengths = apply_rules(
        latest, prev, config,
    )
    buy_features = signal_like_features(
        reasons, rule_votes, rule_strengths, total_rules, profile, "BUY", pair, interval,
    )
    sell_features = signal_like_features(
        reasons, rule_votes, rule_strengths, total_rules, profile, "SELL", pair, interval,
    )
    buy_score = predict_hit_probability(resolved_signals, buy_features)
    sell_score = predict_hit_probability(resolved_signals, sell_features)
    market_state = states[-1] + [
        buy_score if buy_score is not None else 0.5,
        sell_score if sell_score is not None else 0.5,
    ]
    return market_state, df, buy_score, sell_score


# Restored to all 5 intervals (was narrowed to 5min/15min only as part of the swing pause,
# see PROGRESS.md) -- drives every automated RL loop (Train all, Sync now) that iterates
# pairs x intervals, and GET /rl/insights' scan.
RL_INTERVALS = ["5min", "15min", "1h", "4h", "1day"]


async def _find_warm_start_policy(pair: str, interval: str) -> tuple[str | None, bytes | None]:
    """
    Finds the most recent usable PPOPolicy for this pair/interval to continue training from,
    or (None, None) if there isn't one. "Usable" means both:
      1. feature_names still matches RL_FEATURE_NAMES -- same staleness guard
         choose_action_ppo already applies for live inference; a policy trained against an
         older/different state schema can't have its weights meaningfully continued.
      2. its OWN test-slice evaluation (the linked BacktestRun) shows at least one
         directional_signals -- otherwise it's a policy that converged to always-HOLD, and
         warm-starting from it would continue training from that same stuck point rather
         than giving this combo a fresh random initialization's chance to find a real
         trading policy. This is the fix for training runs landing on an all-HOLD policy
         for some pair/interval combos (which one varies run to run -- see PROGRESS.md) --
         skip it here rather than perpetuating it.
    Returns (policy_id, model_bytes) on success so the caller can pass both through to
    train_ppo_policy for warm_started_from tracking.
    """
    prev = await ppo_policies_collection.find_one({"pair": pair, "interval": interval}, sort=[("created_at", -1)])
    if not prev or prev.get("feature_names") != RL_FEATURE_NAMES:
        return None, None
    eval_run = await backtest_runs_collection.find_one({"run_id": prev.get("eval_run_id")})
    if not eval_run or not eval_run.get("directional_signals"):
        return None, None
    return prev["policy_id"], prev["model_bytes"]


async def _run_rl_training(
    pair: str, interval: str, total_timesteps: int, train_frac: float, max_lookforward: int, starting_balance: float,
    random_seed: int | None = None, target_atr_mult: float | None = None, stop_atr_mult: float | None = None,
):
    """
    Shared by POST /rl/train and the /rl/train-all background job below -- fetches candle
    history, runs PPO training (app/services/ppo_engine.py), and persists the policy/
    evaluation/trade log. Raises ValueError for the caller to turn into whatever error shape
    fits its own endpoint (a 400 for the single endpoint, a per-cell error string for the
    batch job).

    train_ppo_policy itself is a blocking, CPU-bound PPO rollout -- run via run_in_threadpool
    so it doesn't tie up the event loop, which matters a lot more here than it did for a
    single call, since the batch job below calls this 20 times in a row and other requests
    (status polling, live signal generation, the cron) still need to get through while it runs.

    Warm-starts from the most usable prior policy for this pair/interval when one exists (see
    _find_warm_start_policy) -- continues training instead of starting from a fresh randomly-
    initialized model every call, so a policy that's already learned something builds on that
    rather than re-rolling the dice each time. Falls back to a fresh model automatically when
    there's no prior policy, its feature schema is stale, or it degenerated to always-HOLD.

    target_atr_mult/stop_atr_mult: explicit override passed straight through to
    train_ppo_policy/train_and_evaluate_ppo_poc -- omit (recommended) to use
    rl_engine.rl_atr_mults(interval, pair)'s configured default. Exists for sweeping a
    candidate (target, stop) pair against a high-expired-rate interval/pair (see
    GET /rl/resolution-stats) before committing it to RL_ATR_MULT_PAIR_OVERRIDES -- this
    still persists a real policy/eval run like any other call, it's not a dry run.
    """
    if not 0 < train_frac < 1:
        raise ValueError("train_frac must be between 0 and 1 (exclusive).")
    if starting_balance <= 0:
        raise ValueError("starting_balance must be positive.")

    config = default_config_for(rl_config_profile(interval), pair)
    docs = await load_full_candle_history(pair, interval)
    if not docs:
        raise ValueError(f"No candle history for {pair}/{interval}. Run /ingest/{interval} first.")

    # Fetched here (async, before the threadpool call) rather than inside train_ppo_policy
    # itself -- that function is sync/CPU-bound and runs via run_in_threadpool, which can't
    # make its own motor (async) DB calls. Every currently-resolved rule-based signal, same
    # query /ml/predict already uses -- train_ppo_policy filters this down to only the subset
    # resolved before ITS OWN train/test split boundary once it knows where that falls (see
    # rl_engine.frozen_ml_snapshot for why that filtering can't happen here).
    ml_reference_signals = await signals_collection.find(
        {"source": "live", "status": {"$in": ["hit", "miss", "expired"]}}
    ).to_list(length=None)

    warm_start_policy_id, warm_start_model_bytes = await _find_warm_start_policy(pair, interval)

    df = pd.DataFrame(docs)
    policy, eval_run, trade_signals, poc_diagnostics = await run_in_threadpool(
        train_ppo_policy, df, pair, interval, config, total_timesteps=total_timesteps,
        train_frac=train_frac, max_lookforward=max_lookforward, starting_balance=starting_balance,
        ml_reference_signals=ml_reference_signals, random_seed=random_seed,
        target_atr_mult=target_atr_mult, stop_atr_mult=stop_atr_mult,
        warm_start_policy_id=warm_start_policy_id, warm_start_model_bytes=warm_start_model_bytes,
    )

    # Individual test-slice trades reuse backtest_signals_collection (same as run_backtest's
    # own persistence) tagged with eval_run.run_id -- GET /backtest/runs/{run_id}/signals
    # already answers "which trades passed and which failed" for free, no new endpoint.
    if trade_signals:
        await backtest_signals_collection.insert_many([s.model_dump() for s in trade_signals])
    await backtest_runs_collection.insert_one(eval_run.model_dump())
    await ppo_policies_collection.insert_one(policy.model_dump())
    return policy, eval_run, poc_diagnostics


@app.post("/rl/train/{interval}")
async def train_rl(
    interval: str, pair: str, total_timesteps: int = 50_000, train_frac: float = 0.7, max_lookforward: int = 20,
    starting_balance: float = DEFAULT_STARTING_BALANCE, random_seed: int | None = None,
    target_atr_mult: float | None = None, stop_atr_mult: float | None = None,
):
    """
    Trains a PPO policy (app/services/ppo_engine.py) for this pair/interval against
    historical candle replay -- an adaptive alternative to consensus's fixed weighted-vote
    threshold, learning how to weight the same STRATEGIES instead of using a hand-picked
    REQUIRED_WEIGHT_FRACTION, AND how much to risk on each trade (2 size tiers, see
    rl_engine.RISK_FRACTION_BY_TIER) against a compounding account balance starting at
    `starting_balance` (default $50). One independent policy per pair, not shared across
    pairs. Persists both the learned PPOPolicy and its test-slice evaluation (a real
    BacktestRun, profile="rl_ppo", including starting_balance/ending_balance/total_return_pct
    alongside the usual hit_rate/expectancy) so it's directly comparable to every other
    approach via GET /backtest/runs?pair=X&profile=rl_ppo.

    This project's RL agent was linear Q-learning before PPO replaced it (see git history) --
    PPO is now the only RL algorithm here, not an alternative alongside the old one.

    Strategy calls use the intraday RuleConfig (short EMAs + session filter) for every
    interval (rl_config_profile) -- the one cross-pair-validated ruleset, applied
    consistently regardless of interval.

    pair: query param (e.g. ?pair=EUR/USD) — a path param would break on the literal '/'.

    For training every pair x interval at once, see POST /rl/train-all instead -- doing that
    here by simply looping client-side is what used to make the frontend's "Train all" a
    ~15-minute sequence of fetches the browser tab had to hold open the whole time.

    Runs synchronously, in-process, on FastAPI Cloud -- a single combo already takes 60-90s
    and has previously 524'd against the platform's gateway/proxy timeout (see PROGRESS.md's
    2026-09-10 entry). Nothing automated calls this anymore (.github/workflows/train-rl.yml
    and /rl/train-all both go through scripts/train_rl.py on a GitHub Actions runner
    instead) -- kept only as a manual/dev escape hatch, expect it to be slow and treat an
    occasional timeout as a known risk, not a bug to chase here.

    Warm-starts from the most recent usable policy for this pair/interval when one exists
    (see _find_warm_start_policy) -- continues training rather than starting fresh every
    call, unless there's no prior policy, its feature schema is stale, or it degenerated to
    always-HOLD (in which case a fresh model is used automatically, no action needed).

    random_seed: omit for normal training (stays genuinely exploratory). Pass an explicit int
    only when comparing two runs against each other and you need a reproducible baseline to
    isolate a real parameter effect from ordinary training-run variance.

    target_atr_mult/stop_atr_mult: omit (recommended) to use rl_engine.rl_atr_mults'
    interval/pair default. Pass both together to override it for this one call -- for
    sweeping a candidate value against a high-expired-rate combo (see
    GET /rl/resolution-stats) before adding it to RL_ATR_MULT_PAIR_OVERRIDES. This still
    persists a real policy/eval run like any other call; GET /backtest/runs?pair=X&
    profile=rl_ppo (filter the response to this interval yourself -- no server-side
    interval filter exists on that endpoint) lists every past run including these, so
    compare candidates there rather than trusting only this response if running several
    back to back.
    """
    try:
        policy, eval_run, poc_diagnostics = await _run_rl_training(
            pair, interval, total_timesteps, train_frac, max_lookforward, starting_balance,
            random_seed=random_seed, target_atr_mult=target_atr_mult, stop_atr_mult=stop_atr_mult,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "policy": policy.model_dump(exclude={"model_bytes"}),
        "evaluation": eval_run,
        "poc_diagnostics": poc_diagnostics,
    }


async def _trigger_rl_train_workflow(job_id: str, total_timesteps: int, train_frac: float, starting_balance: float):
    """
    Dispatches .github/workflows/train-rl.yml via GitHub's REST API instead of training
    in-process -- see PROGRESS.md's 2026-09-10 entry. Training every pair x interval used to
    run here via BackgroundTasks (run_train_all_job, removed), which works for surviving the
    *triggering request* returning early but not for surviving the FastAPI Cloud instance
    itself being recycled mid-run -- the same failure class that killed the old in-process
    APScheduler ingestion cron. The GitHub Actions runner this dispatches to doesn't have
    that problem, and scripts/train_rl.py there runs the identical training/persistence code
    this endpoint used to call directly.

    Requires settings.github_pat (a PAT with the `workflow` scope -- NOT the read-only
    Actions-log PAT noted in project memory; check scope before reusing it) and
    settings.github_repo ("owner/repo"). Raises HTTPException on any non-204 response so the
    caller gets a real error instead of a job_id nothing will ever update.
    """
    if not settings.github_pat or not settings.github_repo:
        raise HTTPException(
            status_code=500,
            detail="github_pat/github_repo not configured -- can't dispatch .github/workflows/train-rl.yml.",
        )
    url = f"https://api.github.com/repos/{settings.github_repo}/actions/workflows/train-rl.yml/dispatches"
    payload = {
        # master, not main -- see CLAUDE.md's deploy notes for this repo's default branch.
        "ref": "master",
        "inputs": {
            "job_id": job_id,
            "total_timesteps": str(total_timesteps),
            "train_frac": str(train_frac),
            "starting_balance": str(starting_balance),
        },
    }
    headers = {"Authorization": f"Bearer {settings.github_pat}", "Accept": "application/vnd.github+json"}
    # Wrapped specifically because an unhandled exception here (vs. a clean HTTPException)
    # produces a response with no CORS headers -- Starlette's ServerErrorMiddleware sits
    # OUTSIDE CORSMiddleware in the stack, so an uncaught exception's resulting 500 never
    # passes back through CORSMiddleware to get Access-Control-Allow-Origin attached. The
    # browser then reports "blocked by CORS" for what is actually a connection failure to
    # GitHub's API -- this project already hit this exact false-CORS-diagnosis once before
    # for an unrelated endpoint (see SIGNAL_STACK_V2.md). Catching it here and re-raising as
    # HTTPException keeps the response inside normal exception-handling middleware (and
    # therefore CORSMiddleware), so the frontend sees the real error instead of a generic
    # network failure.
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=502, detail=f"Couldn't reach GitHub's API ({type(e).__name__}): {e}",
        )
    if resp.status_code != 204:
        raise HTTPException(
            status_code=502, detail=f"GitHub workflow_dispatch failed ({resp.status_code}): {resp.text[:300]}",
        )


@app.get("/debug/egress-check")
async def debug_egress_check():
    """
    Diagnostic only -- tests whether THIS FastAPI Cloud instance can reach api.github.com at
    all, against api.twelvedata.com (a host already known to work, since ingestion depends on
    it) as a control. Added while debugging POST /rl/train-all consistently 502ing with a
    browser-side CORS error even after _trigger_rl_train_workflow's httpx call was wrapped in
    try/except (01dce7c) -- if that fix didn't change the symptom, the likely explanation is
    FastAPI Cloud's own gateway timing out and returning ITS OWN 502 (no CORS headers, not
    generated by this app at all) before this app's own httpx timeout/exception handling ever
    gets a chance to run, meaning the real problem is outbound network access to GitHub
    specifically, not a bug in the exception handling itself. Read-only, no side effects --
    safe to leave in place as a general ops diagnostic, not just for this one investigation.
    """
    # Bumped by hand on each deploy meant to verify itself this way -- FastAPI Cloud's
    # dashboard "live" label has been observed NOT to match what's actually serving requests
    # (this route 404'd with a clean FastAPI-shaped 404, meaning auth passed and Starlette's
    # own router just didn't know about it, even though the dashboard showed the commit that
    # added it as the live deployment) -- a value baked into the running code itself is the
    # only way to be sure which build is actually answering requests.
    DEPLOY_MARKER = "2026-09-11-e"

    results: dict = {"deploy_marker": DEPLOY_MARKER}
    for name, url in [("github", "https://api.github.com"), ("twelvedata", "https://api.twelvedata.com")]:
        t0 = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.get(url)
            results[name] = {"ok": True, "status_code": resp.status_code, "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1)}
        except httpx.RequestError as e:
            results[name] = {"ok": False, "error": f"{type(e).__name__}: {e}", "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1)}
    return results


@app.get("/debug/archive-check")
async def debug_archive_check(pair: str, interval: str):
    """
    Diagnostic only, read-only -- proves whether THIS deployed backend can actually read the
    committed data/candles/*.csv.gz archive files (FastAPI Cloud deploys from this same repo,
    but that's an assumption, not something confirmed before /candles/prune-history relies on
    it). Reports row count and min/max timestamp found in the archive file, separate from
    whatever's in Mongo -- a row count of 0 here means either no archive file exists for this
    pair/interval, or it exists but the deployed container doesn't actually have it on disk.
    """
    from app.services.candle_archive import load_archived_candles, ARCHIVE_DIR, _slug

    path = ARCHIVE_DIR / f"{_slug(pair)}_{interval}.csv.gz"
    df = load_archived_candles(pair, interval)
    return {
        "archive_dir": str(ARCHIVE_DIR),
        "archive_dir_exists": ARCHIVE_DIR.exists(),
        "expected_file": str(path),
        "file_exists": path.exists(),
        "rows": len(df),
        "min_timestamp": df["timestamp"].min().isoformat() if len(df) else None,
        "max_timestamp": df["timestamp"].max().isoformat() if len(df) else None,
    }


@app.get("/debug/db-stats")
async def debug_db_stats():
    """
    Diagnostic only, read-only -- per-collection storage size, so what's actually consuming
    an Atlas cluster's storage cap can be checked directly instead of guessed at (added while
    figuring out what to prune on Cluster2's M0 tier: 280.89MB used out of a 512MB hard cap,
    before the deep intraday candle backfill added anything more).
    """
    overall = await db.command("dbStats")
    collections = {}
    for name in await db.list_collection_names():
        stats = await db.command("collStats", name)
        collections[name] = {
            "count": stats.get("count"),
            "size_mb": round(stats.get("size", 0) / 1024 / 1024, 2),
            "storage_size_mb": round(stats.get("storageSize", 0) / 1024 / 1024, 2),
            "total_index_size_mb": round(stats.get("totalIndexSize", 0) / 1024 / 1024, 2),
        }
    return {
        "data_size_mb": round(overall.get("dataSize", 0) / 1024 / 1024, 2),
        "storage_size_mb": round(overall.get("storageSize", 0) / 1024 / 1024, 2),
        "index_size_mb": round(overall.get("indexSize", 0) / 1024 / 1024, 2),
        "collections": dict(sorted(collections.items(), key=lambda kv: kv[1]["storage_size_mb"], reverse=True)),
    }


@app.get("/debug/dispatch-check")
async def debug_dispatch_check():
    """
    Diagnostic only -- calls _trigger_rl_train_workflow directly with a throwaway job_id
    (real dispatch, will actually queue a train-rl.yml run -- harmless, same as any other
    manual/dev training trigger) wrapped in a bare except so ANY failure mode is reported
    back instead of producing an unhandled 500/CORS-masking response. Added because
    GET /debug/egress-check showed clean, fast (30ms) connectivity to api.github.com, yet
    POST /rl/train-all still 502s with no CORS headers -- meaning the failure isn't a generic
    GitHub-unreachable problem, and must be something specific to the real dispatch call or
    something else in that endpoint's body. This isolates exactly which.
    """
    job_id = f"debug-{uuid.uuid4().hex[:8]}"
    # github_repo isn't a secret (it's a public repo path) -- included so a 404 from GitHub
    # can be diagnosed as "wrong repo string" vs "PAT lacks access" without guessing. Only
    # the PAT's length/prefix are shown (never the value itself) as a "did it even load"
    # sanity check.
    config_snapshot = {
        "github_repo": settings.github_repo,
        "github_pat_len": len(settings.github_pat or ""),
        "github_pat_prefix": (settings.github_pat or "")[:12],
    }
    try:
        await _trigger_rl_train_workflow(job_id, 50_000, 0.7, DEFAULT_STARTING_BALANCE)
        return {"ok": True, "job_id": job_id, **config_snapshot}
    except HTTPException as e:
        return {"ok": False, "kind": "HTTPException", "status_code": e.status_code, "detail": e.detail, **config_snapshot}
    except Exception as e:
        return {"ok": False, "kind": type(e).__name__, "detail": str(e), **config_snapshot}


@app.post("/rl/train-all")
async def train_rl_all(
    total_timesteps: int = 50_000, train_frac: float = 0.7, starting_balance: float = DEFAULT_STARTING_BALANCE,
):
    """
    Triggers training every pair x interval combination (the same set the daily cron trains)
    on a GitHub Actions runner (.github/workflows/train-rl.yml) and returns immediately with
    a job_id -- poll GET /rl/train-all/{job_id} for progress, same as before. Training used to
    run here in-process via BackgroundTasks; moved off FastAPI Cloud entirely (see
    PROGRESS.md's 2026-09-10 entry) so a 20-combo, ~20-30 minute run can't be cut short by an
    instance recycle or a gateway/proxy timeout on any individual combo.
    """
    if not 0 < train_frac < 1:
        raise HTTPException(status_code=400, detail="train_frac must be between 0 and 1 (exclusive).")
    if starting_balance <= 0:
        raise HTTPException(status_code=400, detail="starting_balance must be positive.")

    job_id = uuid.uuid4().hex[:12]
    await _trigger_rl_train_workflow(job_id, total_timesteps, train_frac, starting_balance)
    job = RLTrainAllJob(
        job_id=job_id, status="running", created_at=datetime.utcnow(),
        total_timesteps=total_timesteps, train_frac=train_frac, starting_balance=starting_balance,
        total=len(settings.pairs_list) * len(RL_INTERVALS),
    )
    await rl_train_jobs_collection.insert_one(job.model_dump())
    return job


@app.get("/rl/train-all/{job_id}")
async def get_train_all_job(job_id: str):
    doc = await rl_train_jobs_collection.find_one({"job_id": job_id})
    if not doc:
        raise HTTPException(status_code=404, detail=f"No train-all job {job_id}.")
    return RLTrainAllJob(**{k: v for k, v in doc.items() if k != "_id"})


@app.post("/rl/train-all/{job_id}/cancel")
async def cancel_train_all_job(job_id: str):
    """
    Immediately marks a running train-all job "cancelled" -- not just a request the loop
    picks up between combos. Originally this only set cancel_requested and waited for the
    old in-process training loop's own between-combo check, but that's a no-op if the
    in-flight combo hangs outright rather than just running long (seen in practice: a single
    pair/interval stuck for 60+ minutes with the whole job normally taking ~15-30). Training
    now runs on a GitHub Actions runner (scripts/train_rl.py, see PROGRESS.md's 2026-09-10
    entry) rather than an in-process background task, but the same cooperative-cancel gap
    applies there too -- the script only checks this job doc's status between combos, so this
    endpoint still updates the document directly rather than waiting for that check to land.

    Already-completed combos and their trained policies are kept, not rolled back; only the
    remaining ones are skipped. If a combo already in flight on the runner finishes anyway
    after this, its leftover write is a no-op -- see scripts/train_rl.py's
    status="running" write guard.
    """
    doc = await rl_train_jobs_collection.find_one({"job_id": job_id})
    if not doc:
        raise HTTPException(status_code=404, detail=f"No train-all job {job_id}.")
    if doc["status"] != "running":
        raise HTTPException(status_code=400, detail=f"Job {job_id} is already {doc['status']}, nothing to cancel.")
    await rl_train_jobs_collection.update_one(
        {"job_id": job_id, "status": "running"},
        {"$set": {"cancel_requested": True, "status": "cancelled", "finished_at": datetime.utcnow()}},
    )
    doc = await rl_train_jobs_collection.find_one({"job_id": job_id})
    return RLTrainAllJob(**{k: v for k, v in doc.items() if k != "_id"})


@app.get("/rl/train-all-latest")
async def get_latest_train_all_job():
    """Lets the frontend rehydrate an in-progress job after a reload or reopen -- otherwise
    there's no way to tell "nothing running" apart from "was running, the tab just reloaded"."""
    doc = await rl_train_jobs_collection.find_one(sort=[("created_at", -1)])
    if not doc:
        return None
    return RLTrainAllJob(**{k: v for k, v in doc.items() if k != "_id"})


@app.get("/rl/policies")
async def list_rl_policies(pair: str | None = None, interval: str | None = None, limit: int = 20):
    """
    Each policy is enriched with its test-slice evaluation summary (hit_rate_pct,
    expectancy_pct, directional_signals, hold_signals -- pulled from the linked BacktestRun
    via eval_run_id) so this list doubles as a learning-progress view: read down the rows for
    a given pair/interval over successive trainings to see whether hit rate/expectancy is
    actually trending anywhere, not just when the most recent training happened.

    model_bytes (the serialized PPO neural network) is excluded from the response -- it's not
    something a caller displays; POST /rl/signal loads it straight from the DB document, not
    from this response.
    """
    query: dict = {}
    if pair:
        query["pair"] = pair
    if interval:
        query["interval"] = interval
    cursor = ppo_policies_collection.find(query, {"model_bytes": 0}).sort("created_at", -1).limit(limit)
    docs = await cursor.to_list(length=limit)

    eval_run_ids = [d["eval_run_id"] for d in docs if d.get("eval_run_id")]
    eval_runs = {}
    if eval_run_ids:
        eval_cursor = backtest_runs_collection.find({"run_id": {"$in": eval_run_ids}})
        async for run in eval_cursor:
            eval_runs[run["run_id"]] = run

    for d in docs:
        d["_id"] = str(d["_id"])
        run = eval_runs.get(d.get("eval_run_id"))
        d["evaluation"] = {
            "hit_rate_pct": run["hit_rate_pct"],
            "expectancy_pct": run["expectancy_pct"],
            "directional_signals": run["directional_signals"],
            "hold_signals": run["hold_signals"],
            "starting_balance": run.get("starting_balance"),
            "ending_balance": run.get("ending_balance"),
            "total_return_pct": run.get("total_return_pct"),
        } if run else None
    return docs


@app.post("/rl/reset")
async def reset_rl(confirm: bool = False):
    """
    Wipes every RL policy and its training artifacts so the next training run starts fresh,
    while keeping the genuine market-outcome history (hit/miss/expired RLSignals) intact --
    that history is real evidence of what actually happened and stays valid regardless of
    which policy generated the trade, so there's no reason to lose it just because the
    policies themselves are being reset.

    Deletes:
      - Every ppo_policies_collection document (all pair/interval policies).
      - Every backtest_runs_collection document with profile="rl_ppo" (RL's own eval-run
        summaries), plus every backtest_signals_collection document linked to one of those
        run_ids via run_id (RL's individual test-slice trade log -- these don't carry
        profile="rl_ppo" themselves, Signal has no RL-specific profile literal; linkage is by
        run_id only).
      - rl_signals_collection documents with status "pending" or "superseded" -- pending ones
        would otherwise reference a policy_id that no longer exists after the reset, and
        superseded ones were never a real market outcome to begin with (the agent changed its
        mind before label_outcome got the chance).

    Explicitly does NOT touch: rl_signals_collection documents with status "hit"/"miss"/
    "expired" (the real outcome history -- case_memory.py's cross-pair pool, GET
    /rl/accuracy's history), the rule-based signals_collection, or anything ML-related
    (ml_runs_collection, signals_collection) -- this is scoped to RL only.

    NOTE: this project's RL agent was linear Q-learning (rl_policies_collection,
    BacktestRun profile="rl") before PPO replaced it. Those old collections/documents are
    dead weight now -- nothing reads or writes them -- but this endpoint doesn't delete them
    itself; they're harmless orphaned history, not something worth adding delete logic for
    here just to tidy up.

    confirm: defaults to False, which runs the exact same queries and returns the counts of
    what WOULD be deleted without deleting anything -- a dry run to sanity-check the numbers
    before committing to an irreversible operation. Pass confirm=true to actually execute.
    """
    rl_run_ids = [
        d["run_id"] async for d in backtest_runs_collection.find({"profile": "rl_ppo"}, {"run_id": 1})
    ]

    policies_count = await ppo_policies_collection.count_documents({})
    runs_count = len(rl_run_ids)
    signals_count = await backtest_signals_collection.count_documents(
        {"run_id": {"$in": rl_run_ids}}
    ) if rl_run_ids else 0
    pending_count = await rl_signals_collection.count_documents({"status": "pending"})
    superseded_count = await rl_signals_collection.count_documents({"status": "superseded"})
    kept_count = await rl_signals_collection.count_documents(
        {"status": {"$in": ["hit", "miss", "expired"]}}
    )

    result = {
        "confirmed": confirm,
        "deleted": {
            "rl_policies": policies_count,
            "rl_eval_runs": runs_count,
            "rl_eval_trade_log_signals": signals_count,
            "rl_signals_pending": pending_count,
            "rl_signals_superseded": superseded_count,
        },
        "kept": {"rl_signals_hit_miss_expired": kept_count},
    }
    if not confirm:
        return result

    if rl_run_ids:
        await backtest_signals_collection.delete_many({"run_id": {"$in": rl_run_ids}})
    await backtest_runs_collection.delete_many({"profile": "rl_ppo"})
    await ppo_policies_collection.delete_many({})
    await rl_signals_collection.delete_many({"status": {"$in": ["pending", "superseded"]}})
    return result


@app.post("/rl/prune-history")
async def prune_rl_history(confirm: bool = False, keep_latest_n: int = 1):
    """
    Reclaims Atlas storage from RL training history WITHOUT losing any feature this app
    actually uses -- unlike /rl/reset, which wipes everything RL-related, this only removes
    the parts nothing ever reads again:

      - ppo_policies_collection: every policy doc except the keep_latest_n most recent per
        pair/interval gets its model_bytes field unset (the doc and its metadata survive).
        Every read of model_bytes in this file (choose_action_ppo, warm-start lookup,
        POST /rl/signal/{interval}) queries sort=[("created_at", -1)] for a single combo --
        an older policy's weights are never loaded again once a newer one exists. GET
        /rl/policies (the learning-progress trend view) already excludes model_bytes from
        its projection, so stripping it there is invisible to that feature.
      - backtest_signals_collection: the individual test-slice trade log for every RL eval
        run (backtest_runs profile="rl_ppo") except the ones still linked to a kept policy
        (their eval_run_id) gets deleted outright. The trend view only reads
        backtest_runs_collection (hit_rate_pct/expectancy_pct summaries, ~1MB total) for
        this, never the per-trade detail -- only a drill-down into one specific historical
        run's trade list would miss it, and only for a run old enough to have been pruned.

    Non-RL backtest_signals (rule-based /backtest and /backtest/optimize runs, profile !=
    "rl_ppo") are untouched -- same scope /rl/reset already uses.

    confirm: defaults to False, a dry run that returns exactly what WOULD change without
    changing anything.
    """
    stripped_policy_ids = []
    kept_eval_run_ids = set()
    prunable_eval_run_ids = set()
    policies_size_before = 0

    for pair in settings.pairs_list:
        for interval in RL_INTERVALS:
            docs = await ppo_policies_collection.find(
                {"pair": pair, "interval": interval}, {"model_bytes": 1, "eval_run_id": 1}
            ).sort("created_at", -1).to_list(length=None)
            for d in docs[:keep_latest_n]:
                if d.get("eval_run_id"):
                    kept_eval_run_ids.add(d["eval_run_id"])
            for d in docs[keep_latest_n:]:
                if d.get("model_bytes") is not None:
                    stripped_policy_ids.append(d["_id"])
                    policies_size_before += len(d["model_bytes"])
                if d.get("eval_run_id"):
                    prunable_eval_run_ids.add(d["eval_run_id"])

    prunable_eval_run_ids -= kept_eval_run_ids
    signals_count = await backtest_signals_collection.count_documents(
        {"run_id": {"$in": list(prunable_eval_run_ids)}}
    ) if prunable_eval_run_ids else 0

    result = {
        "confirmed": confirm,
        "ppo_policies_stripped": len(stripped_policy_ids),
        "ppo_policies_model_bytes_freed_mb": round(policies_size_before / 1024 / 1024, 2),
        "backtest_signals_to_delete": signals_count,
        "eval_runs_affected": len(prunable_eval_run_ids),
    }
    if not confirm:
        return result

    if stripped_policy_ids:
        await ppo_policies_collection.update_many(
            {"_id": {"$in": stripped_policy_ids}}, {"$unset": {"model_bytes": ""}}
        )
    if prunable_eval_run_ids:
        await backtest_signals_collection.delete_many({"run_id": {"$in": list(prunable_eval_run_ids)}})
    return result


@app.post("/rl/signal/{interval}")
async def create_rl_signal(interval: str, pair: str, balance: float = DEFAULT_STARTING_BALANCE):
    """
    Loads the most recently trained PPOPolicy for this pair/interval, computes the current
    state from live strategy calls on the latest candles PLUS the supplied `balance` (same
    encoding used during training, via rl_engine.full_rl_state -- `balance` is your real
    current account balance, not a value this backend tracks itself, since there's no way to
    know whether a prior signal was actually taken or what it filled at on an external
    broker), and picks the greedy (deterministic) action via ppo_engine.choose_action_ppo.
    Only BUY/SELL get stored -- HOLD never produces an RLSignal, same as ConsensusSignal. The
    chosen size tier (SMALL/LARGE) is sized against `balance` via position_size_units, so the
    signal is always grounded in your actual account, not an assumed one.

    The response also includes `memory` (app/services/case_memory.py) -- how similar past
    resolved signals for this exact pair/interval actually turned out (hit rate among decided
    outcomes, average pct move, how many nearest historical cases were used), a k-nearest-
    neighbor lookup against every past state vector alongside whatever the PPO policy's own
    action-probability distribution says. It's informational, not a gate on the trade this
    endpoint returns -- the point is making "have we been here before" inspectable, not
    silently overriding the policy's decision with a separate heuristic.

    Single-position-at-a-time, same discipline _take_action_sized already uses during
    training (it only decides again once a simulated trade has resolved, advancing by
    candles_to_outcome). If a pending RL signal already exists for this pair/interval, this
    endpoint does NOT re-decide -- it first checks whether that signal has genuinely resolved
    against real candles (resolve_rl_signal_real_outcome); if so, that real outcome is recorded
    and a fresh decision proceeds below as normal, and if not, the still-open pending signal is
    simply returned as-is, without recomputing state or q_values. This used to instead
    re-evaluate on every call and mark the old signal "superseded" the moment a new decision
    disagreed with it (including trivial cases like entry_price ticking a fraction on a moving
    price) -- that meant live inference was re-deciding far more often than the policy was ever
    trained to, which is what actually drove the high supersede rate GET /rl/accuracy showed,
    not policy quality. Waiting for genuine resolution instead keeps live behavior consistent
    with training and gives every signal a real chance to become a hit/miss/expired data point.

    pair: query param (e.g. ?pair=EUR/USD) — a path param would break on the literal '/'.
    """
    if balance <= 0:
        raise HTTPException(status_code=400, detail="balance must be positive.")

    pending = await rl_signals_collection.find_one(
        {"pair": pair, "interval": interval, "status": "pending"}, sort=[("timestamp", -1)],
    )
    if pending is not None:
        real_outcome, _ = await resolve_rl_signal_real_outcome(pending)
        if real_outcome is not None:
            await rl_signals_collection.update_one({"_id": pending["_id"]}, {"$set": real_outcome})
        else:
            # Still genuinely open -- don't re-decide, just hand back the live view as-is.
            pending["_id"] = str(pending["_id"])
            return {"signal": pending, "q_values": pending.get("q_values"), "memory": None}

    policy_doc = await ppo_policies_collection.find_one(
        {"pair": pair, "interval": interval}, sort=[("created_at", -1)],
    )
    if policy_doc is None:
        raise HTTPException(
            status_code=400,
            detail=f"No trained RL policy yet for {pair}/{interval}. Call POST /rl/train/{interval}?pair={pair} first.",
        )
    policy = PPOPolicy(**{k: v for k, v in policy_doc.items() if k != "_id"})

    config = default_config_for(rl_config_profile(interval), pair)
    cursor = candles_collection.find({"pair": pair, "interval": interval}).sort("timestamp", -1).limit(500)
    docs = await cursor.to_list(length=500)
    docs.reverse()

    min_needed = max(config.ema_slow, MIN_WARMUP_BARS)
    if len(docs) < min_needed:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough candle history ({len(docs)} rows). Need {min_needed}+ for RL. "
                    f"Run /ingest/{interval} first."
        )

    resolved_signals = await signals_collection.find(
        {"source": "live", "status": {"$in": ["hit", "miss", "expired"]}}
    ).to_list(length=None)
    market_state, df, ml_buy_score, ml_sell_score = _rl_state_from_candles(
        docs, config, pair, interval, rl_config_profile(interval), resolved_signals,
    )
    state = full_rl_state(market_state, balance, policy.starting_balance)
    try:
        action, q_values = choose_action_ppo(policy, state)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    current_price = float(df.iloc[-1]["close"])

    # Live-exclusion gate: a policy whose latest training-time backtest showed a clear losing
    # edge (same STRONG_LOSS_RETURN_PCT threshold GET /rl/insights already flags as
    # "critical") doesn't get to trade live, even though it keeps training normally in the
    # background (the once-daily cron retrain and _check_and_retrain_degraded_policies's
    # degradation-triggered retrain are both untouched by this). Self-correcting, not a
    # manual toggle someone has to remember to flip back -- this re-checks the LATEST
    # policy's own eval fresh on every call, so the next retrain that clears the threshold
    # re-enables live signals automatically.
    eval_run = await backtest_runs_collection.find_one({"run_id": policy.eval_run_id})
    excluded_return = eval_run.get("total_return_pct") if eval_run else None
    if excluded_return is not None and excluded_return <= STRONG_LOSS_RETURN_PCT:
        # No pending signal to retire here -- if one existed, it was already resolved or
        # returned as still-open above, before this policy was even loaded.
        return {
            "signal": None, "q_values": q_values,
            "excluded_reason": (
                f"Latest training run lost {abs(excluded_return):.0f}% of a simulated $50 "
                f"start (hit rate {eval_run.get('hit_rate_pct')}%) -- excluded from live "
                f"signals until a retrain clears this. Training continues normally in the "
                f"background; this re-checks fresh every call, so it re-enables automatically."
            ),
        }

    if action == "HOLD":
        return {"signal": None, "q_values": q_values}

    direction, tier = action.split("_")

    # ML quality gate: same GOOD_SIGNAL_ML_THRESHOLD bar create_signal applies, using the
    # ml_hit_probability_buy/sell already computed for this exact bar (see
    # _rl_state_from_candles) rather than a fresh lookup -- whichever score matches the
    # direction the policy just chose. None (not enough resolved history yet) fails OPEN,
    # same reasoning as create_signal's own gate: absence of data isn't evidence of a bad
    # trade. This sits alongside, not instead of, the live-exclusion gate above (policy-level)
    # and memory_gate below (case-level) -- three independent, layered checks on top of the
    # raw Q-policy action, same "guaranteed by construction, not left for the agent alone to
    # discover" philosophy as the fixed 1.5:1 target:stop floor.
    ml_score = ml_buy_score if direction == "BUY" else ml_sell_score
    if ml_score is not None and ml_score < GOOD_SIGNAL_ML_THRESHOLD:
        return {
            "signal": None, "q_values": q_values,
            "ml_blocked_reason": (
                f"ML rated this {direction} at only {ml_score * 100:.0f}% hit probability "
                f"(below the {GOOD_SIGNAL_ML_THRESHOLD * 100:.0f}% bar for a live signal) "
                f"-- held instead."
            ),
        }

    # "Have we seen a state like this before, and how did it actually turn out" -- see
    # case_memory.py. Pooled across EVERY pair/interval, not just this one, the same way ML's
    # classifier pools across all pairs/profiles into one shared model instead of siloing each
    # one into its own small sample -- state features are already scale-comparable by
    # construction (votes in [-1,1], raw_* features normalized similarly, see
    # rl_engine.RL_MARKET_FEATURE_NAMES's own comment) specifically so a state from one
    # pair/interval means roughly the same thing as a state from another, making this pooling
    # sound rather than apples-to-oranges. Each of the 20 pair/interval policies is otherwise
    # starved for its own history; this is the one place a thin policy gets to borrow from
    # what every OTHER pair/interval has collectively learned about similar-looking setups.
    # Still filtered to the SAME direction the policy just chose (not both directions blended)
    # -- the question memory needs to answer is specifically "how have BUY (or SELL) decisions
    # that looked like this one actually gone," regardless of which pair/interval they fired on.
    # Computed against every resolved (never superseded) past RLSignal with a stored state
    # vector; empty/none for a brand new state shape or one whose history predates the state
    # field, same "surface as absent, not a fabricated number" convention as ml_model's
    # None-when-not-enough-data.
    memory_candidates = await rl_signals_collection.find({
        "direction": direction,
        "status": {"$in": list(RESOLVED_STATUSES)},
    }).to_list(length=None)
    memory = memory_summary(state, memory_candidates)

    # This policy's own OVERALL resolved-trade record for this pair/interval (both directions
    # combined, same directional-hit-rate math as GET /rl/accuracy) -- distinct from the
    # narrow, same-direction "similar cases" number above. A policy can be genuinely strong
    # overall while still showing a weak LOCAL neighborhood for one specific setup (a handful
    # of nearest cases is a small, sometimes-noisy sample) -- surfacing both numbers together,
    # always, means a good policy's real track record doesn't get silently ignored just
    # because memory_gate below is scoped to the narrower question. See memory_gate's own
    # docstring for how the two get reconciled when they disagree.
    overall_hits = await rl_signals_collection.count_documents(
        {"pair": pair, "interval": interval, "status": "hit"}
    )
    overall_misses = await rl_signals_collection.count_documents(
        {"pair": pair, "interval": interval, "status": "miss"}
    )
    overall_decided = overall_hits + overall_misses
    memory["policy_hit_rate_pct"] = round(overall_hits / overall_decided * 100, 1) if overall_decided else None
    memory["policy_decided_trades"] = overall_decided

    # Act on memory, not just report it: memory_gate can downgrade LARGE->SMALL or override
    # the whole trade to HOLD when similar past states have a poor track record -- see that
    # function's own docstring for why this is a layered safety rule on top of the learned
    # policy, same philosophy as the fixed 1.5:1 target:stop floor.
    gated_tier, memory_override = memory_gate(memory, tier)
    if gated_tier == "HOLD":
        return {"signal": None, "q_values": q_values, "memory": memory, "memory_override": memory_override}
    tier = gated_tier
    risk_fraction = RISK_FRACTION_BY_TIER[tier]

    latest = df.iloc[-1]
    entry_price = current_price
    atr_val = float(compute_atr_series(df, config.atr_period).iloc[-1])
    target_atr_mult, stop_atr_mult = rl_atr_mults(interval, pair)
    target_price, stop_price = compute_atr_target_stop(
        entry_price, atr_val, direction, target_atr_mult, stop_atr_mult,
    )
    units = position_size_units(balance, risk_fraction, entry_price, stop_price, pair)

    rl_signal = RLSignal(
        signal_id=uuid.uuid4().hex[:12], algo="ppo",
        pair=pair, interval=interval, timestamp=latest["timestamp"], direction=direction,
        entry_price=entry_price, target_price=target_price, stop_price=stop_price,
        q_values=q_values, policy_id=policy.policy_id,
        size_tier=tier, risk_fraction=risk_fraction, balance_at_signal=round(balance, 2),
        position_size_units=round(units, 2), state=state, memory_override=memory_override,
    )

    # No de-dupe-against-pending check needed here -- by this point any pending signal for
    # this pair/interval was either already returned as-is (still open, above) or already
    # resolved and cleared (the block at the top of this function). There is nothing left to
    # compare against or supersede.
    await rl_signals_collection.insert_one(rl_signal.model_dump())
    return {"signal": rl_signal, "q_values": q_values, "memory": memory}


@app.get("/rl/signals/{signal_id}/explain")
async def explain_rl_signal(signal_id: str, k: int = 10):
    """
    For one resolved RL signal, answers "what was actually different this time" -- finds its
    nearest historical neighbor (same pair/interval, by state-vector distance, see
    app/services/case_memory.py) whose outcome DISAGREED with this one (a hit's nearest miss,
    or a miss's nearest hit), and reports the top feature differences between the two states,
    largest first. Not causal attribution -- a distance-ranked feature list, same
    "inspectable, not a fabricated explanation" spirit as this project's other interpretability
    surfaces (SignalReason.detail, ML's feature_importances). Meant for reviewing a specific
    surprising outcome after the fact, not for live decision-making (see /rl/signal/{interval}
    for the live `memory` summary instead).

    404 if signal_id doesn't exist. 400 if the signal isn't resolved yet (still pending or
    superseded -- there's no real outcome yet to explain), or if no disagreeing neighbor
    exists in its k nearest cases (e.g. every nearby case was expired, or every nearby case
    agreed with this one -- itself a meaningful "this looks like a consistent pattern" result,
    surfaced as such rather than an error).
    """
    doc = await rl_signals_collection.find_one({"signal_id": signal_id})
    if doc is None:
        raise HTTPException(status_code=404, detail=f"No RL signal found with signal_id={signal_id}")
    if doc["status"] not in RESOLVED_STATUSES:
        raise HTTPException(
            status_code=400,
            detail=f"Signal {signal_id} is '{doc['status']}', not yet resolved -- nothing to explain yet.",
        )
    if not doc.get("state"):
        raise HTTPException(
            status_code=400,
            detail=f"Signal {signal_id} predates the state field -- no vector to compare against.",
        )

    candidates = await rl_signals_collection.find({
        "pair": doc["pair"], "interval": doc["interval"],
        "status": {"$in": list(RESOLVED_STATUSES)},
        "signal_id": {"$ne": signal_id},
    }).to_list(length=None)

    nearest = nearest_cases(doc["state"], candidates, k=k)
    neighbor = find_diverging_neighbor(doc["status"], nearest)
    if neighbor is None:
        return {
            "signal_id": signal_id, "status": doc["status"], "cases_checked": len(nearest),
            "diverging_neighbor": None,
            "note": "No disagreeing neighbor among the nearest cases checked -- either every "
                    "nearby case was expired, or every nearby case agreed with this outcome.",
        }

    feature_diffs = explain_divergence(doc["state"], neighbor["state"], RL_FEATURE_NAMES)
    return {
        "signal_id": signal_id, "status": doc["status"],
        "diverging_neighbor": {
            "signal_id": neighbor.get("signal_id"), "status": neighbor.get("status"),
            "outcome_pct_move": neighbor.get("outcome_pct_move"), "distance": neighbor["distance"],
        },
        "feature_diffs": feature_diffs,
    }


@app.get("/rl/signals")
async def list_rl_signals(pair: str | None = None, limit: int = 50):
    query = {"pair": pair} if pair else {}
    cursor = rl_signals_collection.find(query).sort("timestamp", -1).limit(limit)
    docs = await cursor.to_list(length=limit)
    for d in docs:
        d["_id"] = str(d["_id"])
    return docs


@app.get("/rl/accuracy")
async def rl_accuracy(pair: str | None = None, interval: str | None = None, limit: int = 100):
    """
    Overall live-trading accuracy for the RL agent -- same rolling-window-plus-real-total
    shape as GET /signals/accuracy, over rl_signals_collection instead. source="live" only
    (excludes the per-training test-slice trades in backtest_signals_collection, which are a
    different, already-visible thing via Training history's "Show trades").

    hit/miss/expired only -- "superseded" signals (the agent changed its mind before
    label_outcome ever got to judge one, see RLSignal.status's docstring) are deliberately
    excluded from the hit-rate denominator, since they're not a real win/loss/timeout. Their
    count is still surfaced separately (total_superseded) so it's visible why the total
    signal count and total_resolved can differ a lot, especially on faster/less-converged
    intervals whose policy changes its mind often.
    """
    query: dict = {"source": "live", "status": {"$in": ["hit", "miss", "expired"]}}
    if pair:
        query["pair"] = pair
    if interval:
        query["interval"] = interval
    cursor = rl_signals_collection.find(query).sort("timestamp", -1).limit(limit)
    docs = await cursor.to_list(length=limit)

    hits = sum(1 for d in docs if d["status"] == "hit")
    misses = sum(1 for d in docs if d["status"] == "miss")
    expired = sum(1 for d in docs if d["status"] == "expired")
    total = len(docs)

    total_hits = await rl_signals_collection.count_documents({**query, "status": "hit"})
    total_misses = await rl_signals_collection.count_documents({**query, "status": "miss"})
    total_expired = await rl_signals_collection.count_documents({**query, "status": "expired"})
    total_resolved = total_hits + total_misses + total_expired
    total_superseded = await rl_signals_collection.count_documents(
        {**{k: v for k, v in query.items() if k != "status"}, "status": "superseded"}
    )
    total_decided = total_hits + total_misses
    total_all = total_resolved + total_superseded

    return {
        "pair": pair,
        "interval": interval,
        "sample_size": total,
        "hits": hits,
        "misses": misses,
        "expired": expired,
        "hit_rate_pct": round(hits / total * 100, 1) if total else None,
        "total_resolved": total_resolved,
        "total_hits": total_hits,
        "total_misses": total_misses,
        "total_expired": total_expired,
        # total_hit_rate_pct divides by every resolved signal INCLUDING expired (a non-event,
        # neither win nor loss) -- kept for backward compat, but it collapses toward 0 whenever
        # expired dominates (e.g. a fast/unconverged policy on 5min/15min), making it look like
        # "not learning anything" when the real story is "rarely decides." directional_hit_rate_pct
        # below is the number that actually answers "when this agent commits to hit-or-miss, how
        # often is it right."
        "total_hit_rate_pct": round(total_hits / total_resolved * 100, 1) if total_resolved else None,
        "total_superseded": total_superseded,
        "directional_hit_rate_pct": round(total_hits / total_decided * 100, 1) if total_decided else None,
        "resolution_breakdown_pct": {
            "hit": round(total_hits / total_all * 100, 1) if total_all else None,
            "miss": round(total_misses / total_all * 100, 1) if total_all else None,
            "expired": round(total_expired / total_all * 100, 1) if total_all else None,
            "superseded": round(total_superseded / total_all * 100, 1) if total_all else None,
        },
    }


@app.get("/rl/resolution-stats")
async def rl_resolution_stats(pair: str | None = None):
    """
    Diagnostic for the high `expired` fraction GET /rl/accuracy shows: per interval, the
    candles_to_outcome distribution among signals that genuinely resolved (hit/miss),
    compared against that interval's own LIVE_MAX_LOOKFORWARD_BY_INTERVAL window ceiling
    (outcome_scoring.py).

    Answers "is expired a window problem or an ATR-band problem":
    - Resolved trades clustering near the window ceiling (median/p90 close to it) -> the
      window itself is the binding constraint, widen LIVE_MAX_LOOKFORWARD_BY_INTERVAL.
    - Resolved trades resolving well under the ceiling, but expired_fraction_pct still high
      -> price is chopping inside a target/stop band it never reaches within the window --
      RL_ATR_MULTS_BY_INTERVAL's multiples are the actual lever, not the window.

    Best read after create_rl_signal's supersede-churn fix has had a day or so of live cron
    cycles to build up genuinely-resolved (not superseded) history -- older history is
    dominated by supersede noise and won't give a clean answer.

    source="live" only, same scope as GET /rl/accuracy.
    """
    def _pctile(sorted_vals: list[int], p: float) -> Optional[int]:
        if not sorted_vals:
            return None
        idx = min(len(sorted_vals) - 1, int(round(p * (len(sorted_vals) - 1))))
        return sorted_vals[idx]

    by_interval: dict[str, dict] = {}
    for interval, window in LIVE_MAX_LOOKFORWARD_BY_INTERVAL.items():
        base_query: dict = {"source": "live", "interval": interval}
        if pair:
            base_query["pair"] = pair

        resolved_docs = await rl_signals_collection.find(
            {**base_query, "status": {"$in": ["hit", "miss"]}}, {"candles_to_outcome": 1},
        ).to_list(length=None)
        values = sorted(d["candles_to_outcome"] for d in resolved_docs if d.get("candles_to_outcome") is not None)

        expired_count = await rl_signals_collection.count_documents({**base_query, "status": "expired"})
        resolved_count = len(values)
        total = resolved_count + expired_count

        by_interval[interval] = {
            "window_candles": window,
            "resolved_hit_miss": resolved_count,
            "expired": expired_count,
            "expired_fraction_pct": round(expired_count / total * 100, 1) if total else None,
            "candles_to_outcome": {
                "min": values[0] if values else None,
                "median": _pctile(values, 0.5),
                "p90": _pctile(values, 0.9),
                "max": values[-1] if values else None,
            },
        }
    return {"pair": pair, "by_interval": by_interval}


@app.get("/rl/learning-curve")
async def rl_learning_curve(days: int = 30):
    """
    Day-by-day trend of both halves of "is the agent getting smarter" -- a single snapshot
    (GET /rl/accuracy) can't answer that, only whether today looks better or worse than the
    last poll. Two independent series, both derived from data already stored (no new
    collection):

    - live_directional_hit_rate_pct: every resolved (hit/miss, expired excluded -- same
      reasoning as directional_hit_rate_pct above) live RLSignal, grouped by the calendar day
      its outcome actually resolved (outcome_timestamp), not when it was generated -- this is
      "how good were the agent's real decisions that got proven right or wrong on this day,"
      across every pair/interval combined.
    - avg_training_hit_rate_pct / avg_training_return_pct: every RL training run
      (BacktestRun, profile="rl_ppo"), grouped by the day it was trained, averaged across
      whichever pair/interval combos got (re)trained that day -- "how good did the agent's
      own test-slice evaluation look on the policies produced this day." Each PPO retrain
      now warm-starts from the prior usable policy for that pair/interval when one exists
      (see _find_warm_start_policy) -- a rising trend here can reflect genuine accumulated
      weight refinement, not just improving training data/setup, so don't read this as
      "the underlying setup got better" the way the retired linear policy's version of this
      trend could be read; a fresh (non-warm-started) run happens automatically whenever the
      prior policy was stale or degenerated to always-HOLD, so the mix isn't uniform day to
      day either.

    Aggregated in Python, not a Mongo pipeline -- data volume here (signals/training runs
    over `days` days) is small enough that this is simpler to read and maintain, consistent
    with how GET /rl/accuracy above already prefers plain queries over aggregation.
    """
    since = datetime.utcnow() - timedelta(days=days)

    live_cursor = rl_signals_collection.find({
        "source": "live", "status": {"$in": ["hit", "miss"]}, "outcome_timestamp": {"$gte": since},
    })
    live_by_day: dict[str, dict] = {}
    async for d in live_cursor:
        day = d["outcome_timestamp"].strftime("%Y-%m-%d")
        bucket = live_by_day.setdefault(day, {"hits": 0, "misses": 0})
        bucket["hits" if d["status"] == "hit" else "misses"] += 1

    train_cursor = backtest_runs_collection.find({
        "profile": "rl_ppo", "created_at": {"$gte": since}, "hit_rate_pct": {"$ne": None},
    })
    train_by_day: dict[str, dict] = {}
    async for d in train_cursor:
        day = d["created_at"].strftime("%Y-%m-%d")
        bucket = train_by_day.setdefault(day, {"hit_rates": [], "returns": [], "count": 0})
        bucket["hit_rates"].append(d["hit_rate_pct"])
        if d.get("total_return_pct") is not None:
            bucket["returns"].append(d["total_return_pct"])
        bucket["count"] += 1

    points = []
    for day in sorted(set(live_by_day) | set(train_by_day)):
        live = live_by_day.get(day, {"hits": 0, "misses": 0})
        decided = live["hits"] + live["misses"]
        train = train_by_day.get(day)
        points.append({
            "date": day,
            "live_directional_hit_rate_pct": round(live["hits"] / decided * 100, 1) if decided else None,
            "live_decided_trades": decided,
            "avg_training_hit_rate_pct": round(sum(train["hit_rates"]) / len(train["hit_rates"]), 1) if train else None,
            "avg_training_return_pct": (
                round(sum(train["returns"]) / len(train["returns"]), 1) if train and train["returns"] else None
            ),
            "policies_trained": train["count"] if train else 0,
        })

    verdict = {
        # pool's second element must already be on the SAME 0-100 percentage scale as the
        # training case below (which sums hit_rate_pct values directly) -- hits*100 here, not
        # bare hits, so first_pos/first_n in _half_split_verdict comes out as a percentage in
        # both cases instead of a 0-1 fraction for live and a 0-100 percentage for training.
        "live": _half_split_verdict(
            sorted(live_by_day.items()),
            pool=lambda bucket: (bucket["hits"] + bucket["misses"], bucket["hits"] * 100),
            min_n=MIN_TRADES_PER_VERDICT_HALF, improve_delta=5.0,
        ),
        "training": _half_split_verdict(
            sorted(train_by_day.items()),
            pool=lambda bucket: (len(bucket["hit_rates"]), sum(bucket["hit_rates"])),
            min_n=MIN_POLICIES_PER_VERDICT_HALF, improve_delta=2.0,
        ),
    }
    return {"days": days, "points": points, "verdict": verdict}


# Minimum pooled sample size each half of the verdict comparison needs before _half_split_verdict
# will call it "improving"/"declining" rather than "not_enough_data" -- live trades are inherently
# scarce (a handful/day across all 20 pair/intervals combined) so a single noisy day (or even a
# handful) swinging the number is exactly the failure mode this guards against; a real verdict needs
# real volume behind both halves of the comparison. Training runs are far more plentiful (up to 20
# per retrain cycle), so that threshold is set higher in absolute terms but is still cheap to clear.
MIN_TRADES_PER_VERDICT_HALF = 15
MIN_POLICIES_PER_VERDICT_HALF = 20


def _half_split_verdict(
    day_buckets: list[tuple[str, dict]], pool, min_n: int, improve_delta: float,
) -> dict:
    """
    Splits chronologically-sorted (date, bucket) pairs into an earlier and a later half, pools
    each half's raw counts (not an average of daily percentages -- averaging percentages would
    let a 1-trade day and an 18-trade day sway the result equally, which is exactly why the live
    number looks noisy day-to-day even while the underlying policies are genuinely improving) via
    the caller-supplied `pool(bucket) -> (n, positive_count)`, and reports whether the rate moved
    by at least `improve_delta` percentage points between halves. Returns "not_enough_data"
    instead of a real verdict if either half falls short of `min_n` pooled samples -- a confident-
    sounding "declining" built on 3 trades would be actively misleading, not just imprecise.
    """
    if not day_buckets:
        return {
            "status": "not_enough_data", "first_half_rate_pct": None, "second_half_rate_pct": None,
            "first_half_n": 0, "second_half_n": 0,
        }
    mid = len(day_buckets) // 2
    first_n, first_pos = (lambda ns: (sum(n for n, _ in ns), sum(p for _, p in ns)))(
        [pool(b) for _, b in day_buckets[:mid]]
    )
    second_n, second_pos = (lambda ns: (sum(n for n, _ in ns), sum(p for _, p in ns)))(
        [pool(b) for _, b in day_buckets[mid:]]
    )
    first_rate = round(first_pos / first_n, 1) if first_n else None
    second_rate = round(second_pos / second_n, 1) if second_n else None
    status = "not_enough_data"
    if first_n >= min_n and second_n >= min_n and first_rate is not None and second_rate is not None:
        delta = second_rate - first_rate
        status = "improving" if delta >= improve_delta else "declining" if delta <= -improve_delta else "flat"
    return {
        "status": status, "first_half_rate_pct": first_rate, "second_half_rate_pct": second_rate,
        "first_half_n": first_n, "second_half_n": second_n,
    }


# Thresholds for GET /rl/insights below -- starting guesses, not independently tuned, same
# caveat as every other unvalidated constant in this project. MIN_SAMPLE_FOR_INSIGHT gates
# every per-pair/interval finding (not just the live-resolution ones) so a combo with only a
# handful of live signals doesn't generate a confident-sounding finding off noise.
MIN_SAMPLE_FOR_INSIGHT = 15
STRONG_LOSS_RETURN_PCT = -30.0
STRONG_WIN_RETURN_PCT = 30.0
STRONG_WIN_MIN_HIT_RATE_PCT = 40.0
HIGH_SUPERSEDE_FRACTION = 0.4
HIGH_EXPIRED_FRACTION = 0.8
LIVE_VS_TRAINED_GAP_PCT = 15.0  # same margin DEGRADATION_MARGIN_PCT below uses for auto-retrain


def _finding(severity: str, title: str, detail: str, pair: str | None = None, interval: str | None = None) -> dict:
    """Constructs through RLInsightFinding (not a bare dict literal) so a typo in `severity`
    or a missing field fails loudly here instead of silently reaching the frontend."""
    return RLInsightFinding(severity=severity, pair=pair, interval=interval, title=title, detail=detail).model_dump()


@app.get("/rl/insights")
async def rl_insights():
    """
    Deterministic, explainable "what to improve" findings -- not an LLM call, a fixed set of
    rules scanning data this project already collects: every pair/interval's latest trained
    policy and its own backtest eval, that pair/interval's live resolution breakdown
    (hit/miss/expired/superseded), and the overall learning-curve verdict (see
    _half_split_verdict above, reused directly so this doesn't duplicate that logic or drift
    out of sync with it). Same "explainable, not black-box" ethos as the rule engine's
    SignalReason and the ML classifier's feature_importances -- every finding here traces
    back to a specific number, not a model's opaque judgment call.

    Findings are sorted critical-first, then warning, then good -- what needs attention
    surfaces at the top.
    """
    findings: list[dict] = []

    for pair in settings.pairs_list:
        for interval in RL_INTERVALS:
            label = f"{pair} {interval}"

            policy_doc = await ppo_policies_collection.find_one(
                {"pair": pair, "interval": interval}, sort=[("created_at", -1)],
            )
            if policy_doc is not None:
                eval_run = await backtest_runs_collection.find_one({"run_id": policy_doc.get("eval_run_id")})
                hit_rate = eval_run.get("hit_rate_pct") if eval_run else None
                total_return = eval_run.get("total_return_pct") if eval_run else None

                if total_return is not None and total_return <= STRONG_LOSS_RETURN_PCT:
                    findings.append(_finding(
                        "critical", f"{label}: training shows a losing edge",
                        f"Latest retrain's test-slice walk lost {abs(total_return):.0f}% of a "
                        f"simulated $50 start (hit rate {hit_rate}%). Consider excluding this "
                        f"pair/interval from live signals, or capping it to the SMALL size "
                        f"tier, until this turns around.",
                        pair=pair, interval=interval,
                    ))
                elif total_return is not None and total_return >= STRONG_WIN_RETURN_PCT and (hit_rate or 0) >= STRONG_WIN_MIN_HIT_RATE_PCT:
                    findings.append(_finding(
                        "good", f"{label}: strongest performer",
                        f"Latest retrain grew a simulated $50 start by {total_return:.0f}% at a "
                        f"{hit_rate}% hit rate -- your most reliable edge right now.",
                        pair=pair, interval=interval,
                    ))
            else:
                hit_rate = None

            query = {"pair": pair, "interval": interval, "source": "live"}
            hits = await rl_signals_collection.count_documents({**query, "status": "hit"})
            misses = await rl_signals_collection.count_documents({**query, "status": "miss"})
            expired = await rl_signals_collection.count_documents({**query, "status": "expired"})
            superseded = await rl_signals_collection.count_documents({**query, "status": "superseded"})
            decided = hits + misses
            total_all = decided + expired + superseded

            if total_all >= MIN_SAMPLE_FOR_INSIGHT:
                if superseded / total_all >= HIGH_SUPERSEDE_FRACTION:
                    findings.append(_finding(
                        "warning", f"{label}: policy keeps changing its mind",
                        f"{round(superseded / total_all * 100)}% of signals were superseded "
                        f"before ever resolving -- the agent rarely lets a decision play out. "
                        f"Often means it hasn't converged yet; more training episodes may help.",
                        pair=pair, interval=interval,
                    ))
                if expired / total_all >= HIGH_EXPIRED_FRACTION:
                    findings.append(_finding(
                        "warning", f"{label}: signals rarely reach a real outcome",
                        f"{round(expired / total_all * 100)}% of resolved signals timed out "
                        f"without hitting target or stop. On fast intervals this is often the "
                        f"fixed spread cost eating most of the typical move -- worth revisiting "
                        f"target/stop sizing for this interval specifically.",
                        pair=pair, interval=interval,
                    ))

            if decided >= MIN_SAMPLE_FOR_INSIGHT and hit_rate is not None:
                live_rate = round(hits / decided * 100, 1)
                if hit_rate - live_rate >= LIVE_VS_TRAINED_GAP_PCT:
                    findings.append(_finding(
                        "warning", f"{label}: live results lagging the training claim",
                        f"Trained policy's own backtest claimed {hit_rate}% hit rate; live is "
                        f"running {live_rate}% over {decided} resolved trades so far. This "
                        f"already auto-triggers a background retrain -- flagged here so it's "
                        f"visible, not silent.",
                        pair=pair, interval=interval,
                    ))

    learning = await rl_learning_curve(days=30)
    live_v, train_v = learning["verdict"]["live"], learning["verdict"]["training"]
    if train_v["status"] in ("improving", "declining"):
        findings.append(_finding(
            "good" if train_v["status"] == "improving" else "critical",
            f"Overall training quality is {train_v['status']}",
            f"{train_v['first_half_rate_pct']}% -> {train_v['second_half_rate_pct']}% hit rate "
            f"across the earlier vs later half of the last 30 days ({train_v['first_half_n']} vs "
            f"{train_v['second_half_n']} pooled policy runs).",
        ))
    if live_v["status"] in ("improving", "declining"):
        findings.append(_finding(
            "good" if live_v["status"] == "improving" else "warning",
            f"Overall live trading is {live_v['status']}",
            f"{live_v['first_half_rate_pct']}% -> {live_v['second_half_rate_pct']}% hit rate "
            f"across the earlier vs later half of the last 30 days ({live_v['first_half_n']} vs "
            f"{live_v['second_half_n']} decided trades).",
        ))

    if not findings:
        findings.append(_finding(
            "warning", "Not enough data for real findings yet",
            "Keep training and letting live signals resolve -- most findings need at least "
            f"{MIN_SAMPLE_FOR_INSIGHT} resolved/decided signals per pair/interval.",
        ))

    order = {"critical": 0, "warning": 1, "good": 2}
    findings.sort(key=lambda f: order[f["severity"]])
    return {"generated_at": datetime.utcnow(), "findings": findings}


# How far live directional accuracy is allowed to fall below what the currently active
# policy's own training-time backtest eval claimed before that's treated as real degradation
# rather than ordinary live/backtest gap (this project's own /signals/accuracy docstring
# already notes live performance "often won't match [backtest] at first, and that gap is
# itself useful signal, not a bug to explain away" -- some gap is normal, this margin is
# meant to catch something bigger than that). Starting guess, not independently tuned -- same
# caveat as every other unvalidated constant in this project; revisit once enough triggered
# retrains exist to check whether this margin catches real regressions without false-triggering
# on ordinary noise.
DEGRADATION_MARGIN_PCT = 15.0
# Below this many live decided (hit+miss) trades, a live hit rate is too small a sample to
# compare against a trained backtest number at all -- skip the check entirely rather than
# risk retraining off 2-3 lucky/unlucky trades.
MIN_LIVE_DECIDED_FOR_DEGRADATION_CHECK = 10


async def _retrain_degraded_policy_background(pair: str, interval: str) -> None:
    """
    BackgroundTasks target for a degradation-triggered retrain -- a thin wrapper around
    _run_rl_training so a failure here (e.g. a transient DB hiccup) doesn't surface as an
    unhandled exception in server logs with no useful destination; same "don't let a
    best-effort background job crash noisily" reasoning every per-combo try/except in this
    file uses. Nothing to report the error TO here (unlike train-all, there's no job
    document this is updating) -- silently skipping means it simply gets caught again by
    the NEXT /rl/score cycle's degradation check, on already-fresh data.

    Deliberately NOT delegated to the GitHub Actions workflow the way POST /rl/train-all and
    _run_all_flows_job's RL-training phase now are (see PROGRESS.md's 2026-09-10 entry) --
    this is a single combo (60-90s), not a 20-combo/~20-30min batch, so the in-process
    recycle risk that motivated moving those two is far smaller here; revisit if this ever
    proves otherwise in practice.
    """
    try:
        await _run_rl_training(pair, interval, 50_000, 0.7, 20, DEFAULT_STARTING_BALANCE)
    except ValueError:
        pass


async def _check_and_retrain_degraded_policies(background_tasks: BackgroundTasks) -> list[dict]:
    """
    The training-side close of the memory-gate loop: if a pair/interval's LIVE directional
    hit rate has fallen DEGRADATION_MARGIN_PCT or more below what its own currently active
    policy claimed during its training-time backtest eval, kick off a fresh retrain in the
    background right now, instead of waiting for the next scheduled once-daily training slot.
    Checks every pair/interval combination every time this runs (cheap -- a handful of
    count_documents calls each, no heavy computation), not just ones that had a signal
    resolve this cycle, so a policy that's been quietly degrading for a while still gets
    caught the next time /rl/score fires.

    Purely additive to the once-daily cron training -- this can only trigger EXTRA retrains
    sooner, never skip or replace the scheduled one.
    """
    results = []
    for pair in settings.pairs_list:
        for interval in RL_INTERVALS:
            live_hits = await rl_signals_collection.count_documents(
                {"pair": pair, "interval": interval, "status": "hit"}
            )
            live_misses = await rl_signals_collection.count_documents(
                {"pair": pair, "interval": interval, "status": "miss"}
            )
            decided = live_hits + live_misses
            if decided < MIN_LIVE_DECIDED_FOR_DEGRADATION_CHECK:
                continue

            policy_doc = await ppo_policies_collection.find_one(
                {"pair": pair, "interval": interval}, sort=[("created_at", -1)],
            )
            if policy_doc is None:
                continue
            eval_run = await backtest_runs_collection.find_one({"run_id": policy_doc.get("eval_run_id")})
            trained_hit_rate_pct = eval_run.get("hit_rate_pct") if eval_run else None
            if trained_hit_rate_pct is None:
                continue

            live_hit_rate_pct = round(live_hits / decided * 100, 1)
            gap = trained_hit_rate_pct - live_hit_rate_pct
            entry = {
                "pair": pair, "interval": interval,
                "live_hit_rate_pct": live_hit_rate_pct, "live_decided_trades": decided,
                "trained_hit_rate_pct": trained_hit_rate_pct, "retrain_triggered": False,
            }
            if gap >= DEGRADATION_MARGIN_PCT:
                background_tasks.add_task(_retrain_degraded_policy_background, pair, interval)
                entry["retrain_triggered"] = True
            results.append(entry)
    return results


@app.post("/rl/score")
async def score_rl_signals(background_tasks: BackgroundTasks, max_lookforward: int | None = None):
    """
    Closes the loop for live RL signals the same way /signals/score and /consensus/score do
    for their own collections -- the live, forward-going half of "backtest its signals and
    learn" (the historical-replay half is POST /rl/train itself).

    max_lookforward left unset (the default, what the cron always uses) gives each signal its
    own interval-appropriate window instead of one flat candle count for every interval -- see
    outcome_scoring.LIVE_MAX_LOOKFORWARD_BY_INTERVAL.

    Also runs _check_and_retrain_degraded_policies after scoring -- see that function's own
    docstring. Its results are returned under `degradation_check` alongside the usual scoring
    tally so a caller (the cron logs, or a human) can see whether anything got flagged without
    needing a separate endpoint.
    """
    tally = await score_pending_rl_signals(max_lookforward=max_lookforward)
    degradation_check = await _check_and_retrain_degraded_policies(background_tasks)
    return {**tally, "degradation_check": degradation_check}


@app.get("/paper-trade/account")
async def paper_trade_account():
    """
    Authorizes against Deriv with the configured token and returns which account
    it resolved to — use this to confirm you're wired up to a DEMO account before
    running any paper trade. Places no trade.
    """
    try:
        async with deriv_session() as (_api, account):
            return {
                "loginid": account.get("loginid"),
                "is_virtual": bool(account.get("is_virtual")),
                "currency": account.get("currency"),
                "balance": account.get("balance"),
            }
    except DerivAuthError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/paper-trade/{interval}/{profile}")
async def paper_trade(
    interval: str,
    profile: str,
    pair: str,
    stake: float = 10.0,
    multiplier: int = 100,
    target_atr_mult: Optional[float] = None,
    stop_atr_mult: Optional[float] = None,
):
    """
    Generates a fresh live signal from stored candles and, if directional, executes
    it as a Deriv Multipliers contract on your DEMO account. Refuses to run at all
    if the authorized Deriv account isn't virtual (see deriv_client.deriv_session).

    pair: query param (e.g. ?pair=EUR/USD) — a path param would break on the literal '/'.
    stake: amount risked, in your Deriv account's currency.
    multiplier: leverage factor Deriv applies to price moves (must be one of the
    values Deriv allows for this symbol — an invalid value is rejected by Deriv itself).
    target_atr_mult/stop_atr_mult: explicit override — omit to use the profile's own
    config values.
    """
    config = default_config_for(profile, pair)
    cursor = candles_collection.find({"pair": pair, "interval": interval}).sort("timestamp", -1).limit(500)
    docs = await cursor.to_list(length=500)
    docs.reverse()  # find() gave newest-first for the limit to bite correctly; generate_signal wants ascending

    min_needed = config.ema_slow
    if len(docs) < min_needed:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough candle history ({len(docs)} rows). Need {min_needed}+ ({profile} profile). "
                    f"Run /ingest/{interval} first.",
        )

    df = pd.DataFrame(docs)
    signal = generate_signal(df, pair, interval, profile, config)
    await signals_collection.insert_one(signal.model_dump())

    if signal.direction == "HOLD":
        return {"signal": signal, "paper_trade": None, "note": "Signal was HOLD — nothing executed."}

    atr_val = float(compute_atr_series(df, config.atr_period).iloc[-1])
    signal.target_price, signal.stop_price = compute_atr_target_stop(
        signal.price_at_signal, atr_val, signal.direction,
        target_atr_mult if target_atr_mult is not None else config.target_atr_mult,
        stop_atr_mult if stop_atr_mult is not None else config.stop_atr_mult,
    )

    try:
        trade = await execute_paper_trade(signal, stake=stake, multiplier=multiplier)
    except DerivAuthError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    await paper_trades_collection.insert_one(trade.model_dump())
    return {"signal": signal, "paper_trade": trade}


@app.get("/paper-trade/open")
async def list_open_paper_trades():
    """Refreshes each open trade's contract status from Deriv, then returns the list."""
    cursor = paper_trades_collection.find({"status": "open"}).sort("opened_at", -1)
    docs = await cursor.to_list(length=None)

    updated = []
    for doc in docs:
        refreshed = await sync_open_trade(doc)
        if refreshed is not doc:
            await paper_trades_collection.replace_one({"_id": doc["_id"]}, refreshed)
        refreshed["_id"] = str(refreshed["_id"])
        updated.append(refreshed)
    return updated


@app.get("/paper-trade/history")
async def list_paper_trade_history(status: str | None = None, limit: int = 100):
    query = {"status": {"$in": ["won", "lost", "error"]}}
    if status:
        query = {"status": status}
    cursor = paper_trades_collection.find(query).sort("opened_at", -1).limit(limit)
    docs = await cursor.to_list(length=limit)
    for d in docs:
        d["_id"] = str(d["_id"])
    return docs


async def _run_all_flows_step(job_id: str, results: dict, step_label: str) -> bool:
    """
    Shared checkpoint after each unit of work in _run_all_flows_job: persists progress
    (current_step/completed_steps/results so far) and reports whether the job should keep
    going. Every write is filtered on status="running" so a force-cancelled job can't be
    resurrected by writes that happen to still be in flight when the cancel lands -- same
    guard run_train_all_job uses. Returns False (caller should stop) if the job was
    cancelled or the doc has vanished/changed status underneath it.
    """
    doc = await run_all_flows_jobs_collection.find_one_and_update(
        {"job_id": job_id, "status": "running"},
        {"$set": {"current_step": step_label, "results": results}, "$inc": {"completed_steps": 1}},
    )
    if doc is None:
        return False  # already cancelled (or otherwise no longer running) -- stop here
    return not doc.get("cancel_requested")


async def _run_all_flows_job(job_id: str) -> None:
    """
    The actual work behind POST /ops/run-all-flows, run as a background task -- see that
    endpoint's docstring for what this replicates. Runs as a
    BackgroundTasks target (started right after the job doc is created and the response
    already sent) specifically because the full sequence can take well over Cloudflare's
    ~100s proxy timeout, the same 524 this project already hit with a single /rl/train call
    -- a synchronous request/response here would just be a bigger version of that same
    problem, not a fix. Same "persist a job doc, poll it" pattern POST /rl/train-all uses.

    Writes progress after every checkpoint (_run_all_flows_step) and wraps the entire body
    in try/except -- found live: users reported this "sometimes getting stuck or not
    finishing," and the original version only ever wrote to Mongo once, at the very end, with
    no top-level exception handling. Either a genuinely slow run OR an outright crash inside
    it (an exception escaping every individual try/except below -- e.g. a bug in the newer
    case-memory lookups this job also exercises via create_rl_signal) looked EXACTLY the
    same from the outside: the job doc stuck at status="running" forever, no way to tell
    which had happened or to do anything about it.

    Includes RL training now (previously deliberately excluded -- see git history for why),
    per explicit user request that "Sync now" mean everything: backfill, retrain, and
    regenerate signals in one action, not ingest-and-signals with training left as a separate
    manual step. Placed right after ingest and before live signal generation specifically so
    the RL signals this same run produces come from the freshly-trained policy, not the
    previous one.
    """
    results: dict = {
        "ingest": {}, "rl_training": {}, "signals": {}, "consensus": {}, "rl_signals": {}, "score": {}, "ml_train": None,
    }
    # 5 ingest checkpoints + 20 RL-training checkpoints + 20 pair/interval checkpoints
    # (signals+consensus+rl_signals together count as one unit of progress each) + 3 score
    # checkpoints + 1 ml_train.
    total_steps = len(RL_INTERVALS) + 2 * (len(RL_INTERVALS) * len(settings.pairs_list)) + 3 + 1
    await run_all_flows_jobs_collection.update_one(
        {"job_id": job_id, "status": "running"}, {"$set": {"total_steps": total_steps}}
    )

    try:
        for i, interval in enumerate(RL_INTERVALS):
            if i > 0:
                # A normal scheduled cron tick only ever ingests ONE interval group at a time
                # (see keep-fresh.yml's per-interval `if` gates) -- this job is the one path
                # that deliberately ingests every interval in one go ("catch everything up
                # now"), which is exactly what produced a live 429 from Twelve Data the first
                # time this ran (5 intervals x 4 pairs = 20 requests fired back-to-back with
                # zero spacing, landing right after an earlier manual workflow_dispatch had
                # just done the same thing). This doesn't change the daily credit cost (still
                # 1 credit/pair/interval either way) -- it only paces out the per-minute
                # REQUEST rate this job itself generates, unrelated to whatever the normal
                # cron happens to be doing at the same time.
                await asyncio.sleep(8)
            try:
                results["ingest"][interval] = await ingest(interval, output_size=5)
            except Exception as e:
                results["ingest"][interval] = f"error: {e}"
            if not await _run_all_flows_step(job_id, results, f"ingest {interval}"):
                return

        # Delegates to the same GitHub Actions workflow POST /rl/train-all uses (see
        # _trigger_rl_train_workflow and PROGRESS.md's 2026-09-10 entry) instead of calling
        # _run_rl_training in-process -- this loop used to be the other in-process,
        # BackgroundTasks-scheduled 20-combo PPO training path, carrying the same "may not
        # survive an instance recycle" risk /rl/train-all had. Polling rl_train_jobs_collection
        # every 15s is cheap (a single-doc lookup, no CPU-heavy work) so it doesn't reintroduce
        # that risk itself -- and even if THIS job's polling loop were interrupted by a recycle,
        # the actual training keeps running on the GitHub Actions runner and its results still
        # land in ppo_policies_collection regardless, unlike the old in-process version where an
        # interrupted loop meant losing whatever combo was mid-training.
        rl_job_id = uuid.uuid4().hex[:12]
        await _trigger_rl_train_workflow(rl_job_id, 50_000, 0.7, DEFAULT_STARTING_BALANCE)
        seen = 0
        while True:
            await asyncio.sleep(15)
            rl_job_doc = await rl_train_jobs_collection.find_one({"job_id": rl_job_id})
            if not rl_job_doc:
                continue  # scripts/train_rl.py hasn't created its job doc yet -- keep waiting
            cancelled = False
            for cell in rl_job_doc.get("results", [])[seen:]:
                key = f"{cell['pair']}/{cell['interval']}"
                results["rl_training"][key] = (
                    {
                        "policy_id": cell.get("policy_id"), "hit_rate_pct": cell.get("hit_rate_pct"),
                        "total_return_pct": cell.get("total_return_pct"),
                    } if cell.get("ok") else f"error: {cell.get('error')}"
                )
                if not await _run_all_flows_step(job_id, results, f"train {key}"):
                    cancelled = True
                    break
            seen = len(rl_job_doc.get("results", []))
            if cancelled:
                # Cancelling THIS run-all-flows job should also stop the still-running GitHub
                # Actions training run -- same status="running" write guard/cooperative-cancel
                # mechanism POST /rl/train-all/{job_id}/cancel already uses; scripts/train_rl.py
                # checks this before starting each remaining combo.
                await rl_train_jobs_collection.update_one(
                    {"job_id": rl_job_id, "status": "running"},
                    {"$set": {"status": "cancelled", "finished_at": datetime.utcnow()}},
                )
                return
            if rl_job_doc.get("status") != "running":
                break

        for interval in RL_INTERVALS:
            profile = "intraday"
            for pair in settings.pairs_list:
                key = f"{pair}/{interval}"
                try:
                    await create_signal(interval, profile, pair)
                    results["signals"][key] = "ok"
                except Exception as e:
                    results["signals"][key] = f"error: {e}"
                try:
                    await create_consensus_signal(interval, pair)
                    results["consensus"][key] = "ok"
                except Exception as e:
                    results["consensus"][key] = f"error: {e}"
                try:
                    await create_rl_signal(interval, pair)
                    results["rl_signals"][key] = "ok"
                except Exception as e:
                    results["rl_signals"][key] = f"error: {e}"
                if not await _run_all_flows_step(job_id, results, f"signals {key}"):
                    return

        try:
            results["score"]["signals"] = await score_signals()
        except Exception as e:
            results["score"]["signals"] = f"error: {e}"
        if not await _run_all_flows_step(job_id, results, "score signals"):
            return

        try:
            results["score"]["consensus"] = await score_consensus_signals()
        except Exception as e:
            results["score"]["consensus"] = f"error: {e}"
        if not await _run_all_flows_step(job_id, results, "score consensus"):
            return

        try:
            bg = BackgroundTasks()
            results["score"]["rl"] = await score_rl_signals(bg)
            await bg()  # run any degradation-triggered retrains synchronously, same call
        except Exception as e:
            results["score"]["rl"] = f"error: {e}"
        if not await _run_all_flows_step(job_id, results, "score rl"):
            return

        try:
            results["ml_train"] = await train_ml_model()
        except Exception as e:
            results["ml_train"] = f"error: {e}"
        if not await _run_all_flows_step(job_id, results, "ml_train"):
            return

        await run_all_flows_jobs_collection.update_one(
            {"job_id": job_id, "status": "running"},
            {"$set": {"status": "done", "finished_at": datetime.utcnow(), "results": results, "current_step": None}},
        )
    except Exception as e:
        # Something escaped every individual try/except above -- still resolve the job
        # instead of leaving it "running" forever with no explanation.
        results["fatal_error"] = str(e)
        await run_all_flows_jobs_collection.update_one(
            {"job_id": job_id, "status": "running"},
            {"$set": {"status": "done", "finished_at": datetime.utcnow(), "results": results, "current_step": None}},
        )


@app.post("/ops/run-all-flows")
async def run_all_flows(background_tasks: BackgroundTasks):
    """
    Manually replicates a full catch-up cycle -- ingest every interval, retrain every RL
    policy, generate signals (regular + consensus + RL) across every pair/interval, score
    all three, retrain the ML classifier -- as a background job, since the full sequence can
    take well over a minute (see _run_all_flows_job's own docstring). Returns job_id
    immediately; poll GET /ops/run-all-flows/{job_id} for status/results.

    Includes RL training across all 20 pair/interval combos (the same work POST
    /rl/train-all does, ~20-30 min alone) -- by explicit user request that "Sync now" mean
    everything, not ingest-and-signals with training left as a separate manual step. This
    makes the whole job noticeably longer than keep-fresh.yml's own per-cycle work (which
    only trains once daily); current_step/completed_steps/total_steps exist specifically so
    that length is visible progress, not an indistinguishable-from-stuck wait.

    For when the GitHub Actions cron has gone quiet for a while (its own scheduler isn't
    always reliable under load -- confirmed live 2026-08-27, a ~5hr gap with zero runs despite
    an active */20 schedule, see CLAUDE.md) and someone wants today's candles/signals caught
    up right now instead of waiting on it or triggering the workflow on GitHub directly.

    Every step inside the job is individually try/excepted so one pair/interval/step failing
    (a Twelve Data quota hit, "no trained policy yet", etc.) doesn't stop the rest from
    running -- same tolerance keep-fresh.yml's own `|| true` steps already have. Ingest uses
    output_size=5 (a light top-up, matching the cron's own per-cycle amount), not a deep
    backfill.
    """
    job_id = uuid.uuid4().hex[:12]
    job = RunAllFlowsJob(job_id=job_id, status="running", created_at=datetime.utcnow())
    await run_all_flows_jobs_collection.insert_one(job.model_dump())
    background_tasks.add_task(_run_all_flows_job, job_id)
    return {"job_id": job_id, "status": "running"}


@app.get("/ops/run-all-flows/{job_id}")
async def get_run_all_flows_job(job_id: str):
    doc = await run_all_flows_jobs_collection.find_one({"job_id": job_id})
    if doc is None:
        raise HTTPException(status_code=404, detail=f"No run-all-flows job found with job_id={job_id}")
    doc["_id"] = str(doc["_id"])
    return doc


@app.post("/ops/run-all-flows/{job_id}/cancel")
async def cancel_run_all_flows_job(job_id: str):
    """
    Immediately marks a running "Sync now" job cancelled -- not just a request the job picks
    up between checkpoints. Same reasoning as POST /rl/train-all/{job_id}/cancel: a step that
    hangs outright (rather than just running long) means the background task's own coroutine
    never comes back around to check a flag, so this updates the job document directly. If
    the stuck step eventually finishes on its own after this, its leftover write is a no-op
    -- see _run_all_flows_step's status="running" write guard.
    """
    doc = await run_all_flows_jobs_collection.find_one({"job_id": job_id})
    if doc is None:
        raise HTTPException(status_code=404, detail=f"No run-all-flows job found with job_id={job_id}")
    if doc["status"] != "running":
        raise HTTPException(status_code=400, detail=f"Job {job_id} is already {doc['status']}, nothing to cancel.")
    await run_all_flows_jobs_collection.update_one(
        {"job_id": job_id, "status": "running"},
        {"$set": {"cancel_requested": True, "status": "cancelled", "finished_at": datetime.utcnow()}},
    )
    doc = await run_all_flows_jobs_collection.find_one({"job_id": job_id})
    doc["_id"] = str(doc["_id"])
    return doc
