from fastapi import BackgroundTasks, Body, FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from datetime import datetime
import uuid
import pandas as pd

from app.core.config import get_settings
from app.core.auth import AuthMiddleware, create_token, verify_credentials
from app.core.database import (
    init_indexes,
    candles_collection,
    signals_collection,
    backtest_signals_collection,
    backtest_runs_collection,
    paper_trades_collection,
    consensus_signals_collection,
    ml_runs_collection,
    rl_policies_collection,
    rl_signals_collection,
    rl_train_jobs_collection,
)
from app.models.schemas import LoginRequest, RuleConfig, RLPolicy, RLSignal, RLTrainAllJob, RLTrainAllCell
from app.services.data_fetcher import fetch_and_store
from app.services.indicators import atr as compute_atr_series, add_all_indicators
from app.services.signal_engine import generate_signal, compute_atr_target_stop, default_config_for, spread_cost_pct
from app.services.backtester import run_backtest, run_consensus_backtest
from app.services.strategies import STRATEGIES
from app.services.consensus import check_consensus
from app.services.ml_features import extract_features
from app.services.ml_model import train_hit_classifier, predict_hit_probability
from app.services.rl_engine import (
    train_rl_policy, choose_action, compute_strategy_vote_states, rl_config_profile, full_rl_state,
    position_size_units, rl_atr_mults, MIN_WARMUP_BARS, DEFAULT_STARTING_BALANCE,
    RISK_FRACTION_BY_TIER, DEFAULT_EPISODES, RL_FEATURE_NAMES,
)
from app.services.case_memory import (
    memory_summary, nearest_cases, explain_divergence, find_diverging_neighbor, RESOLVED_STATUSES,
)
from app.services.deriv_client import deriv_session, DerivAuthError
from app.services.paper_trading import execute_paper_trade, sync_open_trade
from app.services.outcome_scoring import score_pending_signals, score_pending_consensus_signals, score_pending_rl_signals

settings = get_settings()
app = FastAPI(title="Forex Trading Assistant")

# Default grid for /backtest/optimize when no configs are supplied — covers the knobs
# backtesting has actually shown to matter (EMA responsiveness, RSI sensitivity, and
# target/stop — RuleConfig's own 1.5x/1.0x default turned out to have negative expectancy
# on every EMA/RSI combination on every pair; a closer target with a more generous stop
# (0.5x/1.25x) is a validated, cross-pair-tested improvement, see signal_engine.PROFILE_DEFAULTS
# and README "Intraday vs swing"). Both target/stop ratios are included here so a plain
# "run optimize" has a chance to find the better one instead of only searching a doomed ratio.
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
    """
    results = {}
    for pair in settings.pairs_list:
        try:
            count = await fetch_and_store(pair, interval, output_size=output_size)
            results[pair] = f"{count} candles stored"
        except Exception as e:
            results[pair] = f"error: {e}"
    return results


@app.post("/signals/{interval}/{profile}")
async def create_signal(
    interval: str, profile: str, pair: str,
    target_atr_mult: Optional[float] = None, stop_atr_mult: Optional[float] = None,
):
    """
    Generate a signal for a pair using stored candle data.
    profile: 'intraday' or 'swing'. pair is a query param (e.g. ?pair=EUR/USD) —
    it contains a literal '/', which breaks Starlette path-parameter matching
    even when percent-encoded, so it can't live in the URL path.

    Directional signals get an ATR-based target_price/stop_price attached at creation
    time (same formula the backtester uses) — this is what /signals/score checks stored
    candles against later to resolve the signal's status from "pending" to hit/miss/expired.
    target_atr_mult/stop_atr_mult: explicit override; omit to use that profile's config
    values (RuleConfig.target_atr_mult/stop_atr_mult — see signal_engine.PROFILE_DEFAULTS).
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
async def get_candles(interval: str, pair: str, profile: str = "swing", limit: int = 300):
    """
    Stored candles with EMA/RSI/MACD/ATR attached, for charting — computed with that
    profile's RuleConfig (so the EMA/RSI periods shown match whatever generate_signal
    actually used for that profile). Returns the most recent `limit` candles, ascending
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
    # Consensus has no PROFILE_DEFAULTS entry of its own; swing's config (no session filter)
    # is the more neutral pick of the two since consensus isn't scoped to a session window.
    config = default_config_for("swing", pair)
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

    config = default_config_for("swing", pair)
    cursor = candles_collection.find({"pair": pair, "interval": interval}).sort("timestamp", 1)
    docs = await cursor.to_list(length=None)
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
    cursor = candles_collection.find({"pair": pair, "interval": interval}).sort("timestamp", 1)
    docs = await cursor.to_list(length=None)

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

    cursor = candles_collection.find({"pair": pair, "interval": interval}).sort("timestamp", 1)
    docs = await cursor.to_list(length=None)
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

    cursor = candles_collection.find({"pair": pair, "interval": interval}).sort("timestamp", 1)
    docs = await cursor.to_list(length=None)
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
    Trains the supervised hit/miss classifier (app/services/ml_model.py, LogisticRegression --
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
            last_run["_id"] = str(last_run["_id"])
            last_run["skipped"] = True
            return last_run

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


def _rl_state_from_candles(docs: list[dict], config: RuleConfig) -> tuple[list[float], pd.DataFrame]:
    """Shared by /rl/train and /rl/signal -- builds the indicator dataframe and the current
    (latest-bar) state vector the same way compute_strategy_vote_states does internally,
    without recomputing every prior bar's state just to read the last one."""
    df = pd.DataFrame(docs)
    indicator_df = add_all_indicators(df, config)
    states = compute_strategy_vote_states(indicator_df, config)
    return states[-1], df


RL_INTERVALS = ["5min", "15min", "1h", "4h", "1day"]


async def _run_rl_training(
    pair: str, interval: str, episodes: int, train_frac: float, max_lookforward: int, starting_balance: float,
    reset: bool = False,
):
    """
    Shared by POST /rl/train and the /rl/train-all background job below -- fetches candle
    history, runs training, and persists the policy/evaluation/trade log. Raises ValueError
    for the caller to turn into whatever error shape fits its own endpoint (a 400 for the
    single endpoint, a per-cell error string for the batch job).

    train_rl_policy itself is a blocking, CPU-bound pandas replay (~40-50s per pair/interval
    against the full backfilled history) -- run via run_in_threadpool so it doesn't tie up the
    event loop, which matters a lot more here than it did for a single call, since the batch
    job below calls this 20 times in a row and other requests (status polling, live signal
    generation, the cron) still need to get through during those ~15 minutes.

    Looks up this pair/interval's most recently persisted policy and passes it to
    train_rl_policy as warm_start (unless reset=True) so training continues from what the
    prior run learned instead of starting from zero weights every single time -- see
    train_rl_policy's own docstring for why this is what actually makes "keep training"
    accumulate instead of just re-fitting the same history repeatedly. Safe by construction:
    train_rl_policy itself falls back to a fresh run if the found policy's feature_names don't
    match the current schema, so this lookup never needs its own compatibility check.
    """
    if not 0 < train_frac < 1:
        raise ValueError("train_frac must be between 0 and 1 (exclusive).")
    if starting_balance <= 0:
        raise ValueError("starting_balance must be positive.")

    config = default_config_for(rl_config_profile(interval), pair)
    cursor = candles_collection.find({"pair": pair, "interval": interval}).sort("timestamp", 1)
    docs = await cursor.to_list(length=None)
    if not docs:
        raise ValueError(f"No candle history for {pair}/{interval}. Run /ingest/{interval} first.")

    warm_start = None
    if not reset:
        prior_doc = await rl_policies_collection.find_one(
            {"pair": pair, "interval": interval}, sort=[("created_at", -1)],
        )
        if prior_doc is not None:
            warm_start = RLPolicy(**{k: v for k, v in prior_doc.items() if k != "_id"})

    df = pd.DataFrame(docs)
    policy, eval_run, trade_signals = await run_in_threadpool(
        train_rl_policy, df, pair, interval, config, episodes=episodes, train_frac=train_frac,
        max_lookforward=max_lookforward, starting_balance=starting_balance, warm_start=warm_start,
    )

    # Individual test-slice trades reuse backtest_signals_collection (same as run_backtest's
    # own persistence) tagged with eval_run.run_id -- GET /backtest/runs/{run_id}/signals
    # already answers "which trades passed and which failed" for free, no new endpoint.
    if trade_signals:
        await backtest_signals_collection.insert_many([s.model_dump() for s in trade_signals])
    await backtest_runs_collection.insert_one(eval_run.model_dump())
    await rl_policies_collection.insert_one(policy.model_dump())
    return policy, eval_run


@app.post("/rl/train/{interval}")
async def train_rl(
    interval: str, pair: str, episodes: int = DEFAULT_EPISODES, train_frac: float = 0.7, max_lookforward: int = 20,
    starting_balance: float = DEFAULT_STARTING_BALANCE, reset: bool = False,
):
    """
    Trains a linear Q-policy (app/services/rl_engine.py) for this pair/interval via
    epsilon-greedy Q-learning over historical candle replay -- an adaptive alternative to
    consensus's fixed weighted-vote threshold, learning how to weight the same 7 strategies
    instead of using a hand-picked REQUIRED_WEIGHT_FRACTION, AND how much to risk on each
    trade (2 size tiers, see rl_engine.RISK_FRACTION_BY_TIER) against a compounding account
    balance starting at `starting_balance` (default $50). One independent policy per pair,
    not shared across pairs. Persists both the learned RLPolicy and its test-slice evaluation
    (a real BacktestRun, profile="rl", now including starting_balance/ending_balance/
    total_return_pct alongside the usual hit_rate/expectancy) so it's directly comparable to
    every other approach via GET /backtest/runs?pair=X&profile=rl.

    Strategy calls use the intraday RuleConfig (short EMAs + session filter) for 5min/15min
    and swing for everything else (rl_config_profile) -- same interval grouping the regular
    signal engine already uses, so a 5-minute chart isn't read with EMA periods tuned for
    multi-day trends.

    pair: query param (e.g. ?pair=EUR/USD) — a path param would break on the literal '/'.

    For training every pair x interval at once, see POST /rl/train-all instead -- doing that
    here by simply looping client-side is what used to make the frontend's "Train all" a
    ~15-minute sequence of fetches the browser tab had to hold open the whole time.

    By default this continues from this pair/interval's most recently persisted policy (warm
    start -- see train_rl_policy's docstring) rather than retraining from zero weights every
    call, so repeated training actually builds on prior runs instead of just re-fitting the
    same growing candle history from scratch each time. Pass reset=true to force a fresh
    zero-initialized run instead (e.g. after deliberately changing RL_ATR_MULTS_BY_PROFILE or
    another training-behavior constant, where continuing from the old policy's weights isn't
    desired even though the feature schema itself hasn't changed).
    """
    try:
        policy, eval_run = await _run_rl_training(
            pair, interval, episodes, train_frac, max_lookforward, starting_balance, reset=reset,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"policy": policy, "evaluation": eval_run}


async def run_train_all_job(job_id: str, episodes: int, train_frac: float, starting_balance: float):
    """
    The actual batch loop, scheduled via BackgroundTasks from POST /rl/train-all so it keeps
    running after that request has already returned -- trains every pair x interval
    combination sequentially (same set .github/workflows/keep-fresh.yml's cron trains once
    daily), writing progress to rl_train_jobs_collection after each combo so GET
    /rl/train-all/{job_id} always reflects real progress, not just "still running somewhere."

    Checks cancel_requested before starting each combo (cheap, single-doc lookup) so
    POST /rl/train-all/{job_id}/cancel can stop it -- cooperative, not preemptive: a combo
    already in flight always finishes (train_rl_policy can't be interrupted mid-call without
    much more complexity) UNLESS it hangs outright (seen in practice: a single combo running
    60+ minutes when the whole 20-combo job should take ~15-30), in which case this loop's own
    "next combo" check never runs at all. POST .../cancel therefore also force-marks the job
    cancelled immediately, not just requests it -- see that endpoint's docstring. Every write
    here is filtered on {"job_id": job_id, "status": "running"} specifically so that if a
    force-cancelled job's stuck combo eventually wakes up and finishes on its own, its
    leftover result/completion writes become no-ops instead of silently resurrecting a job
    the user already told to stop.
    """
    for pair in settings.pairs_list:
        for interval in RL_INTERVALS:
            job_doc = await rl_train_jobs_collection.find_one({"job_id": job_id}, {"status": 1})
            if not job_doc or job_doc.get("status") != "running":
                return  # already cancelled (cooperatively or forced) -- stop here
            try:
                policy, eval_run = await _run_rl_training(
                    pair, interval, episodes, train_frac, max_lookforward=20, starting_balance=starting_balance,
                )
                cell = RLTrainAllCell(
                    pair=pair, interval=interval, ok=True, policy_id=policy.policy_id,
                    hit_rate_pct=eval_run.hit_rate_pct, expectancy_pct=eval_run.expectancy_pct,
                    directional_signals=eval_run.directional_signals, hold_signals=eval_run.hold_signals,
                    starting_balance=eval_run.starting_balance, ending_balance=eval_run.ending_balance,
                    total_return_pct=eval_run.total_return_pct,
                )
            except Exception as e:
                cell = RLTrainAllCell(pair=pair, interval=interval, ok=False, error=str(e))
            await rl_train_jobs_collection.update_one(
                {"job_id": job_id, "status": "running"},
                {"$push": {"results": cell.model_dump()}, "$inc": {"completed": 1}},
            )
    await rl_train_jobs_collection.update_one(
        {"job_id": job_id, "status": "running"},
        {"$set": {"status": "done", "finished_at": datetime.utcnow()}},
    )


@app.post("/rl/train-all")
async def train_rl_all(
    background_tasks: BackgroundTasks,
    episodes: int = DEFAULT_EPISODES, train_frac: float = 0.7, starting_balance: float = DEFAULT_STARTING_BALANCE,
):
    """
    Starts training every pair x interval combination (the same set the daily cron trains) as
    a server-side background job and returns immediately with a job_id -- poll GET
    /rl/train-all/{job_id} for progress. Runs server-side specifically so the ~15-minute total
    duration (20 combos, ~40-50s each) doesn't depend on the triggering browser tab staying
    open, foregrounded, or connected the whole time; the previous frontend-driven version held
    20 sequential fetches open in the tab and a lost connection partway through (mobile screen
    lock, backgrounding, a network switch) would abandon the run silently.
    """
    if not 0 < train_frac < 1:
        raise HTTPException(status_code=400, detail="train_frac must be between 0 and 1 (exclusive).")
    if starting_balance <= 0:
        raise HTTPException(status_code=400, detail="starting_balance must be positive.")

    job_id = uuid.uuid4().hex[:12]
    job = RLTrainAllJob(
        job_id=job_id, status="running", created_at=datetime.utcnow(),
        episodes=episodes, train_frac=train_frac, starting_balance=starting_balance,
        total=len(settings.pairs_list) * len(RL_INTERVALS),
    )
    await rl_train_jobs_collection.insert_one(job.model_dump())
    background_tasks.add_task(run_train_all_job, job_id, episodes, train_frac, starting_balance)
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
    picks up between combos. Originally this only set cancel_requested and waited for
    run_train_all_job's own between-combo check, but that's a no-op if the in-flight combo
    hangs outright rather than just running long (seen in practice: a single pair/interval
    stuck for 60+ minutes with the whole job normally taking ~15-30). Since a hung combo
    means that background task's own coroutine will never come back around to check a flag,
    this endpoint updates the job document directly instead of waiting for it to.

    Already-completed combos and their trained policies are kept, not rolled back; only the
    remaining ones are skipped. If the stuck combo eventually finishes on its own after this,
    its leftover write is a no-op -- see run_train_all_job's status="running" write guard.
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
    """
    query = {}
    if pair:
        query["pair"] = pair
    if interval:
        query["interval"] = interval
    cursor = rl_policies_collection.find(query).sort("created_at", -1).limit(limit)
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


async def _supersede_pending_rl_signal(pending: dict, current_price: float) -> None:
    """
    Marks a pending RL signal "superseded" because the policy's live view has moved on (a
    newer decision at the same pair/interval disagrees with it), rather than leaving it to
    resolve naturally against label_outcome's live window (see
    outcome_scoring.LIVE_MAX_LOOKFORWARD_BY_INTERVAL -- several days at most intervals).
    This project treats "pending" as "still the agent's current live view" for RL signals
    specifically, since they're meant to be traded manually and soon -- a stale one hanging
    around for most of a day isn't a real trade opportunity anymore. The record is UPDATED,
    never deleted -- same "keep history, don't erase it" convention as every other status
    transition in this project.

    Deliberately a status distinct from "expired" (see RLSignal.status's docstring) -- this
    was originally folded into "expired" and it quietly wrecked GET /rl/accuracy: a policy
    that changes its mind often (most likely on the faster, less-converged 5min/15min
    intervals) superseded the large majority of its own signals well before label_outcome
    ever got a chance to judge them, making the agent look far less accurate than its actual
    resolved (hit/miss/genuinely-expired) trades show.
    """
    pct_move = ((current_price - pending["entry_price"]) / pending["entry_price"]) * 100
    if pending["direction"] == "SELL":
        pct_move = -pct_move
    pct_move -= spread_cost_pct(pending["pair"], pending["entry_price"])
    await rl_signals_collection.update_one(
        {"_id": pending["_id"]},
        {"$set": {
            "status": "superseded",
            "outcome_price": round(current_price, 5),
            "outcome_timestamp": datetime.utcnow(),
            "outcome_pct_move": round(pct_move, 5),
            "candles_to_outcome": None,  # superseded, not resolved by walking forward N candles
        }},
    )


@app.post("/rl/signal/{interval}")
async def create_rl_signal(interval: str, pair: str, balance: float = DEFAULT_STARTING_BALANCE):
    """
    Loads the most recently trained RLPolicy for this pair/interval, computes the current
    state from live strategy calls on the latest candles PLUS the supplied `balance` (same
    encoding used during training, via rl_engine.full_rl_state -- `balance` is your real
    current account balance, not a value this backend tracks itself, since there's no way to
    know whether a prior signal was actually taken or what it filled at on an external
    broker), and picks the greedy action. Only BUY/SELL get stored -- HOLD never produces an
    RLSignal, same as ConsensusSignal. The chosen size tier (SMALL/LARGE) is sized against
    `balance` via position_size_units, so the signal is always grounded in your actual
    account, not an assumed one.

    The response also includes `memory` (app/services/case_memory.py) -- how similar past
    resolved signals for this exact pair/interval actually turned out (hit rate among decided
    outcomes, average pct move, how many nearest historical cases were used), a k-nearest-
    neighbor lookup against every past state vector alongside whatever the linear Q-policy's
    own weights say. It's informational, not a gate on the trade this endpoint returns -- the
    point is making "have we been here before" inspectable, not silently overriding the
    policy's decision with a separate heuristic.

    If a different decision now disagrees with whatever RL signal is still "pending" for this
    pair/interval (new direction, new action is HOLD, or price has moved enough that the
    entry itself changed), that old pending signal is marked "superseded" immediately (see
    _supersede_pending_rl_signal) instead of being left to resolve on its own hours later --
    "pending" should mean "still the agent's current live view," not "might still resolve
    eventually." An exact repeat of the still-pending signal (same direction/entry/target) is
    left alone and returned as-is (same de-dupe idea create_consensus_signal already uses,
    intentionally not re-sizing an already-shown pending signal just because `balance`
    happened to differ on this call), so re-running this before the underlying candle has
    advanced doesn't spuriously expire-then-recreate it.

    pair: query param (e.g. ?pair=EUR/USD) — a path param would break on the literal '/'.
    """
    if balance <= 0:
        raise HTTPException(status_code=400, detail="balance must be positive.")

    policy_doc = await rl_policies_collection.find_one(
        {"pair": pair, "interval": interval}, sort=[("created_at", -1)],
    )
    if policy_doc is None:
        raise HTTPException(
            status_code=400,
            detail=f"No trained RL policy yet for {pair}/{interval}. Call POST /rl/train/{interval}?pair={pair} first.",
        )
    policy = RLPolicy(**{k: v for k, v in policy_doc.items() if k != "_id"})

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

    market_state, df = _rl_state_from_candles(docs, config)
    state = full_rl_state(market_state, balance, policy.starting_balance)
    try:
        action, q_values = choose_action(policy, state)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    current_price = float(df.iloc[-1]["close"])

    pending = await rl_signals_collection.find_one(
        {"pair": pair, "interval": interval, "status": "pending"}, sort=[("timestamp", -1)],
    )

    if action == "HOLD":
        if pending is not None:
            await _supersede_pending_rl_signal(pending, current_price)
        return {"signal": None, "q_values": q_values}

    direction, tier = action.split("_")
    risk_fraction = RISK_FRACTION_BY_TIER[tier]

    latest = df.iloc[-1]
    entry_price = current_price
    atr_val = float(compute_atr_series(df, config.atr_period).iloc[-1])
    target_atr_mult, stop_atr_mult = rl_atr_mults(rl_config_profile(interval))
    target_price, stop_price = compute_atr_target_stop(
        entry_price, atr_val, direction, target_atr_mult, stop_atr_mult,
    )
    units = position_size_units(balance, risk_fraction, entry_price, stop_price, pair)

    # "Have we seen a state like this before, and how did it actually turn out" -- see
    # case_memory.py. Computed against every resolved (never superseded) past RLSignal for
    # this exact pair/interval that has a stored state vector; empty/none for a brand new
    # pair/interval or one whose history predates the state field, same "surface as absent,
    # not a fabricated number" convention as ml_model's None-when-not-enough-data.
    memory_candidates = await rl_signals_collection.find(
        {"pair": pair, "interval": interval, "status": {"$in": list(RESOLVED_STATUSES)}}
    ).to_list(length=None)
    memory = memory_summary(state, memory_candidates)

    rl_signal = RLSignal(
        signal_id=uuid.uuid4().hex[:12],
        pair=pair, interval=interval, timestamp=latest["timestamp"], direction=direction,
        entry_price=entry_price, target_price=target_price, stop_price=stop_price,
        q_values=q_values, policy_id=policy.policy_id,
        size_tier=tier, risk_fraction=risk_fraction, balance_at_signal=round(balance, 2),
        position_size_units=round(units, 2), state=state,
    )

    if (
        pending is not None
        and pending["direction"] == rl_signal.direction
        and pending["entry_price"] == rl_signal.entry_price
        and pending["target_price"] == rl_signal.target_price
    ):
        # Exact repeat of the still-live signal -- nothing has changed, return it as-is.
        pending["_id"] = str(pending["_id"])
        return {"signal": pending, "q_values": q_values, "memory": memory}

    if pending is not None:
        # The agent's view has moved on (different direction or entry) -- the old signal is
        # no longer what the agent would trade right now.
        await _supersede_pending_rl_signal(pending, current_price)

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
    surfaces (SignalReason.detail, ML's feature_coefficients). Meant for reviewing a specific
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


@app.post("/rl/score")
async def score_rl_signals(max_lookforward: int | None = None):
    """
    Closes the loop for live RL signals the same way /signals/score and /consensus/score do
    for their own collections -- the live, forward-going half of "backtest its signals and
    learn" (the historical-replay half is POST /rl/train itself).

    max_lookforward left unset (the default, what the cron always uses) gives each signal its
    own interval-appropriate window instead of one flat candle count for every interval -- see
    outcome_scoring.LIVE_MAX_LOOKFORWARD_BY_INTERVAL.
    """
    return await score_pending_rl_signals(max_lookforward=max_lookforward)


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
