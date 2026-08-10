from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import pandas as pd

from app.core.config import get_settings
from app.core.database import (
    init_indexes,
    candles_collection,
    signals_collection,
    backtest_signals_collection,
    backtest_runs_collection,
    paper_trades_collection,
)
from app.models.schemas import RuleConfig
from app.services.data_fetcher import fetch_and_store
from app.services.indicators import atr as compute_atr_series, add_all_indicators
from app.services.signal_engine import generate_signal, compute_atr_target_stop, default_config_for
from app.services.backtester import run_backtest
from app.services.deriv_client import deriv_session, DerivAuthError
from app.services.paper_trading import execute_paper_trade, sync_open_trade
from app.services.outcome_scoring import score_pending_signals

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

    # The latest stored candle only advances when /ingest brings in a new one — calling
    # this endpoint again before that (e.g. every dashboard load) would otherwise insert
    # an identical duplicate for the same candle, inflating /signals/accuracy's counts.
    last = await signals_collection.find_one(
        {"pair": pair, "interval": interval, "profile": profile, "source": "live"},
        sort=[("timestamp", -1)],
    )
    if (
        last is not None
        and last["direction"] == signal.direction
        and last["price_at_signal"] == signal.price_at_signal
    ):
        last["_id"] = str(last["_id"])
        return last

    if signal.direction in ("BUY", "SELL"):
        atr_val = float(compute_atr_series(df, config.atr_period).iloc[-1])
        signal.target_price, signal.stop_price = compute_atr_target_stop(
            signal.price_at_signal, atr_val, signal.direction,
            target_atr_mult if target_atr_mult is not None else config.target_atr_mult,
            stop_atr_mult if stop_atr_mult is not None else config.stop_atr_mult,
        )

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
async def score_signals(max_lookforward: int = 20):
    """
    Checks every pending live signal against candles that have arrived since it fired,
    resolving status to hit/miss/expired wherever enough real data now exists — the live
    equivalent of what the backtester does against fixed history. Run this after each
    /ingest so newly-arrived candles get checked; the .github/workflows/keep-fresh.yml
    cron does both automatically every 15 minutes, this is for triggering it on demand.
    """
    return await score_pending_signals(max_lookforward=max_lookforward)


@app.get("/signals/accuracy")
async def signal_accuracy(pair: str | None = None, profile: str | None = None, limit: int = 100):
    """
    Rolling hit-rate over the most recent *resolved* live signals (hit/miss/expired) —
    excludes still-pending signals and anything from a backtest run. This is what tells
    you whether live performance is actually tracking what was backtested; it often won't
    match at first, and that gap is itself useful signal, not a bug to explain away.
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

    return {
        "pair": pair,
        "profile": profile,
        "sample_size": total,
        "hits": hits,
        "misses": misses,
        "expired": expired,
        "hit_rate_pct": round(hits / total * 100, 1) if total else None,
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
