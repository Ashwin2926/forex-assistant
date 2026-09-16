import uuid
import pandas as pd
from datetime import datetime
from app.models.schemas import ConsensusSignal, BacktestRun, RuleStat, RuleConfig
from app.services.indicators import add_all_indicators
from app.services.signal_engine import label_outcome, spread_cost_pct
from app.services.strategies import STRATEGIES
from app.services.consensus import check_consensus


def run_consensus_backtest(
    df: pd.DataFrame,
    pair: str,
    interval: str,
    config: RuleConfig = RuleConfig(),
    max_lookforward: int = 20,
    eval_start_index: int | None = None,
) -> tuple[BacktestRun, list[ConsensusSignal]]:
    """
    Walk-forward replay of the SMC consensus engine against stored history: at each bar
    (i >= min_warmup), runs every STRATEGIES entry and only scores a bar where check_consensus
    actually fires (weighted majority, see consensus.STRATEGY_WEIGHTS/REQUIRED_WEIGHT_FRACTION).
    Reuses label_outcome/spread_cost_pct so a consensus signal is judged by the exact same
    real-cost standard as everything else -- expected to fire rarely, since none of the
    strategies individually has shown a reliable edge in prior backtests, so a majority landing
    on the same trade at the same time should be uncommon by construction, not routine.

    rule_stats is repurposed here: "rule" is each strategy's name, "fired_count" counts how
    often that strategy gave any directional call at all (consensus or not),
    "agreed"/"hits_when_agreed" count how often that strategy's direction matched the
    eventual consensus direction on a bar where consensus fired, and how often those
    agreements hit -- shows which strategies actually pull their weight in a consensus vs
    which rarely participate.

    avg_confidence_hit/avg_confidence_miss are repurposed to hold the mean agreeing_count
    rather than a 0-100 confidence score, since consensus signals have no analogous
    confidence number of their own.
    """
    df = df.reset_index(drop=True)
    indicator_df = add_all_indicators(df, config)

    # 30 covers find_swing_levels' own 20-bar lookback (used by market_structure and
    # liquidity_sweep) plus a buffer, none of which are tied to config.ema_slow.
    min_warmup = max(config.ema_slow, 30, eval_start_index or 0)
    last_evaluable = len(df) - 1 - max_lookforward
    if last_evaluable < min_warmup:
        raise ValueError(
            f"Not enough candles for a consensus backtest: have {len(df)}, need at least "
            f"{min_warmup + max_lookforward + 1} (warmup + max_lookforward)."
        )

    run_id = uuid.uuid4().hex[:12]
    signals: list[ConsensusSignal] = []

    hold_count = 0
    hits = misses = expired = 0
    win_pcts: list[float] = []
    loss_pcts: list[float] = []
    all_pcts: list[float] = []
    confidence_hit: list[float] = []
    confidence_miss: list[float] = []

    strategy_stats: dict[str, dict[str, int]] = {}

    for i in range(min_warmup, last_evaluable + 1):
        window = indicator_df.iloc[: i + 1]
        latest = indicator_df.iloc[i]
        calls = [fn(window, config) for fn in STRATEGIES]

        for c in calls:
            if c.direction != "HOLD":
                strategy_stats.setdefault(c.strategy, {"fired": 0, "agreed": 0, "hits_when_agreed": 0})["fired"] += 1

        consensus = check_consensus(calls, pair, interval, latest["timestamp"], float(latest["atr"]))
        if consensus is None:
            hold_count += 1
            continue

        direction = consensus.direction
        entry_price = consensus.entry_price

        future_candles = df.iloc[i + 1: i + 1 + max_lookforward]
        status, outcome_price, outcome_timestamp, candles_to_outcome = label_outcome(
            future_candles, direction, consensus.target_price, consensus.stop_price, max_lookforward
        )
        # last_evaluable's bound guarantees max_lookforward future candles always exist here,
        # so label_outcome can never return "pending" in a backtest context.
        assert status != "pending", "internal error: backtest ran out of candles unexpectedly"

        pct_move = ((outcome_price - entry_price) / entry_price) * 100
        if direction == "SELL":
            pct_move = -pct_move  # normalize: positive = favorable move regardless of direction
        pct_move -= spread_cost_pct(pair, entry_price)  # every real trade pays this, win or lose

        if status == "hit":
            hits += 1
            win_pcts.append(pct_move)
            confidence_hit.append(float(consensus.agreeing_count))
        elif status == "miss":
            misses += 1
            loss_pcts.append(pct_move)
            confidence_miss.append(float(consensus.agreeing_count))
        else:
            expired += 1
            (win_pcts if pct_move >= 0 else loss_pcts).append(pct_move)
        all_pcts.append(pct_move)

        for c in calls:
            if c.direction != direction:
                continue
            stat = strategy_stats.setdefault(c.strategy, {"fired": 0, "agreed": 0, "hits_when_agreed": 0})
            stat["agreed"] += 1
            if status == "hit":
                stat["hits_when_agreed"] += 1

        consensus.status = status
        consensus.outcome_price = round(float(outcome_price), 5)
        consensus.outcome_timestamp = outcome_timestamp
        consensus.outcome_pct_move = round(pct_move, 5)
        consensus.candles_to_outcome = candles_to_outcome
        consensus.source = "backtest"
        consensus.run_id = run_id
        signals.append(consensus)

    directional_signals = hits + misses + expired
    rule_stat_list = [
        RuleStat(
            rule=strategy,
            fired_count=stat["fired"],
            directional_signals_agreed=stat["agreed"],
            hits_when_agreed=stat["hits_when_agreed"],
            hit_rate_when_agreed_pct=(
                round(stat["hits_when_agreed"] / stat["agreed"] * 100, 1) if stat["agreed"] else None
            ),
        )
        for strategy, stat in sorted(strategy_stats.items())
    ]

    run = BacktestRun(
        run_id=run_id,
        pair=pair,
        interval=interval,
        profile="consensus",
        created_at=datetime.utcnow(),
        rule_config=config,
        target_atr_mult=config.target_atr_mult,
        stop_atr_mult=config.stop_atr_mult,
        max_lookforward=max_lookforward,
        candles_evaluated=last_evaluable - min_warmup + 1,
        total_signals=directional_signals + hold_count,
        hold_signals=hold_count,
        directional_signals=directional_signals,
        hits=hits,
        misses=misses,
        expired=expired,
        hit_rate_pct=round(hits / directional_signals * 100, 1) if directional_signals else None,
        avg_win_pct=round(sum(win_pcts) / len(win_pcts), 4) if win_pcts else None,
        avg_loss_pct=round(sum(loss_pcts) / len(loss_pcts), 4) if loss_pcts else None,
        expectancy_pct=round(sum(all_pcts) / len(all_pcts), 4) if all_pcts else None,
        avg_confidence_hit=round(sum(confidence_hit) / len(confidence_hit), 1) if confidence_hit else None,
        avg_confidence_miss=round(sum(confidence_miss) / len(confidence_miss), 1) if confidence_miss else None,
        rule_stats=rule_stat_list,
    )

    return run, signals
