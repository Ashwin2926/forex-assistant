import uuid
import pandas as pd
from datetime import datetime
from typing import Optional
from app.models.schemas import Signal, BacktestRun, RuleStat, RuleConfig
from app.services.indicators import add_all_indicators
from app.services.signal_engine import apply_rules, decide, compute_atr_target_stop, label_outcome, spread_cost_pct


def run_backtest(
    df: pd.DataFrame,
    pair: str,
    interval: str,
    profile: str,
    config: RuleConfig = RuleConfig(),
    target_atr_mult: Optional[float] = None,
    stop_atr_mult: Optional[float] = None,
    max_lookforward: int = 20,
    eval_start_index: int | None = None,
) -> tuple[BacktestRun, list[Signal]]:
    """
    Walk-forward replay of the live rule engine against stored history.

    At each candle i (i >= config.ema_slow), evaluate apply_rules()/decide() — the exact
    same code the live engine uses — using only data up to and including candle i.
    EMA/RSI/MACD/ATR are all causal (ewm/rolling only look backward), so computing
    them once over the full df and reading row i is equivalent to recomputing them
    on df.iloc[:i+1]; this avoids O(n^2) recomputation without introducing lookahead.

    Every directional (BUY/SELL) signal is labeled by walking forward up to
    max_lookforward candles: "hit" if price reaches an ATR-based target before an
    ATR-based stop, "miss" if the stop is hit first, "expired" if neither happens
    inside the window. If a single candle's high/low range contains both the target
    and the stop, it's counted as a miss — OHLC data alone can't tell us which was
    touched first intra-candle, so we assume the worse outcome rather than the
    optimistic one.

    target_atr_mult/stop_atr_mult: explicit override; omit (None) to use config's own
    target_atr_mult/stop_atr_mult — this is what lets /backtest/optimize search target/stop
    as part of the same RuleConfig grid instead of holding it fixed across every candidate.

    eval_start_index: if given, no signal is scored before this row index, though
    indicators are still computed from the start of df — this is what lets a caller
    hold out a trailing slice of history as an out-of-sample test set (indicators need
    the preceding data for warmup; the score just shouldn't include it). See
    /backtest/optimize, which tunes RuleConfig on an earlier slice and validates the
    winner here on a later one it never saw during tuning.
    """
    target_atr_mult = target_atr_mult if target_atr_mult is not None else config.target_atr_mult
    stop_atr_mult = stop_atr_mult if stop_atr_mult is not None else config.stop_atr_mult

    df = df.reset_index(drop=True)
    indicator_df = add_all_indicators(df, config)

    min_warmup = max(config.ema_slow, eval_start_index or 0)
    last_evaluable = len(df) - 1 - max_lookforward
    if last_evaluable < min_warmup:
        raise ValueError(
            f"Not enough candles for a backtest: have {len(df)}, need at least "
            f"{min_warmup + max_lookforward + 1} (warmup + max_lookforward)."
        )

    run_id = uuid.uuid4().hex[:12]
    signals: list[Signal] = []

    hold_count = 0
    hits = misses = expired = 0
    win_pcts: list[float] = []
    loss_pcts: list[float] = []
    all_pcts: list[float] = []  # every directional signal's pct_move, for expectancy
    confidence_hit: list[float] = []
    confidence_miss: list[float] = []

    # rule -> {"fired": count rule triggered at all, "agreed": count it agreed with the
    # final direction, "hits_when_agreed": how often those agreeing signals hit}
    rule_stats: dict[str, dict[str, int]] = {}

    for i in range(min_warmup, last_evaluable + 1):
        latest = indicator_df.iloc[i]
        prev = indicator_df.iloc[i - 1]

        reasons, bullish_votes, bearish_votes, total_rules, rule_votes, rule_strengths = apply_rules(latest, prev, config)
        volatility_ok = next(r.passed for r in reasons if r.rule == "volatility_filter")
        session_ok = next((r.passed for r in reasons if r.rule == "session_filter"), True)
        direction, confidence = decide(
            bullish_votes, bearish_votes, total_rules, volatility_ok and session_ok, rule_votes, rule_strengths
        )

        for r in reasons:
            if r.passed:
                rule_stats.setdefault(r.rule, {"fired": 0, "agreed": 0, "hits_when_agreed": 0})["fired"] += 1

        if direction == "HOLD":
            hold_count += 1
            continue

        entry_price = float(latest["close"])
        signal_timestamp = latest["timestamp"]
        atr_val = float(latest["atr"])

        target_price, stop_price = compute_atr_target_stop(
            entry_price, atr_val, direction, target_atr_mult, stop_atr_mult
        )

        future_candles = df.iloc[i + 1: i + 1 + max_lookforward]
        status, outcome_price, outcome_timestamp, candles_to_outcome = label_outcome(
            future_candles, direction, target_price, stop_price, max_lookforward
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
            confidence_hit.append(confidence)
        elif status == "miss":
            misses += 1
            loss_pcts.append(pct_move)
            confidence_miss.append(confidence)
        else:
            expired += 1
            (win_pcts if pct_move >= 0 else loss_pcts).append(pct_move)
        all_pcts.append(pct_move)

        for rule, voted_direction in rule_votes.items():
            if voted_direction != direction:
                continue
            stat = rule_stats.setdefault(rule, {"fired": 0, "agreed": 0, "hits_when_agreed": 0})
            stat["agreed"] += 1
            if status == "hit":
                stat["hits_when_agreed"] += 1

        signals.append(Signal(
            pair=pair,
            profile=profile,
            interval=interval,
            timestamp=signal_timestamp,
            direction=direction,
            confidence=confidence,
            reasons=reasons,
            price_at_signal=entry_price,
            status=status,
            outcome_price=round(float(outcome_price), 5),
            outcome_timestamp=outcome_timestamp,
            outcome_pct_move=round(pct_move, 5),
            source="backtest",
            run_id=run_id,
            target_price=round(target_price, 5),
            stop_price=round(stop_price, 5),
            candles_to_outcome=candles_to_outcome,
        ))

    directional_signals = hits + misses + expired
    rule_stat_list = [
        RuleStat(
            rule=rule,
            fired_count=stat["fired"],
            directional_signals_agreed=stat["agreed"],
            hits_when_agreed=stat["hits_when_agreed"],
            hit_rate_when_agreed_pct=(
                round(stat["hits_when_agreed"] / stat["agreed"] * 100, 1) if stat["agreed"] else None
            ),
        )
        for rule, stat in sorted(rule_stats.items())
    ]

    run = BacktestRun(
        run_id=run_id,
        pair=pair,
        interval=interval,
        profile=profile,
        created_at=datetime.utcnow(),
        rule_config=config,
        target_atr_mult=target_atr_mult,
        stop_atr_mult=stop_atr_mult,
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
