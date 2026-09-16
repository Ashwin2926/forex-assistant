from datetime import datetime
from typing import Optional
from app.models.schemas import ConsensusSignal, StrategyCall
from app.services.strategies import STRATEGIES

# Per-strategy vote weight -- equal by default, NOT tuned from real data (that would be
# overfitting a small backtest sample). Weighting exists so a strategy that's shown it doesn't
# belong in the majority can be dialed down later by editing one number here, instead of
# hand-coding exceptions into check_consensus itself. Derived from function name
# (call_X -> "X") so it can't drift out of sync with STRATEGIES/StrategyCall.strategy.
STRATEGY_WEIGHTS: dict[str, float] = {fn.__name__.removeprefix("call_"): 1.0 for fn in STRATEGIES}

# Fraction of total vote weight that must agree on one direction. Re-derived when the SMC
# strategy lineup (market_structure/order_blocks/fair_value_gap/liquidity_sweep/supply_demand)
# replaced the prior 7-strategy mix (trend/bollinger/support_resistance/candlestick/stoch_adx/
# volume_momentum/smart_money) -- all 5 current strategies are equally weighted at 1.0 (no
# volume_momentum-style permanent-HOLD exception left to zero out), so 0.6 needs 3 of 5 to
# agree (0.6 * 5.0 = 3.0). This is a proportional bar, not a fixed headcount -- kept at the
# prior 0.6 fraction as a starting point rather than re-derived from scratch, since the
# original 0.6 was itself chosen (2026-08-24) to avoid an unreachably rare consensus across a
# philosophically diverse strategy mix, which is equally true of this all-SMC lineup.
# UNVALIDATED against the new strategies -- re-backtest across all pairs/intervals
# (backtester.run_consensus_backtest) before trusting this bar, not just a vibe check.
REQUIRED_WEIGHT_FRACTION = 0.6

# How close (in ATR multiples) agreeing strategies' entry/exit prices must be to count as
# genuinely describing the same trade, not just the same direction -- two strategies both
# saying "BUY" with wildly different entries/targets aren't actually agreeing on anything
# tradeable. This is a starting guess, not derived from any backtest -- same caveat as
# TYPICAL_SPREAD_PRICE in signal_engine.py: treat it as tunable and unvalidated until the
# consensus backtest (backtester.run_consensus_backtest) says otherwise.
PROXIMITY_ATR_MULT = 0.5


def direction_confidence(calls: list[StrategyCall], direction: str) -> float:
    """
    Weighted agreement strength for a specific direction, using the same STRATEGY_WEIGHTS
    check_consensus itself votes with -- lets a caller (rl_engine.compute_ml_scores, ML
    feature extraction in main.py) score "how strongly do these calls support BUY/SELL"
    without a full check_consensus pass (which also gates on price proximity -- appropriate
    for a real trade decision, unnecessary overhead for scoring a hypothetical direction).
    Same "agreeing weight / total weight" shape signal_engine.decide() used to compute for
    the rule engine, generalized to a caller-chosen direction the way that module's own
    (now-removed) direction_confidence did.
    """
    total_weight = sum(STRATEGY_WEIGHTS.get(c.strategy, 1.0) for c in calls)
    if total_weight == 0:
        return 0.0
    agreeing_weight = sum(
        STRATEGY_WEIGHTS.get(c.strategy, 1.0) * (c.strength or 0.0)
        for c in calls if c.direction == direction
    )
    return round(min(agreeing_weight / total_weight, 1.0) * 100, 1)


def check_consensus(
    calls: list[StrategyCall], pair: str, interval: str, timestamp: datetime, atr: float,
) -> Optional[ConsensusSignal]:
    """
    Groups strategy calls by direction (HOLD is never eligible), requires their combined
    STRATEGY_WEIGHTS to clear REQUIRED_WEIGHT_FRACTION of the total, and requires their entry
    prices AND their target prices to each span no more than PROXIMITY_ATR_MULT * atr.
    Returns None when nothing clears the bar -- expected to be the common case, not the
    exception, given the single-strategy backtests already show no reliable edge from any of
    these strategies in isolation, so consensus firing on every bar would itself be a red
    flag, not a good sign.
    """
    total_weight = sum(STRATEGY_WEIGHTS.get(c.strategy, 1.0) for c in calls)
    required_weight = REQUIRED_WEIGHT_FRACTION * total_weight

    for direction in ("BUY", "SELL"):
        agreeing = [c for c in calls if c.direction == direction]
        agreeing_weight = sum(STRATEGY_WEIGHTS.get(c.strategy, 1.0) for c in agreeing)
        if agreeing_weight < required_weight:
            continue

        entries = [c.entry_price for c in agreeing]
        targets = [c.target_price for c in agreeing if c.target_price is not None]
        stops = [c.stop_price for c in agreeing if c.stop_price is not None]
        if len(targets) < len(agreeing) or len(stops) < len(agreeing):
            continue  # a directional call missing its target/stop shouldn't happen -- don't trust it if it does

        tolerance = PROXIMITY_ATR_MULT * atr
        if (max(entries) - min(entries)) > tolerance:
            continue
        if (max(targets) - min(targets)) > tolerance:
            continue

        mean_entry = sum(entries) / len(entries)
        return ConsensusSignal(
            pair=pair, interval=interval, timestamp=timestamp, direction=direction,
            entry_price=mean_entry,
            target_price=sum(targets) / len(targets),
            stop_price=sum(stops) / len(stops),
            agreeing_count=len(agreeing),  # a real headcount for display, even though the gate above is weighted
            confidence=direction_confidence(calls, direction),
            atr_pct=(atr / mean_entry * 100) if mean_entry else 0.0,
            strategy_calls=calls,
        )

    return None
