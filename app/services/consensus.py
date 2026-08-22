from datetime import datetime
from typing import Optional
from app.models.schemas import ConsensusSignal, StrategyCall
from app.services.strategies import STRATEGIES

# Per-strategy vote weight -- NOT tuned from real data yet, deliberately all equal for now.
# A raw headcount ("at most 1 dissenter") can't adapt to a strategy that's *consistently*
# on the other side of the market from the rest: the first consensus backtest round showed
# Bollinger agreeing with the eventual consensus direction on 0 of the signals that fired,
# across all 4 pairs, both train and test -- a plausible, explainable pattern (mean-reversion
# is philosophically opposed to the other four, which lean trend/momentum), but with an
# 8-13 signal sample per pair it isn't proof, just a real observation worth acting on later.
# Weighting lets that get reflected by editing one number here once there's enough consensus-
# backtest history to justify it, instead of hand-coding which strategies "count less" into
# check_consensus itself. Derived from function name (call_X -> "X") so it can't drift out of
# sync with STRATEGIES/StrategyCall.strategy.
STRATEGY_WEIGHTS: dict[str, float] = {fn.__name__.removeprefix("call_"): 1.0 for fn in STRATEGIES}

# Fraction of total vote weight that must agree on one direction. With today's all-equal
# weights this reproduces the original "at most 1 of N dissents" bar (0.8 * 6 = 4.8, so 5 of
# 6 equal votes is required, same stringency as the previous 4-of-5 design) -- but once a
# weight is lowered for a strategy that's shown it doesn't belong in the majority, that
# strategy alone can no longer single-handedly block every consensus the way a raw headcount
# would let it.
REQUIRED_WEIGHT_FRACTION = 0.8

# How close (in ATR multiples) agreeing strategies' entry/exit prices must be to count as
# genuinely describing the same trade, not just the same direction -- two strategies both
# saying "BUY" with wildly different entries/targets aren't actually agreeing on anything
# tradeable. This is a starting guess, not derived from any backtest -- same caveat as
# TYPICAL_SPREAD_PRICE in signal_engine.py: treat it as tunable and unvalidated until the
# consensus backtest (backtester.run_consensus_backtest) says otherwise.
PROXIMITY_ATR_MULT = 0.5


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

        return ConsensusSignal(
            pair=pair, interval=interval, timestamp=timestamp, direction=direction,
            entry_price=sum(entries) / len(entries),
            target_price=sum(targets) / len(targets),
            stop_price=sum(stops) / len(stops),
            agreeing_count=len(agreeing),  # a real headcount for display, even though the gate above is weighted
            strategy_calls=calls,
        )

    return None
