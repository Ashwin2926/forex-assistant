from datetime import datetime
from typing import Optional
from app.models.schemas import ConsensusSignal, StrategyCall
from app.services.strategies import STRATEGIES

# Per-strategy vote weight -- equal by default, NOT tuned from real data (that would be
# overfitting an 8-13 signal backtest sample). Weighting exists so a strategy that's shown it
# doesn't belong in the majority can be dialed down later by editing one number here, instead
# of hand-coding exceptions into check_consensus itself. Derived from function name
# (call_X -> "X") so it can't drift out of sync with STRATEGIES/StrategyCall.strategy.
STRATEGY_WEIGHTS: dict[str, float] = {fn.__name__.removeprefix("call_"): 1.0 for fn in STRATEGIES}

# volume_momentum is the one exception to "equal by default, unweighted until proven
# otherwise" -- confirmed (not just suspected) that Twelve Data structurally can't supply
# forex volume (their own docs: "available not for all instrument types" -- spot FX has no
# centralized tape to report it from, so no plan tier fixes this), so every call it makes is
# HOLD, permanently. Weighting it 0 means it can't dilute the other strategies' threshold by
# sitting in the denominator doing nothing -- the remaining active strategies land at
# whatever bar REQUIRED_WEIGHT_FRACTION below sets as if it didn't exist at all, instead of
# an unreachable bar. If a real volume source (OANDA/MT5 tick volume, see PROGRESS.md) ever
# replaces Twelve Data for ingestion, raise this back to 1.0 -- the strategy itself doesn't
# need to change, only this number.
STRATEGY_WEIGHTS["volume_momentum"] = 0.0

# Fraction of total vote weight that must agree on one direction. With volume_momentum at 0
# and the other 6 (as of smart_money, added 2026-08-24) equal at 1.0, total_weight is
# effectively 6.0, so 0.6 needs 4 of 6 to agree (0.6 * 6.0 = 3.6, rounds up). This is a
# proportional bar, not a fixed headcount -- adding a real, functioning strategy is expected
# to shift the raw number needed, unlike volume_momentum's permanent-HOLD case which would
# have silently tightened the bar for a voice that can never actually vote. Originally set to
# 0.6 (3-of-5) on 2026-08-24 after 0.8 (4-of-5, before smart_money existed) made consensus
# fire so rarely across 5 philosophically different techniques it was closer to "wait for a
# near-miracle" than "wait for a real majority" (the 2026-08-21 backtest found Bollinger
# never once agreed with the pack, for context on how divergent these techniques can be).
# Re-backtest across all pairs/intervals after any further change here, including strategy
# count changes -- more signals at a lower effective bar trades agreement-strength for
# frequency, so this needs the same train/test validation as everything else, not just a
# vibe check.
REQUIRED_WEIGHT_FRACTION = 0.6

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
