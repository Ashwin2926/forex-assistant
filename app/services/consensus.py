from datetime import datetime
from typing import Optional
from app.models.schemas import ConsensusSignal, StrategyCall

MIN_AGREEING = 4

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
    Groups strategy calls by direction (HOLD is never eligible), requires at least
    MIN_AGREEING calls on one side, and requires their entry prices AND their target prices to
    each span no more than PROXIMITY_ATR_MULT * atr. Returns None when nothing clears the bar
    -- expected to be the common case, not the exception, given the single-strategy backtests
    already show no reliable edge from any one of these five in isolation, so consensus firing
    on every bar would itself be a red flag, not a good sign.
    """
    for direction in ("BUY", "SELL"):
        agreeing = [c for c in calls if c.direction == direction]
        if len(agreeing) < MIN_AGREEING:
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
            agreeing_count=len(agreeing),
            strategy_calls=calls,
        )

    return None
