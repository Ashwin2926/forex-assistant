import pandas as pd
from datetime import datetime
from typing import Optional
from app.models.schemas import RuleConfig


def compute_atr_target_stop(
    entry_price: float, atr: float, direction: str, target_atr_mult: float = 1.5, stop_atr_mult: float = 1.0
) -> tuple[float, float]:
    """
    Shared by the backtester and any live execution path (paper trading) so a
    BUY/SELL signal's target/stop are always derived the same way regardless of
    where it's used — a live trade's risk parameters should never drift from
    what was actually backtested.
    """
    if direction == "BUY":
        return entry_price + target_atr_mult * atr, entry_price - stop_atr_mult * atr
    elif direction == "SELL":
        return entry_price - target_atr_mult * atr, entry_price + stop_atr_mult * atr
    raise ValueError(f"No target/stop for direction={direction!r}; only BUY/SELL have one.")


# Rough typical retail spread per pair, in raw price units rather than pips -- JPY pairs
# quote 2 decimals (1 pip = 0.01) vs 4 for the other majors (1 pip = 0.0001), so price
# units sidestep that difference entirely. These are ballpark figures for a typical
# retail account, NOT sourced from Deriv or any specific broker's actual live spread
# (paper_trading.py already notes it doesn't model Deriv's real spread/commission either,
# for the same reason: nobody's confirmed the real number yet). Until that's confirmed,
# every expectancy number computed with this should be read as an upper bound on the real
# edge, not a validated one -- real spreads also widen outside major sessions and around
# news, which a single fixed number per pair can't capture.
TYPICAL_SPREAD_PRICE: dict[str, float] = {
    "EUR/USD": 0.00010,  # ~1.0 pip
    "GBP/USD": 0.00015,  # ~1.5 pips
    "USD/JPY": 0.010,    # ~1.0 pip -- JPY pip size is 0.01, not 0.0001
    "AUD/USD": 0.00015,  # ~1.5 pips
}
DEFAULT_SPREAD_PRICE = 0.00015  # fallback for any pair not listed above


def spread_cost_pct(pair: str, entry_price: float) -> float:
    """
    Round-trip spread cost as a % of entry price, in the same units as pct_move. Every
    real trade pays this on entry and exit regardless of whether it wins or loses -- the
    backtester and live outcome scoring both previously computed pct_move (and therefore
    expectancy_pct) as if trades executed at the exact recorded close price with zero cost.
    Subtracting this here means expectancy now reflects a realistic cost instead of a
    frictionless simulation; it does NOT change hit/miss/expired classification, since
    that's still purely about whether target_price/stop_price was touched.
    """
    spread = TYPICAL_SPREAD_PRICE.get(pair, DEFAULT_SPREAD_PRICE)
    return (spread / entry_price) * 100


def label_outcome(
    future_candles: pd.DataFrame, direction: str, target_price: float, stop_price: float, max_lookforward: int,
) -> tuple[str, Optional[float], Optional[datetime], Optional[int]]:
    """
    Walk forward through future_candles (strictly after the signal, ascending by timestamp)
    checking for target/stop hit. Shared by the backtester (which always has the full
    max_lookforward window available, since it's replaying fixed history) and live outcome
    scoring (which may only have a handful of real candles so far and needs to know whether
    to keep waiting).

    Returns (status, outcome_price, outcome_timestamp, candles_to_outcome):
      "hit" / "miss"   — target or stop touched; outcome_price is that level exactly.
      "expired"        — max_lookforward candles passed with neither touched; outcome_price
                          is the close of the max_lookforward-th candle.
      "pending"        — fewer than max_lookforward candles available yet and neither has
                          been touched — caller should check again once more data arrives.

    If a single candle's range contains both target and stop, it's counted as a miss —
    OHLC data alone can't tell us which was touched first intra-candle, so we assume the
    worse outcome rather than the optimistic one.
    """
    for j, (_, candle) in enumerate(future_candles.iterrows(), start=1):
        if j > max_lookforward:
            break
        if direction == "BUY":
            stop_hit = candle["low"] <= stop_price
            target_hit = candle["high"] >= target_price
        else:
            stop_hit = candle["high"] >= stop_price
            target_hit = candle["low"] <= target_price

        if stop_hit:
            return "miss", stop_price, candle["timestamp"], j
        elif target_hit:
            return "hit", target_price, candle["timestamp"], j

    if len(future_candles) >= max_lookforward:
        final_candle = future_candles.iloc[max_lookforward - 1]
        return "expired", float(final_candle["close"]), final_candle["timestamp"], max_lookforward

    return "pending", None, None, None


# Per-profile defaults for the EMA/RSI/MACD/ATR periods SMC strategies and ATR target/stop
# math still read off RuleConfig (see strategies.py/indicators.py) -- validated cross-pair via
# the retired rule engine's own /backtest/optimize before that engine was removed (see
# PROGRESS.md's rule-engine-removal entry), not re-derived from scratch for SMC. intraday:
# EMA 9/21 on 15min data, 5000 candles/pair (all 4 majors). This is now the only profile --
# every interval (5min through 1day) is evaluated with this same config rather than a separate
# longer-horizon ruleset.
PROFILE_DEFAULTS: dict[str, RuleConfig] = {
    "intraday": RuleConfig(
        ema_fast=9,
        ema_slow=21,
        rsi_period=14,
        atr_period=14,
        # target_atr_mult/stop_atr_mult left at RuleConfig's own 1.5/1.0 default -- already
        # modestly positive and consistent (train +0.0005%/test +0.0002%, no sign flip on
        # EUR/USD) when this was last validated under the rule engine.
    ),
}


def default_config_for(profile: str, pair: Optional[str] = None) -> RuleConfig:
    return PROFILE_DEFAULTS.get(profile, RuleConfig())
