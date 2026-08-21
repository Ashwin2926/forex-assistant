import pandas as pd
from typing import Callable
from app.models.schemas import RuleConfig, SignalReason, StrategyCall
from app.services.signal_engine import apply_rules, decide, compute_atr_target_stop
from app.services.patterns import find_swing_levels, detect_candlestick_pattern

# Patterns that imply a direction to trade. Doji is deliberately excluded -- it signals
# indecision, not a direction, so it should read as HOLD like "no pattern" does.
BULLISH_PATTERNS = {"bullish_engulfing", "hammer"}
BEARISH_PATTERNS = {"bearish_engulfing", "shooting_star"}

# Below this, ADX says there's no real trend underway -- a stochastic crossover in a
# non-trending market is much more likely to be noise, so call_stoch_adx ignores it entirely
# rather than trading it at reduced confidence.
ADX_TREND_THRESHOLD = 20.0


def call_trend(df: pd.DataFrame, config: RuleConfig = RuleConfig()) -> StrategyCall:
    """Wraps the existing, already-backtested EMA/RSI/MACD engine unmodified -- this strategy
    IS today's live signal_engine.py, just reframed as one voice among five."""
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    reasons, bullish_votes, bearish_votes, total_rules, rule_votes, rule_strengths = apply_rules(latest, prev, config)
    volatility_ok = next(r.passed for r in reasons if r.rule == "volatility_filter")
    session_ok = next((r.passed for r in reasons if r.rule == "session_filter"), True)
    direction, _confidence = decide(
        bullish_votes, bearish_votes, total_rules, volatility_ok and session_ok, rule_votes, rule_strengths
    )

    entry_price = float(latest["close"])
    target_price = stop_price = None
    if direction in ("BUY", "SELL"):
        atr_val = float(latest["atr"])
        target_price, stop_price = compute_atr_target_stop(
            entry_price, atr_val, direction, config.target_atr_mult, config.stop_atr_mult
        )

    return StrategyCall(
        strategy="trend", direction=direction,
        entry_price=entry_price, target_price=target_price, stop_price=stop_price,
        reasons=reasons,
    )


def call_bollinger(df: pd.DataFrame, config: RuleConfig = RuleConfig()) -> StrategyCall:
    """
    Mean-reversion: a close at/beyond a band is treated as an extreme due to revert back
    toward the middle band, gated by band width so a touch during a tight, low-volatility
    squeeze isn't trusted as a real extreme (reuses volatility_threshold_pct the same way the
    trend strategy's volatility_filter does, just measured as band width instead of ATR%).

    Target is the middle band itself, not a generic ATR multiple -- that's the actual
    mean-reversion thesis this strategy is built on, so forcing it into the same
    entry ± target_atr_mult*ATR shape every other strategy uses would misrepresent what it's
    actually predicting. Stop is still ATR-based, same risk-sizing convention as elsewhere.
    """
    latest = df.iloc[-1]
    entry_price = float(latest["close"])
    atr_val = float(latest["atr"])
    band_width_pct = (
        (latest["bb_upper"] - latest["bb_lower"]) / latest["bb_middle"] * 100
        if latest["bb_middle"] else 0.0
    )
    bands_wide_enough = band_width_pct > config.volatility_threshold_pct

    reasons = [SignalReason(
        rule="bollinger_band_width", passed=bands_wide_enough, value=float(band_width_pct),
        detail=f"Band width is {band_width_pct:.4f}% of price — "
               f"{'sufficient' if bands_wide_enough else 'too narrow, likely ranging tightly'}"
    )]

    direction = "HOLD"
    target_price = stop_price = None
    if bands_wide_enough:
        if latest["close"] <= latest["bb_lower"]:
            direction = "BUY"
            reasons.append(SignalReason(
                rule="bollinger_touch", passed=True, value=float(latest["close"] - latest["bb_lower"]),
                detail=f"Close ({latest['close']:.5f}) at/below lower band ({latest['bb_lower']:.5f}) "
                       f"— oversold, reversion to mean expected"
            ))
        elif latest["close"] >= latest["bb_upper"]:
            direction = "SELL"
            reasons.append(SignalReason(
                rule="bollinger_touch", passed=True, value=float(latest["close"] - latest["bb_upper"]),
                detail=f"Close ({latest['close']:.5f}) at/above upper band ({latest['bb_upper']:.5f}) "
                       f"— overbought, reversion to mean expected"
            ))
        else:
            reasons.append(SignalReason(
                rule="bollinger_touch", passed=False, value=float(latest["close"] - latest["bb_middle"]),
                detail=f"Close ({latest['close']:.5f}) inside bands — no extreme"
            ))
    else:
        reasons.append(SignalReason(rule="bollinger_touch", passed=False, detail="Skipped -- bands too narrow to trust a touch"))

    if direction in ("BUY", "SELL"):
        target_price = float(latest["bb_middle"])
        stop_price = (
            entry_price - config.stop_atr_mult * atr_val if direction == "BUY"
            else entry_price + config.stop_atr_mult * atr_val
        )

    return StrategyCall(
        strategy="bollinger", direction=direction,
        entry_price=entry_price, target_price=target_price, stop_price=stop_price,
        reasons=reasons,
    )


def call_support_resistance(df: pd.DataFrame, config: RuleConfig = RuleConfig()) -> StrategyCall:
    """
    Breakout (close clears a level it was previously inside of) or bounce (price touches a
    level intrabar but closes back on the same side) off the nearest recent swing level.
    Levels are computed from history strictly before this bar (df.iloc[:-1]) -- using the
    current bar's own high/low to define "the level it just broke" would be circular.
    """
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    entry_price = float(latest["close"])
    atr_val = float(latest["atr"])
    resistance_levels, support_levels = find_swing_levels(df.iloc[:-1])
    nearest_resistance = resistance_levels[0] if resistance_levels else None
    nearest_support = support_levels[0] if support_levels else None

    res_str = f"{nearest_resistance:.5f}" if nearest_resistance is not None else "n/a"
    sup_str = f"{nearest_support:.5f}" if nearest_support is not None else "n/a"
    reasons = [SignalReason(
        rule="swing_levels", passed=nearest_resistance is not None and nearest_support is not None,
        detail=f"Nearest resistance {res_str}, nearest support {sup_str}"
    )]

    direction = "HOLD"
    target_price = stop_price = None

    if nearest_resistance is not None and latest["close"] > nearest_resistance and prev["close"] <= nearest_resistance:
        direction = "BUY"
        reasons.append(SignalReason(
            rule="resistance_breakout", passed=True, value=float(latest["close"] - nearest_resistance),
            detail=f"Close ({latest['close']:.5f}) broke above resistance ({nearest_resistance:.5f})"
        ))
    elif nearest_support is not None and latest["close"] < nearest_support and prev["close"] >= nearest_support:
        direction = "SELL"
        reasons.append(SignalReason(
            rule="support_breakdown", passed=True, value=float(nearest_support - latest["close"]),
            detail=f"Close ({latest['close']:.5f}) broke below support ({nearest_support:.5f})"
        ))
    elif nearest_support is not None and latest["low"] <= nearest_support and latest["close"] > nearest_support:
        direction = "BUY"
        reasons.append(SignalReason(
            rule="support_bounce", passed=True, value=float(latest["close"] - nearest_support),
            detail=f"Price touched support ({nearest_support:.5f}) and bounced, close ({latest['close']:.5f})"
        ))
    elif nearest_resistance is not None and latest["high"] >= nearest_resistance and latest["close"] < nearest_resistance:
        direction = "SELL"
        reasons.append(SignalReason(
            rule="resistance_bounce", passed=True, value=float(nearest_resistance - latest["close"]),
            detail=f"Price touched resistance ({nearest_resistance:.5f}) and rejected, close ({latest['close']:.5f})"
        ))
    else:
        reasons.append(SignalReason(rule="swing_level_reaction", passed=False, detail="No breakout or bounce at a nearby level"))

    if direction in ("BUY", "SELL"):
        target_price, stop_price = compute_atr_target_stop(
            entry_price, atr_val, direction, config.target_atr_mult, config.stop_atr_mult
        )

    return StrategyCall(
        strategy="support_resistance", direction=direction,
        entry_price=entry_price, target_price=target_price, stop_price=stop_price,
        reasons=reasons,
    )


def call_candlestick(df: pd.DataFrame, config: RuleConfig = RuleConfig()) -> StrategyCall:
    """Engulfing/hammer/shooting-star/doji from patterns.py. Most bars have no pattern at
    all -- that's expected, not a bug, and reads as HOLD like every other strategy's non-signal."""
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    entry_price = float(latest["close"])
    atr_val = float(latest["atr"])
    pattern = detect_candlestick_pattern(latest, prev)

    direction = "HOLD"
    if pattern in BULLISH_PATTERNS:
        direction = "BUY"
    elif pattern in BEARISH_PATTERNS:
        direction = "SELL"

    reasons = [SignalReason(
        rule="candlestick_pattern", passed=direction != "HOLD",
        detail=f"Pattern: {pattern}" if pattern else "No recognized pattern on this bar",
    )]

    target_price = stop_price = None
    if direction in ("BUY", "SELL"):
        target_price, stop_price = compute_atr_target_stop(
            entry_price, atr_val, direction, config.target_atr_mult, config.stop_atr_mult
        )

    return StrategyCall(
        strategy="candlestick", direction=direction,
        entry_price=entry_price, target_price=target_price, stop_price=stop_price,
        reasons=reasons,
    )


def call_stoch_adx(df: pd.DataFrame, config: RuleConfig = RuleConfig()) -> StrategyCall:
    """
    Direction from a %K/%D stochastic crossover (only when not already deep in overbought/
    oversold territory, so it's a fresh cross rather than a stale one); ADX is a trend-STRENGTH
    filter, not a direction source (it can't say which way a strong trend points), so a
    crossover is only trusted when ADX confirms a real trend is underway.
    """
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    entry_price = float(latest["close"])
    atr_val = float(latest["atr"])
    adx_val = float(latest["adx"]) if pd.notna(latest["adx"]) else 0.0
    trending = adx_val > ADX_TREND_THRESHOLD

    reasons = [SignalReason(
        rule="adx_trend_strength", passed=trending, value=adx_val,
        detail=f"ADX at {adx_val:.1f} — "
               f"{'trending, stochastic signal trusted' if trending else 'no clear trend, stochastic signal ignored'}"
    )]

    direction = "HOLD"
    target_price = stop_price = None
    if trending:
        stoch_cross_up = prev["stoch_k"] <= prev["stoch_d"] and latest["stoch_k"] > latest["stoch_d"] and latest["stoch_k"] < 80
        stoch_cross_down = prev["stoch_k"] >= prev["stoch_d"] and latest["stoch_k"] < latest["stoch_d"] and latest["stoch_k"] > 20
        if stoch_cross_up:
            direction = "BUY"
            reasons.append(SignalReason(
                rule="stochastic_cross", passed=True, value=float(latest["stoch_k"]),
                detail=f"%K ({latest['stoch_k']:.1f}) crossed above %D ({latest['stoch_d']:.1f})"
            ))
        elif stoch_cross_down:
            direction = "SELL"
            reasons.append(SignalReason(
                rule="stochastic_cross", passed=True, value=float(latest["stoch_k"]),
                detail=f"%K ({latest['stoch_k']:.1f}) crossed below %D ({latest['stoch_d']:.1f})"
            ))
        else:
            reasons.append(SignalReason(
                rule="stochastic_cross", passed=False, value=float(latest["stoch_k"]),
                detail="No stochastic crossover this bar"
            ))
    else:
        reasons.append(SignalReason(rule="stochastic_cross", passed=False, detail="Skipped -- ADX below trend threshold"))

    if direction in ("BUY", "SELL"):
        target_price, stop_price = compute_atr_target_stop(
            entry_price, atr_val, direction, config.target_atr_mult, config.stop_atr_mult
        )

    return StrategyCall(
        strategy="stoch_adx", direction=direction,
        entry_price=entry_price, target_price=target_price, stop_price=stop_price,
        reasons=reasons,
    )


STRATEGIES: list[Callable[..., StrategyCall]] = [
    call_trend, call_bollinger, call_support_resistance, call_candlestick, call_stoch_adx,
]
