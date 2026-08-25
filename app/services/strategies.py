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

# call_volume_momentum: how large a 5-bar move needs to be (relative to ATR) to count as
# "momentum" at all, and how far above its own 20-bar average volume has to be to "confirm"
# that move rather than trust price alone. Both starting guesses, not backtested.
MOMENTUM_ROC_ATR_MULT = 1.0
VOLUME_CONFIRMATION_MULT = 1.5

# call_smart_money: how much larger a rejection wick must be than the candle's own body to
# count as an aggressive "liquidity sweep" rather than an ordinary bounce. Starting guess,
# matches detect_candlestick_pattern's hammer/shooting-star wick-to-body convention, not
# independently backtested.
SMC_WICK_BODY_MULT = 2.0

# How far beyond the sweep wick's own extreme the stop sits, as an ATR multiple -- a real
# SMC stop goes just past the sweep itself (if price returns there, the liquidity-grab
# thesis was wrong), not a flat multiple from entry the way most other strategies here do.
SMC_STOP_BUFFER_ATR_MULT = 0.25


def _clamp01(x: float) -> float:
    return max(0.0, min(x, 1.0))


def call_trend(df: pd.DataFrame, config: RuleConfig = RuleConfig()) -> StrategyCall:
    """Wraps the existing, already-backtested EMA/RSI/MACD engine unmodified -- this strategy
    IS today's live signal_engine.py, just reframed as one voice among five."""
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    reasons, bullish_votes, bearish_votes, total_rules, rule_votes, rule_strengths = apply_rules(latest, prev, config)
    volatility_ok = next(r.passed for r in reasons if r.rule == "volatility_filter")
    session_ok = next((r.passed for r in reasons if r.rule == "session_filter"), True)
    direction, confidence = decide(
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
        # confidence is already a 0-100 blend of rule_strengths -- the same "how strong was
        # this reading" score this strategy's own reasons are built from, just reused instead
        # of re-deriving a separate strength number.
        strength=_clamp01(confidence / 100) if direction in ("BUY", "SELL") else None,
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

    strength = None
    if direction in ("BUY", "SELL"):
        target_price = float(latest["bb_middle"])
        stop_price = (
            entry_price - config.stop_atr_mult * atr_val if direction == "BUY"
            else entry_price + config.stop_atr_mult * atr_val
        )
        # How far past the band, in ATR terms -- one full ATR beyond the band is already a
        # sizeable extreme, so that's the "full strength" ceiling; a touch that's barely past
        # the band reads as weak.
        band_edge = latest["bb_lower"] if direction == "BUY" else latest["bb_upper"]
        strength = _clamp01(abs(entry_price - band_edge) / atr_val) if atr_val > 0 else None

    return StrategyCall(
        strategy="bollinger", direction=direction,
        entry_price=entry_price, target_price=target_price, stop_price=stop_price,
        reasons=reasons, strength=strength,
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

    strength = None
    if direction in ("BUY", "SELL"):
        target_price, stop_price = compute_atr_target_stop(
            entry_price, atr_val, direction, config.target_atr_mult, config.stop_atr_mult
        )
        # reasons[-1] is whichever of the 4 breakout/bounce branches fired -- its value is
        # already the distance past the level; one full ATR past it is a decisive move.
        strength = _clamp01(abs(reasons[-1].value) / atr_val) if atr_val > 0 else None

    return StrategyCall(
        strategy="support_resistance", direction=direction,
        entry_price=entry_price, target_price=target_price, stop_price=stop_price,
        reasons=reasons, strength=strength,
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
    # Genuinely binary, unlike every other strategy here -- a pattern either matched this bar
    # or it didn't, with no natural in-between reading to grade (detect_candlestick_pattern
    # doesn't expose partial-match info). Full strength on a match rather than inventing a
    # fake gradient; None means "no vote," same as everywhere else.
    strength = None
    if direction in ("BUY", "SELL"):
        target_price, stop_price = compute_atr_target_stop(
            entry_price, atr_val, direction, config.target_atr_mult, config.stop_atr_mult
        )
        strength = 1.0

    return StrategyCall(
        strategy="candlestick", direction=direction,
        entry_price=entry_price, target_price=target_price, stop_price=stop_price,
        reasons=reasons, strength=strength,
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

    strength = None
    if direction in ("BUY", "SELL"):
        target_price, stop_price = compute_atr_target_stop(
            entry_price, atr_val, direction, config.target_atr_mult, config.stop_atr_mult
        )
        # ADX 50+ is already a very strong trend by common technical-analysis convention, so
        # that's the "full strength" ceiling -- not the crossover itself, since ADX is what
        # this strategy actually gates on (the crossover only fires direction, not conviction).
        strength = _clamp01(adx_val / 50)

    return StrategyCall(
        strategy="stoch_adx", direction=direction,
        entry_price=entry_price, target_price=target_price, stop_price=stop_price,
        reasons=reasons, strength=strength,
    )


def call_volume_momentum(df: pd.DataFrame, config: RuleConfig = RuleConfig()) -> StrategyCall:
    """
    Momentum trading: a fast multi-bar price move (rate_of_change, not a single noisy candle),
    confirmed by above-average volume rather than trusted on price alone -- the piece
    call_stoch_adx's ADX-based trend-strength approximation doesn't cover, since ADX has
    nothing to do with volume.

    Forex caveat worth being upfront about: most forex data providers, Twelve Data included,
    report *tick* volume -- how many price updates occurred, not literal traded volume, since
    spot forex is decentralized with no single exchange tape the way a listed stock has. It's
    a real, commonly-used proxy for market activity, just not the same thing a stock trader
    would mean by "volume." Falls back to HOLD when volume data isn't available at all,
    rather than guessing.
    """
    latest = df.iloc[-1]
    entry_price = float(latest["close"])
    atr_val = float(latest["atr"])

    volume_available = pd.notna(latest.get("volume")) and latest.get("volume", 0) > 0 and pd.notna(latest.get("volume_sma")) and latest["volume_sma"] > 0
    if not volume_available:
        return StrategyCall(
            strategy="volume_momentum", direction="HOLD",
            entry_price=entry_price, target_price=None, stop_price=None,
            reasons=[SignalReason(rule="volume_confirmation", passed=False, detail="No reliable volume data for this candle")],
        )

    volume_ratio = float(latest["volume"] / latest["volume_sma"])
    volume_confirmed = volume_ratio >= VOLUME_CONFIRMATION_MULT
    reasons = [SignalReason(
        rule="volume_confirmation", passed=volume_confirmed, value=volume_ratio,
        detail=f"Volume is {volume_ratio:.2f}x the 20-bar average — "
               f"{'confirmed' if volume_confirmed else 'not elevated enough'}"
    )]

    roc = float(latest["roc"]) if pd.notna(latest["roc"]) else 0.0
    roc_price_move = abs(roc) / 100 * entry_price
    momentum_strong = atr_val > 0 and (roc_price_move / atr_val) >= MOMENTUM_ROC_ATR_MULT
    reasons.append(SignalReason(
        rule="momentum_strength", passed=momentum_strong, value=roc,
        detail=f"5-bar rate of change is {roc:.3f}% — "
               f"{'strong' if momentum_strong else 'not strong enough'} relative to ATR"
    ))

    direction = "HOLD"
    target_price = stop_price = None
    strength = None
    if volume_confirmed and momentum_strong:
        direction = "BUY" if roc > 0 else "SELL"
        target_price, stop_price = compute_atr_target_stop(
            entry_price, atr_val, direction, config.target_atr_mult, config.stop_atr_mult
        )
        # Averages two independently-normalized components -- volume ratio against 2x its own
        # confirmation threshold, momentum against 2x its own ATR-relative threshold -- rather
        # than either alone, since this strategy's whole thesis is that both need to agree
        # (currently moot: volume_momentum is weighted 0 in consensus.STRATEGY_WEIGHTS since
        # Twelve Data can't supply real forex volume, but this stays ready for if that changes).
        volume_strength = _clamp01(volume_ratio / (VOLUME_CONFIRMATION_MULT * 2))
        momentum_strength = _clamp01((roc_price_move / atr_val) / (MOMENTUM_ROC_ATR_MULT * 2)) if atr_val > 0 else 0.0
        strength = (volume_strength + momentum_strength) / 2

    return StrategyCall(
        strategy="volume_momentum", direction=direction,
        entry_price=entry_price, target_price=target_price, stop_price=stop_price,
        reasons=reasons, strength=strength,
    )


def call_smart_money(df: pd.DataFrame, config: RuleConfig = RuleConfig()) -> StrategyCall:
    """
    A mechanized, deliberately narrow approximation of one piece of ICT/"Smart Money
    Concepts": a liquidity sweep -- price wicks through a recent swing level (where
    stop-losses are assumed to cluster) with a wick disproportionately larger than its own
    body, then closes back on the original side. This differs from
    call_support_resistance's bounce case specifically by requiring that wick dominance
    (support_resistance fires on any close-back-inside touch, regardless of wick size) --
    the actual visual signature SMC traders look for ("stop hunt"), not just "price touched
    a level."

    Target is the opposite recent swing level (the next liquidity pool), not a generic ATR
    multiple -- same precedent as call_bollinger using its middle band, since that's what
    this strategy's actual thesis predicts price will travel to. Stop sits just beyond the
    sweep wick's own extreme (plus a small ATR buffer), not a flat multiple from entry --
    that's where a real SMC stop goes, since a return past the sweep means the liquidity-grab
    read was wrong.

    Full SMC also covers market structure (BOS/CHoCH), order blocks, and fair value gaps --
    none of that is modeled here. This mechanizes the liquidity-sweep trigger only.
    """
    latest = df.iloc[-1]
    entry_price = float(latest["close"])
    atr_val = float(latest["atr"])
    resistance_levels, support_levels = find_swing_levels(df.iloc[:-1])
    nearest_resistance = resistance_levels[0] if resistance_levels else None
    nearest_support = support_levels[0] if support_levels else None

    body = abs(latest["close"] - latest["open"])
    upper_wick = latest["high"] - max(latest["close"], latest["open"])
    lower_wick = min(latest["close"], latest["open"]) - latest["low"]

    res_str = f"{nearest_resistance:.5f}" if nearest_resistance is not None else "n/a"
    sup_str = f"{nearest_support:.5f}" if nearest_support is not None else "n/a"
    reasons = [SignalReason(
        rule="smc_levels", passed=nearest_resistance is not None and nearest_support is not None,
        detail=f"Liquidity reference levels -- recent high {res_str}, recent low {sup_str}"
    )]

    direction = "HOLD"
    target_price = stop_price = None

    # Both a sweep AND a target need to exist -- a sweep with nowhere (no opposite level) to
    # send the reversal toward isn't a tradeable setup here.
    swept_high = nearest_resistance is not None and nearest_support is not None and latest["high"] > nearest_resistance and latest["close"] < nearest_resistance
    swept_low = nearest_support is not None and nearest_resistance is not None and latest["low"] < nearest_support and latest["close"] > nearest_support

    if swept_high and body > 0 and upper_wick >= body * SMC_WICK_BODY_MULT:
        direction = "SELL"
        target_price = nearest_support
        stop_price = float(latest["high"]) + SMC_STOP_BUFFER_ATR_MULT * atr_val
        reasons.append(SignalReason(
            rule="liquidity_sweep", passed=True, value=float(upper_wick / body),
            detail=f"Wicked above {res_str} (high {latest['high']:.5f}) then rejected, closing at "
                   f"{latest['close']:.5f} -- wick {upper_wick / body:.1f}x the candle's body"
        ))
    elif swept_low and body > 0 and lower_wick >= body * SMC_WICK_BODY_MULT:
        direction = "BUY"
        target_price = nearest_resistance
        stop_price = float(latest["low"]) - SMC_STOP_BUFFER_ATR_MULT * atr_val
        reasons.append(SignalReason(
            rule="liquidity_sweep", passed=True, value=float(lower_wick / body),
            detail=f"Wicked below {sup_str} (low {latest['low']:.5f}) then rejected, closing at "
                   f"{latest['close']:.5f} -- wick {lower_wick / body:.1f}x the candle's body"
        ))
    else:
        reasons.append(SignalReason(rule="liquidity_sweep", passed=False, detail="No wick-dominant rejection at a recent swing level"))

    # reasons[-1].value is the wick/body ratio when a sweep fired -- 3x the minimum threshold
    # (SMC_WICK_BODY_MULT) is treated as a maximally decisive rejection, self-referential to
    # this strategy's own trigger the same way stoch_adx's ceiling relates to its own gate.
    strength = _clamp01(reasons[-1].value / (SMC_WICK_BODY_MULT * 3)) if direction in ("BUY", "SELL") else None

    return StrategyCall(
        strategy="smart_money", direction=direction,
        entry_price=entry_price, target_price=target_price, stop_price=stop_price,
        reasons=reasons, strength=strength,
    )


STRATEGIES: list[Callable[..., StrategyCall]] = [
    call_trend, call_bollinger, call_support_resistance, call_candlestick, call_stoch_adx,
    call_volume_momentum, call_smart_money,
]
