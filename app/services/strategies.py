import pandas as pd
from typing import Callable, Optional
from app.models.schemas import RuleConfig, SignalReason, StrategyCall
from app.services.signal_engine import compute_atr_target_stop
from app.services.patterns import find_swing_levels

# Full Smart Money Concepts (ICT) stack -- five independent strategies, replacing the single
# call_smart_money liquidity-sweep approximation that used to be the only SMC voice in this
# codebase. Each one is a mechanized, deliberately narrow reading of one SMC concept, not a
# full discretionary implementation -- same "mechanize the trigger, not the whole judgment
# call" philosophy every other strategy in this module already follows.


def _clamp01(x: float) -> float:
    return max(0.0, min(x, 1.0))


# How much larger a candle's body needs to be than its own ATR to count as a "strong impulse"
# move for call_order_blocks/call_supply_demand -- ATR-relative, never a fixed pip count, the
# same convention every threshold in this module follows.
IMPULSE_ATR_MULT = 1.5

# How many bars back of the evaluation window call_order_blocks/call_supply_demand will search
# for the most recent qualifying impulse -- bounded so a zone that formed long ago (long since
# irrelevant to current price) doesn't still fire a call; a real order block/demand zone is
# either retested soon after forming or it isn't, in practice.
SMC_ZONE_LOOKBACK_BARS = 15

# How far beyond a zone's own far edge a stop sits, as an ATR multiple -- a real SMC stop goes
# just past the level itself (if price returns there, the zone's thesis was wrong), not a flat
# multiple from entry the way most non-SMC strategies use. Shared by every SMC strategy below
# that anchors its stop to a zone edge rather than entry price.
SMC_STOP_BUFFER_ATR_MULT = 0.25

# call_liquidity_sweep: how much larger a rejection wick must be than the candle's own body to
# count as an aggressive "liquidity sweep" rather than an ordinary bounce. Starting guess,
# matches detect_candlestick_pattern's hammer/shooting-star wick-to-body convention, not
# independently backtested.
SMC_WICK_BODY_MULT = 2.0

# call_supply_demand: a bar's body below this ATR multiple counts as "small" enough to belong
# to a consolidation range rather than being a directional move in its own right.
CONSOLIDATION_BODY_ATR_MULT = 0.5

# call_supply_demand needs at least this many consecutive small-bodied bars immediately before
# an impulse to call it a genuine consolidation zone -- one bar alone is just an order block
# (call_order_blocks), not "one level up" from it.
CONSOLIDATION_MIN_BARS = 2

# call_fair_value_gap: gap size (candle 1's edge to candle 3's edge) at this multiple of ATR is
# treated as a maximally significant imbalance for strength-scoring purposes -- a self-
# referential ceiling, same pattern as call_liquidity_sweep's own strength ceiling below.
FVG_STRENGTH_ATR_CEILING_MULT = 2.0

# call_market_structure: a reversal call (Change of Character -- a swing level taken out
# AGAINST the prevailing EMA trend) is inherently less certain than a continuation call (Break
# of Structure -- taken out WITH it), since it's arguing the trend is ending rather than
# confirming it. Discounted, not dropped -- CHoCH is still a real, tradeable SMC concept.
CHOCH_STRENGTH_DISCOUNT = 0.7


def call_market_structure(df: pd.DataFrame, config: RuleConfig = RuleConfig()) -> StrategyCall:
    """
    Break of Structure (BOS) vs Change of Character (CHoCH): a recent swing level -- the same
    swing-pivot detection call_liquidity_sweep and (formerly) call_support_resistance already
    used, see patterns.find_swing_levels -- taken out in the direction of the prevailing EMA
    trend reads as continuation (BOS); taken out AGAINST it reads as an early reversal (CHoCH).
    Trend context is ema_fast vs ema_slow, the same trend read this codebase already uses
    elsewhere (signal_engine's own trend_ema rule) -- not re-derived independently.

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
    uptrend = bool(pd.notna(latest["ema_fast"]) and pd.notna(latest["ema_slow"]) and latest["ema_fast"] > latest["ema_slow"])

    res_str = f"{nearest_resistance:.5f}" if nearest_resistance is not None else "n/a"
    sup_str = f"{nearest_support:.5f}" if nearest_support is not None else "n/a"
    reasons = [SignalReason(
        rule="structure_levels", passed=nearest_resistance is not None and nearest_support is not None,
        detail=f"Nearest structural high {res_str}, nearest structural low {sup_str} -- "
               f"{'uptrend' if uptrend else 'downtrend'} context (EMA{config.ema_fast} vs EMA{config.ema_slow})"
    )]

    direction = "HOLD"
    target_price = stop_price = None
    is_choch = False

    broke_high = nearest_resistance is not None and latest["close"] > nearest_resistance and prev["close"] <= nearest_resistance
    broke_low = nearest_support is not None and latest["close"] < nearest_support and prev["close"] >= nearest_support

    if broke_high:
        direction = "BUY"
        is_choch = not uptrend
        reasons.append(SignalReason(
            rule="change_of_character" if is_choch else "break_of_structure", passed=True,
            value=float(latest["close"] - nearest_resistance),
            detail=f"Close ({latest['close']:.5f}) broke above structural high ({nearest_resistance:.5f}) -- "
                   f"{'reversal against the trend (CHoCH)' if is_choch else 'continuation with the trend (BOS)'}"
        ))
    elif broke_low:
        direction = "SELL"
        is_choch = uptrend
        reasons.append(SignalReason(
            rule="change_of_character" if is_choch else "break_of_structure", passed=True,
            value=float(nearest_support - latest["close"]),
            detail=f"Close ({latest['close']:.5f}) broke below structural low ({nearest_support:.5f}) -- "
                   f"{'reversal against the trend (CHoCH)' if is_choch else 'continuation with the trend (BOS)'}"
        ))
    else:
        reasons.append(SignalReason(rule="structure_break", passed=False, detail="No structural level broken this bar"))

    strength = None
    if direction in ("BUY", "SELL"):
        target_price, stop_price = compute_atr_target_stop(
            entry_price, atr_val, direction, config.target_atr_mult, config.stop_atr_mult
        )
        strength = _clamp01(abs(reasons[-1].value) / atr_val) if atr_val > 0 else None
        if strength is not None and is_choch:
            strength = _clamp01(strength * CHOCH_STRENGTH_DISCOUNT)

    return StrategyCall(
        strategy="market_structure", direction=direction,
        entry_price=entry_price, target_price=target_price, stop_price=stop_price,
        reasons=reasons, strength=strength,
    )


def _find_recent_impulse(window: pd.DataFrame, lookback: int) -> Optional[tuple[int, str, float, float]]:
    """
    Scans the last `lookback` bars of `window`, EXCLUDING its own final row (that's "now" --
    an order block/demand zone needs at least one bar of reaction after the impulse that
    formed it to mean anything), for the most recent single candle whose body is at least
    IMPULSE_ATR_MULT times that bar's own ATR. Returns (positional index within window,
    impulse direction, that bar's open, that bar's close) for the nearest qualifying bar, or
    None if none of the last `lookback` bars qualify. Shared by call_order_blocks and
    call_supply_demand -- both need "find the most recent strong move" as their first step,
    differing only in what they treat as the zone that preceded it.
    """
    start = max(0, len(window) - 1 - lookback)
    for pos in range(len(window) - 2, start - 1, -1):
        bar = window.iloc[pos]
        atr_val = bar["atr"]
        if pd.isna(atr_val) or atr_val <= 0:
            continue
        body = abs(bar["close"] - bar["open"])
        if body >= IMPULSE_ATR_MULT * atr_val:
            direction = "BUY" if bar["close"] > bar["open"] else "SELL"
            return pos, direction, float(bar["open"]), float(bar["close"])
    return None


def call_order_blocks(df: pd.DataFrame, config: RuleConfig = RuleConfig()) -> StrategyCall:
    """
    An order block is the last opposite-direction candle immediately before a strong impulse
    move -- the last sellers (or buyers) absorbed right before price left them behind.
    "Strong" is ATR-relative (IMPULSE_ATR_MULT), never a fixed pip count, matching every other
    threshold in this codebase. Fires when price is currently trading back inside that
    candle's own high/low range (a retest) without having closed through its far edge -- the
    thesis is continuation in the impulse's original direction from the retest, not the
    retest itself being a reversal.
    """
    latest = df.iloc[-1]
    entry_price = float(latest["close"])
    atr_val = float(latest["atr"])

    impulse = _find_recent_impulse(df, SMC_ZONE_LOOKBACK_BARS)
    if impulse is None:
        return StrategyCall(
            strategy="order_blocks", direction="HOLD", entry_price=entry_price,
            reasons=[SignalReason(rule="order_block_impulse", passed=False, detail="No qualifying impulse move in recent history")],
        )

    impulse_pos, impulse_direction, impulse_open, impulse_close = impulse
    if impulse_pos == 0:
        return StrategyCall(
            strategy="order_blocks", direction="HOLD", entry_price=entry_price,
            reasons=[SignalReason(rule="order_block_impulse", passed=False, detail="Impulse candle has no preceding bar to form an order block from")],
        )

    ob_candle = df.iloc[impulse_pos - 1]
    ob_bullish = bool(ob_candle["close"] > ob_candle["open"])
    # A genuine order block is the OPPOSITE color of the impulse it precedes -- a bullish
    # impulse should be preceded by the last bearish candle (the last sellers), not another
    # bullish one (that would just be more of the same move, not a distinct absorption zone).
    is_valid_ob = (impulse_direction == "BUY" and not ob_bullish) or (impulse_direction == "SELL" and ob_bullish)
    ob_high = float(ob_candle["high"])
    ob_low = float(ob_candle["low"])

    reasons = [SignalReason(
        rule="order_block_impulse", passed=is_valid_ob, value=float(abs(ob_candle["close"] - ob_candle["open"])),
        detail=f"{impulse_direction} impulse {len(df) - 1 - impulse_pos} bar(s) ago, preceding candle "
               f"[{ob_low:.5f}, {ob_high:.5f}] {'is' if is_valid_ob else 'is not'} an opposite-direction order block"
    )]

    direction = "HOLD"
    target_price = stop_price = None
    if is_valid_ob:
        in_zone = bool(latest["low"] <= ob_high and latest["high"] >= ob_low)
        if impulse_direction == "BUY" and in_zone and latest["close"] > ob_low:
            direction = "BUY"
            target_price = float(df.iloc[impulse_pos]["high"])
            stop_price = ob_low - SMC_STOP_BUFFER_ATR_MULT * atr_val
        elif impulse_direction == "SELL" and in_zone and latest["close"] < ob_high:
            direction = "SELL"
            target_price = float(df.iloc[impulse_pos]["low"])
            stop_price = ob_high + SMC_STOP_BUFFER_ATR_MULT * atr_val

        reasons.append(SignalReason(
            rule="order_block_retest", passed=direction != "HOLD",
            detail=(f"Price retesting the order block zone [{ob_low:.5f}, {ob_high:.5f}], "
                    f"close {latest['close']:.5f} -- continuation {direction} expected")
            if direction != "HOLD" else "Price not currently retesting the order block zone"
        ))

    strength = None
    if direction in ("BUY", "SELL"):
        impulse_atr = float(df.iloc[impulse_pos]["atr"])
        impulse_body = abs(impulse_close - impulse_open)
        # Ceiling at 2x the qualifying threshold -- self-referential to this strategy's own
        # trigger, same pattern as call_liquidity_sweep's wick-ratio ceiling below.
        strength = _clamp01(impulse_body / (impulse_atr * IMPULSE_ATR_MULT * 2)) if impulse_atr > 0 else None

    return StrategyCall(
        strategy="order_blocks", direction=direction,
        entry_price=entry_price, target_price=target_price, stop_price=stop_price,
        reasons=reasons, strength=strength,
    )


def call_fair_value_gap(df: pd.DataFrame, config: RuleConfig = RuleConfig()) -> StrategyCall:
    """
    The standard 3-candle ICT imbalance: candle 1's range doesn't overlap candle 3's range,
    leaving a gap the middle candle's impulse punched straight through. A bullish gap
    (candle 1's high below candle 3's low) reads as an up-move that left support beneath
    current price; a bearish gap (candle 1's low above candle 3's high) reads as a down-move
    that left resistance above it. Target is the next opposing liquidity level
    (find_swing_levels, same swing-pivot source every other strategy here uses), falling back
    to the gap's own size projected forward when no such level exists yet -- never a generic
    ATR multiple, since the gap's own geometry IS this strategy's thesis. Stop sits just
    beyond the gap's near edge (where it formed) -- a full retrace back through the gap
    invalidates the imbalance.
    """
    if len(df) < 3:
        return StrategyCall(
            strategy="fair_value_gap", direction="HOLD", entry_price=float(df.iloc[-1]["close"]),
            reasons=[SignalReason(rule="fvg_geometry", passed=False, detail="Not enough history for a 3-candle gap")],
        )

    candle1 = df.iloc[-3]
    candle3 = df.iloc[-1]
    latest = candle3
    entry_price = float(latest["close"])
    atr_val = float(latest["atr"])

    bullish_gap = bool(candle1["high"] < candle3["low"])
    bearish_gap = bool(candle1["low"] > candle3["high"])

    reasons = [SignalReason(
        rule="fvg_geometry", passed=bullish_gap or bearish_gap,
        detail=(f"3-candle imbalance: candle1 high {candle1['high']:.5f} vs candle3 low {candle3['low']:.5f}"
                if bullish_gap else
                f"3-candle imbalance: candle1 low {candle1['low']:.5f} vs candle3 high {candle3['high']:.5f}"
                if bearish_gap else "No imbalance -- candle1 and candle3 ranges overlap")
    )]

    direction = "HOLD"
    target_price = stop_price = None
    gap_size = 0.0

    if bullish_gap:
        direction = "BUY"
        gap_low, gap_high = float(candle1["high"]), float(candle3["low"])
        gap_size = gap_high - gap_low
        resistance_levels, _ = find_swing_levels(df.iloc[:-1])
        next_liquidity = next((r for r in resistance_levels if r > entry_price), None)
        target_price = next_liquidity if next_liquidity is not None else gap_high + gap_size
        stop_price = gap_low - SMC_STOP_BUFFER_ATR_MULT * atr_val
        reasons.append(SignalReason(
            rule="fair_value_gap", passed=True, value=gap_size,
            detail=f"Bullish gap [{gap_low:.5f}, {gap_high:.5f}] acts as support beneath price"
        ))
    elif bearish_gap:
        direction = "SELL"
        gap_high, gap_low = float(candle1["low"]), float(candle3["high"])
        gap_size = gap_high - gap_low
        _, support_levels = find_swing_levels(df.iloc[:-1])
        next_liquidity = next((s for s in support_levels if s < entry_price), None)
        target_price = next_liquidity if next_liquidity is not None else gap_low - gap_size
        stop_price = gap_high + SMC_STOP_BUFFER_ATR_MULT * atr_val
        reasons.append(SignalReason(
            rule="fair_value_gap", passed=True, value=gap_size,
            detail=f"Bearish gap [{gap_low:.5f}, {gap_high:.5f}] acts as resistance above price"
        ))
    else:
        reasons.append(SignalReason(rule="fair_value_gap", passed=False, detail="No fair value gap on this 3-candle sequence"))

    strength = _clamp01(gap_size / (atr_val * FVG_STRENGTH_ATR_CEILING_MULT)) if direction in ("BUY", "SELL") and atr_val > 0 else None

    return StrategyCall(
        strategy="fair_value_gap", direction=direction,
        entry_price=entry_price, target_price=target_price, stop_price=stop_price,
        reasons=reasons, strength=strength,
    )


def call_liquidity_sweep(df: pd.DataFrame, config: RuleConfig = RuleConfig()) -> StrategyCall:
    """
    A mechanized, deliberately narrow approximation of one piece of ICT/"Smart Money
    Concepts": price wicks through a recent swing level (where stop-losses are assumed to
    cluster) with a wick disproportionately larger than its own body, then closes back on the
    original side -- the actual visual signature SMC traders look for ("stop hunt"), not just
    "price touched a level" (the plain swing-level bounce this codebase used to cover via
    call_support_resistance, before that strategy was retired in favor of this SMC lineup).

    Target is the opposite recent swing level (the next liquidity pool), not a generic ATR
    multiple -- that's what this strategy's actual thesis predicts price will travel to. Stop
    sits just beyond the sweep wick's own extreme (plus a small ATR buffer), not a flat
    multiple from entry -- that's where a real SMC stop goes, since a return past the sweep
    means the liquidity-grab read was wrong.

    Formerly named call_smart_money, when it was this module's only SMC strategy; extracted
    under its own name now that market structure, order blocks, fair value gaps, and
    supply/demand zones are each modeled separately.
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
    # this strategy's own trigger the same way call_stoch_adx's ceiling used to relate to ADX.
    strength = _clamp01(reasons[-1].value / (SMC_WICK_BODY_MULT * 3)) if direction in ("BUY", "SELL") else None

    return StrategyCall(
        strategy="liquidity_sweep", direction=direction,
        entry_price=entry_price, target_price=target_price, stop_price=stop_price,
        reasons=reasons, strength=strength,
    )


def call_supply_demand(df: pd.DataFrame, config: RuleConfig = RuleConfig()) -> StrategyCall:
    """
    One level up from an order block: instead of just the single last opposite-direction
    candle before a strong impulse (call_order_blocks), this looks for a genuine
    consolidation -- CONSOLIDATION_MIN_BARS+ consecutive small-bodied bars (body under
    CONSOLIDATION_BODY_ATR_MULT * that bar's own ATR each) immediately preceding the impulse
    -- and treats the whole consolidation's high/low range as the demand/supply zone, not
    just one candle's. Fires the same way call_order_blocks does: price back inside that
    zone without having broken through its far edge, continuation in the impulse's original
    direction expected.
    """
    latest = df.iloc[-1]
    entry_price = float(latest["close"])
    atr_val = float(latest["atr"])

    impulse = _find_recent_impulse(df, SMC_ZONE_LOOKBACK_BARS)
    if impulse is None:
        return StrategyCall(
            strategy="supply_demand", direction="HOLD", entry_price=entry_price,
            reasons=[SignalReason(rule="supply_demand_impulse", passed=False, detail="No qualifying impulse move in recent history")],
        )

    impulse_pos, impulse_direction, impulse_open, impulse_close = impulse

    # Walk backward from just before the impulse, collecting consecutive small-bodied bars.
    zone_bars = []
    pos = impulse_pos - 1
    while pos >= 0:
        bar = df.iloc[pos]
        bar_atr = bar["atr"]
        if pd.isna(bar_atr) or bar_atr <= 0:
            break
        body = abs(bar["close"] - bar["open"])
        if body > CONSOLIDATION_BODY_ATR_MULT * bar_atr:
            break
        zone_bars.append(bar)
        pos -= 1

    if len(zone_bars) < CONSOLIDATION_MIN_BARS:
        return StrategyCall(
            strategy="supply_demand", direction="HOLD", entry_price=entry_price,
            reasons=[SignalReason(
                rule="supply_demand_impulse", passed=False, value=float(len(zone_bars)),
                detail=f"Only {len(zone_bars)} consolidating bar(s) before the {impulse_direction} impulse -- "
                       f"need {CONSOLIDATION_MIN_BARS}+ for a genuine zone, not just an order block"
            )],
        )

    zone_high = max(float(b["high"]) for b in zone_bars)
    zone_low = min(float(b["low"]) for b in zone_bars)

    reasons = [SignalReason(
        rule="supply_demand_impulse", passed=True, value=float(len(zone_bars)),
        detail=f"{impulse_direction} impulse preceded by a {len(zone_bars)}-bar consolidation "
               f"[{zone_low:.5f}, {zone_high:.5f}]"
    )]

    direction = "HOLD"
    target_price = stop_price = None
    in_zone = bool(latest["low"] <= zone_high and latest["high"] >= zone_low)
    if impulse_direction == "BUY" and in_zone and latest["close"] > zone_low:
        direction = "BUY"
        target_price = float(df.iloc[impulse_pos]["high"])
        stop_price = zone_low - SMC_STOP_BUFFER_ATR_MULT * atr_val
    elif impulse_direction == "SELL" and in_zone and latest["close"] < zone_high:
        direction = "SELL"
        target_price = float(df.iloc[impulse_pos]["low"])
        stop_price = zone_high + SMC_STOP_BUFFER_ATR_MULT * atr_val

    reasons.append(SignalReason(
        rule="supply_demand_retest", passed=direction != "HOLD",
        detail=(f"Price retesting the {'demand' if impulse_direction == 'BUY' else 'supply'} zone "
                f"[{zone_low:.5f}, {zone_high:.5f}], continuation {direction} expected")
        if direction != "HOLD" else "Price not currently retesting the consolidation zone"
    ))

    strength = None
    if direction in ("BUY", "SELL"):
        impulse_atr = float(df.iloc[impulse_pos]["atr"])
        impulse_body = abs(impulse_close - impulse_open)
        strength = _clamp01(impulse_body / (impulse_atr * IMPULSE_ATR_MULT * 2)) if impulse_atr > 0 else None

    return StrategyCall(
        strategy="supply_demand", direction=direction,
        entry_price=entry_price, target_price=target_price, stop_price=stop_price,
        reasons=reasons, strength=strength,
    )


STRATEGIES: list[Callable[..., StrategyCall]] = [
    call_market_structure, call_order_blocks, call_fair_value_gap, call_liquidity_sweep, call_supply_demand,
]
