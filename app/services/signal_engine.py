import pandas as pd
from datetime import datetime
from typing import Optional
from app.models.schemas import Signal, SignalReason, RuleConfig
from app.services.indicators import add_all_indicators


def apply_rules(
    latest: pd.Series, prev: pd.Series, config: RuleConfig = RuleConfig()
) -> tuple[list[SignalReason], int, int, int, dict[str, str], dict[str, float]]:
    """
    Core rule logic, isolated from indicator computation and dataframe slicing so
    both the live engine and the backtester evaluate the exact same decision code
    against a row of already-computed indicators. Returns (reasons, bullish_votes,
    bearish_votes, total_rules, rule_votes, rule_strengths):
      rule_votes maps each rule that fired to the direction ("BUY"/"SELL") it voted
      for — used by the backtester to score which specific rules were pulling their
      weight vs. dead weight.
      rule_strengths maps each voting rule to a normalized [0, 1] "how strong was
      this reading" score — e.g. an RSI of 15 is much stronger oversold evidence
      than an RSI of 29 even though both cast the same vote at the old flat-count
      confidence. decide() uses this to weight confidence instead of just counting
      votes, so two signals with the same direction and vote count no longer always
      report the exact same confidence.
    """
    reasons: list[SignalReason] = []
    rule_votes: dict[str, str] = {}
    rule_strengths: dict[str, float] = {}
    bullish_votes = 0
    bearish_votes = 0
    total_rules = 0

    # ATR% computed up front — used both as its own filter (rule 4, below) and as the
    # volatility yardstick trend_ema/macd_cross strength is normalized against, so an EMA
    # spread or MACD histogram reads as "big" relative to how much this pair/interval is
    # actually moving right now, not against one fixed number that wouldn't generalize
    # across pairs with very different typical price scales (EUR/USD ~1.1 vs USD/JPY ~150).
    atr_pct = (latest["atr"] / latest["close"]) * 100

    # Rule 1: Trend - price vs fast/slow EMA. value is the EMA spread normalized by price
    # (%) rather than the raw EMA levels themselves — raw levels aren't comparable across
    # pairs or over time as a price drifts, so they're useless as a feature; the
    # normalized spread is scale-invariant and its sign alone already encodes
    # uptrend/downtrend/flat.
    total_rules += 1
    ema_spread_pct = (latest["ema_fast"] - latest["ema_slow"]) / latest["close"] * 100
    if latest["ema_fast"] > latest["ema_slow"]:
        bullish_votes += 1
        rule_votes["trend_ema"] = "BUY"
        rule_strengths["trend_ema"] = min(abs(ema_spread_pct) / atr_pct, 1.0) if atr_pct > 0 else 0.0
        reasons.append(SignalReason(
            rule="trend_ema", passed=True, value=ema_spread_pct,
            detail=f"EMA{config.ema_fast} ({latest['ema_fast']:.5f}) above EMA{config.ema_slow} "
                   f"({latest['ema_slow']:.5f}) — uptrend context"
        ))
    elif latest["ema_fast"] < latest["ema_slow"]:
        bearish_votes += 1
        rule_votes["trend_ema"] = "SELL"
        rule_strengths["trend_ema"] = min(abs(ema_spread_pct) / atr_pct, 1.0) if atr_pct > 0 else 0.0
        reasons.append(SignalReason(
            rule="trend_ema", passed=True, value=ema_spread_pct,
            detail=f"EMA{config.ema_fast} ({latest['ema_fast']:.5f}) below EMA{config.ema_slow} "
                   f"({latest['ema_slow']:.5f}) — downtrend context"
        ))
    else:
        reasons.append(SignalReason(
            rule="trend_ema", passed=False, value=ema_spread_pct, detail="EMAs flat, no clear trend"
        ))

    # Rule 2: Momentum - RSI. strength is how far past its threshold the reading is,
    # scaled against the natural 0-100 RSI bound (no ATR normalization needed here —
    # RSI is already unitless and self-bounded).
    total_rules += 1
    if latest["rsi"] < config.rsi_oversold:
        bullish_votes += 1
        rule_votes["rsi_oversold"] = "BUY"
        rule_strengths["rsi_oversold"] = max(0.0, min(
            (config.rsi_oversold - latest["rsi"]) / config.rsi_oversold, 1.0
        ))
        reasons.append(SignalReason(
            rule="rsi_oversold", passed=True, value=float(latest["rsi"]),
            detail=f"RSI at {latest['rsi']:.1f} — oversold, potential bounce"
        ))
    elif latest["rsi"] > config.rsi_overbought:
        bearish_votes += 1
        rule_votes["rsi_overbought"] = "SELL"
        rule_strengths["rsi_overbought"] = max(0.0, min(
            (latest["rsi"] - config.rsi_overbought) / (100 - config.rsi_overbought), 1.0
        ))
        reasons.append(SignalReason(
            rule="rsi_overbought", passed=True, value=float(latest["rsi"]),
            detail=f"RSI at {latest['rsi']:.1f} — overbought, potential pullback"
        ))
    else:
        reasons.append(SignalReason(
            rule="rsi_neutral", passed=False, value=float(latest["rsi"]),
            detail=f"RSI at {latest['rsi']:.1f} — no extreme"
        ))

    # Rule 3: MACD crossover. value is the histogram (macd - signal) rather than a bare
    # crossed/didn't-cross boolean — its sign and magnitude carry real information that a
    # pass/fail flag alone throws away. strength normalizes that histogram (converted to
    # a % of price, same units as ema_spread_pct) against ATR%, same reasoning as trend_ema.
    total_rules += 1
    macd_hist = float(latest["macd"] - latest["macd_signal"])
    macd_hist_pct = macd_hist / latest["close"] * 100
    macd_cross_up = prev["macd"] <= prev["macd_signal"] and latest["macd"] > latest["macd_signal"]
    macd_cross_down = prev["macd"] >= prev["macd_signal"] and latest["macd"] < latest["macd_signal"]
    if macd_cross_up:
        bullish_votes += 1
        rule_votes["macd_cross"] = "BUY"
        rule_strengths["macd_cross"] = min(abs(macd_hist_pct) / atr_pct, 1.0) if atr_pct > 0 else 0.0
        reasons.append(SignalReason(
            rule="macd_cross", passed=True, value=macd_hist, detail="MACD crossed above signal line"
        ))
    elif macd_cross_down:
        bearish_votes += 1
        rule_votes["macd_cross"] = "SELL"
        rule_strengths["macd_cross"] = min(abs(macd_hist_pct) / atr_pct, 1.0) if atr_pct > 0 else 0.0
        reasons.append(SignalReason(
            rule="macd_cross", passed=True, value=macd_hist, detail="MACD crossed below signal line"
        ))
    else:
        reasons.append(SignalReason(
            rule="macd_cross", passed=False, value=macd_hist, detail="No recent MACD crossover"
        ))

    # Rule 4: Volatility filter - skip signals when ATR indicates dead market
    volatility_ok = atr_pct > config.volatility_threshold_pct
    reasons.append(SignalReason(
        rule="volatility_filter", passed=volatility_ok, value=float(atr_pct),
        detail=f"ATR is {atr_pct:.4f}% of price — {'sufficient' if volatility_ok else 'too low, likely illiquid session'}"
    ))

    # Rule 5: Session filter - skip signals outside the highest-liquidity window. Not a
    # vote — like volatility_filter, a failure here forces HOLD regardless of what rules
    # 1-3 say. Only applies when config.session_filter_enabled is set; a config with it
    # off leaves this disabled and always passes.
    session_ok = True
    if config.session_filter_enabled:
        hour = pd.Timestamp(latest["timestamp"]).hour
        session_ok = config.session_start_hour_utc <= hour < config.session_end_hour_utc
        reasons.append(SignalReason(
            rule="session_filter", passed=session_ok, value=float(hour),
            detail=f"{hour:02d}:00 UTC — {'within' if session_ok else 'outside'} the "
                   f"{config.session_start_hour_utc:02d}:00-{config.session_end_hour_utc:02d}:00 UTC window"
        ))

    return reasons, bullish_votes, bearish_votes, total_rules, rule_votes, rule_strengths


def decide(
    bullish_votes: int, bearish_votes: int, total_rules: int, gate_ok: bool,
    rule_votes: dict[str, str], rule_strengths: dict[str, float],
) -> tuple[str, float]:
    """
    Turn rule votes into a direction + confidence. gate_ok combines every pass/fail
    gating condition (volatility_filter and, for intraday, session_filter) — if any
    gate fails, force HOLD regardless of vote counts. Shared by live engine and backtester.

    Direction is still decided purely by vote count (bullish_votes vs bearish_votes) —
    unchanged from before, so every backtested hit-rate number in signal_engine.py's
    PROFILE_DEFAULTS history stays valid; only how *confidence* is computed changed.
    Previously confidence was max(bullish_votes, bearish_votes) / total_rules, which can
    only ever take 4 values for total_rules=3 (0%, 33.3%, 66.7%, 100%) — every signal
    where e.g. 2 of 3 rules agreed reported the identical 66.7% regardless of whether
    those rules were barely over their threshold or deep into it. Now it's the *sum of
    rule_strengths* for whichever rules agreed with the winning direction, divided by
    total_rules — a continuous value in the same [0, 100] range, and equal to the old
    formula exactly when every agreeing rule's strength is 1.0.
    """
    if not gate_ok:
        return "HOLD", 0.0

    if bullish_votes > bearish_votes:
        direction = "BUY"
    elif bearish_votes > bullish_votes:
        direction = "SELL"
    else:
        # Tie (including 0-0, no rule voting either way) -- no winning side to weight
        # confidence against, so fall back to the plain vote-count formula.
        confidence = round((max(bullish_votes, bearish_votes) / total_rules) * 100, 1)
        return "HOLD", confidence

    agreeing_strength = sum(
        rule_strengths.get(rule, 1.0) for rule, voted in rule_votes.items() if voted == direction
    )
    confidence = round(min(agreeing_strength / total_rules, 1.0) * 100, 1)
    return direction, confidence


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


def generate_signal(
    df: pd.DataFrame, pair: str, interval: str, profile: str, config: RuleConfig = RuleConfig()
) -> Signal:
    """
    Rule-based signal generation. Fully deterministic and explainable -
    every decision is logged as a SignalReason so you can audit *why*
    a signal fired, and later score whether it was right.

    df: OHLCV dataframe, ascending by timestamp, at least config.ema_slow rows for the
    slow EMA to be meaningful.
    """
    df = add_all_indicators(df, config)
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    reasons, bullish_votes, bearish_votes, total_rules, rule_votes, rule_strengths = apply_rules(latest, prev, config)
    volatility_ok = next(r.passed for r in reasons if r.rule == "volatility_filter")
    session_ok = next((r.passed for r in reasons if r.rule == "session_filter"), True)
    direction, confidence = decide(
        bullish_votes, bearish_votes, total_rules, volatility_ok and session_ok, rule_votes, rule_strengths
    )

    return Signal(
        pair=pair,
        profile=profile,
        interval=interval,
        timestamp=datetime.utcnow(),
        direction=direction,
        confidence=confidence,
        reasons=reasons,
        price_at_signal=float(latest["close"]),
    )


# Per-profile defaults, validated (not guessed) via /backtest/optimize with a train/test
# split. Re-run optimize and update these if the ruleset changes or a wider candle sample
# tells a different story; don't hand-edit them back to a guess.
#
# intraday: cross-pair validated on 15min data, 5000 candles/pair (all 4 majors) — an
# earlier pass on only 2000 candles/pair had picked EMA 12/26 based on EUR/USD alone and
# didn't hold up cross-pair. With enough data, all four pairs converge on EMA 9/21 +
# session filter (12:00-16:00 UTC) + a tighter volatility_threshold_pct=0.02, with a
# believable 4-6pt train->test drop: EUR/USD 37.9%/32.4%, GBP/USD 44.6%/38.6%, USD/JPY
# 41.9%/37.7%, AUD/USD 37.4%/31.3%. This is now the only profile — every interval
# (5min through 1day) is evaluated with this same validated config rather than a
# separate longer-horizon ruleset, so a signal on any timeframe is judged by the one
# ruleset that's actually been cross-pair backtest-validated.
PROFILE_DEFAULTS: dict[str, RuleConfig] = {
    "intraday": RuleConfig(
        ema_fast=9,
        ema_slow=21,
        rsi_period=14,
        atr_period=14,
        volatility_threshold_pct=0.02,
        session_filter_enabled=True,
        session_start_hour_utc=12,
        session_end_hour_utc=16,
        # target_atr_mult/stop_atr_mult left at RuleConfig's own 1.5/1.0 default — already
        # modestly positive and consistent (train +0.0005%/test +0.0002%, no sign flip on
        # EUR/USD).
    ),
}


def default_config_for(profile: str, pair: Optional[str] = None) -> RuleConfig:
    return PROFILE_DEFAULTS.get(profile, RuleConfig())
