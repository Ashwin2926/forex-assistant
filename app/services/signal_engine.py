import pandas as pd
from datetime import datetime
from typing import Optional
from app.models.schemas import Signal, SignalReason, RuleConfig
from app.services.indicators import add_all_indicators


def apply_rules(
    latest: pd.Series, prev: pd.Series, config: RuleConfig = RuleConfig()
) -> tuple[list[SignalReason], int, int, int, dict[str, str]]:
    """
    Core rule logic, isolated from indicator computation and dataframe slicing so
    both the live engine and the backtester evaluate the exact same decision code
    against a row of already-computed indicators. Returns (reasons, bullish_votes,
    bearish_votes, total_rules, rule_votes) where rule_votes maps each rule that
    fired to the direction ("BUY"/"SELL") it voted for — used by the backtester to
    score which specific rules were pulling their weight vs. dead weight.
    """
    reasons: list[SignalReason] = []
    rule_votes: dict[str, str] = {}
    bullish_votes = 0
    bearish_votes = 0
    total_rules = 0

    # Rule 1: Trend - price vs fast/slow EMA. value is the EMA spread normalized by price
    # (%) rather than the raw EMA levels themselves — raw levels aren't comparable across
    # pairs (EUR/USD ~1.1 vs USD/JPY ~150) or over time as a price drifts, so they're
    # useless as a feature; the normalized spread is scale-invariant and its sign alone
    # already encodes uptrend/downtrend/flat.
    total_rules += 1
    ema_spread_pct = (latest["ema_fast"] - latest["ema_slow"]) / latest["close"] * 100
    if latest["ema_fast"] > latest["ema_slow"]:
        bullish_votes += 1
        rule_votes["trend_ema"] = "BUY"
        reasons.append(SignalReason(
            rule="trend_ema", passed=True, value=ema_spread_pct,
            detail=f"EMA{config.ema_fast} ({latest['ema_fast']:.5f}) above EMA{config.ema_slow} "
                   f"({latest['ema_slow']:.5f}) — uptrend context"
        ))
    elif latest["ema_fast"] < latest["ema_slow"]:
        bearish_votes += 1
        rule_votes["trend_ema"] = "SELL"
        reasons.append(SignalReason(
            rule="trend_ema", passed=True, value=ema_spread_pct,
            detail=f"EMA{config.ema_fast} ({latest['ema_fast']:.5f}) below EMA{config.ema_slow} "
                   f"({latest['ema_slow']:.5f}) — downtrend context"
        ))
    else:
        reasons.append(SignalReason(
            rule="trend_ema", passed=False, value=ema_spread_pct, detail="EMAs flat, no clear trend"
        ))

    # Rule 2: Momentum - RSI
    total_rules += 1
    if latest["rsi"] < config.rsi_oversold:
        bullish_votes += 1
        rule_votes["rsi_oversold"] = "BUY"
        reasons.append(SignalReason(
            rule="rsi_oversold", passed=True, value=float(latest["rsi"]),
            detail=f"RSI at {latest['rsi']:.1f} — oversold, potential bounce"
        ))
    elif latest["rsi"] > config.rsi_overbought:
        bearish_votes += 1
        rule_votes["rsi_overbought"] = "SELL"
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
    # crossed/didn't-cross boolean — its sign and magnitude carry real information (how
    # far above/below the signal line, not just whether a cross happened this bar) that
    # the pass/fail flag alone throws away.
    total_rules += 1
    macd_hist = float(latest["macd"] - latest["macd_signal"])
    macd_cross_up = prev["macd"] <= prev["macd_signal"] and latest["macd"] > latest["macd_signal"]
    macd_cross_down = prev["macd"] >= prev["macd_signal"] and latest["macd"] < latest["macd_signal"]
    if macd_cross_up:
        bullish_votes += 1
        rule_votes["macd_cross"] = "BUY"
        reasons.append(SignalReason(
            rule="macd_cross", passed=True, value=macd_hist, detail="MACD crossed above signal line"
        ))
    elif macd_cross_down:
        bearish_votes += 1
        rule_votes["macd_cross"] = "SELL"
        reasons.append(SignalReason(
            rule="macd_cross", passed=True, value=macd_hist, detail="MACD crossed below signal line"
        ))
    else:
        reasons.append(SignalReason(
            rule="macd_cross", passed=False, value=macd_hist, detail="No recent MACD crossover"
        ))

    # Rule 4: Volatility filter - skip signals when ATR indicates dead market
    atr_pct = (latest["atr"] / latest["close"]) * 100
    volatility_ok = atr_pct > config.volatility_threshold_pct
    reasons.append(SignalReason(
        rule="volatility_filter", passed=volatility_ok, value=float(atr_pct),
        detail=f"ATR is {atr_pct:.4f}% of price — {'sufficient' if volatility_ok else 'too low, likely illiquid session'}"
    ))

    # Rule 5: Session filter (intraday only) - skip signals outside the highest-liquidity
    # window. Not a vote — like volatility_filter, a failure here forces HOLD regardless
    # of what rules 1-3 say. Swing profile leaves this disabled and always passes.
    session_ok = True
    if config.session_filter_enabled:
        hour = pd.Timestamp(latest["timestamp"]).hour
        session_ok = config.session_start_hour_utc <= hour < config.session_end_hour_utc
        reasons.append(SignalReason(
            rule="session_filter", passed=session_ok, value=float(hour),
            detail=f"{hour:02d}:00 UTC — {'within' if session_ok else 'outside'} the "
                   f"{config.session_start_hour_utc:02d}:00-{config.session_end_hour_utc:02d}:00 UTC window"
        ))

    return reasons, bullish_votes, bearish_votes, total_rules, rule_votes


def decide(bullish_votes: int, bearish_votes: int, total_rules: int, gate_ok: bool) -> tuple[str, float]:
    """
    Turn rule votes into a direction + confidence. gate_ok combines every pass/fail
    gating condition (volatility_filter and, for intraday, session_filter) — if any
    gate fails, force HOLD regardless of vote counts. Shared by live engine and backtester.
    """
    confidence = round((max(bullish_votes, bearish_votes) / total_rules) * 100, 1)

    if not gate_ok:
        return "HOLD", 0.0
    elif bullish_votes > bearish_votes:
        return "BUY", confidence
    elif bearish_votes > bullish_votes:
        return "SELL", confidence
    else:
        return "HOLD", confidence


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

    reasons, bullish_votes, bearish_votes, total_rules, _rule_votes = apply_rules(latest, prev, config)
    volatility_ok = next(r.passed for r in reasons if r.rule == "volatility_filter")
    session_ok = next((r.passed for r in reasons if r.rule == "session_filter"), True)
    direction, confidence = decide(bullish_votes, bearish_votes, total_rules, volatility_ok and session_ok)

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
# split — see README "Intraday vs swing" for the numbers. Re-run optimize and update these
# if the ruleset changes or a wider candle sample tells a different story; don't hand-edit
# them back to a guess.
#
# intraday: cross-pair validated on 15min data, 5000 candles/pair (all 4 majors) — an
# earlier pass on only 2000 candles/pair had picked EMA 12/26 based on EUR/USD alone and
# didn't hold up cross-pair (see README "Intraday vs swing" for that history). With enough
# data, all four pairs converge on EMA 9/21 + session filter (12:00-16:00 UTC) + a tighter
# volatility_threshold_pct=0.02, each clearly beating swing's ~30% baseline with a believable
# 4-6pt train->test drop: EUR/USD 37.9%/32.4%, GBP/USD 44.6%/38.6%, USD/JPY 41.9%/37.7%,
# AUD/USD 37.4%/31.3%.
# swing: EUR/USD 1h, 1780 candles — EMA 50/200 (hit-rate winner) still won on hit_rate_pct,
# but adding expectancy_pct (mean pct_move per signal, not just win/loss-bucket averages)
# revealed the ORIGINAL target_atr_mult=1.5/stop_atr_mult=1.0 was negative-expectancy on
# every single grid candidate, on every pair — a 29-31% hit-rate looks plausible in
# isolation but is well under the 40% breakeven that ratio requires (stop/(target+stop)).
# target/stop themselves are now RuleConfig fields (not fixed endpoint params) specifically
# so /backtest/optimize can search them. Sweeping found target=0.5/stop=1.25 (a much closer
# target, a more generous stop) as a real, cross-pair-validated improvement: EUR/USD went
# clearly positive (train +0.0024%/test +0.0051%, 486 test signals), GBP/USD went from
# -0.004% to roughly breakeven (-0.0001%/+0.0002%), USD/JPY improved but stayed negative
# (-0.0072%/-0.0046% — consistent both sides, a real pair-specific shortfall, not noise),
# AUD/USD showed a train/test sign flip (+0.007%/-0.0032%) — the same overfitting signature
# as the earlier intraday lesson, so don't fully trust that pair's number. Net: strictly
# better than the original 1.5/1.0 (negative everywhere) on every pair, genuinely profitable
# on EUR/USD, but swing is not uniformly profitable across all four pairs yet — per-pair
# target/stop tuning is a real, still-open gap, not a solved problem.
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
        # EUR/USD), unlike swing this didn't need retuning.
    ),
    "swing": RuleConfig(
        target_atr_mult=0.5,
        stop_atr_mult=1.25,
    ),
}

# Per-pair swing overrides, closing the gap the global 0.5/1.25 default left open (see
# README "Per-pair swing target/stop tuning"). Found via /backtest/optimize on
# ~2500-2700 candles/pair, 1h, train_frac=0.7. Only overriding where train and test
# *agree in sign* — a pair whose best train candidate flips sign on the untouched test
# slice is the same overfitting signature already documented for AUD/USD's intraday
# result, and is not trustworthy just because one slice looks good.
#
# Round 1 (target/stop only, EMA 50/200 held fixed):
#   EUR/USD 0.75/1.25 — train +0.0025%/test +0.0037% (2518/1260 signals): real improvement,
#     same sign, better than the 0.5/1.25 global default's +0.0006% here.
#   GBP/USD 0.75/1.5  — train -0.0002%/test -0.0012% (2564/1133 signals): still negative,
#     but consistent and less negative than the global default's -0.0046% here.
#   USD/JPY 0.5/2.0   — train -0.0074%/test -0.0023% (2566/1151 signals): still negative,
#     but consistent and meaningfully better than the global default's -0.0123% here.
# AUD/USD had no override after round 1: its best train candidate (1.0/1.25, -0.0023%)
# flipped positive on test (+0.0021%, 2681/1249 signals) — a sign flip, not a validated
# improvement.
#
# Round 2 (EMA x RSI grid, each pair's round-1 target/stop held fixed): tested 9
# combinations (EMA 50/200, 20/100, 10/50 x RSI 30/70, 25/75, 35/65) per pair.
#   EUR/USD's best (EMA 20/100) flipped sign: train +0.0045%/test -0.0016% — rejected,
#     kept EMA 50/200.
#   GBP/USD's best (EMA 10/50, RSI 35/65) flipped sign: train +0.0017%/test -0.0053% —
#     rejected, kept EMA 50/200.
#   AUD/USD's best was just the unmodified global default re-evaluated (EMA 50/200,
#     RSI 30/70) and still flipped sign (train -0.0047%/test +0.0033%) — still no
#     override, same open gap as round 1.
#   USD/JPY's best (EMA 10/50, RSI 25/75) held sign: train -0.0045%/test -0.0011%
#     (2709/1160 signals) — a real improvement over round 1's -0.0074%/-0.0023%, so
#     re-ran a target/stop refinement under this new EMA/RSI (round 3 below) instead of
#     stopping here.
#
# Round 3 (USD/JPY only — target/stop refined under its new EMA 10/50 + RSI 25/75):
#   0.75/2.0 won: train -0.0030%/test -0.0014% (2709/1160 signals) — same sign, the best
#   result found for USD/JPY across all three rounds. Still net-negative — this is a
#   real, still-open pair-specific shortfall, not a solved problem — but each round made
#   it measurably less bad without ever trusting a sign-flipped "winner."
#
# Round 4 (MACD period tuning, all 4 pairs except EUR/USD — already profitable, not a
# tuning target here — each pair's round 1-3 config held fixed; then
# volatility_threshold_pct sweep, all 4 pairs, same discipline):
#   MACD: every pair's best candidate either showed a negligible train/test delta
#   (GBP/USD 8/17/9: train -0.0004%/test -0.0011%, barely different from the 12/26/9
#   default's train -0.0006%) or flipped sign (USD/JPY 5/13/6: train -0.0028%/test
#   +0.0005%; AUD/USD 8/17/9: train -0.0038%/test +0.0049%) — no pair got a MACD
#   override, all keep the RuleConfig default (12/26/9).
#   volatility_threshold_pct (grid: 0.01/0.02/0.03/0.05, 0.02 is RuleConfig's untouched
#   default that swing had never actually searched before this round):
#     EUR/USD 0.05 — train +0.0031%/test +0.0032% (2214/1090 signals): same sign, stable,
#       a real improvement over 0.02's train +0.0025% here — adopted.
#     GBP/USD 0.03 — train -0.0001%/test -0.0006% (2477/1058 signals): same sign, both
#       numbers improved over 0.02's train -0.0006%/test -0.0012% (from round 1) — still
#       net-negative but measurably less bad — adopted.
#     USD/JPY 0.05 — train -0.0018%/test +0.0024%: sign flip — rejected, kept vol=0.02.
#     AUD/USD 0.05 — train -0.0046%/test +0.0049%: sign flip — rejected, kept vol=0.02.
#   AUD/USD re-evaluation (per README "What's next" item 3): across all four rounds
#   (EMA/RSI, target/stop, MACD, volatility) every single "winning" candidate for AUD/USD
#   flipped sign train→test — never once a validated same-sign improvement. Per the plan
#   set out after round 2, that's accepted as real evidence swing has no edge for this
#   pair in this rule set, not just an under-searched grid — AUD/USD keeps the global
#   default with no override, and the search is considered closed rather than open-ended.
SWING_PAIR_OVERRIDES: dict[str, dict] = {
    "EUR/USD": {"target_atr_mult": 0.75, "stop_atr_mult": 1.25, "volatility_threshold_pct": 0.05},
    "GBP/USD": {"target_atr_mult": 0.75, "stop_atr_mult": 1.5, "volatility_threshold_pct": 0.03},
    "USD/JPY": {
        "ema_fast": 10, "ema_slow": 50,
        "rsi_oversold": 25, "rsi_overbought": 75,
        "target_atr_mult": 0.75, "stop_atr_mult": 2.0,
    },
}


def default_config_for(profile: str, pair: Optional[str] = None) -> RuleConfig:
    config = PROFILE_DEFAULTS.get(profile, RuleConfig())
    if profile == "swing" and pair in SWING_PAIR_OVERRIDES:
        config = config.model_copy(update=SWING_PAIR_OVERRIDES[pair])
    return config
