import pandas as pd
from typing import Optional


def find_swing_levels(df: pd.DataFrame, lookback: int = 20) -> tuple[list[float], list[float]]:
    """
    Resistance/support from recent price structure, causal only -- df is expected to already
    be sliced up to "now" (the caller's job, same convention as every other function in this
    codebase that walks bar-by-bar), so this never sees a future high/low. A true swing-pivot
    detector (requiring N bars on both sides to confirm a peak) would lag further behind price
    before confirming a level; taking the highest few highs / lowest few lows in the lookback
    window is a simpler approximation of "recent significant levels" that stays honest about
    only using information available at the time.

    Returns (resistance_levels, support_levels), each sorted nearest-to-price first.
    """
    window = df.iloc[-lookback:] if len(df) >= lookback else df
    resistance = sorted(window["high"].nlargest(3).unique().tolist(), reverse=True)
    support = sorted(window["low"].nsmallest(3).unique().tolist())
    return resistance, support


def detect_candlestick_pattern(latest: pd.Series, prev: pd.Series) -> Optional[str]:
    """
    Classic single/two-candle shape patterns from raw OHLC. Unlike find_swing_levels this
    needs no lookback window, just the current and immediately preceding bar. Returns the
    first matching pattern (checked in a fixed priority order: engulfing, then doji, then
    hammer/shooting star) or None -- a real pattern shouldn't fire on every candle, and a
    strategy built on this should treat "no pattern" as a HOLD, not force one.
    """
    body = abs(latest["close"] - latest["open"])
    candle_range = latest["high"] - latest["low"]
    if candle_range == 0:
        return None
    upper_wick = latest["high"] - max(latest["close"], latest["open"])
    lower_wick = min(latest["close"], latest["open"]) - latest["low"]

    prev_bullish = prev["close"] > prev["open"]
    latest_bullish = latest["close"] > latest["open"]

    # Engulfing: latest's real body fully contains prev's real body, opposite direction.
    if (
        prev_bullish and not latest_bullish
        and latest["open"] > prev["close"] and latest["close"] < prev["open"]
    ):
        return "bearish_engulfing"
    if (
        not prev_bullish and latest_bullish
        and latest["open"] < prev["close"] and latest["close"] > prev["open"]
    ):
        return "bullish_engulfing"

    # Doji: body is a small fraction of the bar's total range -- indecision.
    if body <= candle_range * 0.1:
        return "doji"

    # Hammer / shooting star: small body, one wick at least 2x the body, the other wick small.
    if body <= candle_range * 0.3:
        if lower_wick >= body * 2 and upper_wick <= body * 0.5:
            return "hammer"
        if upper_wick >= body * 2 and lower_wick <= body * 0.5:
            return "shooting_star"

    return None
