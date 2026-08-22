# Maps each rule name that can appear in a Signal's `reasons` to one canonical feature name.
# Three different RSI rule names (rsi_oversold/rsi_overbought/rsi_neutral -- only one ever
# fires per signal, see signal_engine.apply_rules) collapse to the same "rsi" feature, since
# they're all just "the RSI reading," not three different quantities.
FEATURE_RULE_MAP: dict[str, str] = {
    "trend_ema": "ema_spread_pct",
    "rsi_oversold": "rsi",
    "rsi_overbought": "rsi",
    "rsi_neutral": "rsi",
    "macd_cross": "macd_hist",
    "volatility_filter": "atr_pct",
    "session_filter": "session_hour",  # intraday only -- imputed 0.0 for swing, see below
}

# session_hour/session_filter never appears on a swing signal (no session_filter rule for
# that profile) -- 0.0 there means "not applicable," a real and meaningful state, not a
# stand-in for data that should have existed but is missing.
FEATURE_NAMES: list[str] = [
    "ema_spread_pct", "rsi", "macd_hist", "atr_pct", "session_hour",
    "confidence", "profile_intraday", "direction_buy",
]


def extract_features(signal: dict) -> dict[str, float]:
    """
    Maps a stored Signal document (a raw dict, as returned by signals_collection.find() --
    not the Pydantic model) to a flat numeric feature dict for the ML classifier. The single
    place this mapping lives, so training (ml_model.py) and live prediction (/ml/predict) can
    never drift out of sync with each other.

    confidence/profile_intraday/direction_buy come from the signal itself, not its reasons --
    including the existing rule-based confidence score as a feature lets the model learn
    whether that score is itself predictive, rather than assuming it and hand-coding the
    relationship.
    """
    features = {name: 0.0 for name in FEATURE_NAMES}

    for reason in signal.get("reasons", []):
        feature_name = FEATURE_RULE_MAP.get(reason.get("rule"))
        value = reason.get("value")
        if feature_name and value is not None:
            features[feature_name] = float(value)

    features["confidence"] = float(signal.get("confidence") or 0.0)
    features["profile_intraday"] = 1.0 if signal.get("profile") == "intraday" else 0.0
    features["direction_buy"] = 1.0 if signal.get("direction") == "BUY" else 0.0

    return features
