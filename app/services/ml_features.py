from app.models.schemas import SignalReason

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

# The known pair/interval universe this project trades -- duplicated here (rather than
# imported from app.core.config/app.main) to avoid a circular import, since app.main already
# imports this module by way of ml_model. Keep in sync with Settings.pairs_list's default and
# app.main.RL_INTERVALS if either ever changes. A pair/interval outside these lists (shouldn't
# happen, but data can outlive config changes) simply gets an all-zero one-hot -- "unknown",
# not an error.
KNOWN_PAIRS: list[str] = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD"]
KNOWN_INTERVALS: list[str] = ["5min", "15min", "1h", "4h", "1day"]

# session_hour/session_filter never appears on a swing signal (no session_filter rule for
# that profile) -- 0.0 there means "not applicable," a real and meaningful state, not a
# stand-in for data that should have existed but is missing.
#
# pair_*/interval_* one-hots were added because a single shared model previously had no way
# to tell EUR/USD 5min apart from USD/JPY 1h -- profile_intraday only distinguishes the two
# broad buckets, not the individual pair/interval, even though live accuracy is known to
# differ a lot between them (see /rl/accuracy, /signals/accuracy per-pair/interval).
FEATURE_NAMES: list[str] = [
    "ema_spread_pct", "rsi", "macd_hist", "atr_pct", "session_hour",
    "confidence", "profile_intraday", "direction_buy",
    *[f"pair_{p.replace('/', '')}" for p in KNOWN_PAIRS],
    *[f"interval_{iv}" for iv in KNOWN_INTERVALS],
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

    pair_feature = f"pair_{(signal.get('pair') or '').replace('/', '')}"
    if pair_feature in features:
        features[pair_feature] = 1.0
    interval_feature = f"interval_{signal.get('interval')}"
    if interval_feature in features:
        features[interval_feature] = 1.0

    return features


def direction_confidence(
    direction: str, gate_ok: bool, rule_votes: dict[str, str], rule_strengths: dict[str, float], total_rules: int,
) -> float:
    """
    Same rule-agreement-strength formula signal_engine.decide() uses for whichever direction
    the vote count actually won, generalized to a CALLER-CHOSEN direction instead. Needed
    because rl_engine.compute_ml_scores scores what a BUY *and* a SELL would each look like
    at every bar (RL hasn't committed to a direction yet when this runs, unlike a real
    generated Signal which only ever reports confidence for the direction it settled on).
    Mirrors decide()'s "gate fails -> zero confidence, regardless of direction" rule exactly,
    and its "confidence" formula (sum of agreeing rules' strengths / total_rules, capped at
    1.0) for the requested direction specifically rather than whichever direction won the
    vote count.
    """
    if not gate_ok or total_rules == 0:
        return 0.0
    agreeing_strength = sum(
        rule_strengths.get(rule, 1.0) for rule, voted in rule_votes.items() if voted == direction
    )
    return round(min(agreeing_strength / total_rules, 1.0) * 100, 1)


def signal_like_features(
    reasons: list[SignalReason], rule_votes: dict[str, str], rule_strengths: dict[str, float], total_rules: int,
    profile: str, direction: str, pair: str, interval: str,
) -> dict[str, float]:
    """
    Builds the same dict shape extract_features() expects (a stored Signal document), from
    apply_rules()'s raw return values directly instead of a full persisted Signal -- for
    callers scoring a hypothetical BUY/SELL at a bar that was never actually turned into a
    real Signal (see rl_engine.compute_ml_scores: calling signal_engine.generate_signal per
    training bar would recompute every indicator from scratch on every call, prohibitively
    expensive inside a loop that already visits hundreds of bars x hundreds of episodes --
    apply_rules alone is cheap, pure row arithmetic against an already-computed indicator_df).

    direction: "BUY" or "SELL", the hypothetical this call is scoring -- NOT necessarily
    whatever direction the vote count would have picked (see direction_confidence).
    """
    volatility_ok = next((r.passed for r in reasons if r.rule == "volatility_filter"), True)
    session_ok = next((r.passed for r in reasons if r.rule == "session_filter"), True)
    confidence = direction_confidence(direction, volatility_ok and session_ok, rule_votes, rule_strengths, total_rules)
    signal_like = {
        "reasons": [r.model_dump() for r in reasons],
        "confidence": confidence,
        "profile": profile,
        "direction": direction,
        "pair": pair,
        "interval": interval,
    }
    return extract_features(signal_like)
