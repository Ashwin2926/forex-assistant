from app.models.schemas import StrategyCall
from app.services.strategies import STRATEGY_NAMES

# One feature per SMC strategy vote, reusing strategies.STRATEGY_NAMES rather than a second
# hand-typed list, so this can't drift out of sync with RL's own vote feature names
# (rl_engine.RL_MARKET_FEATURE_NAMES uses the exact same names with a "_vote" suffix, also
# imported from strategies.py). Signed strength in [-1, 1] -- positive for a BUY-aligned
# call, negative for SELL-aligned, 0 for HOLD/disagreement with whichever direction is being
# scored -- same encoding rl_engine.compute_strategy_vote_states already uses.
STRATEGY_VOTE_FEATURES: list[str] = [f"{name}_vote" for name in STRATEGY_NAMES]

# The known pair/interval universe this project trades -- duplicated here (rather than
# imported from app.core.config/app.main) to avoid a circular import, since app.main already
# imports this module by way of ml_model. Keep in sync with Settings.pairs_list's default and
# app.main.RL_INTERVALS if either ever changes. A pair/interval outside these lists (shouldn't
# happen, but data can outlive config changes) simply gets an all-zero one-hot -- "unknown",
# not an error.
KNOWN_PAIRS: list[str] = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD"]
KNOWN_INTERVALS: list[str] = ["5min", "15min", "1h", "4h", "1day"]

# pair_*/interval_* one-hots let a single shared model tell EUR/USD 5min apart from USD/JPY
# 1h -- profile_intraday is currently constant (there's only one profile), so it's these
# one-hots that carry the meaningful pair/interval distinction, even though live accuracy is
# known to differ a lot between them (see /rl/accuracy, /rl/insights per-pair/interval).
FEATURE_NAMES: list[str] = [
    *STRATEGY_VOTE_FEATURES, "atr_pct",
    "confidence", "profile_intraday", "direction_buy",
    *[f"pair_{p.replace('/', '')}" for p in KNOWN_PAIRS],
    *[f"interval_{iv}" for iv in KNOWN_INTERVALS],
]


def extract_features(signal: dict) -> dict[str, float]:
    """
    Maps a stored ConsensusSignal document (a raw dict, as returned by
    consensus_signals_collection.find() -- not the Pydantic model) to a flat numeric feature
    dict for the ML classifier. The single place this mapping lives, so training
    (ml_model.py) and live prediction can never drift out of sync with each other.

    Reads each strategy_call's direction/strength into its own `{strategy}_vote` feature
    (signed: positive for a call agreeing with `signal["direction"]`, negative for a call
    opposing it, 0 for HOLD) -- the same signed-strength encoding RL's own market state
    already uses (rl_engine.compute_strategy_vote_states), so ML and RL read the same market
    evidence the same way.

    confidence/profile_intraday/direction_buy come from the signal itself, not its strategy
    calls -- including the consensus's own agreeing-weight-fraction confidence as a feature
    lets the model learn whether that score is itself predictive, rather than assuming it and
    hand-coding the relationship.
    """
    features = {name: 0.0 for name in FEATURE_NAMES}
    direction = signal.get("direction")

    for call in signal.get("strategy_calls", []):
        vote_feature = f"{call.get('strategy')}_vote"
        if vote_feature not in features:
            continue
        call_direction = call.get("direction")
        if call_direction == "HOLD" or call_direction is None:
            continue
        strength = call.get("strength") or 0.0
        features[vote_feature] = float(strength) if call_direction == direction else -float(strength)

    atr_pct = signal.get("atr_pct")
    if atr_pct is not None:
        features["atr_pct"] = float(atr_pct)

    features["confidence"] = float(signal.get("confidence") or 0.0)
    features["profile_intraday"] = 1.0 if signal.get("profile", "intraday") == "intraday" else 0.0
    features["direction_buy"] = 1.0 if direction == "BUY" else 0.0

    pair_feature = f"pair_{(signal.get('pair') or '').replace('/', '')}"
    if pair_feature in features:
        features[pair_feature] = 1.0
    interval_feature = f"interval_{signal.get('interval')}"
    if interval_feature in features:
        features[interval_feature] = 1.0

    return features


def is_current_smc_signal(signal: dict) -> bool:
    """
    Whether a stored ConsensusSignal's strategy_calls match today's 5 SMC strategies exactly
    -- False for a signal generated under the RETIRED 7-strategy lineup (trend/bollinger/
    support_resistance/candlestick/stoch_adx/volume_momentum/smart_money), which still sits in
    consensus_signals_collection from before strategies.py's SMC rewrite (confirmed live: GET
    /consensus still returns several of these old-shaped records). Filtering these out matters
    the same way the old rule engine's EMA12/26-vs-EMA9/21 archive contamination did (see
    PROGRESS.md): a stale-shaped record's strategy_calls have names that don't match any
    current `{name}_vote` feature, so every vote feature would silently read as 0.0/HOLD for
    it -- misleading training noise, not a crash, but noise that should be kept out.
    """
    calls = signal.get("strategy_calls") or []
    return bool(calls) and all(c.get("strategy") in STRATEGY_NAMES for c in calls)


def signal_like_features(
    strategy_calls: list[StrategyCall], confidence: float, atr_pct: float,
    direction: str, pair: str, interval: str,
) -> dict[str, float]:
    """
    Builds the same dict shape extract_features() expects (a stored ConsensusSignal
    document), from a bar's raw StrategyCall list directly instead of a full persisted
    ConsensusSignal -- for callers scoring a hypothetical BUY/SELL at a bar that was never
    actually turned into a real signal (see rl_engine.compute_ml_scores: recomputing SMC
    strategies inside a training loop that already visits hundreds of bars x hundreds of
    episodes needs the cheap, already-computed StrategyCall list, not a fresh
    consensus.check_consensus roundtrip).

    direction: "BUY" or "SELL", the hypothetical this call is scoring -- NOT necessarily
    whatever direction a real consensus would have picked.
    """
    signal_like = {
        "strategy_calls": [c.model_dump() if hasattr(c, "model_dump") else c for c in strategy_calls],
        "confidence": confidence,
        "atr_pct": atr_pct,
        "direction": direction,
        "pair": pair,
        "interval": interval,
    }
    return extract_features(signal_like)
