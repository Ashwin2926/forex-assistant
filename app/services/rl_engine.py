import math
import pandas as pd
from typing import Optional
from xgboost import XGBClassifier
from app.models.schemas import RuleConfig
from app.services.indicators import add_all_indicators
from app.services.signal_engine import compute_atr_target_stop, label_outcome, spread_cost_pct
from app.services.strategies import STRATEGIES, STRATEGY_NAMES
from app.services.consensus import direction_confidence
from app.services.ml_features import signal_like_features, FEATURE_NAMES as ML_FEATURE_NAMES
from app.services.ml_model import fit_hit_classifier

# Direction+size actions -- 2 tiers (not 3) deliberately, to keep the action space close to
# the original 3 (HOLD/BUY/SELL) rather than 7. Every extra action means fewer training
# samples per action -- keeping the space small matters for any RL algorithm trained on this
# project's real, limited candle history, PPO included.
ACTIONS = ["HOLD", "BUY_SMALL", "BUY_LARGE", "SELL_SMALL", "SELL_LARGE"]

# Starting guesses, not backtested -- same caveat as every other unvalidated constant in this
# project (PROXIMITY_ATR_MULT, SMC_WICK_BODY_MULT, etc.).
RISK_FRACTION_BY_TIER = {"SMALL": 0.01, "LARGE": 0.03}  # % of current balance risked
DEFAULT_STARTING_BALANCE = 50.0
MIN_VIABLE_BALANCE = 1.0  # below this, ruin -- can't size a real position, walk/episode ends
RUIN_REWARD = -10.0  # large fixed penalty when a trade would wipe the balance out entirely

# Same threshold app/main.py's /rl/signal live-exclusion gate and GET /rl/insights use to flag
# a policy's own training-time eval as a clear losing edge, not noise -- reused here (not
# redefined a second time in app/main.py, which now imports it from here instead) so "good
# enough to warm-start from" and "good enough to serve live" can never silently drift apart.
STRONG_LOSS_RETURN_PCT = -30.0


def is_usable_warm_start(eval_run: Optional[dict]) -> bool:
    """
    Whether a prior policy's own training-time eval is a starting point worth continuing
    training from, vs. giving the next run a fresh random init instead. Two independent
    disqualifiers: converged to always-HOLD (no directional_signals at all -- continuing from
    this just perpetuates the same stuck point, the original reason this check existed), or
    its own eval already crossed the same STRONG_LOSS_RETURN_PCT threshold that excludes a
    policy from live trading anyway -- continuing training from an already-catastrophic policy
    has no particular reason to recover, and every day spent doing so is a day a genuinely
    fresh attempt doesn't get. See PROGRESS.md's 2026-09-15 entry: GBP/USD 15min's full policy
    lineage showed 14 straight daily warm-started generations, ALL net negative and getting
    WORSE, after the one run that actually worked (+10.2% return) -- nothing before this ever
    stopped a training run from continuing off a policy that had already gone bad.
    """
    if not eval_run or not eval_run.get("directional_signals"):
        return False
    return_pct = eval_run.get("total_return_pct")
    return return_pct is None or return_pct > STRONG_LOSS_RETURN_PCT


def should_keep_new_policy(new_return_pct: Optional[float], baseline_return_pct: float) -> bool:
    """
    The actual fix for the warm-start-drift pattern PROGRESS.md's 2026-09-15 entry documents:
    a freshly trained policy only replaces the one already live if it's at least as good, by
    the same total_return_pct this project already trusts everywhere else (the live-exclusion
    gate, GET /rl/insights, RLTrainAllCell reporting) -- not unconditionally, which is what let
    a single bad day's continuation permanently overwrite a policy that had been working.
    baseline_return_pct is 0.0 (breakeven/hold) when there's no usable prior policy to compare
    against (see is_usable_warm_start) -- a brand new policy still shouldn't go live if it
    lost money outright. Ties keep the new policy (>=, not >) -- deliberately not stacking a
    second, unvalidated margin requirement on top of an already-real metric.
    """
    return new_return_pct is None or new_return_pct >= baseline_return_pct

# STRATEGY_NAMES itself now lives in strategies.py (imported above) -- shared with
# ml_features.py's ML feature names too, so neither can drift out of sync with STRATEGIES.

# The market-only state -- precomputable once per bar, independent of the policy or any
# balance (see compute_strategy_vote_states). balance_log_ratio is appended separately per
# decision (full_rl_state) since it changes trade-to-trade and can't be precomputed the same way.
#
# The raw_* features were added alongside the *_vote features (not instead of -- both are
# kept) because every vote is already a lossy [0,1] "strength" ratio each strategy computes
# against its own gating threshold (see strategies.py), collapsing e.g. a raw ADX of 45 and a
# raw ADX of 90 to whatever fraction of ADX_TREND_THRESHOLD*2.5 they happen to land at, or
# discarding which of several branches a market-structure call fired from entirely. Originally
# added because the linear Q-learning policy this project used before PPO could only ever
# combine whatever was actually present in the state vector, with no way to reconstruct
# information a vote already threw away -- PPO's neural-network policy could in principle
# re-derive some of this nonlinearly from the votes alone, but there's no reason to make it
# work harder than it has to, so these stay in the state:
#   raw_adx_norm    -- adx/100, continuous trend strength (votes only ever gate it at >20)
#   raw_rsi_centered -- (rsi-50)/50, signed distance from neutral (votes only gate <30/>70)
#   raw_stoch_spread -- (stoch_k-stoch_d)/100, momentum direction+magnitude in one number
#   raw_macd_hist_pct -- macd_hist as % of close, comparable across pairs at very different price scales
#   raw_bb_width_pct -- Bollinger band width as % of price, a volatility-regime read distinct from atr_pct
#   raw_volume_ratio -- volume / 20-bar volume SMA, capped to keep an illiquid-bar spike from dominating a gradient step
#   raw_roc_pct -- 5-bar rate-of-change (indicators.rate_of_change), already a % so no further
#                  scaling needed unlike macd_hist -- pure momentum magnitude, distinct from
#                  raw_volume_ratio's confirmation signal and from any single strategy's vote
#   raw_stoch_k_norm -- stoch_k/100, the oscillator's absolute LEVEL (near-0/near-1 = an
#                       overbought/oversold extreme) -- distinct from raw_stoch_spread, which
#                       only captures %K vs %D's relative direction, not where in [0,100] they sit
RL_RAW_FEATURE_NAMES = [
    "raw_adx_norm", "raw_rsi_centered", "raw_stoch_spread",
    "raw_macd_hist_pct", "raw_bb_width_pct", "raw_volume_ratio",
    "raw_roc_pct", "raw_stoch_k_norm",
]
# ml_hit_probability_buy/sell -- what the ML classifier (app/services/ml_model.py, an
# XGBoost classifier trained on ALL resolved rule-based signals pooled across every
# pair/interval) would estimate for a hypothetical BUY, and separately a hypothetical SELL,
# at this bar. Added because ML's own calibration is demonstrably better than any single RL
# policy's (GET /ml/runs' calibration table is genuinely monotonic; each RL policy trains on
# a much thinner pair/interval-scoped slice) -- this hands each thin RL policy a synthesis of
# what every OTHER pair/interval has collectively taught the shared ML model, the same
# pooling idea behind case_memory's cross-pair lookup, but baked into the trained policy
# itself instead of a live-only overlay. Two scores, not one, because ML's own direction_buy
# feature requires knowing the direction -- RL hasn't committed to one yet when this state is
# being computed, so both hypotheticals are scored and the policy learns how much to weight
# each depending on which action (BUY_* vs SELL_*) it's evaluating. See compute_ml_scores for
# how these get computed without the O(n^2) cost of calling signal_engine.generate_signal per
# bar, and ppo_engine.py's training path for the lookahead-bias-safe "frozen snapshot" fit
# this depends on (frozen_ml_snapshot, below).
RL_MARKET_FEATURE_NAMES = (
    [f"{name}_vote" for name in STRATEGY_NAMES] + ["atr_pct"] + RL_RAW_FEATURE_NAMES
    + ["ml_hit_probability_buy", "ml_hit_probability_sell"]
)
RL_FEATURE_NAMES = RL_MARKET_FEATURE_NAMES + ["balance_log_ratio"]

# Fixed, not sourced from RuleConfig -- some validated target/stop ratios elsewhere in this
# project risk MORE than they target, which is exactly what the user said this agent must
# never do. A 1.5:1 reward:risk floor is guaranteed by construction here, not left for the
# agent to discover through reward alone.
# Sizing (RISK_FRACTION_BY_TIER) is additive on top of this -- it changes how much capital is
# committed to the trade, never this ratio.
#
# HISTORICAL NOTE: these per-interval values were tuned against this project's PRIOR linear
# Q-learning policy (see git history) via a walk-forward-style sweep (5 intervals x 4 pairs x
# 5 target:stop grid points, each preserving the >=1.5:1 ratio) plus real, persisted retrains.
# That policy is retired now (PPO, via app/services/ppo_engine.py, replaced it) -- per the plan
# this replacement was built against, ATR multiples tuned for the linear policy's behavior are
# NOT guaranteed to transfer to PPO's; these numbers are a starting point inherited from that
# tuning work, not a validated-for-PPO result. Re-run the same kind of sweep against PPO before
# trusting these values the way the linear policy's own retrains eventually did. The
# per-pair/per-interval detail behind the numbers below (which pairs/intervals improved or
# didn't under the wider 4h band, and the run-to-run exploration-variance caveat that motivated
# random_seed for reproducible comparisons) is preserved in this project's git history rather
# than repeated here, since it specifically describes the linear policy's behavior.
# Any future change here must keep the same >=1.5:1 target:stop ratio the comment above
# requires -- e.g. halving both to 1.0/0.667 preserves the ratio while changing how much price
# movement is needed to resolve within max_lookforward candles; changing the ratio itself would
# reopen the "risks more than it targets" problem this constant was written to prevent.
RL_ATR_MULTS_BY_INTERVAL: dict[str, tuple[float, float]] = {
    "5min": (1.5, 1.0),
    "15min": (1.5, 1.0),
    "1h": (1.5, 1.0),
    "4h": (3.75, 2.5),
    "1day": (1.5, 1.0),
}

# Pair-specific exceptions to the interval default above -- GBP/USD 4h specifically showed no
# stable edge under either 1.5/1.0 or 4h's new 3.75/2.5 when this was tuned against the prior
# linear policy (see RL_ATR_MULTS_BY_INTERVAL's own note on why these are inherited, not
# PPO-validated), so it was kept at the original value rather than forced onto a default that
# didn't actually help it.
#
# GBP/USD 15min -- added 2026-09-11 after GET /rl/resolution-stats showed GBP/USD resolving
# to hit/miss far less often than EUR/USD at 5min/15min (63%/50% expired for EUR/USD vs
# 82%/88% for GBP/USD) despite an identical lookforward window, pointing at the fixed 1.5:1.0
# target:stop being too wide for GBP/USD's actual 15min movement rather than "not waiting long
# enough." Swept two tighter candidates (keeping the same 1.5:1 ratio) via
# POST /rl/train/15min?pair=GBP%2FUSD: (0.75, 0.5) produced ZERO directional signals on the
# test slice (tight enough that spread cost apparently makes every setup net-negative, so the
# trained policy just learned to hold instead of trade) -- worse than the default, not
# better, and the reason this isn't simply "tighter is always better." (1.0, 0.667) produced
# 111 directional signals at 60.4% hit rate / +0.0057% expectancy, a real sample, clearly
# better than the near-total-expiry status quo -- adopted here. GBP/USD 5min was swept the
# same way but only produced 4 directional signals for (1.0, 0.667) -- too small a sample to
# act on (same "don't tune on a handful of trades" floor as case_memory.MIN_CASES_FOR_GATING
# elsewhere in this project) -- left at the interval default pending more data, not silently
# dropped.
#
# USD/JPY and AUD/USD -- same 2026-09-11 sweep extended to the other two pairs GET
# /rl/insights flagged as expired-heavy at 5min/15min. Results did NOT generalize from
# GBP/USD's "tighter is better" pattern -- each pair/interval needed checking on its own:
#   - USD/JPY 5min: (1.0, 0.667) only 2 signals (discard); (0.75, 0.5) gave 101 signals at
#     54.5% hit / +0.005% expectancy -- real sample, modest positive edge. Adopted.
#   - USD/JPY 15min: (1.0, 0.667) gave a well-sampled 122 signals but -0.0165% expectancy
#     (net losing); (0.75, 0.5) looked great (66.7% hit / +0.0582%) but was only 3 signals --
#     noise, not a real edge. Neither candidate is usable -- left at the interval default,
#     unresolved (not "fixed," a genuinely open problem for this combo).
#   - AUD/USD 5min: both candidates produced ZERO directional signals -- left at default.
#   - AUD/USD 15min: (1.0, 0.667) zero signals; (0.75, 0.5) had a real sample (10 signals)
#     but -0.0322% expectancy, i.e. worse than default, not better. Left at default.
# Same lesson as GBP/USD's own (0.75, 0.5) failure above: this is a per-(interval, pair)
# empirical question, not a formula -- don't extrapolate one pair's working override to
# another without sweeping it separately.
RL_ATR_MULT_PAIR_OVERRIDES: dict[tuple[str, str], tuple[float, float]] = {
    ("4h", "GBP/USD"): (1.5, 1.0),
    ("15min", "GBP/USD"): (1.0, 0.667),
    ("5min", "USD/JPY"): (0.75, 0.5),
}


def rl_atr_mults(interval: str, pair: str) -> tuple[float, float]:
    """
    (target_atr_mult, stop_atr_mult) for this interval/pair -- see RL_ATR_MULTS_BY_INTERVAL,
    and RL_ATR_MULT_PAIR_OVERRIDES for the handful of pair-specific exceptions to it.
    """
    override = RL_ATR_MULT_PAIR_OVERRIDES.get((interval, pair))
    if override is not None:
        return override
    return RL_ATR_MULTS_BY_INTERVAL[interval]


# How many rows of tail context each bar's strategy evaluation gets -- comfortably covers
# find_swing_levels' own 20-bar lookback (patterns.py) plus a buffer, without passing the
# full growing window every bar (which would make compute_strategy_vote_states O(n^2) in
# slicing cost for no benefit -- every strategy here only ever reads the last 1-2 rows or a
# bounded lookback, never the full window).
STATE_WINDOW_BARS = 30

MIN_WARMUP_BARS = 30  # covers find_swing_levels/stochastic/ADX's own warmup needs

# Which RuleConfig profile RL's strategy calls should use -- RL previously used a longer-
# horizon config (EMA 50/200, no session filter) unconditionally for every interval,
# including 5min/15min. A 200-period EMA on 5-minute candles spans ~16.7 hours (multiple
# sessions), far too slow to say anything meaningful about a 5-minute chart -- this project
# already solved exactly this mismatch for the regular signal engine (intraday: EMA 9/21 +
# session filter, cross-pair backtest-validated, see signal_engine.PROFILE_DEFAULTS), RL
# just never adopted it. There's now only one validated profile, so every interval uses it.
# target/stop are separately interval-aware via RL_ATR_MULTS_BY_INTERVAL/rl_atr_mults above --
# this function only decides which EMA/RSI/MACD periods and session gating STRATEGIES
# compute their votes with.


def rl_config_profile(interval: str) -> str:
    return "intraday"


def usd_per_unit(pair: str, entry_price: float) -> float:
    """
    Ports trading-signals/page.tsx's calculateLotSize convention server-side -- same 4-pair
    quote-currency assumption (USD/JPY converts via its own rate since it quotes in JPY; the
    other 3 quote in USD directly). Needed here because sizing now drives the reward itself,
    not just a display number.
    """
    return 1 / entry_price if pair == "USD/JPY" else 1.0


def position_size_units(balance: float, risk_fraction: float, entry_price: float, stop_price: float, pair: str) -> float:
    stop_distance = abs(entry_price - stop_price)
    if stop_distance == 0 or balance <= 0:
        return 0.0
    return (balance * risk_fraction) / (stop_distance * usd_per_unit(pair, entry_price))


def compute_strategy_vote_states(indicator_df: pd.DataFrame, config: RuleConfig) -> list[list[float]]:
    """
    One market-state vector per bar (RL_MARKET_FEATURE_NAMES): each of the STRATEGIES'
    current direction, encoded as direction x strength (e.g. -0.8 for a strong
    SELL, -0.2 for a barely-there one, 0.0 for HOLD) instead of a flat +-1, plus atr_pct for
    volatility context. `strength` (see StrategyCall.strength / strategies.py) is each
    strategy's own normalized [0, 1] "how strong was THIS bar's reading" -- a strategy that's
    barely triggered no longer looks identical to one firing at full conviction. This is the
    literal mechanization of "use the existing strategies to generate signals, weighting by
    how strong each one currently is" -- the agent learns how to weight/combine them, an
    adaptive version of what consensus's fixed REQUIRED_WEIGHT_FRACTION already does with a
    hand-picked threshold.

    Doesn't include the balance feature -- see full_rl_state, which appends it per-decision,
    since balance changes trade-to-trade and can't be precomputed the same way these can.

    Computed once for the whole df up front, not recomputed per training pass -- strategy
    outputs are deterministic given a bar's window and don't depend on the policy at all, so
    recomputing them on every one of a training run's many timesteps would be pure waste.
    """
    states: list[list[float]] = []
    n = len(indicator_df)
    for i in range(n):
        if i < 1:
            # Most strategies need at least a prev bar (candlestick patterns, MACD cross);
            # these earliest rows are always before MIN_WARMUP_BARS anyway, never selected
            # as a real decision point -- a zero state here is a placeholder, not a live read.
            states.append([0.0] * len(RL_MARKET_FEATURE_NAMES))
            continue
        window_start = max(0, i + 1 - STATE_WINDOW_BARS)
        window = indicator_df.iloc[window_start: i + 1]
        calls = [fn(window, config) for fn in STRATEGIES]
        call_by_strategy = {c.strategy: c for c in calls}
        votes = []
        for name in STRATEGY_NAMES:
            call = call_by_strategy.get(name)
            if call is None or call.direction == "HOLD":
                votes.append(0.0)
                continue
            strength = call.strength if call.strength is not None else 1.0
            votes.append(strength if call.direction == "BUY" else -strength)
        latest = indicator_df.iloc[i]
        atr_pct = float(latest["atr"] / latest["close"] * 100) if pd.notna(latest["atr"]) and latest["close"] else 0.0

        raw_adx_norm = float(latest["adx"] / 100) if pd.notna(latest["adx"]) else 0.0
        raw_rsi_centered = float((latest["rsi"] - 50) / 50) if pd.notna(latest["rsi"]) else 0.0
        raw_stoch_spread = (
            float((latest["stoch_k"] - latest["stoch_d"]) / 100)
            if pd.notna(latest["stoch_k"]) and pd.notna(latest["stoch_d"]) else 0.0
        )
        raw_macd_hist_pct = (
            float(latest["macd_hist"] / latest["close"] * 100)
            if pd.notna(latest["macd_hist"]) and latest["close"] else 0.0
        )
        raw_bb_width_pct = (
            float((latest["bb_upper"] - latest["bb_lower"]) / latest["bb_middle"] * 100)
            if pd.notna(latest["bb_upper"]) and pd.notna(latest["bb_lower"])
            and pd.notna(latest["bb_middle"]) and latest["bb_middle"] else 0.0
        )
        # Capped at 5x -- an illiquid/off-hours bar's volume can spike far past its 20-bar
        # average (thin liquidity, not a real momentum signal), and an uncapped ratio would
        # otherwise dominate that bar's contribution to a gradient step.
        raw_volume_ratio = (
            min(float(latest["volume"] / latest["volume_sma"]), 5.0)
            if pd.notna(latest["volume_sma"]) and latest["volume_sma"] else 0.0
        )
        raw_roc_pct = float(latest["roc"]) if pd.notna(latest["roc"]) else 0.0
        raw_stoch_k_norm = float(latest["stoch_k"] / 100) if pd.notna(latest["stoch_k"]) else 0.0

        states.append(votes + [
            atr_pct, raw_adx_norm, raw_rsi_centered, raw_stoch_spread,
            raw_macd_hist_pct, raw_bb_width_pct, raw_volume_ratio,
            raw_roc_pct, raw_stoch_k_norm,
        ])
    return states


def compute_ml_scores(
    indicator_df: pd.DataFrame, config: RuleConfig, pair: str, interval: str,
    model: Optional[XGBClassifier],
) -> list[tuple[float, float]]:
    """
    One (ml_hit_probability_buy, ml_hit_probability_sell) pair per bar -- see
    RL_MARKET_FEATURE_NAMES's own comment for why this exists. Computed via the same 5 SMC
    STRATEGIES calls compute_strategy_vote_states makes per bar (not signal_engine's retired
    rule engine) directly against the already-computed indicator_df -- recomputed here rather
    than shared with compute_strategy_vote_states's own per-bar loop (a possible future
    optimization) since the two functions are called independently by this module's callers.

    model=None (not enough resolved signals yet to fit a classifier, or this training run's
    train slice starts before enough of them existed) -> (0.5, 0.5) neutral placeholder for
    every bar -- every other state feature is always a real number, and 0.5 is the honest "no
    information" value for a probability, not a fabricated confident one.

    Bars before max(config.ema_slow, MIN_WARMUP_BARS)'s warmup threshold also get the neutral
    placeholder -- the RL training path's own episode/eval loops never visit these bars as
    decision points anyway (they start from that same min_warmup), so this isn't losing any
    real information -- these bars were always going to be placeholder-only.

    Computed once per training run and reused throughout it -- deterministic given a bar's
    window and the frozen model, same "compute once, don't redo per pass" discipline as
    compute_strategy_vote_states. `model` is a snapshot fit by the caller (see
    frozen_ml_snapshot, below) on only signals resolved before this run's own train/test split
    boundary -- computing that boundary is the caller's job (it already knows split_idx), not
    this function's.

    predict_proba is called ONCE on a batched matrix of every warmed-up bar's features (twice
    total -- once for BUY, once for SELL), not once per bar. Calling it n times in a Python
    loop is what actually caused a live 524 gateway timeout on EUR/USD 5min (thousands of
    individual tiny sklearn calls, each paying real per-call overhead) -- the per-bar STRATEGIES
    calls themselves stay a per-bar loop since they can't be vectorized the same way, but
    there's no reason the classifier call has to be.
    """
    n = len(indicator_df)
    if model is None:
        return [(0.5, 0.5)] * n

    min_warmup = max(config.ema_slow, MIN_WARMUP_BARS)
    warm_indices = list(range(max(1, min_warmup), n))

    buy_matrix: list[list[float]] = []
    sell_matrix: list[list[float]] = []
    for i in warm_indices:
        window_start = max(0, i + 1 - STATE_WINDOW_BARS)
        window = indicator_df.iloc[window_start: i + 1]
        calls = [fn(window, config) for fn in STRATEGIES]
        latest = indicator_df.iloc[i]
        atr_pct = float(latest["atr"] / latest["close"] * 100) if pd.notna(latest["atr"]) and latest["close"] else 0.0
        buy_features = signal_like_features(calls, direction_confidence(calls, "BUY"), atr_pct, "BUY", pair, interval)
        sell_features = signal_like_features(calls, direction_confidence(calls, "SELL"), atr_pct, "SELL", pair, interval)
        buy_matrix.append([buy_features[name] for name in ML_FEATURE_NAMES])
        sell_matrix.append([sell_features[name] for name in ML_FEATURE_NAMES])

    scores: list[tuple[float, float]] = [(0.5, 0.5)] * n
    if warm_indices:
        buy_probs = model.predict_proba(buy_matrix)[:, 1]
        sell_probs = model.predict_proba(sell_matrix)[:, 1]
        for i, buy_p, sell_p in zip(warm_indices, buy_probs, sell_probs):
            scores[i] = (round(float(buy_p), 4), round(float(sell_p), 4))
    return scores


def full_rl_state(market_state: list[float], balance: float, starting_balance: float) -> list[float]:
    """
    Appends the one balance-dependent feature to an otherwise-precomputed market state --
    log(balance / starting_balance), 0 at the reference point, positive when ahead, negative
    when behind. Without this the agent has no way to condition its sizing choice on how the
    account is actually doing (e.g. sizing down after a drawdown). Guards balance<=0 (already
    at/past ruin) with a large negative stand-in rather than crashing on log(0).
    """
    balance_log_ratio = math.log(balance / starting_balance) if balance > 0 else -10.0
    return market_state + [balance_log_ratio]


def _take_action_sized(
    df: pd.DataFrame, indicator_df: pd.DataFrame, i: int, action: str, pair: str, max_lookforward: int, balance: float,
    target_atr_mult: float, stop_atr_mult: float,
) -> tuple[float, int, float]:
    """
    Executes one sized action at bar i, returns (reward, bars_to_advance, new_balance). HOLD
    advances 1 bar, balance unchanged, reward 0.0. A BUY_TIER/SELL_TIER action sizes a real
    position against the CURRENT balance (position_size_units), resolves it via the same
    label_outcome/spread_cost_pct every other part of this project uses, and converts the
    resulting pct_move into a dollar P&L against that position size.

    Reward is log(new_balance / balance) -- the Kelly-criterion-standard objective for
    compounding growth, which also naturally and severely penalizes ruin (log of a near-zero
    balance is deeply negative) without needing a bolted-on penalty; RUIN_REWARD is only a
    guard for the literal balance<=0 edge the log can't represent. Single-position-at-a-time
    via candles_to_outcome, same as every other trading concept here. Used identically by
    both a training step (ppo_engine.ForexTradingEnv.step) and greedy evaluation
    (ppo_engine._evaluate_policy) -- the same reward/balance math either way.
    """
    if action == "HOLD":
        return 0.0, 1, balance

    direction, tier = action.split("_")
    risk_fraction = RISK_FRACTION_BY_TIER[tier]

    entry_price = float(df.loc[i, "close"])
    atr_val = float(indicator_df.loc[i, "atr"])
    target_price, stop_price = compute_atr_target_stop(entry_price, atr_val, direction, target_atr_mult, stop_atr_mult)
    units = position_size_units(balance, risk_fraction, entry_price, stop_price, pair)

    future_candles = df.iloc[i + 1: i + 1 + max_lookforward]
    status, outcome_price, _outcome_ts, candles_to_outcome = label_outcome(
        future_candles, direction, target_price, stop_price, max_lookforward
    )
    # Callers only ever invoke this with i bounded so max_lookforward future candles exist
    # (same guarantee run_backtest's last_evaluable bound already relies on).
    assert status != "pending", "internal error: rl_engine ran out of candles unexpectedly"

    pct_move = ((outcome_price - entry_price) / entry_price) * 100
    if direction == "SELL":
        pct_move = -pct_move
    pct_move -= spread_cost_pct(pair, entry_price)

    dollar_pnl = units * usd_per_unit(pair, entry_price) * (pct_move / 100) * entry_price
    new_balance = balance + dollar_pnl

    if new_balance <= 0:
        return RUIN_REWARD, candles_to_outcome, 0.0
    reward = math.log(new_balance / balance) if balance > 0 else RUIN_REWARD
    return reward, candles_to_outcome, new_balance


def chronological_train_test_split(
    df: pd.DataFrame, train_frac: float, max_lookforward: int, min_warmup: int,
) -> tuple[int, int, int]:
    """
    Used by ppo_engine.py's training path (train_and_evaluate_ppo_poc/train_ppo_policy) for
    the same chronological train/test boundary and "too short to be meaningful" guardrails
    this project has always applied to RL training, walk-forward, not a random shuffle, same
    lookahead-bias discipline as run_backtest's own eval_start_index. Kept in this shared
    module (not ppo_engine.py itself) since it operates on the same df/max_lookforward/warmup
    concepts every other function here does.
    Returns (train_last, test_start, test_last), all inclusive positional indices into df.
    """
    split_idx = int(len(df) * train_frac)
    last_evaluable = len(df) - 1 - max_lookforward

    train_last = min(split_idx - 1, last_evaluable)
    if train_last < min_warmup:
        raise ValueError(
            f"Train slice too short for RL's warmup needs: have up to index {train_last}, "
            f"need at least {min_warmup}. Ingest more history or lower train_frac."
        )
    test_start = split_idx
    test_last = last_evaluable
    if test_last < test_start:
        raise ValueError(
            f"Test slice too short ({max(0, test_last - test_start + 1)} candles) for "
            f"max_lookforward={max_lookforward}. Ingest more history or raise train_frac."
        )
    return train_last, test_start, test_last


def frozen_ml_snapshot(df: pd.DataFrame, split_idx: int, ml_reference_signals: Optional[list[dict]]):
    """
    Used by ppo_engine.py's training path -- fits a lookahead-bias-safe ML snapshot: only
    signals resolved strictly before this run's own split boundary are allowed to inform the
    classifier that scores ml_hit_probability_buy/sell (see compute_ml_scores), so the
    reported test-slice metrics never reflect a model that secretly knows about outcomes from
    its own future test window.
    """
    split_timestamp = df.loc[split_idx, "timestamp"]
    ml_training_signals = [
        s for s in (ml_reference_signals or []) if s.get("timestamp") is not None and s["timestamp"] < split_timestamp
    ]
    return fit_hit_classifier(ml_training_signals)
