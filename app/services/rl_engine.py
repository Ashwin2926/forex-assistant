import math
import random
import uuid
import pandas as pd
from datetime import datetime
from typing import Optional
from app.models.schemas import BacktestRun, RLPolicy, RuleConfig, Signal
from app.services.indicators import add_all_indicators
from app.services.signal_engine import compute_atr_target_stop, label_outcome, spread_cost_pct
from app.services.strategies import STRATEGIES

# Direction+size actions -- 2 tiers (not 3) deliberately, to keep the action space close to
# the original 3 (HOLD/BUY/SELL) rather than 7. Every extra action means fewer training
# samples per action, the exact problem Adagrad and the recent history backfill were just
# built to address -- adding more tiers would directly work against that.
ACTIONS = ["HOLD", "BUY_SMALL", "BUY_LARGE", "SELL_SMALL", "SELL_LARGE"]

# Starting guesses, not backtested -- same caveat as every other unvalidated constant in this
# project (PROXIMITY_ATR_MULT, SMC_WICK_BODY_MULT, etc.).
RISK_FRACTION_BY_TIER = {"SMALL": 0.01, "LARGE": 0.03}  # % of current balance risked
DEFAULT_STARTING_BALANCE = 50.0
MIN_VIABLE_BALANCE = 1.0  # below this, ruin -- can't size a real position, walk/episode ends
RUIN_REWARD = -10.0  # large fixed penalty when a trade would wipe the balance out entirely

# Names in the same order STRATEGIES itself is declared -- derived, not hand-typed, so this
# can't drift out of sync the same way consensus.py's STRATEGY_WEIGHTS already avoids that.
STRATEGY_NAMES = [fn.__name__.removeprefix("call_") for fn in STRATEGIES]

# The market-only state -- precomputable once per bar, independent of the policy or any
# balance (see compute_strategy_vote_states). balance_log_ratio is appended separately per
# decision (full_rl_state) since it changes trade-to-trade and can't be precomputed the same way.
#
# The 6 raw_* features were added alongside the 7 *_vote features (not instead of -- both are
# kept) because every vote is already a lossy [0,1] "strength" ratio each strategy computes
# against its own gating threshold (see strategies.py), collapsing e.g. a raw ADX of 45 and a
# raw ADX of 90 to whatever fraction of ADX_TREND_THRESHOLD*2.5 they happen to land at, or
# discarding which of 4 branches a support/resistance call fired from entirely. Since
# LinearQPolicy is a strictly linear function of the state (see its own docstring), it can
# only ever combine whatever's actually present in the vector -- it can't reconstruct
# information a vote already threw away. These give the linear policy direct access to a few
# of the more informative continuous readings that were being discarded, at whatever scale
# keeps them roughly comparable to the existing votes/atr_pct (which live in roughly [-1, 1]
# and [0, small %] respectively):
#   raw_adx_norm    -- adx/100, continuous trend strength (votes only ever gate it at >20)
#   raw_rsi_centered -- (rsi-50)/50, signed distance from neutral (votes only gate <30/>70)
#   raw_stoch_spread -- (stoch_k-stoch_d)/100, momentum direction+magnitude in one number
#   raw_macd_hist_pct -- macd_hist as % of close, comparable across pairs at very different price scales
#   raw_bb_width_pct -- Bollinger band width as % of price, a volatility-regime read distinct from atr_pct
#   raw_volume_ratio -- volume / 20-bar volume SMA, capped to keep an illiquid-bar spike from dominating a gradient step
RL_RAW_FEATURE_NAMES = [
    "raw_adx_norm", "raw_rsi_centered", "raw_stoch_spread",
    "raw_macd_hist_pct", "raw_bb_width_pct", "raw_volume_ratio",
]
RL_MARKET_FEATURE_NAMES = [f"{name}_vote" for name in STRATEGY_NAMES] + ["atr_pct"] + RL_RAW_FEATURE_NAMES
RL_FEATURE_NAMES = RL_MARKET_FEATURE_NAMES + ["balance_log_ratio"]

# Fixed, not sourced from RuleConfig/SWING_PAIR_OVERRIDES -- some of those (e.g. swing's
# global default target_atr_mult=0.5/stop_atr_mult=1.25) risk MORE than they target, which
# is exactly what the user said this agent must never do. A 1.5:1 reward:risk floor is
# guaranteed by construction here, not left for the agent to discover through reward alone.
# Sizing (RISK_FRACTION_BY_TIER) is additive on top of this -- it changes how much capital is
# committed to the trade, never this ratio.
#
# Per-profile, not one global pair -- every 5min/15min policy trained against the fixed
# 1.5/1.0 has come back negative-expectancy on every pair (see GET /rl/policies), and this
# was never independently swept for intraday the way swing's rule-based target/stop was
# (signal_engine.PROFILE_DEFAULTS). Both profiles start at the same 1.5/1.0 values below --
# this is a structural change (makes per-profile tuning possible), not a claim that intraday
# needs different numbers yet. Any future intraday-specific value must keep the same >=1.5:1
# target:stop ratio the comment above requires -- e.g. halving both to 1.0/0.667 preserves
# the ratio while changing how much price movement is needed to resolve within
# max_lookforward candles; changing the ratio itself would reopen the "risks more than it
# targets" problem this constant was written to prevent.
RL_ATR_MULTS_BY_PROFILE: dict[str, tuple[float, float]] = {
    "swing": (1.5, 1.0),
    "intraday": (1.5, 1.0),
}


def rl_atr_mults(profile: str) -> tuple[float, float]:
    """(target_atr_mult, stop_atr_mult) for this RL profile -- see RL_ATR_MULTS_BY_PROFILE."""
    return RL_ATR_MULTS_BY_PROFILE[profile]


# How many rows of tail context each bar's strategy evaluation gets -- comfortably covers
# find_swing_levels' own 20-bar lookback (patterns.py) plus a buffer, without passing the
# full growing window every bar (which would make compute_strategy_vote_states O(n^2) in
# slicing cost for no benefit -- every strategy here only ever reads the last 1-2 rows or a
# bounded lookback, never the full window).
STATE_WINDOW_BARS = 30

# Starting guesses, not tuned -- same caveat as every other unvalidated constant in this
# project (PROXIMITY_ATR_MULT, SMC_WICK_BODY_MULT, etc.). Revisit once a policy's live
# performance gives something real to tune against.
LEARNING_RATE = 0.1
DISCOUNT_GAMMA = 0.9
EPSILON_START = 1.0
EPSILON_MIN = 0.05
# Raised from 100 (the pre-sizing value, when ACTIONS had 3 entries) after the first sizing-
# aware EUR/USD 5min/15min policies came back showing total_return_pct of -64% to -79% on a
# $50 start despite a near-flat expectancy_pct (-0.01% to -0.02%) -- the gap between those two
# numbers is exactly what you'd expect from geometric compounding under a weak/negative edge
# (variance drag: even a strategy with ~breakeven arithmetic returns produces negative
# geometric growth once you're actually sizing and compounding real bets, and 5min/15min
# already run well below the ~40% hit rate this project's fixed 1.5:1 target:stop needs to
# break even -- see PROGRESS.md). Going from 3 actions (HOLD/BUY/SELL) to 5 (adding the
# SMALL/LARGE split) roughly halves how many training samples each BUY/SELL action sees per
# episode at a fixed episode count, so on top of the weak-edge problem itself, the agent
# likely hadn't converged enough yet to learn "avoid this, or at least don't size up into it"
# -- LARGE in particular is the newest, least-visited action and the most expensive to get
# wrong. Doubling episodes doesn't fix a genuinely weak edge, but it does give the agent a
# real chance to learn to stay away from it (Q(HOLD)=0 should dominate once the negative
# expected log-reward of trading is well-estimated) instead of still exploring into it.
DEFAULT_EPISODES = 200

MIN_WARMUP_BARS = 30  # covers find_swing_levels/stochastic/ADX's own warmup needs

# Which RuleConfig profile each interval's strategy calls should use -- RL previously used
# "swing" (EMA 50/200, no session filter) unconditionally for every interval, including
# 5min/15min. A 200-period EMA on 5-minute candles spans ~16.7 hours (multiple sessions),
# far too slow to say anything meaningful about a 5-minute chart -- this project already
# solved exactly this mismatch for the regular signal engine (intraday: EMA 9/21 + session
# filter, cross-pair backtest-validated, see signal_engine.PROFILE_DEFAULTS), RL just never
# adopted it. Same interval grouping the cron already uses for the regular engine/consensus.
# target/stop are separately profile-aware via RL_ATR_MULTS_BY_PROFILE/rl_atr_mults above --
# this constant only decides which EMA/RSI/MACD periods and session gating the 7 strategies
# compute their votes with.
INTRADAY_INTERVALS = {"5min", "15min"}


def rl_config_profile(interval: str) -> str:
    return "intraday" if interval in INTRADAY_INTERVALS else "swing"


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
    One market-state vector per bar (RL_MARKET_FEATURE_NAMES, 8 features): each of the 7
    strategies' current direction, encoded as direction x strength (e.g. -0.8 for a strong
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

    Computed once for the whole df up front, not recomputed per training episode -- strategy
    outputs are deterministic given a bar's window and don't depend on the policy at all, so
    recomputing them on every one of `episodes` passes would be pure waste.
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
        # otherwise dominate that bar's TD-error gradient across every feature via Adagrad's
        # shared per-step scaling.
        raw_volume_ratio = (
            min(float(latest["volume"] / latest["volume_sma"]), 5.0)
            if pd.notna(latest["volume_sma"]) and latest["volume_sma"] else 0.0
        )

        states.append(votes + [
            atr_pct, raw_adx_norm, raw_rsi_centered, raw_stoch_spread,
            raw_macd_hist_pct, raw_bb_width_pct, raw_volume_ratio,
        ])
    return states


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


ADAGRAD_EPSILON = 1e-8  # avoids division by zero on a feature's very first update


class LinearQPolicy:
    """
    q(state, action) = dot(weights[action], state). One weight vector per action, same
    length/order as RL_FEATURE_NAMES -- as interpretable as the ML page's
    feature_coefficients table, and simple enough that pure numpy-free Python is plenty fast
    for a 9-feature state.

    Updates use Adagrad (per-weight adaptive learning rate, accumulated sum of squared past
    gradients) instead of one flat alpha for every feature -- the strategies here fire at very
    different rates by design (trend votes on ~68% of bars, smart_money on ~4%, see
    PROGRESS.md), so a flat learning rate lets frequently-firing strategies dominate the
    learned weights mostly through sheer repetition, not necessarily through being more
    predictive per occurrence. A feature that's 0 on every bar a strategy doesn't fire (which
    is most bars, for the rare ones) contributes 0 to its own gradient on those steps, so its
    Adagrad accumulator only grows on the bars it actually votes -- giving rare strategies a
    larger effective step size per occurrence instead of quietly lagging behind on raw
    exposure alone.
    """

    def __init__(self, feature_count: int):
        self.weights: dict[str, list[float]] = {a: [0.0] * feature_count for a in ACTIONS}
        self.sum_sq_grad: dict[str, list[float]] = {a: [0.0] * feature_count for a in ACTIONS}

    def q_values(self, state: list[float]) -> dict[str, float]:
        return {a: sum(w * s for w, s in zip(self.weights[a], state)) for a in ACTIONS}

    def epsilon_greedy(self, state: list[float], epsilon: float) -> str:
        if random.random() < epsilon:
            return random.choice(ACTIONS)
        q = self.q_values(state)
        return max(q, key=q.get)

    def update(self, state: list[float], action: str, reward: float, next_state: Optional[list[float]], alpha: float, gamma: float) -> None:
        best_next_q = max(self.q_values(next_state).values()) if next_state is not None else 0.0
        current_q = sum(w * s for w, s in zip(self.weights[action], state))
        td_error = (reward + gamma * best_next_q) - current_q

        grads = [td_error * s for s in state]
        self.sum_sq_grad[action] = [sq + g * g for sq, g in zip(self.sum_sq_grad[action], grads)]
        self.weights[action] = [
            w + (alpha / ((sq ** 0.5) + ADAGRAD_EPSILON)) * g
            for w, sq, g in zip(self.weights[action], self.sum_sq_grad[action], grads)
        ]


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
    via candles_to_outcome, same as every other trading concept here.
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


def train_rl_policy(
    df: pd.DataFrame, pair: str, interval: str, config: RuleConfig = RuleConfig(),
    episodes: int = DEFAULT_EPISODES, train_frac: float = 0.7, max_lookforward: int = 20,
    starting_balance: float = DEFAULT_STARTING_BALANCE,
) -> tuple[RLPolicy, BacktestRun, list[Signal]]:
    """
    Trains a LinearQPolicy via epsilon-greedy Q-learning over the train slice (chronological
    split, same discipline as run_backtest's eval_start_index -- NOT a random shuffle, which
    would leak future information into training), then evaluates it greedily (epsilon=0, no
    exploration) on the untouched test slice -- returned as a real BacktestRun (profile="rl"),
    reusing BacktestRun.profile the same way run_consensus_backtest reuses it for
    "consensus", so RL's expectancy_pct is directly comparable to every other approach via
    the existing GET /backtest/runs?pair=X&profile=rl. The eval run also carries
    starting_balance/ending_balance/total_return_pct -- a real "what would $X have grown to"
    simulation, not just an average per-trade return, since sizing makes growth compounding
    and path-dependent rather than a flat mean.

    Also returns one Signal per individual test-slice trade (hit/miss/expired), tagged with
    the eval BacktestRun's run_id -- reuses run_backtest's own persistence path
    (backtest_signals_collection + GET /backtest/runs/{run_id}/signals) instead of inventing
    a separate trade-log mechanism, so "which trades passed and which failed" is answered by
    an endpoint that already exists. profile="swing" is a compatibility value only (Signal
    has no RL-specific profile literal), same convention already used elsewhere when an
    RL-generated decision needs to pass through a Signal-shaped interface; reasons=[] since
    there's no rule-by-rule breakdown for a learned policy the way there is for the rule
    engine.
    """
    if not 0 < train_frac < 1:
        raise ValueError("train_frac must be between 0 and 1 (exclusive).")
    if starting_balance <= 0:
        raise ValueError("starting_balance must be positive.")

    target_atr_mult, stop_atr_mult = rl_atr_mults(rl_config_profile(interval))

    df = df.reset_index(drop=True)
    indicator_df = add_all_indicators(df, config)
    market_states = compute_strategy_vote_states(indicator_df, config)

    min_warmup = max(config.ema_slow, MIN_WARMUP_BARS)
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

    policy = LinearQPolicy(len(RL_FEATURE_NAMES))
    epsilon = EPSILON_START
    epsilon_decay = (EPSILON_MIN / EPSILON_START) ** (1 / max(episodes, 1))

    for _episode in range(episodes):
        i = min_warmup
        balance = starting_balance
        while i <= train_last:
            state = full_rl_state(market_states[i], balance, starting_balance)
            action = policy.epsilon_greedy(state, epsilon)
            reward, advance, balance = _take_action_sized(
                df, indicator_df, i, action, pair, max_lookforward, balance, target_atr_mult, stop_atr_mult,
            )
            next_i = i + advance
            ruined = balance < MIN_VIABLE_BALANCE
            next_state = full_rl_state(market_states[next_i], balance, starting_balance) if (next_i <= train_last and not ruined) else None
            policy.update(state, action, reward, next_state, LEARNING_RATE, DISCOUNT_GAMMA)
            if ruined:
                break  # out of capital -- nothing left to trade with for the rest of this episode
            i = next_i
        epsilon = max(EPSILON_MIN, epsilon * epsilon_decay)

    # Greedy evaluation on the untouched test slice -- same walk-forward, single-position-at-
    # a-time shape as training, just epsilon=0 and no weight updates. Tallied the same way
    # run_backtest tallies hits/misses/expired/pct_move, for a directly comparable BacktestRun,
    # plus a real balance walk (starting_balance -> ending_balance) since sizing makes growth
    # compounding rather than a flat per-trade average.
    eval_run_id = uuid.uuid4().hex[:12]
    hold_count = 0
    hits = misses = expired = 0
    win_pcts: list[float] = []
    loss_pcts: list[float] = []
    all_pcts: list[float] = []
    trade_signals: list[Signal] = []

    balance = starting_balance
    i = test_start
    while i <= test_last:
        state = full_rl_state(market_states[i], balance, starting_balance)
        q = policy.q_values(state)
        action = max(q, key=q.get)
        if action == "HOLD":
            hold_count += 1
            i += 1
            continue

        direction, tier = action.split("_")
        risk_fraction = RISK_FRACTION_BY_TIER[tier]
        entry_price = float(df.loc[i, "close"])
        atr_val = float(indicator_df.loc[i, "atr"])
        target_price, stop_price = compute_atr_target_stop(entry_price, atr_val, direction, target_atr_mult, stop_atr_mult)
        units = position_size_units(balance, risk_fraction, entry_price, stop_price, pair)
        future_candles = df.iloc[i + 1: i + 1 + max_lookforward]
        status, outcome_price, outcome_ts, candles_to_outcome = label_outcome(
            future_candles, direction, target_price, stop_price, max_lookforward
        )
        assert status != "pending", "internal error: rl_engine ran out of candles unexpectedly"

        pct_move = ((outcome_price - entry_price) / entry_price) * 100
        if direction == "SELL":
            pct_move = -pct_move
        pct_move -= spread_cost_pct(pair, entry_price)

        dollar_pnl = units * usd_per_unit(pair, entry_price) * (pct_move / 100) * entry_price
        balance_before = balance
        balance = max(balance + dollar_pnl, 0.0)

        if status == "hit":
            hits += 1
            win_pcts.append(pct_move)
        elif status == "miss":
            misses += 1
            loss_pcts.append(pct_move)
        else:
            expired += 1
            (win_pcts if pct_move >= 0 else loss_pcts).append(pct_move)
        all_pcts.append(pct_move)

        trade_signals.append(Signal(
            pair=pair, profile="swing", interval=interval, timestamp=df.loc[i, "timestamp"],
            direction=direction, confidence=0.0, reasons=[], price_at_signal=round(entry_price, 5),
            status=status, outcome_price=round(float(outcome_price), 5), outcome_timestamp=outcome_ts,
            outcome_pct_move=round(pct_move, 5), source="backtest", run_id=eval_run_id,
            target_price=round(target_price, 5), stop_price=round(stop_price, 5),
            candles_to_outcome=candles_to_outcome,
            size_tier=tier, risk_fraction=risk_fraction, balance_at_signal=round(balance_before, 2),
            position_size_units=round(units, 2),
        ))

        if balance < MIN_VIABLE_BALANCE:
            break  # ruined on the test walk itself -- out of capital, stop evaluating further
        i += candles_to_outcome

    directional_signals = hits + misses + expired
    ending_balance = round(balance, 2)
    total_return_pct = round((ending_balance - starting_balance) / starting_balance * 100, 2)
    eval_run = BacktestRun(
        run_id=eval_run_id,
        pair=pair,
        interval=interval,
        profile="rl",
        created_at=datetime.utcnow(),
        rule_config=config,
        target_atr_mult=target_atr_mult,
        stop_atr_mult=stop_atr_mult,
        max_lookforward=max_lookforward,
        candles_evaluated=test_last - test_start + 1,
        total_signals=directional_signals + hold_count,
        hold_signals=hold_count,
        directional_signals=directional_signals,
        hits=hits,
        misses=misses,
        expired=expired,
        hit_rate_pct=round(hits / directional_signals * 100, 1) if directional_signals else None,
        avg_win_pct=round(sum(win_pcts) / len(win_pcts), 4) if win_pcts else None,
        avg_loss_pct=round(sum(loss_pcts) / len(loss_pcts), 4) if loss_pcts else None,
        expectancy_pct=round(sum(all_pcts) / len(all_pcts), 4) if all_pcts else None,
        rule_stats=[],  # no per-rule attribution concept here, unlike run_backtest/run_consensus_backtest
        starting_balance=starting_balance,
        ending_balance=ending_balance,
        total_return_pct=total_return_pct,
    )

    rl_policy = RLPolicy(
        policy_id=uuid.uuid4().hex[:12],
        pair=pair,
        interval=interval,
        created_at=datetime.utcnow(),
        episodes=episodes,
        train_frac=train_frac,
        weights=policy.weights,
        feature_names=RL_FEATURE_NAMES,
        eval_run_id=eval_run.run_id,
        starting_balance=starting_balance,
    )

    return rl_policy, eval_run, trade_signals


def choose_action(policy: RLPolicy, state: list[float]) -> tuple[str, dict[str, float]]:
    """
    Greedy action selection from a persisted policy -- used by the signal/predict endpoints.
    Raises ValueError (turned into a clean 400 by the caller) if the policy's weights don't
    match the current ACTIONS set -- a policy trained before an action-space change (like
    HOLD/BUY/SELL -> the 5 sizing actions) is stale, not usable as-is, and a raw KeyError from
    a mismatched lookup would be a confusing way to find that out.

    Same reasoning applies to the state feature schema: a policy trained before
    RL_FEATURE_NAMES grew (e.g. the raw_* features added alongside the vote features) has
    fewer weights per action than `state` now has entries. Without this check, q_values'
    zip(self.weights[a], state) would silently stop at the shorter length instead of raising,
    quietly ignoring every feature past the old policy's count and producing a meaningless
    but not-obviously-wrong Q-value -- a policy.feature_names length check turns that into the
    same clean "retrain it" 400 the action-set check already gives.
    """
    if set(policy.weights.keys()) != set(ACTIONS):
        raise ValueError(
            f"RL policy {policy.policy_id} for {policy.pair}/{policy.interval} was trained "
            f"against a different action set (stale after an RL update) -- retrain via "
            f"POST /rl/train/{policy.interval}?pair={policy.pair} first."
        )
    if len(policy.feature_names) != len(state):
        raise ValueError(
            f"RL policy {policy.policy_id} for {policy.pair}/{policy.interval} was trained "
            f"against a different state feature set ({len(policy.feature_names)} features, "
            f"current code produces {len(state)}) -- stale after a feature-set change, "
            f"retrain via POST /rl/train/{policy.interval}?pair={policy.pair} first."
        )
    q_policy = LinearQPolicy(len(policy.feature_names))
    q_policy.weights = policy.weights
    q_values = q_policy.q_values(state)
    action = max(q_values, key=q_values.get)
    return action, q_values
