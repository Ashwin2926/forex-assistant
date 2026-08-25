import random
import uuid
import pandas as pd
from datetime import datetime
from typing import Optional
from app.models.schemas import BacktestRun, RLPolicy, RuleConfig, Signal
from app.services.indicators import add_all_indicators
from app.services.signal_engine import compute_atr_target_stop, label_outcome, spread_cost_pct
from app.services.strategies import STRATEGIES

ACTIONS = ["HOLD", "BUY", "SELL"]

# Names in the same order STRATEGIES itself is declared -- derived, not hand-typed, so this
# can't drift out of sync the same way consensus.py's STRATEGY_WEIGHTS already avoids that.
STRATEGY_NAMES = [fn.__name__.removeprefix("call_") for fn in STRATEGIES]
RL_FEATURE_NAMES = [f"{name}_vote" for name in STRATEGY_NAMES] + ["atr_pct"]

# Fixed, not sourced from RuleConfig/SWING_PAIR_OVERRIDES -- some of those (e.g. swing's
# global default target_atr_mult=0.5/stop_atr_mult=1.25) risk MORE than they target, which
# is exactly what the user said this agent must never do. A 1.5:1 reward:risk floor is
# guaranteed by construction here, not left for the agent to discover through reward alone.
RL_TARGET_ATR_MULT = 1.5
RL_STOP_ATR_MULT = 1.0

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
DEFAULT_EPISODES = 100

MIN_WARMUP_BARS = 30  # covers find_swing_levels/stochastic/ADX's own warmup needs


def compute_strategy_vote_states(indicator_df: pd.DataFrame, config: RuleConfig) -> list[list[float]]:
    """
    One state vector per bar: each of the 7 strategies' current direction, encoded as
    direction x strength (e.g. -0.8 for a strong SELL, -0.2 for a barely-there one, 0.0 for
    HOLD) instead of a flat +-1, plus atr_pct for volatility context. `strength` (see
    StrategyCall.strength / strategies.py) is each strategy's own normalized [0, 1] "how
    strong was THIS bar's reading" -- a strategy that's barely triggered no longer looks
    identical to one firing at full conviction. This is the literal mechanization of "use the
    existing strategies to generate signals, weighting by how strong each one currently is" --
    the agent learns how to weight/combine them, an adaptive version of what consensus's fixed
    REQUIRED_WEIGHT_FRACTION already does with a hand-picked threshold.

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
            states.append([0.0] * len(RL_FEATURE_NAMES))
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
        states.append(votes + [atr_pct])
    return states


ADAGRAD_EPSILON = 1e-8  # avoids division by zero on a feature's very first update


class LinearQPolicy:
    """
    q(state, action) = dot(weights[action], state). One weight vector per action, same
    length/order as RL_FEATURE_NAMES -- as interpretable as the ML page's
    feature_coefficients table, and simple enough that pure numpy-free Python is plenty fast
    for an 8-feature state.

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


def _take_action(
    df: pd.DataFrame, indicator_df: pd.DataFrame, i: int, action: str, pair: str, max_lookforward: int,
) -> tuple[float, int]:
    """
    Executes one action at bar i, returns (reward, bars_to_advance). HOLD advances by 1 bar
    with reward 0.0. BUY/SELL opens a position at RL_TARGET_ATR_MULT/RL_STOP_ATR_MULT,
    resolves it via the same label_outcome/spread_cost_pct every other part of this project
    uses, and advances by candles_to_outcome -- single-position-at-a-time, matching every
    other trading concept here (paper trading, live signals): the agent can't open a second
    position while one is still open.
    """
    if action == "HOLD":
        return 0.0, 1

    entry_price = float(df.loc[i, "close"])
    atr_val = float(indicator_df.loc[i, "atr"])
    target_price, stop_price = compute_atr_target_stop(entry_price, atr_val, action, RL_TARGET_ATR_MULT, RL_STOP_ATR_MULT)
    future_candles = df.iloc[i + 1: i + 1 + max_lookforward]
    status, outcome_price, _outcome_ts, candles_to_outcome = label_outcome(
        future_candles, action, target_price, stop_price, max_lookforward
    )
    # Callers only ever invoke this with i bounded so max_lookforward future candles exist
    # (same guarantee run_backtest's last_evaluable bound already relies on).
    assert status != "pending", "internal error: rl_engine ran out of candles unexpectedly"

    pct_move = ((outcome_price - entry_price) / entry_price) * 100
    if action == "SELL":
        pct_move = -pct_move
    pct_move -= spread_cost_pct(pair, entry_price)
    return pct_move, candles_to_outcome


def train_rl_policy(
    df: pd.DataFrame, pair: str, interval: str, config: RuleConfig = RuleConfig(),
    episodes: int = DEFAULT_EPISODES, train_frac: float = 0.7, max_lookforward: int = 20,
) -> tuple[RLPolicy, BacktestRun, list[Signal]]:
    """
    Trains a LinearQPolicy via epsilon-greedy Q-learning over the train slice (chronological
    split, same discipline as run_backtest's eval_start_index -- NOT a random shuffle, which
    would leak future information into training), then evaluates it greedily (epsilon=0, no
    exploration) on the untouched test slice -- returned as a real BacktestRun (profile="rl"),
    reusing BacktestRun.profile the same way run_consensus_backtest reuses it for
    "consensus", so RL's expectancy_pct is directly comparable to every other approach via
    the existing GET /backtest/runs?pair=X&profile=rl.

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

    df = df.reset_index(drop=True)
    indicator_df = add_all_indicators(df, config)
    states = compute_strategy_vote_states(indicator_df, config)

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
        while i <= train_last:
            state = states[i]
            action = policy.epsilon_greedy(state, epsilon)
            reward, advance = _take_action(df, indicator_df, i, action, pair, max_lookforward)
            next_i = i + advance
            next_state = states[next_i] if next_i <= train_last else None
            policy.update(state, action, reward, next_state, LEARNING_RATE, DISCOUNT_GAMMA)
            i = next_i
        epsilon = max(EPSILON_MIN, epsilon * epsilon_decay)

    # Greedy evaluation on the untouched test slice -- same walk-forward, single-position-at-
    # a-time shape as training, just epsilon=0 and no weight updates. Tallied the same way
    # run_backtest tallies hits/misses/expired/pct_move, for a directly comparable BacktestRun.
    eval_run_id = uuid.uuid4().hex[:12]
    hold_count = 0
    hits = misses = expired = 0
    win_pcts: list[float] = []
    loss_pcts: list[float] = []
    all_pcts: list[float] = []
    trade_signals: list[Signal] = []

    i = test_start
    while i <= test_last:
        state = states[i]
        q = policy.q_values(state)
        action = max(q, key=q.get)
        if action == "HOLD":
            hold_count += 1
            i += 1
            continue

        entry_price = float(df.loc[i, "close"])
        atr_val = float(indicator_df.loc[i, "atr"])
        target_price, stop_price = compute_atr_target_stop(entry_price, atr_val, action, RL_TARGET_ATR_MULT, RL_STOP_ATR_MULT)
        future_candles = df.iloc[i + 1: i + 1 + max_lookforward]
        status, outcome_price, outcome_ts, candles_to_outcome = label_outcome(
            future_candles, action, target_price, stop_price, max_lookforward
        )
        assert status != "pending", "internal error: rl_engine ran out of candles unexpectedly"

        pct_move = ((outcome_price - entry_price) / entry_price) * 100
        if action == "SELL":
            pct_move = -pct_move
        pct_move -= spread_cost_pct(pair, entry_price)

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
            direction=action, confidence=0.0, reasons=[], price_at_signal=round(entry_price, 5),
            status=status, outcome_price=round(float(outcome_price), 5), outcome_timestamp=outcome_ts,
            outcome_pct_move=round(pct_move, 5), source="backtest", run_id=eval_run_id,
            target_price=round(target_price, 5), stop_price=round(stop_price, 5),
            candles_to_outcome=candles_to_outcome,
        ))

        i += candles_to_outcome

    directional_signals = hits + misses + expired
    eval_run = BacktestRun(
        run_id=eval_run_id,
        pair=pair,
        interval=interval,
        profile="rl",
        created_at=datetime.utcnow(),
        rule_config=config,
        target_atr_mult=RL_TARGET_ATR_MULT,
        stop_atr_mult=RL_STOP_ATR_MULT,
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
    )

    return rl_policy, eval_run, trade_signals


def choose_action(policy: RLPolicy, state: list[float]) -> tuple[str, dict[str, float]]:
    """Greedy action selection from a persisted policy -- used by the signal/predict endpoints."""
    q_policy = LinearQPolicy(len(policy.feature_names))
    q_policy.weights = policy.weights
    q_values = q_policy.q_values(state)
    action = max(q_values, key=q_values.get)
    return action, q_values
