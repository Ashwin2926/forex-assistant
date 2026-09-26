"""
Signal Stack v2, phase 3 -- this project's RL agent, PPO (stable-baselines3) via a thin
gymnasium.Env adapter, trained against the exact same replay mechanics this project's PRIOR
linear Q-learning policy used (see rl_engine.py, and git history for the retired
LinearQPolicy/train_rl_policy/choose_action): the same _take_action_sized reward logic, the
same compute_strategy_vote_states/compute_ml_scores market-state features, the same
chronological_train_test_split/frozen_ml_snapshot lookahead-bias discipline. This module
reuses every one of those pieces rather than reimplementing them -- the env is an adapter,
not a rewrite, per the plan this was built against.

PPO is now the ONLY RL algorithm in this project -- the linear Q-learning system was fully
retired, not kept alongside this as a second option. POST /rl/train/{interval} and
POST /rl/signal/{interval} (app/main.py) both call into this module; there is no separate
"-ppo"-suffixed endpoint namespace. train_and_evaluate_ppo_poc remains from this module's
original proof-of-concept phase (does this train at all, does it beat random, is a saved
model small/fast enough to serve live) -- train_ppo_policy/choose_action_ppo are the live
path built on top of it once that proof-of-concept cleared.

STILL UNVERIFIED as of this module's own introduction: real-market-data validation at the
scale a full retrain-all needs (every check this module's own tests ran during development
used synthetic OHLCV, since that development sandbox had no Twelve Data/Mongo access) --
treat a freshly-trained PPO policy as unproven until POST /rl/train has actually been run
against real ingested candle history and its eval_run's hit_rate_pct/total_return_pct
reviewed. The PyTorch dependency itself (build time, image size, cold-start latency on
FastAPI Cloud -- see requirements.txt's own comment) has since been confirmed to deploy and
import successfully (other endpoints in the same process depend on it starting up cleanly),
but was likewise unverified before that first real deploy.
"""
import io
import time
import uuid
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from app.models.schemas import BacktestRun, PPOPolicy, RuleConfig, Signal
from app.services.indicators import add_all_indicators
from app.services.signal_engine import spread_cost_pct
from app.services import rl_engine as rl


class ForexTradingEnv(gym.Env):
    """
    Thin gymnasium adapter around rl_engine's position-mechanics functions -- NOT a
    reimplementation. Stepping is bar-by-bar (the learned-exit architecture -- see
    rl_engine.OpenPosition's own docstring): while flat, a BUY/SELL action opens a position
    via rl_engine._open_position and HOLD/EXIT are both no-ops; while a position is open,
    every bar first checks the hard target/stop backstop (rl_engine._check_bar_backstop) and
    max_lookforward expiry, closing via rl_engine._close_position if either trips, and
    otherwise reads the agent's own action -- EXIT closes early at that bar's close, anything
    else (including a BUY/SELL, since only one position is held at a time) counts as staying
    in the trade. Reward is 0 on every bar that doesn't close a position and the real
    log-balance-growth reward (unchanged formula, see _close_position) on the bar that does --
    a real change from the old one-decision-one-reward shape, since a trade can now span many
    bars before either the agent or a hard backstop ends it.

    market_states/df/indicator_df are precomputed ONCE by the caller (train_and_evaluate_ppo_
    poc) and passed in already -- same "compute once, reuse across every episode" discipline
    compute_strategy_vote_states' own docstring documents; this class only walks an index
    through them, it never recomputes a strategy vote or an indicator itself.
    """

    metadata = {"render_modes": []}

    def __init__(
        self, df: pd.DataFrame, indicator_df: pd.DataFrame, market_states: list[list[float]],
        pair: str, start_index: int, end_index: int, max_lookforward: int,
        target_atr_mult: float, stop_atr_mult: float, starting_balance: float,
    ):
        super().__init__()
        self.df = df
        self.indicator_df = indicator_df
        self.market_states = market_states
        self.pair = pair
        self.start_index = start_index
        self.end_index = end_index  # last index this env is allowed to open a new position at
        self.max_lookforward = max_lookforward
        self.target_atr_mult = target_atr_mult
        self.stop_atr_mult = stop_atr_mult
        self.starting_balance = starting_balance

        self.action_space = spaces.Discrete(len(rl.ACTIONS))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(rl.RL_FEATURE_NAMES),), dtype=np.float32,
        )
        self._i = start_index
        self._balance = starting_balance
        self._position: Optional[rl.OpenPosition] = None

    def _obs(self) -> np.ndarray:
        current_price = float(self.df.loc[self._i, "close"]) if self._position is not None else None
        return np.asarray(
            rl.full_rl_state(
                self.market_states[self._i], self._balance, self.starting_balance,
                self._position, current_price, self.max_lookforward,
            ),
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._i = self.start_index
        self._balance = self.starting_balance
        self._position = None
        return self._obs(), {}

    def step(self, action_idx: int):
        action = rl.ACTIONS[int(action_idx)]
        reward = 0.0

        if self._position is None:
            if action not in ("HOLD", "EXIT"):
                self._position = rl._open_position(
                    self.df, self.indicator_df, self._i, action, self.pair, self._balance,
                    self.target_atr_mult, self.stop_atr_mult,
                )
        else:
            candle = self.df.loc[self._i]
            self._position.bars_held += 1
            backstop = rl._check_bar_backstop(
                candle, self._position.direction, self._position.target_price, self._position.stop_price,
            )
            if backstop is not None:
                _status, outcome_price = backstop
                reward, self._balance = rl._close_position(self._position, outcome_price, self._balance)
                self._position = None
            elif self._position.bars_held >= self.max_lookforward or action == "EXIT":
                outcome_price = float(candle["close"])
                reward, self._balance = rl._close_position(self._position, outcome_price, self._balance)
                self._position = None
            # else: still open -- reward stays 0.0, bars_held already incremented above.

        next_i = self._i + 1
        ruined = self._balance < rl.MIN_VIABLE_BALANCE
        terminated = bool(ruined or next_i > self.end_index)
        # An episode boundary can land mid-trade now that a trade spans many bars instead of
        # resolving atomically -- force-close (mark-to-market at the last available close)
        # rather than silently dropping the position's unrealized P&L from the reward signal,
        # which would otherwise teach the agent that trades still open at episode-end are free.
        if terminated and not ruined and self._position is not None:
            close_reward, self._balance = rl._close_position(
                self._position, float(self.df.loc[self._i, "close"]), self._balance
            )
            reward += close_reward
            self._position = None
        # Clamp rather than index past end_index -- SB3 still wants a valid observation on the
        # terminal step even though it won't be bootstrapped from (terminated=True).
        self._i = min(next_i, self.end_index) if not ruined else self._i
        return self._obs(), float(reward), terminated, False, {}


def _build_market_states(
    df: pd.DataFrame, config: RuleConfig, pair: str, interval: str,
    split_idx: int, ml_reference_signals: Optional[list[dict]],
) -> tuple[pd.DataFrame, list[list[float]]]:
    """Same market-state construction this project's RL training has always used -- reused, not reimplemented."""
    indicator_df = add_all_indicators(df, config)
    ml_model = rl.frozen_ml_snapshot(df, split_idx, ml_reference_signals)
    ml_scores = rl.compute_ml_scores(indicator_df, config, pair, interval, ml_model)
    market_states = [
        ms + [buy_score, sell_score]
        for ms, (buy_score, sell_score) in zip(rl.compute_strategy_vote_states(indicator_df, config), ml_scores)
    ]
    return indicator_df, market_states


def select_action(action_probs: dict[str, float], flat: bool) -> str:
    """
    Turns PPO's action-probability distribution into one action. While flat, probability is
    pooled per DIRECTION before picking, instead of a plain argmax over all six actions: the
    action space splits every direction into two size tiers (BUY_SMALL/BUY_LARGE, ...) and
    EXIT is a no-op when flat (ForexTradingEnv.step treats it exactly like HOLD), so a plain
    argmax let HOLD win against a direction the policy actually preferred overall just because
    that direction's mass was split in two -- confirmed live 2026-09-26, e.g. AUD/USD/1day at
    SELL_SMALL 0.15 + SELL_LARGE 0.32 = 0.47 vs HOLD 0.35 (+ EXIT 0.08), and several policies
    whose single largest action while flat was EXIT. Flat choice is among HOLD (HOLD+EXIT),
    BUY (both tiers) and SELL (both tiers), ties going to HOLD; a chosen direction takes
    whichever of its two tiers is itself more probable. With a position open, plain argmax
    (unchanged) -- only EXIT vs. not-EXIT matters there.

    Used by BOTH live inference (choose_action_ppo) and the test-slice evaluation
    (_evaluate_policy via train_ppo_policy), so a policy is judged by exactly the rule it
    trades with live.
    """
    if not flat:
        return max(action_probs, key=action_probs.get)
    hold = action_probs["HOLD"] + action_probs["EXIT"]
    buy = action_probs["BUY_SMALL"] + action_probs["BUY_LARGE"]
    sell = action_probs["SELL_SMALL"] + action_probs["SELL_LARGE"]
    if hold >= buy and hold >= sell:
        return "HOLD"
    direction = "BUY" if buy > sell else "SELL"
    return f"{direction}_LARGE" if action_probs[f"{direction}_LARGE"] > action_probs[f"{direction}_SMALL"] else f"{direction}_SMALL"


def _action_probs(model: PPO, obs: np.ndarray) -> dict[str, float]:
    obs_tensor, _ = model.policy.obs_to_tensor(obs.reshape(1, -1))
    distribution = model.policy.get_distribution(obs_tensor)
    probs = distribution.distribution.probs.detach().cpu().numpy().flatten()
    return {a: float(p) for a, p in zip(rl.ACTIONS, probs)}


def _greedy_ppo_action(model: PPO, obs: np.ndarray, flat: bool = False) -> tuple[str, dict[str, float]]:
    """
    Interpretability replacement for the retired linear policy's per-feature weight table
    (see "Signal Stack v2" phase 3, section 04's interpretability decision) -- PPO's policy
    network has no per-feature weight to show, but it does expose an action-probability
    distribution at every decision, which is the lighter-weight of the two options that plan
    raised (vs. a heavier SHAP-based explainer). Returns the greedy (argmax-probability)
    action plus the full action -> probability dict, deliberately shaped like the retired
    choose_action's (action, q_values) return so a caller can display it the same way --
    RLSignal.q_values holds this dict now regardless of caller.
    """
    action_probs = _action_probs(model, obs)
    action = select_action(action_probs, flat)
    return action, {a: round(p, 4) for a, p in action_probs.items()}


def _evaluate_policy(
    action_fn, df: pd.DataFrame, indicator_df: pd.DataFrame, market_states: list[list[float]],
    pair: str, interval: str, test_start: int, test_last: int, max_lookforward: int,
    target_atr_mult: float, stop_atr_mult: float, starting_balance: float, run_id: str,
) -> tuple[BacktestRun, list[Signal]]:
    """
    Greedy evaluation on the untouched test slice -- same single-position-at-a-time shape and
    tallying this project's RL eval has always used, factored out so both the PPO policy and
    the random baseline (train_and_evaluate_ppo_poc's "does it beat random" check) are scored
    by the exact same yardstick. action_fn(obs: np.ndarray, flat: bool) -> str picks the action for
    a given observation (flat: no position open -- see select_action) -- the only thing that differs between callers.

    Bar-by-bar walk mirroring ForexTradingEnv.step's own mechanics (see its docstring): once a
    position opens, action_fn is re-consulted every subsequent bar it survives, so an EXIT can
    end it early instead of always riding to target/stop/expiry. A closed-early trade counts
    toward hit_rate_pct by the sign of its realized pct_move -- usually a win taken early, per
    the same reasoning app/models/schemas.py's RLSignal.closed_early docstring uses for the
    live dashboard's own accounting -- rather than being excluded from the denominator the way
    a "superseded" trade is.
    """
    hold_count = 0
    hits = misses = expired = 0
    win_pcts: list[float] = []
    loss_pcts: list[float] = []
    all_pcts: list[float] = []
    trade_signals: list[Signal] = []

    balance = starting_balance
    position: Optional[rl.OpenPosition] = None
    entry_i: Optional[int] = None
    i = test_start
    while i <= test_last:
        current_price = float(df.loc[i, "close"]) if position is not None else None
        obs = np.asarray(
            rl.full_rl_state(market_states[i], balance, starting_balance, position, current_price, max_lookforward),
            dtype=np.float32,
        )
        action = action_fn(obs, position is None)

        if position is None:
            if action in ("HOLD", "EXIT"):
                hold_count += 1
                i += 1
                continue
            position = rl._open_position(df, indicator_df, i, action, pair, balance, target_atr_mult, stop_atr_mult)
            entry_i = i
            i += 1
            continue

        candle = df.loc[i]
        position.bars_held += 1
        backstop = rl._check_bar_backstop(candle, position.direction, position.target_price, position.stop_price)
        closed_early = False
        if backstop is not None:
            status, outcome_price = backstop
        elif position.bars_held >= max_lookforward:
            status, outcome_price = "expired", float(candle["close"])
        elif action == "EXIT":
            status, outcome_price, closed_early = None, float(candle["close"]), True
        else:
            i += 1
            continue  # still open, reward/outcome not yet determined

        outcome_ts = candle["timestamp"]
        candles_to_outcome = position.bars_held
        direction, tier, entry_price = position.direction, position.tier, position.entry_price

        pct_move = ((outcome_price - entry_price) / entry_price) * 100
        if direction == "SELL":
            pct_move = -pct_move
        pct_move -= spread_cost_pct(pair, entry_price)

        if closed_early:
            status = "hit" if pct_move >= 0 else "miss"

        units, target_price, stop_price = position.units, position.target_price, position.stop_price
        balance_before = balance
        _reward, balance = rl._close_position(position, outcome_price, balance)
        position = None

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
            pair=pair, profile="intraday", interval=interval, timestamp=df.loc[entry_i, "timestamp"],
            direction=direction, confidence=0.0, reasons=[], price_at_signal=round(entry_price, 5),
            status=status, outcome_price=round(float(outcome_price), 5), outcome_timestamp=outcome_ts,
            outcome_pct_move=round(pct_move, 5), source="backtest", run_id=run_id,
            target_price=round(target_price, 5), stop_price=round(stop_price, 5),
            candles_to_outcome=candles_to_outcome,
            size_tier=tier, risk_fraction=rl.RISK_FRACTION_BY_TIER[tier], balance_at_signal=round(balance_before, 2),
            position_size_units=round(units, 2),
        ))
        entry_i = None

        if balance < rl.MIN_VIABLE_BALANCE:
            break
        i += 1

    directional_signals = hits + misses + expired
    ending_balance = round(balance, 2)
    total_return_pct = round((ending_balance - starting_balance) / starting_balance * 100, 2)
    eval_run = BacktestRun(
        run_id=run_id, pair=pair, interval=interval, profile="rl_ppo", created_at=datetime.utcnow(),
        rule_config=RuleConfig(), target_atr_mult=target_atr_mult, stop_atr_mult=stop_atr_mult,
        max_lookforward=max_lookforward, candles_evaluated=test_last - test_start + 1,
        total_signals=directional_signals + hold_count, hold_signals=hold_count,
        directional_signals=directional_signals, hits=hits, misses=misses, expired=expired,
        hit_rate_pct=round(hits / directional_signals * 100, 1) if directional_signals else None,
        avg_win_pct=round(sum(win_pcts) / len(win_pcts), 4) if win_pcts else None,
        avg_loss_pct=round(sum(loss_pcts) / len(loss_pcts), 4) if loss_pcts else None,
        expectancy_pct=round(sum(all_pcts) / len(all_pcts), 4) if all_pcts else None,
        rule_stats=[], starting_balance=starting_balance, ending_balance=ending_balance,
        total_return_pct=total_return_pct,
    )
    return eval_run, trade_signals


def train_and_evaluate_ppo_poc(
    df: pd.DataFrame, pair: str, interval: str, config: RuleConfig = RuleConfig(),
    total_timesteps: int = 50_000, train_frac: float = 0.7, max_lookforward: int = 20,
    starting_balance: float = rl.DEFAULT_STARTING_BALANCE,
    ml_reference_signals: Optional[list[dict]] = None, random_seed: Optional[int] = 0,
    ppo_kwargs: Optional[dict] = None,
    target_atr_mult: Optional[float] = None, stop_atr_mult: Optional[float] = None,
    warm_start_model_bytes: Optional[bytes] = None,
) -> dict:
    """
    The four-check proof-of-concept "Signal Stack v2" phase 3a calls for, steps 2-3 (step 1,
    "does the PyTorch dependency actually deploy on FastAPI Cloud," can only be answered by a
    real deploy -- see requirements.txt's own comment; step 4, the interpretability decision,
    is answered structurally by _greedy_ppo_action existing at all, but which of the two
    options the plan raised gets used live is still a call for the plan's decisions section).

    Returns a dict rather than a narrower type since this is explicitly POC-only, not a
    persisted/API-facing shape yet:
      ppo_eval_run / ppo_trade_signals   -- PPO's own greedy test-slice evaluation, the same
                                             BacktestRun/Signal shape this project's RL eval
                                             has always used (profile="rl_ppo" -- kept
                                             distinct from the retired linear policy's "rl"
                                             profile still sitting in old BacktestRun history,
                                             see GET /backtest/runs).
      random_baseline_eval_run           -- a uniformly-random policy replayed over the exact
                                             same test slice via the exact same _evaluate_policy
                                             tallying -- the "beats random" check.
      beats_random                       -- PPO's total_return_pct > random baseline's.
      model_size_bytes                   -- size of model.save()'s output (an in-memory zip,
                                             via io.BytesIO -- nothing written to disk here).
      save_load_latency_ms               -- wall-clock time to serialize + deserialize once,
                                             a proxy for live-serving cold-start cost.
      model                              -- the trained PPO object itself, for a caller that
                                             wants to inspect it further (e.g. via
                                             _greedy_ppo_action) -- NOT persisted anywhere by
                                             this function.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # target_atr_mult/stop_atr_mult: explicit override, applied instead of rl.rl_atr_mults'
    # interval/pair default -- same "explicit override for sweeping a candidate value" pattern
    # main.py's POST /backtest/optimize already uses for RuleConfig's own target/stop. Omit
    # (recommended for real training) to use rl_atr_mults(interval, pair) as normal; pass both
    # only when comparing a candidate (target, stop) pair against the current default, e.g.
    # investigating a high-expired-rate interval/pair via GET /rl/resolution-stats first.
    if target_atr_mult is None or stop_atr_mult is None:
        target_atr_mult, stop_atr_mult = rl.rl_atr_mults(interval, pair)
    df = df.reset_index(drop=True)
    min_warmup = max(config.ema_slow, rl.MIN_WARMUP_BARS)
    split_idx = int(len(df) * train_frac)
    train_last, test_start, test_last = rl.chronological_train_test_split(df, train_frac, max_lookforward, min_warmup)

    indicator_df, market_states = _build_market_states(df, config, pair, interval, split_idx, ml_reference_signals)

    def make_env():
        return ForexTradingEnv(
            df, indicator_df, market_states, pair, min_warmup, train_last,
            max_lookforward, target_atr_mult, stop_atr_mult, starting_balance,
        )

    vec_env = DummyVecEnv([make_env])
    kwargs = dict(ppo_kwargs or {})
    # warm_start_model_bytes: continue training an existing policy instead of starting a
    # fresh randomly-initialized one every call -- stable-baselines3's documented resume
    # pattern (PPO.load then .learn(reset_num_timesteps=False)), the "real analog to the
    # retired linear policy's warm start" the original Signal Stack v2 plan called out.
    # Re-attaches THIS call's env (fresh candle data, possibly a different target/stop) to
    # the loaded model rather than reusing whatever env it was saved with. Falls back to a
    # fresh model exactly like before when omitted (e.g. no prior policy exists yet, or the
    # caller decided the prior one's feature schema is stale -- see main.py's caller).
    warm_started = warm_start_model_bytes is not None
    if warm_started:
        model = PPO.load(io.BytesIO(warm_start_model_bytes), env=vec_env, device="cpu", **kwargs)
        model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False)
    else:
        model = PPO("MlpPolicy", vec_env, seed=random_seed, verbose=0, **kwargs)
        model.learn(total_timesteps=total_timesteps)

    # Greedy (deterministic) PPO evaluation on the untouched test slice.
    def ppo_action_fn(obs: np.ndarray, flat: bool) -> str:
        return select_action(_action_probs(model, obs), flat)

    ppo_run_id = uuid.uuid4().hex[:12]
    ppo_eval_run, ppo_trade_signals = _evaluate_policy(
        ppo_action_fn, df, indicator_df, market_states, pair, interval, test_start, test_last,
        max_lookforward, target_atr_mult, stop_atr_mult, starting_balance, ppo_run_id,
    )

    # Random baseline -- same test slice, same _evaluate_policy tallying, uniformly random
    # action choice instead of a trained policy. This is the actual "beats random" gate.
    rng = np.random.default_rng(random_seed)

    def random_action_fn(_obs: np.ndarray, _flat: bool) -> str:
        return rng.choice(rl.ACTIONS)

    random_run_id = uuid.uuid4().hex[:12]
    random_eval_run, _random_trade_signals = _evaluate_policy(
        random_action_fn, df, indicator_df, market_states, pair, interval, test_start, test_last,
        max_lookforward, target_atr_mult, stop_atr_mult, starting_balance, random_run_id,
    )

    # Save/load size + latency -- a live-serving proxy check (step 3's "workable for live
    # serving" half), entirely in-memory (io.BytesIO) so this never touches disk.
    buffer = io.BytesIO()
    t0 = time.perf_counter()
    model.save(buffer)
    save_ms = (time.perf_counter() - t0) * 1000
    model_size_bytes = buffer.getbuffer().nbytes
    buffer.seek(0)
    t0 = time.perf_counter()
    PPO.load(buffer, device="cpu")
    load_ms = (time.perf_counter() - t0) * 1000

    return {
        "ppo_eval_run": ppo_eval_run,
        "ppo_trade_signals": ppo_trade_signals,
        "random_baseline_eval_run": random_eval_run,
        "beats_random": ppo_eval_run.total_return_pct > random_eval_run.total_return_pct,
        "model_size_bytes": model_size_bytes,
        "save_load_latency_ms": round(save_ms + load_ms, 2),
        "model": model,
        "warm_started": warm_started,
    }


def train_ppo_policy(
    df: pd.DataFrame, pair: str, interval: str, config: RuleConfig = RuleConfig(),
    total_timesteps: int = 50_000, train_frac: float = 0.7, max_lookforward: int = 20,
    starting_balance: float = rl.DEFAULT_STARTING_BALANCE,
    ml_reference_signals: Optional[list[dict]] = None, random_seed: Optional[int] = None,
    ppo_kwargs: Optional[dict] = None,
    target_atr_mult: Optional[float] = None, stop_atr_mult: Optional[float] = None,
    warm_start_policy_id: Optional[str] = None, warm_start_model_bytes: Optional[bytes] = None,
) -> tuple[PPOPolicy, BacktestRun, list[Signal], dict]:
    """
    The live-persistence wrapper around train_and_evaluate_ppo_poc -- same training/evaluation,
    plus serializing the trained model into a PPOPolicy ready for a caller (main.py's
    POST /rl/train/{interval}) to insert into ppo_policies_collection. Kept as a thin
    wrapper rather than folding serialization into train_and_evaluate_ppo_poc itself, since a
    caller doing a quick POC check has no reason to pay for serializing a model it's about to
    discard.

    warm_start_policy_id/warm_start_model_bytes: pass both together (the caller's job -- fetch
    the most recent PPOPolicy for this pair/interval, check its feature_names still match
    rl.RL_FEATURE_NAMES, same staleness guard choose_action_ppo already uses for live
    inference) to continue training that policy (stable-baselines3's PPO.load +
    .learn(reset_num_timesteps=False)) instead of starting a fresh randomly-initialized model
    -- a real analog to the retired linear policy's warm start. Omit both (the caller has no
    usable prior policy) for a fresh model, same as every training call before this existed.
    Caution: warm-starting FROM a policy that's degenerated to always-HOLD continues learning
    from that same stuck point rather than escaping it via a fresh random initialization --
    if a pair/interval's policy is currently all-HOLD, a deliberate fresh retrain (omit both
    args, or POST /rl/reset) may be needed before warm-starting is safe to rely on again.

    Returns (policy, eval_run, trade_signals, poc_diagnostics) -- poc_diagnostics carries
    beats_random/model_size_bytes/save_load_latency_ms/warm_started forward from
    train_and_evaluate_ppo_poc so a caller (or the API response) can still see them even
    though the model itself is now serialized into `policy` rather than returned raw.

    target_atr_mult/stop_atr_mult: see train_and_evaluate_ppo_poc's own comment -- passed
    straight through, omit for normal training.
    """
    result = train_and_evaluate_ppo_poc(
        df, pair, interval, config, total_timesteps=total_timesteps, train_frac=train_frac,
        max_lookforward=max_lookforward, starting_balance=starting_balance,
        ml_reference_signals=ml_reference_signals, random_seed=random_seed, ppo_kwargs=ppo_kwargs,
        target_atr_mult=target_atr_mult, stop_atr_mult=stop_atr_mult,
        warm_start_model_bytes=warm_start_model_bytes,
    )
    model: PPO = result["model"]
    buffer = io.BytesIO()
    model.save(buffer)

    policy = PPOPolicy(
        policy_id=uuid.uuid4().hex[:12],
        pair=pair,
        interval=interval,
        created_at=datetime.utcnow(),
        feature_names=rl.RL_FEATURE_NAMES,
        eval_run_id=result["ppo_eval_run"].run_id,
        starting_balance=starting_balance,
        total_timesteps=total_timesteps,
        model_bytes=buffer.getvalue(),
        warm_started_from=warm_start_policy_id if result["warm_started"] else None,
        exit_action_enabled=True,
    )
    poc_diagnostics = {
        "beats_random": result["beats_random"],
        "model_size_bytes": result["model_size_bytes"],
        "save_load_latency_ms": result["save_load_latency_ms"],
        "random_baseline_total_return_pct": result["random_baseline_eval_run"].total_return_pct,
        "warm_started": result["warm_started"],
    }
    return policy, result["ppo_eval_run"], result["ppo_trade_signals"], poc_diagnostics


def choose_action_ppo(policy: PPOPolicy, state: list[float], flat: bool = False) -> tuple[str, dict[str, float]]:
    """
    Live-inference for a persisted PPOPolicy -- called by main.py's POST /rl/signal/{interval}.
    Same staleness guard this project's retired linear choose_action used (a policy trained
    under an older/different state schema raises rather than silently producing a meaningless
    result), same (action, action -> float) return shape.

    Loads the model fresh from policy.model_bytes on every call rather than caching a loaded
    model across requests -- see train_and_evaluate_ppo_poc's own save/load latency
    measurement (tens of milliseconds) for why this is workable for a live signal endpoint
    that's called at most a few times a minute, not a hot path needing a persistent in-memory
    model cache.
    """
    if policy.feature_names != rl.RL_FEATURE_NAMES:
        raise ValueError(
            f"PPO policy {policy.policy_id} for {policy.pair}/{policy.interval} was trained "
            f"against a different state feature set ({len(policy.feature_names)} features, "
            f"current code produces {len(rl.RL_FEATURE_NAMES)}) -- stale after a feature-set "
            f"change, retrain via POST /rl/train/{policy.interval}?pair={policy.pair} first."
        )
    model = PPO.load(io.BytesIO(policy.model_bytes), device="cpu")
    obs = np.asarray(state, dtype=np.float32)
    return _greedy_ppo_action(model, obs, flat)
