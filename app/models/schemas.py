from pydantic import BaseModel, ConfigDict
from datetime import datetime
from typing import Optional, Literal


class LoginRequest(BaseModel):
    username: str
    password: str


class Candle(BaseModel):
    pair: str
    interval: str  # e.g. "5min", "1h", "4h"
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None


class RuleConfig(BaseModel):
    """
    Every tunable threshold in the rule engine, in one place, so a parameter sweep
    can vary them and compare backtested hit-rate without touching code. Defaults
    match the original hardcoded values.
    """
    ema_fast: int = 50
    ema_slow: int = 200
    rsi_period: int = 14
    rsi_oversold: float = 30
    rsi_overbought: float = 70
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    atr_period: int = 14
    volatility_threshold_pct: float = 0.02  # ATR as % of price; below this, market is considered too quiet

    # Session-awareness (intraday only) — outside the window, force HOLD regardless of
    # votes. Default window is the London/NY overlap (12:00-16:00 UTC), the highest-liquidity
    # stretch for majors; Asian-session hours tend to be too quiet for intraday setups.
    session_filter_enabled: bool = False
    session_start_hour_utc: int = 12
    session_end_hour_utc: int = 16

    # Target/stop distance as a multiple of ATR(14) at signal time. Part of RuleConfig (not
    # a separate endpoint param) specifically so /backtest/optimize can search it alongside
    # everything else — the breakeven hit-rate is stop/(target+stop), so these two numbers
    # determine how good a hit-rate actually needs to be before a config can be profitable,
    # not just "right often."
    target_atr_mult: float = 1.5
    stop_atr_mult: float = 1.0


class SignalReason(BaseModel):
    rule: str
    passed: bool
    detail: str
    # The single most decision-relevant number behind this rule's verdict — RSI reading,
    # normalized EMA spread (%), MACD histogram, ATR%, or session hour, depending on the
    # rule. `detail` stays human-readable prose; `value` exists so this data doesn't have
    # to be re-parsed out of that prose later for backtesting analysis or as an ML feature.
    # Optional/defaulted so older stored signals (no `value`) still validate.
    value: Optional[float] = None


class Signal(BaseModel):
    pair: str
    profile: Literal["intraday"]
    interval: str
    timestamp: datetime
    direction: Literal["BUY", "SELL", "HOLD"]
    confidence: float  # 0-100, based on how many rules aligned
    reasons: list[SignalReason]
    price_at_signal: float

    # Filled in later once we know what happened (for scoring accuracy)
    status: Literal["pending", "hit", "miss", "expired"] = "pending"
    outcome_price: Optional[float] = None
    outcome_timestamp: Optional[datetime] = None
    outcome_pct_move: Optional[float] = None

    # Backtest-only fields — live signals leave these at their defaults.
    source: Literal["live", "backtest"] = "live"
    run_id: Optional[str] = None
    target_price: Optional[float] = None
    stop_price: Optional[float] = None
    candles_to_outcome: Optional[int] = None

    # RL sizing-aware trade log only (ppo_engine.py's test-slice trades, persisted to
    # backtest_signals_collection the same way regular backtest signals are) -- None for
    # every other Signal. Same "reuse the existing model, add optional fields" precedent as
    # BacktestRun's starting_balance/ending_balance/total_return_pct.
    size_tier: Optional[Literal["SMALL", "LARGE"]] = None
    risk_fraction: Optional[float] = None
    balance_at_signal: Optional[float] = None
    position_size_units: Optional[float] = None

    # Set by create_signal's ML quality gate (main.py) -- the calibrated hit-probability
    # (app/services/ml_model.py) for whatever direction the rule engine picked, populated
    # whenever a directional evaluation happened (BUY/SELL, even one that passed the gate).
    # None for a signal the rule engine itself decided was HOLD (nothing to score), or one
    # generated before this field existed.
    ml_hit_probability: Optional[float] = None
    # Set only when the gate downgraded a would-be BUY/SELL to HOLD because
    # ml_hit_probability fell below GOOD_SIGNAL_ML_THRESHOLD -- kept on the record itself
    # (not just the API response) so a blocked signal is auditable later via GET /signals,
    # same "inspectable, not silently overriding" precedent as RLSignal.memory_override.
    ml_override: Optional[str] = None


class StrategyCall(BaseModel):
    """
    One strategy's independent verdict at a single bar -- the unit the consensus engine votes
    over (see app/services/consensus.py). Every strategy in app/services/strategies.py returns
    one of these with the same shape (including `reasons`, reusing SignalReason so every
    strategy stays explainable and ML-feature-ready the same way the trend strategy already
    is, not just a bare direction).
    """
    strategy: str  # "market_structure" | "order_blocks" | "fair_value_gap" | "liquidity_sweep" | "supply_demand"
    direction: Literal["BUY", "SELL", "HOLD"]
    entry_price: float
    target_price: Optional[float] = None  # None when direction == HOLD
    stop_price: Optional[float] = None
    reasons: list[SignalReason]
    # How strong THIS bar's reading is for this strategy, normalized to roughly [0, 1] --
    # 0/None when direction is HOLD (no conviction to grade). Each strategy defines its own
    # normalization against its own natural scale (see strategies.py) since e.g. ADX's 0-100
    # range and a wick/body ratio aren't comparable without one. Consumed by the RL agent
    # (rl_engine.py) as vote strength instead of a flat +-1; consensus doesn't currently use
    # this (its threshold is headcount/weight-based, not per-vote strength).
    strength: Optional[float] = None


class ConsensusSignal(BaseModel):
    """
    Fires only when a weighted majority of strategies (see consensus.py's STRATEGY_WEIGHTS/
    REQUIRED_WEIGHT_FRACTION) agree on direction AND their entry/exit prices land within
    PROXIMITY_ATR_MULT of each other -- a separate, additive layer on top of the
    single-strategy Signal model above, not a replacement. entry_price/target_price/
    stop_price are the mean of the agreeing strategies' own numbers.
    """
    pair: str
    interval: str
    timestamp: datetime
    direction: Literal["BUY", "SELL"]  # HOLD never produces a consensus signal
    entry_price: float
    target_price: float
    stop_price: float
    agreeing_count: int
    strategy_calls: list[StrategyCall]  # all 5, so disagreement is visible too, not just the winners

    status: Literal["pending", "hit", "miss", "expired"] = "pending"
    outcome_price: Optional[float] = None
    outcome_timestamp: Optional[datetime] = None
    outcome_pct_move: Optional[float] = None
    candles_to_outcome: Optional[int] = None

    source: Literal["live", "backtest"] = "live"
    run_id: Optional[str] = None


class RuleStat(BaseModel):
    """How often a rule fired, and how often signals it agreed with turned out to hit."""
    rule: str
    fired_count: int
    directional_signals_agreed: int  # signals where this rule's vote matched the final direction
    hits_when_agreed: int
    hit_rate_when_agreed_pct: Optional[float] = None


class BacktestRun(BaseModel):
    run_id: str
    pair: str
    interval: str
    profile: str
    created_at: datetime

    # Parameters used for this run
    rule_config: RuleConfig = RuleConfig()
    target_atr_mult: float
    stop_atr_mult: float
    max_lookforward: int

    # Volume
    candles_evaluated: int
    total_signals: int
    hold_signals: int
    directional_signals: int  # BUY + SELL, excludes HOLD

    # Outcome breakdown (directional signals only)
    hits: int
    misses: int
    expired: int
    hit_rate_pct: Optional[float] = None

    # Move sizing
    avg_win_pct: Optional[float] = None
    avg_loss_pct: Optional[float] = None

    # Expected value per directional signal — mean pct_move across ALL of them (hit, miss,
    # AND expired, unlike avg_win_pct/avg_loss_pct which only average within their own
    # bucket). This is what actually answers "is this profitable on average," since a low
    # hit-rate config can still have positive expectancy if wins are much bigger than losses.
    expectancy_pct: Optional[float] = None

    # Confidence calibration — does higher confidence actually mean more hits?
    avg_confidence_hit: Optional[float] = None
    avg_confidence_miss: Optional[float] = None

    rule_stats: list[RuleStat] = []

    # RL sizing-aware runs only (profile="rl_ppo", see ppo_engine.py; older documents may
    # carry the retired linear policy's profile="rl") -- None for every other backtest type.
    # expectancy_pct above is still a flat per-trade average;
    # these three track the actual compounding walk (starting_balance -> ending_balance),
    # since sizing makes growth path-dependent rather than reducible to a mean.
    starting_balance: Optional[float] = None
    ending_balance: Optional[float] = None
    total_return_pct: Optional[float] = None


class MLCalibrationBucket(BaseModel):
    """
    One predicted-probability range's actual outcome rate on the held-out test slice --
    answers "does a signal the model scores 60-70% actually land above the ~50% baseline,"
    which accuracy/precision/recall alone don't (a model can be 55% accurate overall while
    still being a genuinely useful filter at its high-confidence end, or vice versa).
    """
    range_label: str  # e.g. "60-70%"
    count: int
    actual_hit_rate_pct: float


class MLTrainResult(BaseModel):
    """
    Result of training the supervised hit/miss classifier (app/services/ml_model.py) on the
    current set of resolved live signals -- a sibling to BacktestRun, not a reuse of it: the
    metrics are genuinely different (accuracy/precision/recall against a chronological
    train/test split, not hit_rate/expectancy against a rule-config). feature_importances is
    what an XGBoost classifier actually leaned on, for the same interpretability every other
    part of this project already treats as non-negotiable (SignalReason.detail, rule_stats) --
    named honestly as gain-based IMPORTANCE, not signed coefficients: unlike the prior
    LogisticRegression's coef_, XGBoost's feature_importances_ has no sign or "pushes toward
    hit vs. away from it" direction, only "how much the model relied on this feature."
    """
    run_id: str
    created_at: datetime
    train_samples: int
    test_samples: int
    train_accuracy: float
    test_accuracy: float
    test_precision: Optional[float] = None
    test_recall: Optional[float] = None
    feature_importances: dict[str, float] = {}
    test_calibration: list[MLCalibrationBucket] = []
    # True when /ml/train found the resolved-signal count unchanged since the last run and
    # returned that prior result as-is instead of refitting on identical data (see
    # /ml/train's own docstring) -- lets a caller (the cron, or a human) tell "retrained on
    # more data" apart from "nothing new to learn from yet" instead of silently getting a
    # byte-identical result either way.
    skipped: bool = False


# NOTE: this project's RL agent was linear Q-learning before PPO (see PPOPolicy, below)
# replaced it -- that policy's schema (RLPolicy: a weight vector per action, Adagrad's
# accumulated-squared-gradient state, warm-start lineage) is retired along with the algorithm
# itself; see git history if it's ever needed for reference. rl_policies_collection may still
# hold old documents shaped like it, but nothing in this codebase reads or writes them anymore.


class PPOPolicy(BaseModel):
    """
    Signal Stack v2 phase 3 -- this project's second RL agent, PPO (stable-baselines3) via a
    gymnasium.Env adapter (app/services/ppo_engine.py), trained against the exact same replay
    mechanics RLPolicy's linear Q-learning already uses (_take_action_sized's reward logic,
    compute_strategy_vote_states/compute_ml_scores/full_rl_state's market state) -- only the
    learning algorithm differs. Kept as its own model/collection (ppo_policies_collection),
    not merged into RLPolicy, since the two have genuinely different shapes: a PPO policy has
    no per-feature weight vector to persist, only a serialized neural network.

    model_bytes is stable_baselines3's own model.save() output (a zip archive) persisted as
    raw bytes -- FastAPI Cloud containers aren't guaranteed persistent local disk across
    restarts/redeploys, the same reasoning RLPolicy already persists its weights into Mongo
    rather than a pickle file on disk. feature_names exists for the same staleness guard
    rl_engine.choose_action already performs for the linear policy (see
    ppo_engine.choose_action_ppo) -- a policy trained under an older/different state schema
    shouldn't silently load as if it still matched RL_FEATURE_NAMES.
    """
    # pydantic reserves the "model_" attribute prefix for its own API by default and warns on
    # a field name colliding with it -- protected_namespaces=() opts out, since "model_bytes"
    # (stable-baselines3's own model.save() output) is the honest name for this field, not
    # something worth renaming around a pydantic default.
    model_config = ConfigDict(protected_namespaces=())

    policy_id: str
    pair: str
    interval: str
    created_at: datetime
    feature_names: list[str]
    eval_run_id: str  # the BacktestRun (profile="rl_ppo") that evaluated this policy on the test slice
    starting_balance: float
    total_timesteps: int
    model_bytes: bytes


class RLSignal(BaseModel):
    """
    A signal generated by an RL policy's greedy action -- a third independent signal source
    alongside Signal (rule engine) and ConsensusSignal (fixed-threshold vote), same
    "separate, clearly-labeled, easy to remove" principle. q_values is this signal's
    interpretability surface -- for the linear policy (algo="linear_q"), the actual per-action
    Q-values; for PPO (algo="ppo"), the policy's action-probability distribution instead (see
    ppo_engine.choose_action_ppo) -- structurally identical (action -> float), so this field
    is shared rather than duplicated per algorithm, the same role feature_importances/
    strategy_calls play elsewhere. HOLD never produces a stored RLSignal, same as
    ConsensusSignal.
    """
    # Which RL agent produced this signal -- policy_id alone would require a join against
    # either rl_policies_collection or ppo_policies_collection to tell them apart; this makes
    # it a plain field lookup instead. Defaults to "linear_q" so every signal stored before
    # PPO existed still validates as exactly what it was.
    algo: Literal["linear_q", "ppo"] = "linear_q"
    # A stable business key (uuid hex, same convention as policy_id/run_id/job_id elsewhere)
    # -- needed so GET /rl/signals/{signal_id}/explain (case_memory.py) can look one up
    # without exposing/parsing raw Mongo ObjectIds, which nothing else in this project does.
    signal_id: str
    pair: str
    interval: str
    timestamp: datetime
    direction: Literal["BUY", "SELL"]
    entry_price: float
    target_price: float
    stop_price: float
    q_values: dict[str, float]
    policy_id: str
    # The exact RL_FEATURE_NAMES-ordered state vector this decision was made from -- the raw
    # material for case_memory.py's "have we seen something like this before, and how did it
    # turn out" lookup. Empty for any signal generated before this field existed (state-less
    # legacy records) -- case_memory simply skips those as candidates, never as an error.
    state: list[float] = []
    # Set when case_memory.memory_gate() downsized or blocked this trade against what the raw
    # Q-policy action would have been -- None means memory either had nothing to say (too few
    # similar cases) or agreed with the policy's own choice. Kept on the record itself (not
    # just the API response) so a downsized/blocked trade is auditable later via GET
    # /rl/signals the same way every other decision here already is.
    memory_override: Optional[str] = None

    # What the agent chose to risk, and against what balance -- makes the trade log (and the
    # live signal itself) fully auditable now that sizing isn't a separate, fixed calculator.
    size_tier: Literal["SMALL", "LARGE"]
    risk_fraction: float
    balance_at_signal: float
    position_size_units: float

    # "superseded" is distinct from "expired": expired means label_outcome walked the full
    # max_lookforward window and neither target nor stop was touched (a real "the market
    # didn't move enough" outcome). superseded means a later decision at the same
    # pair/interval disagreed with this one before it ever got that chance (see
    # _supersede_pending_rl_signal in main.py) -- not a real win/loss/timeout, just the
    # agent changing its mind. Conflating the two into one "expired" bucket was making
    # GET /rl/accuracy's hit rate look far worse than the agent's actual resolved
    # performance, since a fast-moving policy on a volatile interval can supersede most of
    # its own signals well before label_outcome would ever have judged them.
    status: Literal["pending", "hit", "miss", "expired", "superseded"] = "pending"
    outcome_price: Optional[float] = None
    outcome_timestamp: Optional[datetime] = None
    outcome_pct_move: Optional[float] = None
    candles_to_outcome: Optional[int] = None
    source: Literal["live", "backtest"] = "live"


class RLInsightFinding(BaseModel):
    """
    One deterministic, explainable "what to improve" finding from GET /rl/insights --
    synthesized from data this project already collects (trained-policy backtest evals, live
    resolution breakdowns, the learning-curve verdict), not an LLM call. Same "explainable,
    not black-box" ethos as SignalReason/feature_importances elsewhere in this project.
    pair/interval are None for a system-wide finding (e.g. the overall learning verdict).
    """
    severity: Literal["good", "warning", "critical"]
    pair: Optional[str] = None
    interval: Optional[str] = None
    title: str
    detail: str


class RLTrainAllCell(BaseModel):
    """One pair/interval's outcome within an RLTrainAllJob -- mirrors what the single
    POST /rl/train endpoint returns, flattened for the progress table."""
    pair: str
    interval: str
    ok: bool
    error: Optional[str] = None
    policy_id: Optional[str] = None
    hit_rate_pct: Optional[float] = None
    expectancy_pct: Optional[float] = None
    directional_signals: Optional[int] = None
    hold_signals: Optional[int] = None
    starting_balance: Optional[float] = None
    ending_balance: Optional[float] = None
    total_return_pct: Optional[float] = None


class RLTrainAllJob(BaseModel):
    """
    Tracks a "train every pair x interval" batch run server-side (app/main.py's
    run_train_all_job background task), so the ~15-minute-total run survives the
    triggering browser tab being closed, backgrounded, or losing connectivity -- the
    frontend just starts a job and polls GET /rl/train-all/{job_id} instead of holding 20
    sequential fetches open itself. Persisted in Mongo rather than kept in-process memory
    because FastAPI Cloud can recycle the instance between requests; an in-memory dict
    would silently lose progress the same way the old in-process APScheduler did (see
    .github/workflows/keep-fresh.yml's history note).
    """
    job_id: str
    status: Literal["running", "done", "cancelled"]
    created_at: datetime
    finished_at: Optional[datetime] = None
    total_timesteps: int
    train_frac: float
    starting_balance: float
    total: int
    completed: int = 0
    results: list[RLTrainAllCell] = []


class RunAllFlowsJob(BaseModel):
    """
    Tracks the manual "Sync now" catch-up job (app/main.py's _run_all_flows_job background
    task) the same way RLTrainAllJob tracks "train all" -- persisted in Mongo, not
    in-process memory, for the same FastAPI-Cloud-can-recycle-the-instance reason. Needed
    because the full ingest+generate+score+ml_train sequence across every pair/interval can
    take well over Cloudflare's ~100s proxy timeout, so it can't just be a synchronous
    request/response the way POST /signals/score alone can.

    current_step/completed_steps/total_steps are written after each checkpoint inside
    _run_all_flows_job (not just once at the end, unlike this job's original version) --
    without incremental progress, a slow-but-working run and a genuinely stuck one look
    identical from the outside (both just "running" forever), which is exactly what users
    reported ("sometimes getting stuck or not finishing") with no way to tell which one it
    actually was, or to do anything about it either way.
    """
    job_id: str
    status: Literal["running", "done", "cancelled"]
    created_at: datetime
    finished_at: Optional[datetime] = None
    results: dict = {}
    current_step: Optional[str] = None
    completed_steps: int = 0
    total_steps: int = 0
    # Set immediately (not just requested) by POST /ops/run-all-flows/{job_id}/cancel -- see
    # that endpoint's docstring for why force-immediate beats cooperative-only (a stuck step
    # would never come back around to check a flag on its own).
    cancel_requested: bool = False


class PaperTrade(BaseModel):
    """
    A live signal executed as a Deriv Multipliers contract on a virtual (demo) account.
    Multipliers are a leveraged derivative, not a 1:1 unit trade — stake/multiplier/
    take_profit_amount/stop_loss_amount describe that contract's own mechanics.
    signal_price/target_price/stop_price are kept for comparison against what our own
    rule engine predicted, but Deriv's contract is what actually executed.
    """
    pair: str
    interval: str
    profile: Literal["intraday"]
    direction: Literal["BUY", "SELL"]

    deriv_symbol: str
    signal_price: float
    target_price: float
    stop_price: float

    stake: float
    multiplier: int
    currency: str
    take_profit_amount: float
    stop_loss_amount: float

    contract_id: Optional[int] = None
    entry_spot: Optional[float] = None
    buy_price: Optional[float] = None

    status: Literal["open", "won", "lost", "error"] = "open"
    pnl: Optional[float] = None
    sell_price: Optional[float] = None

    is_virtual: bool  # always required True by paper_trading.py before a trade is placed
    deriv_loginid: str

    opened_at: datetime
    closed_at: Optional[datetime] = None
    error: Optional[str] = None
