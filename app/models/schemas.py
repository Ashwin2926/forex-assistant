from pydantic import BaseModel
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
    profile: Literal["intraday", "swing"]
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


class StrategyCall(BaseModel):
    """
    One strategy's independent verdict at a single bar -- the unit the consensus engine votes
    over (see app/services/consensus.py). Every strategy in app/services/strategies.py returns
    one of these with the same shape (including `reasons`, reusing SignalReason so every
    strategy stays explainable and ML-feature-ready the same way the trend strategy already
    is, not just a bare direction).
    """
    strategy: str  # "trend" | "bollinger" | "support_resistance" | "candlestick" | "stoch_adx" | "volume_momentum" | "smart_money"
    direction: Literal["BUY", "SELL", "HOLD"]
    entry_price: float
    target_price: Optional[float] = None  # None when direction == HOLD
    stop_price: Optional[float] = None
    reasons: list[SignalReason]


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
    train/test split, not hit_rate/expectancy against a rule-config). feature_coefficients is
    what a LogisticRegression actually leaned on, for the same interpretability every other
    part of this project already treats as non-negotiable (SignalReason.detail, rule_stats).
    """
    run_id: str
    created_at: datetime
    train_samples: int
    test_samples: int
    train_accuracy: float
    test_accuracy: float
    test_precision: Optional[float] = None
    test_recall: Optional[float] = None
    feature_coefficients: dict[str, float] = {}
    test_calibration: list[MLCalibrationBucket] = []


class RLPolicy(BaseModel):
    """
    The learned brain of the RL agent (app/services/rl_engine.py) -- a linear Q-function, one
    weight vector per action (BUY/SELL/HOLD), over the 7 strategies' votes plus atr_pct.
    Persisted because, unlike the ML classifier's sub-second refit on ~270 rows, training
    (many epsilon-greedy episodes over thousands of candles) isn't cheap enough to redo on
    every request -- predict-time just loads the latest one and picks the greedy action.
    """
    policy_id: str
    pair: str
    interval: str
    created_at: datetime
    episodes: int
    train_frac: float
    weights: dict[str, list[float]]  # action -> weight vector, same order as feature_names
    feature_names: list[str]
    eval_run_id: str  # the BacktestRun (profile="rl") that evaluated this policy on the test slice


class RLSignal(BaseModel):
    """
    A signal generated by the RL policy's greedy action -- a third independent signal source
    alongside Signal (rule engine) and ConsensusSignal (fixed-threshold vote), same
    "separate, clearly-labeled, easy to remove" principle. q_values is this signal's
    interpretability surface (which action the policy favored and by how much), the same
    role feature_coefficients/strategy_calls play elsewhere. HOLD never produces a stored
    RLSignal, same as ConsensusSignal.
    """
    pair: str
    interval: str
    timestamp: datetime
    direction: Literal["BUY", "SELL"]
    entry_price: float
    target_price: float
    stop_price: float
    q_values: dict[str, float]
    policy_id: str

    status: Literal["pending", "hit", "miss", "expired"] = "pending"
    outcome_price: Optional[float] = None
    outcome_timestamp: Optional[datetime] = None
    outcome_pct_move: Optional[float] = None
    candles_to_outcome: Optional[int] = None
    source: Literal["live", "backtest"] = "live"


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
    profile: Literal["intraday", "swing"]
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
