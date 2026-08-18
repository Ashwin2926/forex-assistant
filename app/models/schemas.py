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
